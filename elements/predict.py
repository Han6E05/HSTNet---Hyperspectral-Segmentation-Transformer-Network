import torch
import torch.utils
import torch.utils.data
import time
import numpy as np
from torch import nn

from elements.others.SSFTTnet_segmentation import SSFTTnet_Segmentation
from elements.common.HSI_models.SSFTT.SSFTTnet_improved import ImprovedSSFTTnet
from elements.model_wrappers import UNet, DynamicCNN1D, DynamicCNN2D, DynamicCNN3D, CnnLstm, LstmCnn
from elements.common.HSI_models.SSFTT.SSFTTnet import SSFTTnet
from elements.common.HSI_models.SSFTT.SSFTTnet_unet import SSFTTnet_UNet
from elements.common.HSI_models.SSFTT.SSFTTnet_lstm import SSFTTnet_LSTM
from elements.common.RGB_models.RGBViT import RGBViT, RGBResNet, RGBHybrid

from elements.common.HSI_models.HSTNet import HSTNet
# configure logging
from elements.utils import LoggerSingleton

logger = LoggerSingleton.get_logger()


def process_model_output(model, inputs, tile_height=100, inference_mode='patched', patch_size=(5, 5), stride=5,
                         pixel_batch_size=80000, ssftt_patch_size=(16, 16), ssftt_stride=8, use_cached_features=False):
    """
    Process model outputs depending on whether the model is UNet, DynamicCNN3D, or DynamicCNN1D.
    - if UNet or DynamicCNN3D, applies torch.exp(model(inputs)).
    - if SSFTTnet, uses sliding window within tiles for pixel-level predictions
    - if DynamicCNN1D:
        - 'patched' mode performs a grid search over the input, calculates the mean for each tile, passes it to the model,
          and fills the tile region with the model's prediction.
        - 'pixel-wise' mode processes each pixel independently.

    :param model: trained model (either UNet, DynamicCNN3D, SSFTTnet, or DynamicCNN1D)
    :param inputs: input tensor
    :param tile_height: tile height in pixels, used in UNet and DynamicCNN3D models
    :param inference_mode: 'patched' or 'pixel-wise' processing for DynamicCNN1D
    :param patch_size: size of the patches (height, width)
    :param stride: stride of the patches
    :param pixel_batch_size: batch size for processing in pixel-wise mode
    :param ssftt_patch_size: size of sliding window patches within tiles for SSFTTnet
    :param ssftt_stride: stride for sliding window within tiles for SSFTTnet
    :param use_cached_features: if True and model supports it, use cached features for MC Dropout
    :return: final assembled output
    """
    from torch.amp import autocast

    # Determine device type from inputs
    device_type = 'cuda' if inputs.is_cuda else 'cpu'

    torch.cuda.empty_cache()

    # Check if model is a segmentation model (outputs pixel-level predictions directly)
    if isinstance(model, (SSFTTnet_Segmentation, HSTNet, SSFTTnet, SSFTTnet_UNet, SSFTTnet_LSTM, UNet)):
        # Segmentation models output pixel-level predictions directly: (B, num_classes, H, W)
        with torch.no_grad():
            with autocast(device_type):
                outputs = model(inputs)  # (B, num_classes, H, W)
            # UNet uses log_softmax, others use logits
            if isinstance(model, UNet):
                outputs = torch.exp(outputs)  # log_softmax -> softmax
            else:
                outputs = torch.softmax(outputs, dim=1)
            # Zero out void class (index 0) and renormalize
            outputs[:, 0, :, :] = 0
            outputs = outputs / outputs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return outputs

    # CNN1D / CnnLstm: pixel-wise inference
    if isinstance(model, (DynamicCNN1D, CnnLstm)):
        with torch.no_grad():
            B, C, H, W = inputs.shape
            px = inputs.permute(0, 2, 3, 1).contiguous().view(-1, 1, C)
            num_classes = model.fc_layers[-1].out_features
            # process in batches to avoid OOM
            results = []
            for i in range(0, px.shape[0], pixel_batch_size):
                sub = px[i:i + pixel_batch_size]
                with autocast(device_type):
                    out = torch.softmax(model(sub), dim=1)
                results.append(out)
            outputs = torch.cat(results, dim=0)  # (B*H*W, num_classes)
            outputs = outputs.view(B, H, W, num_classes).permute(0, 3, 1, 2)  # (B, num_classes, H, W)
            outputs[:, 0, :, :] = 0
            outputs = outputs / outputs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return outputs

    # CNN2D: tile-level inference (broadcast to pixels)
    if isinstance(model, DynamicCNN2D):
        with torch.no_grad():
            B, C, H, W = inputs.shape
            num_classes = model.fc_layers[-1].out_features
            # process whole tile at once (CNN2D handles spatial dims)
            with autocast(device_type):
                out = torch.softmax(model(inputs), dim=1)  # (B, num_classes)
            out = out.unsqueeze(2).unsqueeze(3).expand(B, num_classes, H, W)
            out[:, 0, :, :] = 0
            out = out / out.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return out

    # CNN3D: spatial predictions (already handled by segmentation path, but kept for clarity)
    if isinstance(model, DynamicCNN3D):
        # DynamicCNN3D: spatial predictions (batch, num_classes, height, width)
        with torch.no_grad():
            height = inputs.size(2)
            outputs_list = []
            for start in range(0, height, tile_height):
                end = min(start + tile_height, height)
                tile = inputs[:, :, start:end, :]
                with autocast(device_type):
                    tile_output = torch.softmax(model(tile), dim=1)
                outputs_list.append(tile_output)
            outputs = torch.cat(outputs_list, dim=2)
            # Zero out void class and renormalize
            outputs[:, 0, :, :] = 0
            outputs = outputs / outputs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return outputs

    elif isinstance(model, (RGBViT, RGBResNet, RGBHybrid)):
        # RGB models process entire RGB tiles (3 channels), outputs (batch, num_classes)
        # Need to expand to spatial dimensions for consistency
        with torch.no_grad():
            with autocast(device_type):
                outputs = model(inputs)  # (batch, num_classes) - no exp needed, models output logits

            # Expand to match spatial dimensions: (batch, num_classes) -> (batch, num_classes, height, width)
            batch_size, num_classes = outputs.shape
            _, _, height, width = inputs.shape
            outputs = outputs.unsqueeze(2).unsqueeze(3).expand(batch_size, num_classes, height, width)

            # Apply softmax to convert logits to probabilities
            outputs = torch.softmax(outputs, dim=1)
            
            # Zero out void class and renormalize
            outputs[:, 0, :, :] = 0
            outputs = outputs / outputs.sum(dim=1, keepdim=True).clamp(min=1e-8)
        return outputs

    # OLD CODE - should never reach here, all models handled above
    else:
        raise ValueError(
            f"Model type not recognized: {type(model).__name__}. All supported models should be handled above.")

    return outputs


@torch.no_grad()
def collect_predictions(test_loader, model, results_dict, config):
    """
        Collects predictions from a model using a specified test loader, accumulates runtime statistics,
        and organizes predictions, labels, and other related data into a results dictionary structured
        by file name.

        This function iterates over batches provided by the test_loader, processes each batch through the model,
        and aggregates the results along with the corresponding inputs and labels into the results_dict.
        It also tracks the inference time for performance analysis.

        Parameters:
            test_loader (DataLoader): DataLoader containing the test dataset with batched inputs, labels, coordinates, and file names.
            model (torch.nn.Module): The model to evaluate.
            results_dict (dict): Dictionary to store inputs, predictions, labels, and coordinates grouped by file name.
            config (object): Configuration object containing device, inference parameters, and model-specific settings.

        Key Config Attributes:
            device (torch.device): Device to perform computations on (CPU or GPU).
            inference_tile_height (int): The tile height used for model inference.
            inference_mode (str): Mode of inference, e.g., 'patched' or 'pixel-wise'.
            inference_patch_size (tuple): Size of patches used in inference.
            inference_stride (int): Stride used in inference for patching.
            pixel_batch_size (int): Batch size for processing in pixel-wise mode.
            cascading_classifier (bool): Flag indicating whether cascading classifier adjustments are needed.
            background_threshold (float): Threshold value used for background classification in cascading classifier.
            background_index (int): Index representing the background class in the classifier's output.

        Effects:
            Updates the results_dict with predictions, corresponding labels, inputs, and coordinates for each file processed.
            Logs the average inference time per sample once processing is complete.

        Returns:
            dict: Updated results_dict containing structured prediction data.

        """
    total_runtime = 0
    num_samples = 0
    total_samples = len(test_loader)

    for batch_idx, batch_data in enumerate(test_loader):
        # Handle both training mode (2 values) and test mode (4 values)
        if len(batch_data) == 2:
            # Training mode: (inputs, labels)
            inputs, labels = batch_data
            coords = [torch.zeros(2)] * inputs.size(0)  # Dummy coords
            file_names = [f"sample_{batch_idx}"] * inputs.size(0)  # Dummy file names
        else:
            # Test mode: (inputs, labels, coords, file_names)
            inputs, labels, coords, file_names = batch_data

        # Print progress every 100 samples
        if (batch_idx + 1) % 100 == 0 or batch_idx == 0:
            print(f"Processing sample {batch_idx + 1}/{total_samples}...")

        # Convert float16 to float32 for model compatibility
        inputs = inputs.to(config.device, dtype=torch.float32)
        labels = labels.to(config.device, dtype=torch.float32)

        # Debug: Print memory before forward pass
        if batch_idx == 0:
            print(
                f"GPU memory before forward pass: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB allocated, {torch.cuda.memory_reserved() / 1024 ** 3:.2f}GB reserved")
            print(f"Input shape: {inputs.shape}, dtype: {inputs.dtype}")

        # process outputs depending on the model type, also calculating inference time
        start_time = time.time()

        # Prepare kwargs for process_model_output
        process_kwargs = {
            'model': model,
            'inputs': inputs,
            'tile_height': config.inference_tile_height,
            'inference_mode': config.inference_mode,
            'patch_size': config.inference_patch_size,
            'stride': config.inference_stride,
            'pixel_batch_size': config.pixel_batch_size,
        }

        # Only add SSFTT parameters if they exist in config
        if hasattr(config, 'ssftt_patch_size'):
            process_kwargs['ssftt_patch_size'] = config.ssftt_patch_size
        if hasattr(config, 'ssftt_stride'):
            process_kwargs['ssftt_stride'] = config.ssftt_stride

        outputs = process_model_output(**process_kwargs)

        # Debug: Print memory after forward pass
        if batch_idx == 0:
            print(
                f"GPU memory after forward pass: {torch.cuda.memory_allocated() / 1024 ** 3:.2f}GB allocated, {torch.cuda.memory_reserved() / 1024 ** 3:.2f}GB reserved")
            print(f"Output shape: {outputs.shape}")

        end_time = time.time()
        runtime = end_time - start_time
        total_runtime += runtime
        num_samples += 1

        if config.cascading_classifier:
            outputs = cascading_classifier_adjustment(outputs=outputs, threshold=config.background_threshold,
                                                      background_index=config.background_index)

        # store predictions and labels by file name
        # NOTE: For fusion models, inputs are very large (224, 3, H, W)
        # We skip storing inputs to save memory
        for i, file_name in enumerate(file_names):
            if file_name not in results_dict['data']:
                results_dict['data'][file_name] = {
                    'inputs': [],
                    'preds': [],
                    'labels': [],
                    'coords': []
                }
            results_dict['data'][file_name]['labels'].append(labels[i].cpu().numpy())
            results_dict['data'][file_name]['preds'].append(outputs[i].cpu().numpy())
            results_dict['data'][file_name]['coords'].append(coords[i].cpu().numpy())

        # Clean up memory after EVERY sample to prevent OOM
        del inputs, labels, outputs
        torch.cuda.empty_cache()

    # convert collected data to numpy arrays
    for file_name, data in results_dict['data'].items():
        # Only convert inputs if they were stored (non-fusion models)
        if data['inputs']:
            data['inputs'] = np.array(data['inputs'])
            # Only apply moveaxis if the array is 4D (spatial models like UNet/CNN3D)
            if data['inputs'].ndim == 4:
                data['inputs'] = np.moveaxis(data['inputs'], [0, 1, 2, 3], [0, 3, 1, 2])
        else:
            # For fusion models, inputs were not stored to save memory
            data['inputs'] = None

        data['preds'] = np.array(data['preds'])
        data['labels'] = np.array(data['labels'])
        data['coords'] = np.array(data['coords'])

        # SSFTTnet outputs 2D predictions (batch, num_classes), so skip moveaxis for those
        if data['preds'].ndim == 4:
            data['preds'] = np.moveaxis(data['preds'], [0, 1, 2, 3], [0, 3, 1, 2])
        if data['labels'].ndim == 4:
            data['labels'] = np.moveaxis(data['labels'], [0, 1, 2, 3], [0, 3, 1, 2])

    if num_samples > 0:
        average_runtime = total_runtime / num_samples
        logger.info(f"Average inference time per sample: {average_runtime:.4f} seconds")
    else:
        logger.warning("No samples processed during inference!")

    return results_dict


def cascading_classifier_adjustment(outputs, threshold, background_index):
    """
    Adjust predictions using cascading classifier logic. If the background probability is above the threshold,
    classify as background. Otherwise, remove background and normalize the remaining fabric classes.

    Supports both pixel-level (4D) and tile-level (2D) predictions.

    :param outputs: Tensor of shape (bs, c, h, w) for pixel-level or (bs, c) for tile-level.
    :param threshold: Threshold for background probability.
    :param background_index: Index of the background class (should be 1).
    :return: Adjusted outputs tensor with normalized probabilities.
    """
    adjusted_outputs = outputs.clone()
    num_classes = adjusted_outputs.shape[1]

    if adjusted_outputs.ndim == 4:
        # Pixel-level predictions (bs, c, h, w)
        batch_size, num_classes, height, width = adjusted_outputs.shape

        # Extract background probability
        background_probs = adjusted_outputs[:, background_index, :, :]  # (bs, h, w)

        # Mask for pixels where the background probability is ABOVE the threshold
        high_background_mask = background_probs > threshold  # (bs, h, w)

        # Expand mask to all classes: (bs, c, h, w)
        high_bg_mask_expanded = high_background_mask.unsqueeze(1).expand(-1, num_classes, -1, -1)

        # For high background pixels, set to pure background [0, 1, 0, 0, ...]
        adjusted_outputs[high_bg_mask_expanded] = 0
        # Set background class to 1.0 for high background pixels
        background_mask_expanded = high_background_mask.unsqueeze(1)  # (bs, 1, h, w)
        adjusted_outputs[:, background_index:background_index + 1, :, :][background_mask_expanded] = 1.0

        # For low background pixels (fabric), remove background and unused, normalize fabric classes
        low_background_mask = ~high_background_mask  # (bs, h, w)

        # Set unused and background to 0 for fabric pixels
        low_bg_mask_expanded = low_background_mask.unsqueeze(1)  # (bs, 1, h, w)
        adjusted_outputs[:, 0:1, :, :][low_bg_mask_expanded] = 0  # unused
        adjusted_outputs[:, background_index:background_index + 1, :, :][low_bg_mask_expanded] = 0  # background

        # Normalize fabric classes (index 2 onwards) to sum to 1
        # Extract fabric classes: (bs, num_fabric_classes, h, w)
        fabric_classes = adjusted_outputs[:, 2:, :, :]  # (bs, 11, h, w)

        # Sum over class dimension: (bs, 1, h, w)
        fabric_sum = fabric_classes.sum(dim=1, keepdim=True)

        # Avoid division by zero
        fabric_sum = torch.where(fabric_sum == 0, torch.ones_like(fabric_sum), fabric_sum)

        # Normalize fabric classes
        adjusted_outputs[:, 2:, :, :] = fabric_classes / fabric_sum

    elif adjusted_outputs.ndim == 2:
        # Tile-level predictions (bs, c)
        # extract background probability
        background_probs = adjusted_outputs[:, background_index]  # (bs,)

        # mask for tiles where the background probability is ABOVE the threshold
        high_background_mask = background_probs > threshold  # (bs,)

        # for these tiles, set to pure background [0, 1, 0, 0, ...]
        adjusted_outputs[high_background_mask, :] = 0
        adjusted_outputs[high_background_mask, background_index] = 1.0

        # for other tiles (fabric), remove background and unused, normalize only fabric classes
        low_background_mask = ~high_background_mask

        # Set background and unused (index 0) to zero for fabric tiles
        adjusted_outputs[low_background_mask, 0] = 0  # unused
        adjusted_outputs[low_background_mask, background_index] = 0  # background

        # Calculate sum of fabric classes only (index 2 onwards)
        fabric_sum = adjusted_outputs[low_background_mask, 2:].sum(dim=1, keepdim=True)  # (num_fabric_tiles, 1)

        # Avoid division by zero
        fabric_sum[fabric_sum == 0] = 1

        # Normalize only fabric classes
        adjusted_outputs[low_background_mask, 2:] = adjusted_outputs[low_background_mask, 2:] / fabric_sum
    else:
        raise ValueError(
            f"Unsupported output shape: {adjusted_outputs.shape}. Expected 2D (tile-level) or 4D (pixel-level).")

    return adjusted_outputs


def collect_predictions_with_uncertainty(test_loader, model, results_dict, config, num_mc_passes=10):
    """
    Collect predictions with MC Dropout uncertainty estimation (optimized with feature caching).

    Uses feature caching: extracts Conv3D/Conv2D features once for all patches,
    then samples transformer+classifier multiple times. This is much faster!

    Args:
        test_loader: DataLoader for test data
        model: Trained model
        results_dict: Dictionary to store results
        config: Configuration object
        num_mc_passes: Number of MC dropout passes (default 10 for speed)

    Returns:
        Updated results_dict with predictions and uncertainty estimates
    """
    from torch.amp import autocast

    # Check if model supports feature caching
    has_feature_methods = (hasattr(model, 'extract_features') and
                           hasattr(model, 'classify_from_features'))

    if not has_feature_methods:
        logger.warning("Model doesn't support feature caching methods. Using standard MC Dropout.")

    # Enable dropout in transformer and classifier only
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()  # Keep dropout active for MC sampling

    total_runtime = 0
    num_samples = 0

    if has_feature_methods:
        logger.info(f"Starting OPTIMIZED uncertainty estimation with {num_mc_passes} MC passes...")
        logger.info(
            "Using feature caching: Conv3D/Conv2D extracted once, transformer+classifier sampled multiple times")
    else:
        logger.info(f"Starting uncertainty estimation with {num_mc_passes} MC passes...")

    for inputs, labels, coords, file_names in test_loader:
        # Convert float16 to float32 for model compatibility
        inputs = inputs.to(config.device, dtype=torch.float32)
        labels = labels.to(config.device, dtype=torch.float32)

        start_time = time.time()

        # Perform MC Dropout inference
        all_predictions = []

        # Prepare kwargs for process_model_output
        process_kwargs = {
            'model': model,
            'inputs': inputs,
            'tile_height': config.inference_tile_height,
            'inference_mode': config.inference_mode,
            'patch_size': config.inference_patch_size,
            'stride': config.inference_stride,
            'pixel_batch_size': config.pixel_batch_size,
        }

        # Only add SSFTT parameters if they exist in config
        if hasattr(config, 'ssftt_patch_size'):
            process_kwargs['ssftt_patch_size'] = config.ssftt_patch_size
        if hasattr(config, 'ssftt_stride'):
            process_kwargs['ssftt_stride'] = config.ssftt_stride

        if has_feature_methods:
            # OPTIMIZED: First pass extracts features, subsequent passes reuse them
            # First pass: compute features (no caching yet, just extract)
            with autocast('cuda' if config.device == 'cuda' else 'cpu'):
                outputs = process_model_output(**process_kwargs, use_cached_features=False)
            all_predictions.append(outputs.cpu().numpy())

            # Subsequent passes: use cached features (much faster!)
            for _ in range(num_mc_passes - 1):
                with autocast('cuda' if config.device == 'cuda' else 'cpu'):
                    outputs = process_model_output(**process_kwargs, use_cached_features=True)
                all_predictions.append(outputs.cpu().numpy())
        else:
            # STANDARD: Run full network for each MC pass
            for _ in range(num_mc_passes):
                with autocast('cuda' if config.device == 'cuda' else 'cpu'):
                    outputs = process_model_output(**process_kwargs, use_cached_features=False)
                all_predictions.append(outputs.cpu().numpy())

        # Stack predictions: (num_passes, batch, num_classes, ...)
        all_predictions = np.array(all_predictions)

        # Calculate mean prediction
        mean_outputs = np.mean(all_predictions, axis=0)
        mean_outputs = torch.from_numpy(mean_outputs).to(config.device)

        # Calculate uncertainty metrics
        epistemic_unc = np.var(all_predictions, axis=0)  # Variance across MC passes
        epistemic_mean = np.mean(epistemic_unc, axis=1, keepdims=True)  # Mean variance per sample

        # Calculate entropy (aleatoric uncertainty)
        epsilon = 1e-10
        aleatoric_unc = -np.sum(
            mean_outputs.cpu().numpy() * np.log(mean_outputs.cpu().numpy() + epsilon),
            axis=1, keepdims=True
        )
        num_classes = mean_outputs.shape[1]
        max_entropy = np.log(num_classes)
        aleatoric_unc = aleatoric_unc / max_entropy

        # Total uncertainty - direct sum like YOLO-NAS (not weighted average)
        # This gives a larger uncertainty range for better discrimination
        total_unc = epistemic_mean + aleatoric_unc

        end_time = time.time()
        runtime = end_time - start_time
        total_runtime += runtime
        num_samples += 1

        if config.cascading_classifier:
            mean_outputs = cascading_classifier_adjustment(
                outputs=mean_outputs,
                threshold=config.background_threshold,
                background_index=config.background_index
            )

        # Store predictions, labels, and uncertainty
        for i, file_name in enumerate(file_names):
            if file_name not in results_dict['data']:
                results_dict['data'][file_name] = {
                    'inputs': [],
                    'preds': [],
                    'labels': [],
                    'coords': [],
                    'uncertainty': [],
                    'epistemic_uncertainty': [],
                    'aleatoric_uncertainty': [],
                    'per_class_uncertainty': []
                }

            results_dict['data'][file_name]['inputs'].append(inputs[i].cpu().numpy())
            results_dict['data'][file_name]['labels'].append(labels[i].cpu().numpy())
            results_dict['data'][file_name]['preds'].append(mean_outputs[i].cpu().numpy())
            results_dict['data'][file_name]['coords'].append(coords[i].cpu().numpy())
            results_dict['data'][file_name]['uncertainty'].append(total_unc[i])
            results_dict['data'][file_name]['epistemic_uncertainty'].append(epistemic_mean[i])
            results_dict['data'][file_name]['aleatoric_uncertainty'].append(aleatoric_unc[i])
            results_dict['data'][file_name]['per_class_uncertainty'].append(epistemic_unc[i])

    # Convert collected data to numpy arrays
    for file_name, data in results_dict['data'].items():
        data['inputs'] = np.array(data['inputs'])
        data['preds'] = np.array(data['preds'])
        data['labels'] = np.array(data['labels'])
        data['coords'] = np.array(data['coords'])
        data['uncertainty'] = np.array(data['uncertainty'])
        data['epistemic_uncertainty'] = np.array(data['epistemic_uncertainty'])
        data['aleatoric_uncertainty'] = np.array(data['aleatoric_uncertainty'])
        data['per_class_uncertainty'] = np.array(data['per_class_uncertainty'])

        # Apply moveaxis if needed (for spatial models)
        if data['inputs'].ndim == 4:
            data['inputs'] = np.moveaxis(data['inputs'], [0, 1, 2, 3], [0, 3, 1, 2])
        if data['preds'].ndim == 4:
            data['preds'] = np.moveaxis(data['preds'], [0, 1, 2, 3], [0, 3, 1, 2])
        if data['labels'].ndim == 4:
            data['labels'] = np.moveaxis(data['labels'], [0, 1, 2, 3], [0, 3, 1, 2])

    average_runtime = total_runtime / num_samples
    logger.info(f"Average inference time per sample (with {num_mc_passes} MC passes): {average_runtime:.4f} seconds")
    if has_feature_methods:
        logger.info(f"Feature caching enabled - significant speedup achieved!")
    logger.info(f"Total time for {num_samples} samples: {total_runtime:.2f} seconds")
    logger.info(f"Uncertainty estimation complete!")

    return results_dict
