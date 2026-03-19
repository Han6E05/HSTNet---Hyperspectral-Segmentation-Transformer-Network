"""
Unified Hyperspectral Image Inference Pipeline

This script handles inference for both traditional models (CNN, U-Net) and SSFTT models.
Automatically detects model type and applies appropriate inference strategy.

Processes include:
1. Loading hyperspectral image
2. Save false-RGB color representation
3. Extract tiles (if needed for SSFTT models)
4. Infer the image with pretrained model
5. Show and save the prediction

Usage:
    python inference_unified.py

Author:
    [Tianhan]
"""

import os
import time
import torch
import warnings
import numpy as np
from pathlib import Path

from elements.others.load_model_utils import load_segmentation_model


# Working directory
def _get_working_dir():
    return Path(__file__).resolve().parent

data_path = BASE_DIR = Path(__file__).resolve().parent

# Configure logging
from elements.utils import LoggerSingleton
LoggerSingleton.setup_logger(_get_working_dir())
logger = LoggerSingleton.get_logger()

# Pipeline imports
from elements.load_model import initialize_model
from elements.common.data.datatypes.scandata import ScanData
from elements.preprocess import convert_hsi_to_rgb, apply_flatfield_correction
import matplotlib.pyplot as plt
from matplotlib.table import Table

# Disable specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


def extract_tiles_from_image(image, tile_size=64, stride=64):
    """
    Extract tiles from a hyperspectral image.
    
    Args:
        image: numpy array of shape (H, W, C)
        tile_size: size of each tile
        stride: stride for tile extraction
    
    Returns:
        tiles: list of tile arrays
        coords: list of tile coordinates (x1, y1, x2, y2)
    """
    height, width, channels = image.shape
    tiles = []
    coords = []
    
    for y in range(0, height - tile_size + 1, stride):
        for x in range(0, width - tile_size + 1, stride):
            tile = image[y:y+tile_size, x:x+tile_size, :]
            tiles.append(tile)
            coords.append([x, y, x+tile_size, y+tile_size])
    
    return tiles, coords


@torch.no_grad()
def process_cnn_output(model, inputs, tile_height=100, inference_mode='patched', 
                      patch_size=(5, 5), stride=5, pixel_batch_size=80000):
    """
    Process model outputs for CNN-based models (CNN1D, CNN-LSTM, etc.).
    
    Args:
        model: trained CNN model
        inputs: input tensor
        tile_height: tile height in pixels
        inference_mode: 'patched' or 'pixel-wise'
        patch_size: size of the patches
        stride: stride of the patches
        pixel_batch_size: batch size for pixel-wise mode
    
    Returns:
        outputs: processed output tensor
    """
    torch.cuda.empty_cache()

    if inference_mode == 'patched':
        batch_size, _, input_height, input_width = inputs.shape
        num_classes = model.fc_layers[-1].out_features
        outputs = torch.zeros((batch_size, num_classes, input_height, input_width), device=inputs.device)
        patch_means = []
        patch_locations = []
        
        for y in range(0, input_height - patch_size[0] + 1, stride):
            for x in range(0, input_width - patch_size[1] + 1, stride):
                patch = inputs[:, :, y:y + patch_size[0], x:x + patch_size[1]]
                patch_mean = patch.mean(dim=[2, 3]).unsqueeze(1)
                patch_means.append(patch_mean)
                patch_locations.append((y, x))
        
        patch_means = torch.cat(patch_means, dim=0)
        patch_outputs = torch.exp(model(patch_means))
        patch_outputs = patch_outputs.view(-1, num_classes, 1, 1)
        
        for i, (y, x) in enumerate(patch_locations):
            tiled_output = patch_outputs[i].repeat(1, 1, patch_size[0], patch_size[1])
            outputs[:, :, y:y + patch_size[0], x:x + patch_size[1]] = tiled_output
            
    elif inference_mode == 'pixel-wise':
        batch_size, channels, height, width = inputs.shape
        num_classes = model.fc_layers[-1].out_features
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        inputs = inputs.view(-1, channels).unsqueeze(1)
        outputs = []
        
        for i in range(0, inputs.size(0), pixel_batch_size):
            sub_batch = inputs[i:i + pixel_batch_size]
            sub_output = torch.exp(model(sub_batch))
            outputs.append(sub_output)
        
        outputs = torch.cat(outputs, dim=0)
        outputs = outputs.view(batch_size, height, width, num_classes)
        outputs = outputs.permute(0, 3, 1, 2)
    else:
        raise ValueError("Invalid inference_mode. Should be 'patched' or 'pixel-wise'.")

    return outputs


@torch.no_grad()
def process_ssftt_output(model, tiles_tensor, batch_size=32, device='cuda', is_segmentation=False, coords=None):
    """
    Process tiles through SSFTT model in batches.
    
    Args:
        model: trained SSFTT model
        tiles_tensor: tensor of shape (N, C, H, W)
        batch_size: batch size for inference (default 32)
        device: device to run inference on
        is_segmentation: whether model outputs pixel-level predictions
        coords: coordinates of tiles for diagnostic logging
    
    Returns:
        outputs: tensor of shape (N, num_classes) for tile-level
                 or (N, num_classes, H, W) for pixel-level
    """
    model.eval()
    num_tiles = tiles_tensor.shape[0]
    outputs = []
    
    logger.info(f"Processing {num_tiles} tiles in batches of {batch_size}...")
    if is_segmentation:
        logger.info("  Using pixel-level segmentation model")
    else:
        logger.info("  Using tile-level classification model")
    
    # 诊断：记录中间tile的详细信息
    mid_idx = len(tiles_tensor) // 2
    mid_coord = coords[mid_idx] if coords and len(coords) > mid_idx else None
    
    for i in range(0, num_tiles, batch_size):
        end_idx = min(i + batch_size, num_tiles)
        if (end_idx) % 200 == 0 or end_idx == num_tiles:
            logger.info(f"  Processed {end_idx}/{num_tiles} tiles...")
        
        # Process batch
        batch = tiles_tensor[i:end_idx].to(device, dtype=torch.float32)
        
        # 正确的解决方案：Per-tile min-max 归一化
        # 
        # 问题：训练数据没有做归一化，导致 S002_1（mean 0.04）是 OOD 输入
        #   - 训练集 tile mean 中位数: 0.45
        #   - S002_1 在训练集的 0.2th percentile
        #   - 固定缩放因子（7.5x）只对 S002_1 有效，对其他图片会失败
        #
        # 唯一正确的解决方案：
        #   1. 训练时：在生成 train_set.npz 时使用 per-tile 归一化
        #   2. 推理时：也使用相同的 per-tile 归一化
        #   3. 这样深色布、浅色布、不同光照条件都能正常工作
        #
        # 注意：当前模型是用未归一化的数据训练的，所以这个归一化会导致性能下降
        #       需要重新训练才能获得最佳性能
        
        # Per-tile min-max normalization: (tile - min) / (max - min)
        # 在 (B, C, H, W) 上对每个 tile 的所有像素归一化
        batch_min = batch.amin(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
        batch_max = batch.amax(dim=(1, 2, 3), keepdim=True)  # (B, 1, 1, 1)
        batch = (batch - batch_min) / (batch_max - batch_min + 1e-8)
        
        # 诊断：检查中间tile
        if i <= mid_idx < end_idx:
            batch_mid_idx = mid_idx - i
            if 0 <= batch_mid_idx < batch.size(0):
                logger.info(f"  Batch {i//batch_size}: Processing tile at index {mid_idx} (coords: {mid_coord})")
                mid_tile = batch[batch_mid_idx:batch_mid_idx+1]
                logger.info(f"    Tile shape: {mid_tile.shape}")
                logger.info(f"    Tile range after normalization: [{mid_tile.min():.3f}, {mid_tile.max():.3f}]")
                logger.info(f"    Tile mean after normalization: {mid_tile.mean():.3f}")
                logger.info(f"    Note: Using per-tile min-max normalization (requires retraining for best results)")
        
        batch_output = model(batch)
        
        # 诊断：检查中间tile的logits
        if i <= mid_idx < end_idx:
            batch_mid_idx = mid_idx - i
            if 0 <= batch_mid_idx < batch.size(0):
                mid_output = batch_output[batch_mid_idx:batch_mid_idx+1]
                logger.info(f"  Mid-tile (index {mid_idx}) raw logits shape: {mid_output.shape}")
                logger.info(f"  Mid-tile logits mean: {mid_output.mean().item():.4f}, std: {mid_output.std().item():.4f}")
                
                # 如果是分割模型，检查每个像素的预测
                if is_segmentation and mid_output.dim() == 4:
                    pixel_logits = mid_output[0, :, 32, 32]  # 中心像素
                    logger.info(f"  Center pixel logits (first 5 classes): {pixel_logits[:5].cpu().numpy()}")
        
        # Apply softmax on class dimension (dim=1)
        # Works for both pixel-level (B, C, H, W) and tile-level (B, C)
        batch_output = torch.softmax(batch_output, dim=1)
        
        # 诊断：检查中间tile的softmax输出
        if i <= mid_idx < end_idx:
            batch_mid_idx = mid_idx - i
            if 0 <= batch_mid_idx < batch_output.size(0):
                mid_softmax = batch_output[batch_mid_idx]
                if is_segmentation:
                    # 对于分割模型，检查中心像素
                    center_softmax = mid_softmax[:, 32, 32] if mid_softmax.dim() == 4 else mid_softmax
                    logger.info(f"  Mid-tile center pixel softmax (first 5 classes): {center_softmax[:5].cpu().numpy()}")
                else:
                    logger.info(f"  Mid-tile softmax (first 5 classes): {mid_softmax[:5].cpu().numpy()}")
        
        outputs.append(batch_output.cpu())
        
        # Clear cache after each batch
        torch.cuda.empty_cache()
    
    outputs = torch.cat(outputs, dim=0)
    
    # 诊断：分析中间tile的预测结果
    if is_segmentation:
        mid_output = outputs[mid_idx] if mid_idx < len(outputs) else None
        if mid_output is not None:
            logger.info(f"\n=== 诊断信息：中间tile (index {mid_idx}, coords: {mid_coord}) ===")
            logger.info(f"Output shape: {mid_output.shape}")
            
            # 如果是分割模型，检查argmax分布
            if mid_output.dim() == 4:  # (C, H, W)
                pred_class = mid_output.argmax(dim=0)  # (H, W)
                unique_classes, counts = torch.unique(pred_class, return_counts=True)
                logger.info(f"  Unique predicted classes: {unique_classes.tolist()}")
                logger.info(f"  Class distribution: {dict(zip(unique_classes.tolist(), counts.tolist()))}")
                
                # 检查中心像素
                center_class = pred_class[32, 32].item()
                logger.info(f"  Center pixel (32,32) class: {center_class}")
                
                # 检查每个类的概率
                center_probs = mid_output[:, 32, 32]  # (C,)
                top5_probs, top5_indices = torch.topk(center_probs, k=5)
                logger.info(f"  Center pixel top-5 classes: {top5_indices.tolist()}")
                logger.info(f"  Center pixel top-5 probs: {top5_probs.tolist()}")
    
    # 诊断：检查模型输出的多样性
    logger.info(f"\n=== 模型输出多样性诊断 ===")
    if is_segmentation and outputs.dim() == 4:  # (N, C, H, W)
        # 检查几个随机tile的输出是否相似
        sample_indices = [0, mid_idx, min(mid_idx//2, len(outputs)-1), min(mid_idx*3//2, len(outputs)-1)]
        for idx in sample_indices:
            if idx < len(outputs):
                tile_output = outputs[idx]
                # 检查中心像素的logits分布
                center_pixel = tile_output[:, 32, 32]  # (C,)
                top3_probs, top3_indices = torch.topk(center_pixel, k=3)
                logger.info(f"  Tile {idx} (coords: {coords[idx] if idx < len(coords) else 'N/A'}):")
                logger.info(f"    Top-3 classes: {top3_indices.tolist()}")
                logger.info(f"    Top-3 probs: {top3_probs.tolist()}")
        
        # 关键诊断：检查logits分布是否异常稳定
        logger.info(f"\n=== Logits分布异常性诊断 ===")
        # 检查第一个batch的第一个tile的logits分布
        if len(outputs) > 0:
            first_tile = outputs[0]
            if first_tile.dim() == 4:  # (C, H, W)
                # 检查中心像素的logits
                center_logits = first_tile[:, 32, 32]
                logger.info(f"  第一个tile中心像素logits (13个类):")
                for class_idx in range(min(13, len(center_logits))):
                    logger.info(f"    Class {class_idx}: {center_logits[class_idx]:.4f}")
                
                # 检查logits是否异常稳定（如你观察到的：background ~5.1-5.3, wool ~1.2-2.6, unused ~0.47-0.51）
                background_logit = center_logits[1].item() if len(center_logits) > 1 else 0
                wool_logit = center_logits[6].item() if len(center_logits) > 6 else 0  # wool是第6个类
                unused_logit = center_logits[0].item() if len(center_logits) > 0 else 0
                
                logger.info(f"  关键logits值:")
                logger.info(f"    Background (class 1): {background_logit:.4f}")
                logger.info(f"    Wool (class 6): {wool_logit:.4f}")
                logger.info(f"    Unused (class 0): {unused_logit:.4f}")
                
                # 检查是否出现你观察到的异常模式
                if 5.0 <= background_logit <= 5.5 and 1.0 <= wool_logit <= 3.0 and 0.4 <= unused_logit <= 0.6:
                    logger.warning("  ⚠️  检测到异常logits分布模式！")
                    logger.warning("  模式: background ~5.1-5.3, wool ~1.2-2.6, unused ~0.47-0.51")
                    logger.warning("  这表明模型对所有输入都产生几乎相同的输出")
                    logger.warning("  可能原因：1) 模型未正确训练 2) 输入数据异常 3) 归一化问题")
    
    return outputs


def run_testing_cnn(input_tensor, model, results_dict, config, file_name, coords):
    """
    Execute testing for CNN-based models.
    """
    logger.info(f"Running CNN inference on {file_name}...")
    
    inputs = torch.stack(input_tensor, dim=0).to(config.device)
    
    start_time = time.time()
    outputs = process_cnn_output(
        model=model,
        inputs=inputs,
        tile_height=config.inference_tile_height,
        inference_mode=config.inference_mode,
        patch_size=config.inference_patch_size,
        stride=config.inference_stride,
        pixel_batch_size=config.pixel_batch_size
    )
    end_time = time.time()
    
    runtime = end_time - start_time
    logger.info(f"Inference time: {runtime:.4f} seconds")
    
    # Store predictions
    if file_name not in results_dict['data']:
        results_dict['data'][file_name] = {
            'inputs': [],
            'preds': [],
            'coords': []
        }
    
    i = 0
    results_dict['data'][file_name]['inputs'].append(inputs[i].cpu().numpy())
    results_dict['data'][file_name]['preds'].append(outputs[i].cpu().numpy())
    results_dict['data'][file_name]['coords'].append(coords[i].cpu().numpy())
    
    # Convert to numpy arrays
    for fname, data in results_dict['data'].items():
        data['inputs'] = np.array(data['inputs'])
        data['preds'] = np.array(data['preds'])
        data['coords'] = np.array(data['coords'])
        
        if data['inputs'].ndim == 4:
            data['inputs'] = np.moveaxis(data['inputs'], [0, 1, 2, 3], [0, 3, 1, 2])
            data['preds'] = np.moveaxis(data['preds'], [0, 1, 2, 3], [0, 3, 1, 2])
    
    return create_output_mapping(results_dict)


def run_testing_ssftt(tiles, coords, model, results_dict, config, file_name):
    """
    Execute testing for SSFTT models (both tile-level and pixel-level).
    """
    logger.info(f"Running SSFTT inference on {len(tiles)} tiles from {file_name}...")

    from elements.common.HSI_models.HSTNet import HSTNet

    
    is_segmentation = isinstance(model, (HSTNet))
    
    # Convert tiles to tensor (N, C, H, W)
    tiles_tensor = torch.tensor(np.array(tiles), dtype=torch.float32).permute(0, 3, 1, 2)
    
    # Convert to fusion format if needed
    if config.model_type == 'ssftt_fusion':
        logger.info("Converting tiles to fusion format (N, C, 3, H, W)...")
        # Add RGB depth dimension: (N, C, H, W) -> (N, C, 3, H, W)
        tiles_tensor = tiles_tensor.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()
        logger.info(f"Tiles shape after fusion conversion: {tiles_tensor.shape}")
    
    # For full-image processing, use batch_size=1
    batch_size = 1 if len(tiles) == 1 else config.batch_size
    
    start_time = time.time()
    outputs = process_ssftt_output(
        model=model,
        tiles_tensor=tiles_tensor,
        batch_size=batch_size,
        device=config.device,
        is_segmentation=is_segmentation,
        coords=coords  # 传递坐标用于诊断
    )
    end_time = time.time()
    
    runtime = end_time - start_time
    if len(tiles) == 1:
        logger.info(f"Inference time: {runtime:.4f} seconds (full image)")
    else:
        logger.info(f"Inference time: {runtime:.4f} seconds ({runtime/len(tiles)*1000:.2f} ms/tile)")
    
    # Store predictions
    if file_name not in results_dict['data']:
        results_dict['data'][file_name] = {
            'preds': [],
            'coords': []
        }
    
    for i, (pred, coord) in enumerate(zip(outputs, coords)):
        results_dict['data'][file_name]['preds'].append(pred.numpy())
        results_dict['data'][file_name]['coords'].append(coord)
    
    return create_output_mapping(results_dict, is_segmentation=is_segmentation)


def create_output_mapping(results_dict, is_segmentation=False):
    """
    Create output mapping from predictions.
    
    Args:
        results_dict: Dictionary containing predictions and metadata
        is_segmentation: Whether predictions are pixel-level (True) or tile-level (False)
    """
    # Create output mapping
    for file_name in results_dict['input_image_mapping'].keys():
        rgb_img = results_dict['input_image_mapping'][file_name]
        height, width, _ = rgb_img.shape
        num_classes = len(results_dict['class_names'])
        
        results_dict['output_image_mapping'][file_name] = np.zeros((height, width, num_classes), dtype=np.float32)
        
        # Count map for averaging overlapping tiles
        count_map = np.zeros((height, width), dtype=np.float32)
        
        if is_segmentation:
            logger.info(f"Creating pixel-level output mapping for {file_name}...")
            # Pixel-level predictions: accumulate and average overlapping tiles
            for i, coords in enumerate(results_dict['data'][file_name]['coords']):
                x1, y1, x2, y2 = coords
                pred = results_dict['data'][file_name]['preds'][i]  # (num_classes, H, W)
                
                # Transpose to (H, W, num_classes)
                tile_pred = np.transpose(pred, (1, 2, 0))
                
                # Accumulate predictions
                results_dict['output_image_mapping'][file_name][y1:y2, x1:x2, :] += tile_pred
                count_map[y1:y2, x1:x2] += 1
            
            # Average overlapping regions
            results_dict['output_image_mapping'][file_name] /= np.maximum(count_map[:, :, np.newaxis], 1)
            logger.info(f"  Averaged {int(count_map.max())} overlapping tiles per pixel")
        else:
            logger.info(f"Creating tile-level output mapping for {file_name}...")
            # Tile-level predictions: expand to all pixels in tile
            for i, coords in enumerate(results_dict['data'][file_name]['coords']):
                x1, y1, x2, y2 = coords
                pred = results_dict['data'][file_name]['preds'][i]
                
                # Handle different prediction shapes
                if pred.ndim == 1:  # SSFTT tile-level: (num_classes,)
                    tile_pred = np.tile(pred[:, np.newaxis, np.newaxis], (1, y2-y1, x2-x1))
                    tile_pred = np.transpose(tile_pred, (1, 2, 0))
                else:  # CNN: (num_classes, H, W)
                    tile_pred = np.transpose(pred, (1, 2, 0))
                
                # Accumulate predictions
                results_dict['output_image_mapping'][file_name][y1:y2, x1:x2, :] += tile_pred
                count_map[y1:y2, x1:x2] += 1
            
            # Average overlapping regions (important for overlapping tiles)
            results_dict['output_image_mapping'][file_name] /= np.maximum(count_map[:, :, np.newaxis], 1)
    
    # Calculate average composition
    for file_name, data in results_dict['data'].items():
        output_image = results_dict['output_image_mapping'][file_name]
        data['predicted_composition'] = np.mean(output_image, axis=(0, 1))
        
        # Log results
        avg_comp_pred = data['predicted_composition']
        comp_str_pred = " ".join([f"{value * 100:.1f}%".ljust(15) for value in avg_comp_pred])
        logger.info(f"{file_name}-pred".ljust(20) + comp_str_pred)
        
        # Log background and fabric percentages
        # Note: index 0 is unused (no void), index 1 is background
        background_pct = avg_comp_pred[1] * 100
        fabric_pct = avg_comp_pred[2:].sum() * 100
        logger.info(f"  -> Background: {background_pct:.1f}%, Fabric: {fabric_pct:.1f}%")
    
    # Write results to file
    results_file_path = os.path.join(_get_working_dir(), 'log', results_dict.get('experiment', 'inference'), 'results.txt')
    os.makedirs(os.path.dirname(results_file_path), exist_ok=True)
    
    with open(results_file_path, 'w') as f:
        f.write("\nAverage predicted composition per sample:\n")
        header = f"{'file name':<20} {' '.join([f'{name:<15}' for name in results_dict['class_names']])}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        for fname, data in results_dict['data'].items():
            avg_comp = data['predicted_composition']
            comp_str = " ".join([f"{value * 100:.1f}%".ljust(15) for value in avg_comp])
            f.write(f"{fname}-pred".ljust(20) + comp_str + "\n")
    
    # Create output dictionary
    output_dict = {
        'class_names': results_dict['class_names'],
        'file_names': list(results_dict['data'].keys())
    }
    
    for fname in results_dict['input_image_mapping'].keys():
        file_data = results_dict['data'][fname]
        output_dict[fname] = {
            'input_image_mapping': results_dict['input_image_mapping'][fname],
            'output_image_mapping': results_dict['output_image_mapping'][fname],
            'predicted_composition': file_data['predicted_composition']
        }
    
    return output_dict


def visualize_segmentation_results(converted_rgb_image, predicted_segmentation, main_title,
                                   red_info, green_info, blue_info):
    """
    Visualize RGB image and predicted segmentation side by side.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(main_title, fontsize=16)

    axes[0].imshow(converted_rgb_image)
    axes[0].set_title('Converted RGB Image')
    axes[0].axis('off')

    axes[1].imshow(predicted_segmentation)
    axes[1].set_title('Predicted Segmentation')
    axes[1].axis('off')

    # Create table for legend
    cell_text = [
        ['    ', 'Red: ' + red_info[0], 'Green: ' + green_info[0], 'Blue: ' + blue_info[0]],
        ['Pred', f"{red_info[1]:.2f}", f"{green_info[1]:.2f}", f"{blue_info[1]:.2f}"]
    ]

    ax_table = fig.add_subplot(111)
    ax_table.axis('off')
    table = Table(ax_table, bbox=[0.1, -0.3, 0.8, 0.2])

    col_width = 0.05
    row_height = 0.05

    for i, row in enumerate(cell_text):
        for j, cell in enumerate(row):
            table.add_cell(i, j, col_width, row_height, text=cell, loc='center', facecolor='white')

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    ax_table.add_table(table)

    plt.tight_layout()
    plt.show()

    return fig


def display(output_dict, output_dir, is_pixel_level=False):
    """
    Display segmentation output for each file.
    
    Args:
        output_dict: Dictionary containing predictions and metadata
        output_dir: Output directory for saving results
        is_pixel_level: Whether to use argmax visualization (True) or probability visualization (False)
    """
    class_names = output_dict['class_names']
    file_names = output_dict['file_names']

    for file_name in file_names:
        all_pred = output_dict[file_name]['predicted_composition']
        output_map = output_dict[file_name]['output_image_mapping']  # (H, W, num_classes)
        converted_rgb_image = output_dict[file_name]['input_image_mapping']
        
        if is_pixel_level:
            logger.info(f"Using argmax visualization for pixel-level model...")
            
            # Use argmax to get class per pixel
            pred_class_map = np.argmax(output_map, axis=2)  # (H, W)
            
            # Build color map for visualization
            # Use top 3 fabric classes for RGB channels
            top_indexes = np.argsort(all_pred)[::-1]
            
            # Find top 3 non-background classes
            fabric_indices = [idx for idx in top_indexes if idx > 1][:3]
            if len(fabric_indices) < 3:
                fabric_indices.extend([2, 3, 4])  # Fallback
            fabric_indices = fabric_indices[:3]
            
            red_idx, green_idx, blue_idx = fabric_indices
            
            # Create RGB visualization: each pixel colored by its predicted class
            vis_image = np.zeros((output_map.shape[0], output_map.shape[1], 3), dtype=np.uint8)
            
            # Assign colors: red channel for red_idx class, green for green_idx, blue for blue_idx
            vis_image[:, :, 0] = (pred_class_map == red_idx).astype(np.uint8) * 255
            vis_image[:, :, 1] = (pred_class_map == green_idx).astype(np.uint8) * 255
            vis_image[:, :, 2] = (pred_class_map == blue_idx).astype(np.uint8) * 255
            
            # For better visualization, also show probability-weighted version
            # This shows where the model is confident
            prob_weighted = np.zeros_like(vis_image, dtype=np.float32)
            for i, idx in enumerate([red_idx, green_idx, blue_idx]):
                prob_weighted[:, :, i] = output_map[:, :, idx]
            
            predicted_segmentation_mask = (prob_weighted * 255).astype(np.uint8)
            
        else:
            logger.info(f"Using probability visualization for tile-level model...")
            
            # Original logic for tile-level models
            top_indexes = np.argsort(all_pred)[::-1]

            # Select top 3 non-background classes for RGB visualization
            red_idx = top_indexes[0]
            if red_idx == 0 or red_idx == 1:
                red_idx = next((idx for idx in top_indexes if idx > 1), 2)
                green_idx = next((idx for idx in top_indexes if idx > 1 and idx != red_idx), 3)
                blue_idx = next((idx for idx in top_indexes if idx > 1 and idx != red_idx and idx != green_idx), 4)
            else:
                green_idx = top_indexes[1]
                if green_idx == 0 or green_idx == 1:
                    green_idx = next((idx for idx in top_indexes if idx > 1 and idx != red_idx), 3)
                    blue_idx = next((idx for idx in top_indexes if idx > 1 and idx != red_idx and idx != green_idx), 4)
                else:
                    blue_idx = top_indexes[2]
                    if blue_idx == 0 or blue_idx == 1:
                        blue_idx = next((idx for idx in top_indexes if idx > 1 and idx != red_idx and idx != green_idx), 4)
                    if blue_idx <= 1:
                        blue_idx = top_indexes[3]

            predicted_segmentation_mask = (output_map[:, :, [red_idx, green_idx, blue_idx]] * 255).astype(np.uint8)

        fabrics = [class_names[red_idx], class_names[green_idx], class_names[blue_idx]]
        pred = np.round([all_pred[red_idx], all_pred[green_idx], all_pred[blue_idx]], 2)

        red_info = (fabrics[0], pred[0])
        green_info = (fabrics[1], pred[1])
        blue_info = (fabrics[2], pred[2])

        main_title = f"Segmentation Results for {file_name}"

        fig = visualize_segmentation_results(
            converted_rgb_image, predicted_segmentation_mask,
            main_title, red_info, green_info, blue_info
        )

        # Save figure
        output_path = os.path.join(output_dir, f"{file_name}_segmentation.png")
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved segmentation result to {output_path}")
        plt.close(fig)


def run_inference(config):
    """
    Main inference pipeline.
    """
    logger.info("=" * 80)
    logger.info("Starting Unified Inference Pipeline")
    logger.info("=" * 80)

    for attr, value in vars(config).items():
        logger.info(f"  {attr}: {value}")

    # Load hyperspectral image
    logger.info(f"\nLoading hyperspectral image: {config.hsimage_path}")
    scan_data = ScanData()
    scan_data.load(config.hsimage_path)

    # Apply flatfield correction
    ffc_image = apply_flatfield_correction(
        scan_data.get_raw(),
        scan_data.get_whiteref(),
        scan_data.get_darkref()
    )
    logger.info(f"Image shape after FFC: {ffc_image.shape}")

    # Load existing RGB image (same filename with .png extension)
    rgb_image_path = config.hsimage_path.replace('.hsimage', '.png')
    if os.path.exists(rgb_image_path):
        logger.info(f"Loading existing RGB image: {rgb_image_path}")
        import cv2
        rgb_image = cv2.imread(rgb_image_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        logger.info(f"RGB image shape: {rgb_image.shape}")
    else:
        logger.warning(f"RGB image not found at {rgb_image_path}, converting from HSI...")
        # Fallback: convert from HSI if RGB doesn't exist
        rgb_image = convert_hsi_to_rgb(ffc_image, red_bin=(70, 70), green_bin=(50, 50), blue_bin=(20, 20))
        logger.info(f"RGB image shape: {rgb_image.shape}")

    # Save RGB image to output directory
    rgb_output_path = os.path.join(config.output_dir, f"{config.file_name}_rgb.png")
    plt.imsave(rgb_output_path, rgb_image)
    logger.info(f"Saved RGB image to {rgb_output_path}")

    # Crop image
    crop_height = (ffc_image.shape[0] // 64) * 64
    ffc_image_cropped = ffc_image[:crop_height, 64:-64]
    rgb_image_cropped = rgb_image[:crop_height, 64:-64]
    logger.info(f"Cropped image shape: {ffc_image_cropped.shape}")

    # Band selection if specified
    if config.selected_bands is not None:
        logger.info(f"Selecting {len(config.selected_bands)} bands...")
        
        # 重要：训练时用了 sorted(config.selected_bands)，所以推理时也必须用 sorted() 来保持一致
        # 训练代码 (train_Tianhan.py) 中：dataset.tiles = dataset.tiles[:, sorted(config.selected_bands), :, :]
        # 因此推理时也必须用相同的顺序
        sorted_bands = sorted(config.selected_bands)
        logger.info(f"Original band order (first 10): {config.selected_bands[:10]}")
        logger.info(f"Sorted band order (first 10): {sorted_bands[:10]}")
        logger.info(f"Using sorted bands to match training pipeline")
        
        ffc_image_cropped = ffc_image_cropped[:, :, sorted_bands]
        logger.info(f"After band selection: {ffc_image_cropped.shape}")

    # Initialize results dictionary
    results_dict = {
        'input_image_mapping': {config.file_name: rgb_image_cropped},
        'output_image_mapping': {},
        'class_names': config.class_names,
        'data': {},
        'experiment': config.experiment
    }

    # Detect model type and run appropriate inference
    if config.model_type in ['hstnet', 'ssftt', 'ssftt_unet', 'ssftt_lstm', 'ssftt_fusion',
                             'ssftt_seg_simple', 'ssftt_seg_full', 'ssftt_twostage']:
        logger.info(f"\nDetected SSFTT-like model type: {config.model_type}")

        # Check if this is a pixel-level segmentation model (outputs per-pixel predictions)
        # Note: Even pixel-level models may need tile-based processing if trained on tiles
        is_pixel_model = config.model_type in ['hstnet', 'ssftt_seg_simple', 'ssftt_seg_full', 'ssftt_twostage']

        # Extract tiles for processing
        logger.info(f"Extracting tiles (size={config.tile_size}, stride={config.stride})...")
        tiles, coords = extract_tiles_from_image(
            ffc_image_cropped,
            tile_size=config.tile_size,
            stride=config.stride
        )
        logger.info(f"Extracted {len(tiles)} tiles")

        # Load model
        logger.info(f"\nLoading SSFTT model from {config.model_path}...")
        spectral_bands = ffc_image_cropped.shape[2]

        # Create model based on type
        if config.model_type == 'hstnet' or config.model_type == 'ssftt_seg_simple':
            from elements.common.HSI_models.HSTNet import HSTNet
            model = HSTNet(
                in_channels=1,
                num_classes=len(config.class_names),
                dim=config.dim,
                depth=config.depth,
                heads=config.heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout,
                emb_dropout=config.emb_dropout,
                patch_size=getattr(config, 'patch_size', 8),
                use_unet=getattr(config, 'use_unet', True),
                spectral_bands=spectral_bands  # Use spectral_bands from ffc_image_cropped
            ).to(config.device)

        else:
            # Original SSFTT models
            model = initialize_model(
                model_type=config.model_type,
                in_channels=1,
                out_classes=len(config.class_names),
                start_filters=None,
                cnn_input_length=None,
                cnn_conv_block_type=None,
                num_tokens=config.num_tokens,
                dim=config.dim,
                depth=config.depth,
                heads=config.heads,
                mlp_dim=config.mlp_dim,
                dropout=config.dropout,
                emb_dropout=config.emb_dropout
            ).to(config.device)

        # Load weights
        if os.path.exists(config.model_path):
            logger.info(f"Loading weights from {config.model_path}...")

            # Use load_model_utils for segmentation models
            if config.model_type in ['hstnet', 'ssftt_seg_simple', 'ssftt_seg_full', 'ssftt_twostage']:

                model, load_info = load_segmentation_model(model, config.model_path, config.device, strict=True)
                if load_info['success']:
                    logger.info("✓ Model weights loaded successfully")
                    if load_info['removed_keys']:
                        logger.info(f"  Removed dynamic keys: {', '.join(load_info['removed_keys'])}")
                else:
                    raise RuntimeError(f"Failed to load model: {load_info['message']}")
            else:
                # Original SSFTT models - lazy initialization
                if config.model_type == 'ssftt_fusion':
                    dummy_input = torch.randn(1, spectral_bands, 3, config.tile_size, config.tile_size).to(config.device)
                else:
                    dummy_input = torch.randn(1, spectral_bands, config.tile_size, config.tile_size).to(config.device)
                with torch.no_grad():
                    _ = model(dummy_input)
                model.load_state_dict(torch.load(config.model_path, map_location=config.device))
                logger.info("✓ Model weights loaded successfully")
        else:
            raise FileNotFoundError(f"Model not found at {config.model_path}")

        # Run inference
        output_dict = run_testing_ssftt(tiles, coords, model, results_dict, config, config.file_name)

    else:
        logger.info(f"\nDetected CNN model type: {config.model_type}")

        # For CNN models, prepare full image tensor
        # Convert (H, W, C) to (1, C, H, W)
        input_tensor = torch.tensor(ffc_image_cropped, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        coords = torch.tensor([[0, 0, ffc_image_cropped.shape[1], ffc_image_cropped.shape[0]]])

        # Load model
        logger.info(f"\nLoading CNN model from {config.model_path}...")
        model = initialize_model(
            model_type=config.model_type,
            in_channels=config.in_channel,
            out_classes=len(config.class_names),
            cnn_conv_block_type=config.cnn_conv_block_type,
            start_filters=config.start_filters,
            cnn_input_length=config.cnn_input_length,
            cnn_conv_layers=config.cnn_conv_layers,
            cnn_fc_layers=config.cnn_fc_layers,
            cnn_dropout=config.cnn_dropout_rate,
            unet_depth=config.unet_depth,
            num_lstm_layers=config.lstm_layers,
            num_lstm_blocks=config.lstm_blocks,
            best_state_path=config.model_path
        ).to(config.device)

        logger.info("Model weights loaded successfully")

        # Run inference
        output_dict = run_testing_cnn([input_tensor], model, results_dict, config, config.file_name, coords)

    # Display and save results
    logger.info("\nGenerating visualization...")
    is_pixel_level = config.model_type in ['hstnet', 'ssftt_seg_simple', 'ssftt_seg_full', 'ssftt_twostage']
    display(output_dict, config.output_dir, is_pixel_level=is_pixel_level)

    logger.info("=" * 80)
    logger.info("Inference completed successfully!")
    logger.info("=" * 80)


if __name__ == '__main__':
    class InferenceConfig:
        def __init__(self):
            # Input/Output paths
            self.hsimage_path = 'data/S002_1.hsimage'
            self.file_name = 'S002_1'
            self.output_dir = 'log/inference_results'
            os.makedirs(self.output_dir, exist_ok=True)

            # Model configuration
            # Options: 'ssftt', 'ssftt_unet', 'ssftt_lstm', 'ssftt_fusion',
            #          'hstnet', 'ssftt_twostage',
            #          'cnn1d', 'unet', etc.
            self.model_type = 'hstnet'  #  Use pixel-level segmentation model

            if self.model_type == 'ssftt':
                self.model_path = 'log/SSFTT_B224/model.npz'
                self.experiment = 'SSFTT_B224'
            elif self.model_type == 'ssftt_unet':
                self.model_path = 'log/SSFTT_U_Net_B224/model.npz'
                self.experiment = 'SSFTT_U_Net_B224'
            elif self.model_type == 'ssftt_lstm':
                self.model_path = 'log/SSFTT_LSTM_B224/model.npz'
                self.experiment = 'SSFTT_LSTM_B224'
            elif self.model_type == 'hstnet':
                self.model_path = 'log/HSTNet_B60/model.npz'
                self.experiment = 'HSTNet_B60'
            elif self.model_type == 'ssftt_seg_simple':
                self.model_path = 'log/SSFTT-Seg-Simple_B224/model.npz'
                self.experiment = 'SSFTT-Seg-Simple_B224'

            # SSFTT model parameters (only for SSFTT models)
            self.num_tokens = 6
            self.dim = 96
            self.depth = 2
            self.heads = 6
            self.mlp_dim = 384
            self.dropout = 0.15
            self.emb_dropout = 0.15
            self.patch_size = 8  # For ssftt_seg_simple
            self.use_unet = False  # Explicit setting for HSTNet (ablation winner: Conv2D is best)

            # CNN model parameters (only for CNN models)
            self.in_channel = 1
            self.start_filters = 32
            self.cnn_conv_block_type = 'A'
            self.cnn_input_length = 224
            self.cnn_conv_layers = 3
            self.cnn_fc_layers = 2
            self.cnn_dropout_rate = 0.3
            self.unet_depth = 4
            self.lstm_layers = None
            self.lstm_blocks = None

            # Tile extraction (for SSFTT models)
            self.tile_size = 64  # Match training tile size (HST trained on 64x64 tiles)
            self.stride = 32  # 50% overlap for smoother results

            # Inference parameters
            self.batch_size = 16  # Reduce if OOM
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # CNN inference parameters
            self.inference_tile_height = 8
            self.inference_mode = 'patched'
            self.inference_patch_size = (8, 8)
            self.inference_stride = 8
            self.pixel_batch_size = 80000

            # Band selection (set to None to use all bands)
            # IMPORTANT: Must match training configuration!
            if self.model_type == 'hstnet':
                # HSTNet_B60 was trained with 60 selected bands
                self.selected_bands = [18, 15, 173, 77, 116, 145, 161, 100, 126, 143,
                                      155, 17, 106, 165, 99, 174, 28, 147, 44, 168,
                                      158, 150, 40, 125, 171, 189, 139, 46, 94, 179,
                                      32, 166, 62, 80, 177, 105, 49, 182, 115, 102,
                                      109, 193, 23, 85, 152, 71, 69, 59, 132, 156,
                                      79, 97, 88, 30, 93, 86, 128, 53, 35, 39]
                # Filter to exclude first 10 (0-9) and last 10 (214-223) bands
                self.selected_bands = [b for b in self.selected_bands if 10 <= b <= 213]
            else:
                # Other models (SSFTT, SSFTT_U_Net, etc.) use all 224 bands
                self.selected_bands = None
            
            # Class names (13 classes total: unused, background, 11 fabric materials)
            self.class_names = ['unused', 'background',
                'cotton', 'polyester', 'viscose', 'nylon', 'acryl', 
                'wool', 'linen', 'modacrylic', 'pla', 'elastane', 'antistatic'
            ]
    
    config = InferenceConfig()
    run_inference(config)
