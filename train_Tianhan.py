"""
HSTNet Training Pipeline

This script trains the HSTNet (Hyperspectral Segmentation Transformer Network) model
for hyperspectral image segmentation and classification.

HSTNet is our novel architecture combining:
- 3D CNN for spectral feature extraction
- U-Net for multi-scale spatial features
- Patch-based Transformer for global context
- Segmentation head for pixel-level predictions

USAGE:
------
To switch between using all bands (224) or selected bands (~90):
1. Open this file
2. Find the ParameterConfig class (around line 560)
3. Set use_band_selection = True  (use selected bands, excluding noisy first/last 10)
   OR use_band_selection = False (use all 224 bands)
4. Run the script

The experiment name will automatically include the band count:
- HSTNet_B90 (with band selection, ~90 bands after filtering)
- HSTNet_B224 (all bands)

MODELS SUPPORTED:
-----------------
- 'hstnet': HSTNet - Our novel architecture (RECOMMENDED)
- 'ssftt': Standard SSFTT baseline
- 'ssftt_unet': SSFTT with U-Net features
- 'ssftt_lstm': SSFTT with LSTM
- 'ssftt_fusion': SSFTT with RGB fusion
- 'ssftt_seg_full': SSFTT with tokenization segmentation
"""

import os
import time
import torch
import warnings
from pathlib import Path
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

# Note: expandable_segments can cause issues in vGPU environments
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'




# ── working dir ──────────────────────────────────────────────────────────────
def _get_working_dir() -> Path:
    return Path(__file__).resolve().parent

data_path = BASE_DIR = Path(__file__).resolve().parent

# ── logging ───────────────────────────────────────────────────────────────────
from elements.utils import LoggerSingleton
LoggerSingleton.setup_logger(_get_working_dir())
logger = LoggerSingleton.get_logger()

# ── pipeline imports ──────────────────────────────────────────────────────────
from elements.common.utils import static_var
from elements.load_data import load_hsi_dataset, stratified_split_by_composition, load_prediction_dict
from elements.optimize import get_adam_optimizer_pt
from elements.predict import collect_predictions
from elements.tune_params import get_step_lr_pt
from elements.calc_loss import calc_divergence_loss
from elements.calc_metrics import calculate_metrics
from elements.visualize import create_tb, show_loss_tb, display_output_mapping, create_output_mapping
from elements.save_model import save_model
from elements.save_results import create_prediction_dict, create_results_dict, save_prediction_dict
# Note: Models are imported dynamically in _build_model() based on model_type



warnings.filterwarnings("ignore", category=UserWarning, module="torch.nn.functional")


# ── helpers ───────────────────────────────────────────────────────────────────
# Note: These are legacy functions, not used with modern architectures like HSTNet
CONV3D_OUT_CHANNELS = 32   # legacy - kept for reference
HSTNET_CONV3D_OUT_CHANNELS = 8  # HSTNet 3D conv output channels

def _calc_conv2d_in_channels(spectral_bands: int) -> int:
    """
    Legacy function - not used with ImprovedSSFTTnet.
    ImprovedSSFTTnet handles dimension calculations internally.
    """
    return CONV3D_OUT_CHANNELS * (spectral_bands - 2)


def _get_spectral_bands(dataset) -> int:
    """Return number of spectral bands from the first tile."""
    # Access tiles directly from dataset.tiles to avoid training_mode issues
    first_tile = dataset.tiles[0]

    # tile shape should be (C, H, W) where C is the number of spectral bands
    if first_tile.ndim == 3:
        return int(first_tile.shape[0])  # C is at index 0
    else:
        raise ValueError(f"Unexpected tile shape: {first_tile.shape}. Expected 3D (C, H, W).")


def _build_model(spectral_bands: int, num_classes: int, config):
    """Build model (HSTNet, SSFTT variants, or other architectures)"""
    model_type = getattr(config, 'model_type', 'hstnet')  # default to HSTNet

    logger.info(
        f"SSFTT config | model_type={model_type} spectral_bands={spectral_bands} "
        f"num_classes={num_classes}"
    )

    if model_type == 'ssftt_fusion':
        from elements.common.HSI_models.Data_Fusion.SSFTTnet_fusion_transformer import SSFTTFusion
        return SSFTTFusion(
            in_channels=224,  # 224 spectral channels for fusion model
            num_classes=num_classes,
            num_tokens=config.num_tokens,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
        ).to(config.device)
    elif model_type == 'ssftt_unet':
        from elements.common.HSI_models.SSFTT.SSFTTnet_unet import SSFTTnet_UNet
        return SSFTTnet_UNet(
            in_channels=1,
            num_classes=num_classes,
            num_tokens=config.num_tokens,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            spectral_bands=spectral_bands,
        ).to(config.device)
    elif model_type == 'ssftt_lstm':
        from elements.common.HSI_models.SSFTT.SSFTTnet_lstm import SSFTTnet_LSTM
        return SSFTTnet_LSTM(
            in_channels=1,
            num_classes=num_classes,
            num_tokens=config.num_tokens,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            spectral_bands=spectral_bands,
        ).to(config.device)
    elif model_type == 'hstnet' or model_type == 'ssftt_seg_simple':
        # HSTNet - Hyperspectral Segmentation Transformer Network
        # Our novel architecture combining 3D CNN + U-Net + Transformer
        from elements.common.HSI_models.HSTNet import HSTNet
        use_unet = getattr(config, 'use_unet', False)
        use_learned_tokens = getattr(config, 'use_learned_tokens', False)
        logger.info(f"Using HSTNet - Hyperspectral Segmentation Transformer Network "
                   f"{'with U-Net' if use_unet else 'with Conv2D'}, "
                   f"{'Learned Tokens' if use_learned_tokens else 'Patch-based'}")
        return HSTNet(
            in_channels=1,
            num_classes=num_classes,
            num_tokens=getattr(config, 'num_tokens', 6),
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            patch_size=8,  # Process in 8×8 patches to save memory
            use_unet=use_unet,
            spectral_bands=spectral_bands,  # Pass the actual number of bands
            conv3d_out_channels=HSTNET_CONV3D_OUT_CHANNELS,  # 8
            use_learned_tokens=use_learned_tokens
        ).to(config.device)

    # ── Baseline models ───────────────────────────────────────────────────────
    elif model_type == 'unet':
        from elements.model_wrappers import UNet
        return UNet(
            num_classes=num_classes,
            in_channels=spectral_bands,
            depth=getattr(config, 'unet_depth', 4),
            start_filters=getattr(config, 'unet_start_filters', 32),
        ).to(config.device)
    elif model_type == 'cnn3d':
        from elements.model_wrappers import DynamicCNN3D
        return DynamicCNN3D(
            in_channels=spectral_bands,
            num_classes=num_classes,
            num_conv_layers=getattr(config, 'num_conv_layers', 3),
            num_fc_layers=getattr(config, 'num_fc_layers', 2),
            start_filters=getattr(config, 'start_filters', 4),
            dropout=getattr(config, 'dropout', 0.3),
            final_activation='none',
            tile_size=getattr(config, 'tile_size', 5),
        ).to(config.device)
    elif model_type == 'cnn2d':
        from elements.model_wrappers import DynamicCNN2D
        return DynamicCNN2D(
            in_channels=spectral_bands,
            num_classes=num_classes,
            num_conv_layers=getattr(config, 'num_conv_layers', 3),
            num_fc_layers=getattr(config, 'num_fc_layers', 2),
            start_filters=getattr(config, 'start_filters', 32),
            dropout=getattr(config, 'dropout', 0.3),
            final_activation='none',
        ).to(config.device)
    elif model_type == 'cnn1d':
        from elements.model_wrappers import DynamicCNN1D
        return DynamicCNN1D(
            in_channels=1,
            num_classes=num_classes,
            input_length=spectral_bands,
            conv_block_type='A',
            num_conv_layers=getattr(config, 'num_conv_layers', 3),
            num_fc_layers=getattr(config, 'num_fc_layers', 2),
            start_filters=getattr(config, 'start_filters', 32),
            dropout=getattr(config, 'dropout', 0.3),
            final_activation='none',
        ).to(config.device)
    elif model_type == 'cnn_lstm':
        from elements.model_wrappers import CnnLstm
        return CnnLstm(
            in_channels=1,
            num_classes=num_classes,
            input_length=spectral_bands,
            conv_block_type='A',
            num_conv_layers=getattr(config, 'num_conv_layers', 3),
            num_fc_layers=getattr(config, 'num_fc_layers', 2),
            start_filters=getattr(config, 'start_filters', 32),
            dropout=getattr(config, 'dropout', 0.3),
            final_activation='none',
            num_lstm_blocks=getattr(config, 'num_lstm_blocks', 1),
            num_lstm_layers=getattr(config, 'num_lstm_layers', 1),
        ).to(config.device)
    else:  # 'ssftt' or default
        from elements.common.HSI_models.SSFTT.SSFTTnet import SSFTTnet
        return SSFTTnet(
            in_channels=1,
            num_classes=num_classes,
            num_tokens=config.num_tokens,
            dim=config.dim,
            depth=config.depth,
            heads=config.heads,
            mlp_dim=config.mlp_dim,
            dropout=config.dropout,
            emb_dropout=config.emb_dropout,
            spectral_bands=spectral_bands,
        ).to(config.device)

# ── training / validation ─────────────────────────────────────────────────────
@static_var(model_saved=False)
@static_var(best_val_loss=9999)
def run_training(train_loader, val_loader, model, optimizer, loss_func, scheduler, config):
    """Train model, validate periodically, save best checkpoint."""
    run_training.best_val_loss = 9999
    train_losses, val_losses = [], []
    scaler = GradScaler('cuda' if config.device == 'cuda' else 'cpu')
    start_time = time.time()

    # Early stopping setup
    early_stop_patience = getattr(config, 'early_stop_patience', 15)  # Default 15 epochs
    early_stop_counter = 0
    early_stop_min_delta = getattr(config, 'early_stop_min_delta', 1e-4)  # Minimum improvement

    for epoch in range(config.num_epochs):
        model.train()
        running_loss, n_batches = 0.0, 0

        for inputs, labels in train_loader:
            # Convert float16 to float32 for model compatibility
            inputs = inputs.to(config.device, dtype=torch.float32)
            labels = labels.to(config.device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)

            with autocast('cuda' if config.device == 'cuda' else 'cpu'):
                # CNN1D/CnnLstm: pixel-wise forward (B, C, H, W) -> (B, num_classes, H, W)
                if config.model_type in ['cnn1d', 'cnn_lstm']:
                    B, C, H, W = inputs.shape
                    # flatten pixels: (B*H*W, 1, C)
                    px = inputs.permute(0, 2, 3, 1).contiguous().view(-1, 1, C)
                    px_out = model(px)  # (B*H*W, num_classes)
                    num_classes = px_out.shape[1]
                    outputs = px_out.view(B, H, W, num_classes).permute(0, 3, 1, 2)  # (B, num_classes, H, W)
                elif config.model_type == 'cnn2d':
                    B, C, H, W = inputs.shape
                    out = model(inputs)  # (B, num_classes)
                    outputs = out.unsqueeze(2).unsqueeze(3).expand(B, out.shape[1], H, W)  # broadcast to (B, num_classes, H, W)
                else:
                    outputs = model(inputs)
                loss = loss_func(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

            # Clear cache periodically to prevent memory fragmentation
            if n_batches % 10 == 0:
                torch.cuda.empty_cache()

        train_losses.append(running_loss / n_batches)

        do_val = (
            (epoch + 1) % config.val_frequency == 0
            or epoch == 0
            or epoch == config.num_epochs - 1
        )

        if do_val:
            val_loss = run_validation(model, val_loader, loss_func, config.device)
            val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < run_training.best_val_loss - early_stop_min_delta:
                run_training.best_val_loss = val_loss
                save_model(model, os.path.join(_get_working_dir(), 'log', config.experiment))
                run_training.model_saved = True
                early_stop_counter = 0  # Reset counter on improvement
            else:
                run_training.model_saved = False
                early_stop_counter += 1

            # Quick cosine similarity on val set (vectorized, GPU, ~zero overhead)
            model.eval()
            cos_sims = []
            with torch.no_grad():
                for inputs_v, labels_v in val_loader:
                    inputs_v = inputs_v.to(config.device, dtype=torch.float32)
                    labels_v = labels_v.to(config.device, dtype=torch.float32)
                    with autocast('cuda' if config.device == 'cuda' else 'cpu'):
                        if config.model_type in ['cnn1d', 'cnn_lstm']:
                            B, C, H, W = inputs_v.shape
                            px = inputs_v.permute(0, 2, 3, 1).contiguous().view(-1, 1, C)
                            px_out = model(px)
                            num_classes = px_out.shape[1]
                            out_v = torch.softmax(px_out.view(B, H, W, num_classes).permute(0, 3, 1, 2), dim=1)
                        elif config.model_type == 'cnn2d':
                            B, C, H, W = inputs_v.shape
                            raw = model(inputs_v)  # (B, num_classes)
                            out_v = torch.softmax(raw, dim=1).unsqueeze(2).unsqueeze(3).expand(B, raw.shape[1], H, W)
                        else:
                            out_v = torch.softmax(model(inputs_v), dim=1)
                    pred_c = out_v.mean(dim=(2, 3))
                    true_c = labels_v.mean(dim=(2, 3))
                    pred_n = pred_c / (pred_c.norm(dim=1, keepdim=True) + 1e-8)
                    true_n = true_c / (true_c.norm(dim=1, keepdim=True) + 1e-8)
                    cos_sims.append((pred_n * true_n).sum(dim=1).mean().item())
            val_cos_sim = sum(cos_sims) / len(cos_sims)
            model.train()

            show_loss_tb(val_loss,            epoch, writer=config.writer, name='Valid Loss')
            show_loss_tb(train_losses[epoch], epoch, writer=config.writer, name='Train Loss')

            logger.info(
                f"Epoch {epoch + 1:<3}/{config.num_epochs:<3} | "
                f"TrainLoss: {train_losses[epoch]:<8.6f} | "
                f"ValidLoss: {val_loss:<8.6f} | "
                f"ValCosSim: {val_cos_sim:<6.4f} | "
                f"LR: {optimizer.param_groups[0]['lr']:<8.6f} | "
                f"ModelSaved: {str(run_training.model_saved):<5} | "
                f"EarlyStop: {early_stop_counter}/{early_stop_patience} | "
                f"Time: {time.time() - start_time:<6.2f}s"
            )

            # Early stopping check
            if early_stop_counter >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epoch + 1} epochs (no improvement for {early_stop_patience} epochs)")
                break
        else:
            val_losses.append(None)


@torch.no_grad()
def run_validation(model, data_loader, loss_func, device) -> float:
    from elements.model_wrappers import DynamicCNN1D, CnnLstm, DynamicCNN2D
    model.eval()
    running_loss, n_batches = 0.0, 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device, dtype=torch.float32)
        labels = labels.to(device, dtype=torch.float32)
        with autocast('cuda' if device == 'cuda' else 'cpu'):
            if isinstance(model, (DynamicCNN1D, CnnLstm)):
                B, C, H, W = inputs.shape
                px = inputs.permute(0, 2, 3, 1).contiguous().view(-1, 1, C)
                px_out = model(px)
                num_classes = px_out.shape[1]
                outputs = px_out.view(B, H, W, num_classes).permute(0, 3, 1, 2)
            elif isinstance(model, DynamicCNN2D):
                B, C, H, W = inputs.shape
                out = model(inputs)  # (B, num_classes)
                outputs = out.unsqueeze(2).unsqueeze(3).expand(B, out.shape[1], H, W)
            else:
                outputs = model(inputs)
            loss = loss_func(outputs, labels)
        running_loss += loss.item()
        n_batches += 1
    return running_loss / n_batches


# ── testing ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_testing(test_loader, model, results_dict, config) -> dict:
    model.eval()

    # Choose prediction function based on uncertainty config
    if config.enable_uncertainty:
        from elements.predict import collect_predictions_with_uncertainty
        logger.info(f"Running inference with uncertainty estimation ({config.num_mc_passes} MC passes)...")
        results_dict = collect_predictions_with_uncertainty(
            test_loader=test_loader, model=model,
            results_dict=results_dict, config=config,
            num_mc_passes=config.num_mc_passes
        )
    else:
        results_dict = collect_predictions(
            test_loader=test_loader, model=model,
            results_dict=results_dict, config=config
        )

    results_dict = create_output_mapping(results_dict=results_dict,
                                         enable_background=config.enable_background)
    results_dict = calculate_metrics(results_dict=results_dict,
                                     saved_dir=_get_working_dir(), config=config)
    return create_prediction_dict(results_dict)


# ── main pipeline ─────────────────────────────────────────────────────────────
def run_pipeline(config):
    logger.info("=" * 80)
    logger.info("Starting HSTNet Training Pipeline")
    logger.info("=" * 80)

    # Log band selection status prominently
    if config.selected_bands is not None:
        logger.info(f"Band Selection: ENABLED - Using {len(config.selected_bands)} selected bands")
    else:
        logger.info(f"Band Selection: DISABLED - Using all 224 bands")

    logger.info(f"Experiment: {config.experiment}")
    logger.info("=" * 80)

    for attr, value in vars(config).items():
        # Skip logging the long selected_bands_list
        if attr == 'selected_bands_list':
            continue
        logger.info(f"  {attr}: {value}")

    # ── TRAIN ─────────────────────────────────────────────────────────────────
    if config.do_train:
        logger.info("=== Training pipeline ===")

        dataset = load_hsi_dataset(
            dataset_path=os.path.join('/home/student/S2/HIT/dataset', config.train_dataset_name)
        )
        dataset.training_mode = True

        if config.selected_bands is not None:
            # 处理波段选择：根据 tiles 的维度选择正确的索引方式
            if dataset.tiles.ndim == 4:  # (N, C, H, W)
                logger.info(f"Selecting {len(config.selected_bands)} bands from {dataset.tiles.shape[1]} total bands...")
                sorted_bands = sorted(config.selected_bands)
                logger.info(f"Original band order (first 10): {config.selected_bands[:10]}")
                logger.info(f"Sorted band order (first 10): {sorted_bands[:10]}")
                dataset.tiles = dataset.tiles[:, sorted_bands, :, :]
                logger.info(f"Tiles shape after band selection: {dataset.tiles.shape}")
            elif dataset.tiles.ndim == 3:  # (N, 1, C)
                dataset.tiles = dataset.tiles[:, :, sorted(config.selected_bands)]
                logger.info(f"Tiles shape after band selection: {dataset.tiles.shape}")
        else:
            logger.info(f"Using all bands - Tiles shape: {dataset.tiles.shape}")

        # Handle labels based on model type
        if config.model_type in ['hstnet', 'ssftt', 'ssftt_unet', 'ssftt_lstm', 'ssftt_seg_simple', 'ssftt_seg_full', 'ssftt_twostage', 'unet', 'cnn3d', 'cnn2d', 'cnn1d', 'cnn_lstm']:
            # Segmentation models need pixel-level labels (B, num_classes, H, W)
            logger.info(f"Preparing pixel-level labels for {config.model_type}...")
            
            if dataset.labels.ndim == 2:
                # Tile-level labels (N, num_classes) -> Pixel-level (N, num_classes, H, W)
                logger.info(f"Expanding tile-level labels {dataset.labels.shape} to pixel-level...")
                N, num_classes = dataset.labels.shape
                _, _, H, W = dataset.tiles.shape
                # Replicate tile label to all pixels
                dataset.labels = dataset.labels.unsqueeze(2).unsqueeze(3).expand(N, num_classes, H, W)
                logger.info(f"Labels shape after expansion: {dataset.labels.shape}")
            elif dataset.labels.ndim == 4:
                # Already pixel-level (N, num_classes, H, W)
                logger.info(f"Labels already pixel-level: {dataset.labels.shape}")
            else:
                raise ValueError(f"Unexpected label shape: {dataset.labels.shape}")
        else:
            # Standard tile-level models use flat (per-sample) labels → aggregate 4-D labels if needed
            if dataset.labels.ndim == 4:
                logger.info("Converting 4D labels → 2D by spatial mean for tile-level models...")
                dataset.labels = dataset.labels.mean(dim=(2, 3))
                logger.info(f"Labels shape: {dataset.labels.shape}")

        # Create custom collate function for fusion model
        if config.model_type == 'ssftt_fusion':
            logger.info("Using fusion collate function for dynamic 5D conversion...")
            def fusion_collate_fn(batch):
                """Convert batch to fusion format on-the-fly"""
                tiles = torch.stack([item[0] for item in batch])  # (B, C, H, W) where C=224
                labels = torch.stack([item[1] for item in batch])  # (B, num_classes)
                # Add RGB depth dimension: (B, C, H, W) -> (B, C, 3, H, W)
                # Expand to create RGB depth (simple replication for now)
                tiles = tiles.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()
                return tiles, labels
            collate_fn = fusion_collate_fn
        else:
            collate_fn = None

        train_dataset, val_dataset = stratified_split_by_composition(
            dataset=dataset,
            train_ratio=config.train_ratio,
            model_type='cnn1d' if config.model_type not in ['ssftt_seg_simple', 'ssftt_seg_full', 'ssftt_twostage'] else 'unet',
            check_stratification=config.check_stratification,
        )

        if not config.enable_background:
            def _remove_background(subset):
                labels = torch.stack([subset.dataset[i][1] for i in subset.indices])
                mask = labels[:, 0] != 1.0
                return Subset(dataset, torch.tensor(subset.indices)[mask].tolist())

            train_dataset = _remove_background(train_dataset)
            val_dataset   = _remove_background(val_dataset)

        loader_kwargs = dict(batch_size=config.batch_size, num_workers=config.num_workers,
                             pin_memory=config.pin_memory, collate_fn=collate_fn)
        train_loader = DataLoader(train_dataset, shuffle=True,  **loader_kwargs)
        val_loader   = DataLoader(val_dataset,   shuffle=False, **loader_kwargs)

        spectral_bands = _get_spectral_bands(dataset)
        model = _build_model(spectral_bands, num_classes=len(dataset.get_class_names()), config=config)

        # Dummy forward pass only needed for SSFTT models (lazy conv2d_features initialization)
        if config.model_type in ['ssftt', 'ssftt_unet', 'ssftt_lstm', 'ssftt_fusion']:
            logger.info("Initializing SSFTT model with dummy forward pass...")
            if config.model_type == 'ssftt_fusion':
                dummy_input = torch.randn(2, spectral_bands, 3, 16, 16).to(config.device)
            else:
                dummy_input = torch.randn(2, spectral_bands, 16, 16).to(config.device)
            with torch.no_grad():
                _ = model(dummy_input)

        # Count and display model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info("=" * 60)
        logger.info(f"Model Parameters Summary:")
        logger.info(f"  Total parameters:     {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        logger.info(f"  Input spectral bands: {spectral_bands}")
        
        logger.info("=" * 60)

        # Calculate class weights if enabled
        class_weights = None
        if config.use_class_weights:
            logger.info("\nCalculating class weights for imbalanced dataset...")
            from elements.calc_loss import calculate_class_weights
            class_weights = calculate_class_weights(
                train_dataset,
                method=config.class_weight_method,
                power=config.class_weight_power,
                min_weight=config.class_weight_min
            )
            class_weights = class_weights.to(config.device)
            logger.info(f"Class weights: {class_weights}")

        loss_func = calc_divergence_loss(ignore_indices=config.ignore_indices, class_weights=class_weights)
        optimizer = get_adam_optimizer_pt(model=model, learning_rate=config.learning_rate)
        scheduler = get_step_lr_pt(optimizer, mode='min', patience=config.patience, factor=config.factor)

        run_training(train_loader=train_loader, val_loader=val_loader, model=model,
                     optimizer=optimizer, loss_func=loss_func, scheduler=scheduler, config=config)

    # ── TEST ──────────────────────────────────────────────────────────────────
    if config.do_test:
        logger.info("=== Inference pipeline ===")

        torch.cuda.empty_cache()

        dataset = load_hsi_dataset(
            dataset_path=os.path.join('/home/student/S2/HIT/dataset', config.test_dataset_name)
        )
        if config.selected_bands is not None:
            logger.info(f"Selecting {len(config.selected_bands)} bands for test dataset...")
            
            # Check current shape
            if len(dataset.tiles) > 0:
                first_tile_shape = dataset.tiles[0].shape
                logger.info(f"Original tile shape: {first_tile_shape}")
                
                # Determine if tiles are list or tensor
                if isinstance(dataset.tiles, list):
                    # List of tensors
                    sorted_bands = sorted(config.selected_bands)
                    logger.info(f"Test dataset - Original band order (first 10): {config.selected_bands[:10]}")
                    logger.info(f"Test dataset - Sorted band order (first 10): {sorted_bands[:10]}")
                    new_tiles = []
                    for i in range(len(dataset.tiles)):
                        # dataset.tiles[i] shape: (C, H, W)
                        selected_tile = dataset.tiles[i][sorted_bands, :, :]
                        new_tiles.append(selected_tile)
                    dataset.tiles = new_tiles
                    logger.info(f"After band selection, tile 0 shape: {dataset.tiles[0].shape}")
                else:
                    # Tensor
                    sorted_bands = sorted(config.selected_bands)
                    logger.info(f"Test dataset - Original band order (first 10): {config.selected_bands[:10]}")
                    logger.info(f"Test dataset - Sorted band order (first 10): {sorted_bands[:10]}")
                    dataset.tiles = dataset.tiles[:, sorted_bands, :, :]
                    logger.info(f"After band selection, tiles shape: {dataset.tiles.shape}")

        # Get spectral bands BEFORE setting training_mode
        # because training_mode affects how dataset[0] returns data
        spectral_bands = _get_spectral_bands(dataset)
        logger.info(f"Detected {spectral_bands} spectral bands from dataset")

        logger.info(f"GPU memory after loading dataset: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")

        dataset.training_mode = False

        # Create custom collate function for fusion model
        if config.model_type == 'ssftt_fusion':
            logger.info("Using fusion collate function for test dataset...")
            def fusion_collate_fn(batch):
                """Convert batch to fusion format on-the-fly"""
                tiles = torch.stack([item[0] for item in batch])  # (B, C, H, W) where C=224
                labels = torch.stack([item[1] for item in batch])  # (B, num_classes, H, W)
                coords = [item[2] for item in batch]  # List of coords
                file_names = [item[3] for item in batch]  # List of file names
                # Add RGB depth dimension: (B, C, H, W) -> (B, C, 3, H, W)
                tiles = tiles.unsqueeze(2).expand(-1, -1, 3, -1, -1).contiguous()
                return tiles, labels, coords, file_names
            collate_fn = fusion_collate_fn
        else:
            collate_fn = None

        # Reduce num_workers for testing to avoid "Too many open files" error
        # Testing uses batch_size=1, so multiple workers can cause file descriptor issues
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False,
                                 num_workers=0, pin_memory=False, collate_fn=collate_fn)

        model = _build_model(spectral_bands, num_classes=len(dataset.get_class_names()), config=config)

        logger.info(f"GPU memory after building model: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")

        model_path = os.path.join(_get_working_dir(), 'log', config.experiment, 'model.npz')
        if os.path.exists(model_path):
            # Dummy forward pass only needed for SSFTT models (lazy conv2d_features initialization)
            if config.model_type in ['ssftt', 'ssftt_unet', 'ssftt_lstm', 'ssftt_fusion']:
                logger.info("Initializing SSFTT model with dummy forward pass...")
                if config.model_type == 'ssftt_fusion':
                    dummy_input = torch.randn(2, spectral_bands, 3, 16, 16).to(config.device)
                else:
                    dummy_input = torch.randn(2, spectral_bands, 16, 16).to(config.device)
                with torch.no_grad():
                    _ = model(dummy_input)

            # Now load the weights
            model.load_state_dict(torch.load(model_path, map_location=config.device))
            logger.info(f"Loaded weights from {model_path}")
            logger.info(f"GPU memory after loading weights: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f}GB reserved")
        else:
            logger.warning(f"No weights found at {model_path}, using untrained model")

        results_dict    = create_results_dict(dataset=dataset)
        prediction_dict = run_testing(test_loader=test_loader, model=model,
                                      results_dict=results_dict, config=config)
        
        save_prediction_dict(
            prediction_dict,
            os.path.join(_get_working_dir(), 'log', config.experiment, 'prediction_dict.npz'),
        )

    # ── DISPLAY ───────────────────────────────────────────────────────────────
    if config.display_mapping:
        prediction_dict = load_prediction_dict(
            os.path.join(_get_working_dir(), 'log', config.experiment, 'prediction_dict.npz')
        )
        results_dir = os.path.join(_get_working_dir(), 'log', config.experiment, 'results')
        os.makedirs(results_dir, exist_ok=True)
        display_output_mapping(prediction_dict, results_dir)
        logger.info(f"Results saved to {results_dir}")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    class ParameterConfig:
        def __init__(self):
            # Model type options:
            # - 'hstnet': HSTNet - Our novel Hyperspectral Segmentation Transformer Network (RECOMMENDED)
            # - 'ssftt': Standard SSFTT (tile-level classification)
            # - 'ssftt_unet': SSFTT with U-Net features (tile-level)
            # - 'ssftt_lstm': SSFTT with LSTM (tile-level)
            # - 'ssftt_fusion': SSFTT with RGB fusion (tile-level)
            # - 'ssftt_twostage': Two-Stage SSFTT (pixel-level, background + fabric)
            # - 'unet': U-Net (pixel-level segmentation)
            # - 'cnn3d': 3D-CNN (pixel-level segmentation)
            # - 'cnn2d': 2D-CNN (pixel-level, tile input)
            # - 'cnn1d': 1D-CNN (pixel-wise spectral classification)
            # - 'cnn_lstm': CNN-LSTM (pixel-wise spectral classification)
            self.model_type = 'hstnet'  # HSTNet - our novel architecture

            self.do_train        = True
            self.do_test         = True
            
            # Transformer hyper-parameters for 64×64 tiles
            self.num_tokens  = 6     # Only used for tokenization-based models (ssftt, ssftt_unet, ssftt_lstm, ssftt_seg_full)
            self.dim         = 96    # 96 dim for balanced model
            self.depth       = 2     # 2 layers for faster training
            self.heads       = 6     # 6 attention heads
            self.mlp_dim     = 384   # 4x dim (96 * 4)
            self.dropout     = 0.15  # Regularization
            self.emb_dropout = 0.15

            # Two-stage model specific parameters
            self.freeze_stage1 = False  # Set to True to freeze background segmentation stage
            self.freeze_stage2 = False  # Set to True to freeze fabric classification stage
            # Training workflow for two-stage:
            # 1. Train stage 1 only: freeze_stage1=False, freeze_stage2=True
            # 2. Train stage 2 only: freeze_stage1=True, freeze_stage2=False
            # 3. Fine-tune both: freeze_stage1=False, freeze_stage2=False
            
            # Segmentation model specific parameters
            self.use_unet = False  # Set to True to use U-Net for spatial feature extraction (better performance)
            self.use_learned_tokens = False  # Set to True to use SSFTT-style learned tokenization instead of patch-based (experimental)

            # Datasets - using 64×64 tiles for better purity
            self.train_dataset_name = 'train_set.npz'
            self.test_dataset_name  = 'test_set.npz'

            # Band selection - set to None to use all 224 bands
            self.use_band_selection = True  # Set to True to use selected bands, False to use all 224 bands

            # 使用 Auto 选择的 100 个波段（从 50 增加到 100 以提高准确率）
            # Only used if use_band_selection = True
            # Original LDA selected bands
            original_selected_bands = [18, 15, 173, 77, 116, 145, 161, 100, 126, 143,
                                      155, 17, 106, 165, 99, 174, 28, 147, 44, 168,
                                      158, 150, 40, 125, 171, 189, 139, 46, 94, 179,
                                      32, 166, 62, 80, 177, 105, 49, 182, 115, 102,
                                      109, 193, 23, 85, 152, 71, 69, 59, 132, 156,
                                      79, 97, 88, 30, 93, 86, 128, 53, 35, 39]

            
            # Filter out first 10 bands (0-9) and last 10 bands (214-223) as they are noisy
            self.selected_bands_list = [b for b in original_selected_bands if 10 <= b <= 213]

            # Apply band selection based on flag
            self.selected_bands = self.selected_bands_list if self.use_band_selection else None

            # Training - optimized for 64×64 tiles and pixel-level models
            self.batch_size    = 32   # Reduced for pixel-level models
            self.num_workers   = 32
            self.pin_memory    = True
            self.num_epochs    = 100
            self.val_frequency = 1
            self.learning_rate = 0.0001  # Good starting point
            self.patience      = 15
            self.factor        = 0.5

            # Legacy model hyperparameters (cnn1d, cnn2d, cnn3d, cnn_lstm, unet)
            self.num_conv_layers    = 3
            self.num_fc_layers      = 2
            self.start_filters      = 16   # 32→16 for cnn2d/cnn1d/cnn_lstm (32 causes 16M params, too large)
            self.num_lstm_blocks    = 1
            self.num_lstm_layers    = 1
            self.unet_depth         = 4
            self.unet_start_filters = 32
            self.tile_size          = 5  # for CNN3D tile reduction

            # Early stopping parameters
            self.early_stop_patience = 15
            self.early_stop_min_delta = 1e-4

            # Data split
            self.train_ratio           = 0.7
            self.check_stratification  = False
            self.enable_background     = True
            self.ignore_indices        = [0]  # Ignore index 0 (unused) in loss calculation

            # Class weights for imbalanced datasets
            # IMPORTANT: For pixel-level segmentation with background, class weights can
            # cause issues because background has low weight (0.487) which makes the model
            # ignore it. Better to use ignore_indices only and let the model learn naturally.
            self.use_class_weights     = False  # Disabled for better background learning
            self.class_weight_method   = 'inverse'
            self.class_weight_power    = 0.3
            self.class_weight_min      = 0.01

            # Device & pipeline control
            self.device          = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.display_mapping = True

            # Experiment name - includes model type and band info
            bands_suffix = f"B{len(self.selected_bands_list)}" if self.use_band_selection else "B224"
            
            # Model name mapping
            model_name_map = {
                'hstnet': 'HSTNet',
                'ssftt': 'SSFTT',
                'ssftt_unet': 'SSFTT_U_Net',
                'ssftt_lstm': 'SSFTT_LSTM',
                'ssftt_fusion': 'SSFTT_Fusion',
                'ssftt_twostage': 'SSFTT_TwoStage',
                'unet': 'UNet',
                'cnn3d': 'CNN3D',
                'cnn2d': 'CNN2D',
                'cnn1d': 'CNN1D',
                'cnn_lstm': 'CNN_LSTM',
            }
            model_name = model_name_map.get(self.model_type, 'SSFTT')
            
            self.experiment      = f'{model_name}_{bands_suffix}'
            self.writer          = None

            # Inference parameters
            self.inference_tile_height = 64
            self.inference_mode = 'patched'
            self.inference_patch_size = (64, 64)
            self.inference_stride = 32  # 50% overlap for smoother predictions
            self.pixel_batch_size = 4096

            # Sliding window parameters (for within-tile predictions)
            self.ssftt_patch_size = (16, 16)
            self.ssftt_stride = 8

            # Cascading classifier - use background (index 1) to separate fabric from background
            # Note: For pixel-level models, this is less important as they handle spatial info better
            self.cascading_classifier = False  # Disabled for pixel-level models
            self.background_threshold = 0.5
            self.background_index = 1
            
            # Uncertainty estimation
            self.enable_uncertainty = False
            self.num_mc_passes = 10

    config = ParameterConfig()
    
    # Ensure experiment directory exists before creating tensorboard
    experiment_dir = os.path.join(_get_working_dir(), 'log', config.experiment)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Don't delete previous tensorboard logs to avoid conflicts with running tensorboard
    config.writer = create_tb(experiment_dir, delete_previous=False)
    run_pipeline(config)