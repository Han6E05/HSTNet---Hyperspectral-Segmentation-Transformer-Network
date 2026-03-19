import torch.utils
import torch.utils.data
import os
import numpy as np
import pandas as pd
import cv2
import torch

from torch.utils.data import Dataset
from collections import defaultdict
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

from elements.common.data.datatypes.scandata import ScanData
from elements.preprocess import apply_flatfield_correction
from elements.visualize import visualize_extracted_tiles

# configure logging
from elements.utils import LoggerSingleton

logger = LoggerSingleton.get_logger()


# Placeholder class for backward compatibility with old datasets
class HSIAugmentation:
    """
    Placeholder class for loading old datasets that reference HSIAugmentation.
    This class is no longer used but is kept for backward compatibility.
    """

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return args[0] if args else None


class HSITileExtractor:
    """
    A class to extract tiles from a binary mask using 'smart_tiling' or 'smart_patching' or 'raw'.
    The class validates parameters, provides optimized region extraction methods, and supports
    settings for stride and tile size.
    """

    def __init__(self, mask, tile_size, extraction_mode, stride):
        """
        Initialize the extractor with necessary parameters.

        :param mask: Mask image with void (0,0,0), background (255,0,0), and fabric (255,255,255) regions
        :param tile_size: Tuple specifying (tile_width, tile_height)
        :param extraction_mode: 'smart_tiling' or 'smart_patching'
        :param stride: Stride for the grid search method
        """
        self.mask = mask
        self.tile_width, self.tile_height = tile_size
        self.mask_height, self.mask_width = mask.shape[:2]
        self.extraction_mode = extraction_mode
        self.stride = stride

        self._validate_parameters()

    def _validate_parameters(self):
        """Validates the initialization parameters."""
        valid_modes = ['smart_tiling', 'smart_patching']
        if self.extraction_mode not in valid_modes:
            raise ValueError(f"Invalid extraction_mode: {self.extraction_mode}. Must be one of: {valid_modes}.")

        if self.stride <= 0:
            raise ValueError(f"Stride must be a positive integer. Received: {self.stride}")

    def extract_tiles(self):
        """Main method to extract regions based on the selected extraction_mode."""
        if self.extraction_mode == 'smart_tiling':
            return self._extract_tiles_smart_tiling()
        elif self.extraction_mode == 'smart_patching':
            return self._extract_tiles_smart_patching()

    def _extract_tiles_smart_tiling(self):
        """
        Perform a grid search over the entire image from (0,0), discarding tiles that are fully void.
        If any part of the tile is non-void, the tile and coordinates are returned.

        :return: List of coordinates for extracted tiles (x1, y1, x2, y2)
        """
        tiles = []
        for y in range(0, self.mask_height - self.tile_height + 1, self.stride):
            for x in range(0, self.mask_width - self.tile_width + 1, self.stride):
                tile = self.mask[y:y + self.tile_height, x:x + self.tile_width]

                # discard tile if it is completely void
                if np.all(tile == [0, 0, 0]):
                    continue

                # return coordinates in (x1, y1, x2, y2) format
                tiles.append((x, y, x + self.tile_width, y + self.tile_height))

        return tiles

    def _extract_tiles_smart_patching(self):
        """
        Start grid search from the first non-void pixel; extract tiles covering only fabric or background
        regions, with no void areas present.

        :return: List of coordinates for extracted tiles (x1, y1, x2, y2)
        """
        tiles = []
        start_y, start_x = np.where(np.any(self.mask != [0, 0, 0], axis=-1))[0][0], 0

        # start grid search over non-void areas
        for y in range(start_y, self.mask_height - self.tile_height + 1, self.stride):
            for x in range(start_x, self.mask_width - self.tile_width + 1, self.stride):
                tile = self.mask[y:y + self.tile_height, x:x + self.tile_width]

                # skip tiles that contain any void pixels, but allow background ([255, 0, 0]) or fabric ([255, 255, 255])
                if np.any((tile == [0, 0, 0]).all(axis=-1)):
                    continue

                # ensure tile is either entirely fabric or entirely background
                if np.all(tile == [255, 255, 255]) or np.all(tile == [255, 0, 0]):
                    tiles.append((x, y, x + self.tile_width, y + self.tile_height))

        return tiles


def stratified_split_by_composition(dataset, train_ratio, model_type, check_stratification=False):
    """
    Perform a stratified split on the dataset based on unique composition labels.

    :param dataset: Instance of HSITileDataset
    :param train_ratio: Ratio of the dataset to allocate to training (default 0.8)
    :param model_type: value indicating if the dataset is for U-Net.txt (with 4D labels) or Cnn (with 2D labels)
    :param check_stratification: Whether to check the results of the stratified split
    :return: train_dataset, val_dataset
    """
    # For large datasets with 4D labels, use a simpler random split to avoid memory issues
    if hasattr(dataset, 'labels') and isinstance(dataset.labels, torch.Tensor) and dataset.labels.ndim == 4:
        num_samples = len(dataset)
        # Use random split for ANY 4D dataset to avoid memory issues during label processing
        print(f"\n  4D labels detected ({num_samples} samples)")
        print(f"  Using simple random split to avoid memory issues...")

        indices = list(range(num_samples))
        train_size = int(num_samples * train_ratio)

        # Use numpy for faster random sampling
        import numpy as np
        np.random.seed(42)
        np.random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        print(f"  Train samples: {len(train_indices)}")
        print(f"  Val samples: {len(val_indices)}")

        return train_dataset, val_dataset

    # Original stratified split logic for smaller datasets
    file_name_hashes = [hash(f) for f in dataset.file_names]
    file_name_tensor = torch.tensor(file_name_hashes).view(-1, 1).float()

    # process labels based on the model structure
    if model_type == 'unet' or model_type == 'cnn3d':
        # Check if labels are already 4D or 2D
        if dataset.labels.ndim == 4:
            # for unet dataset, reshape labels to (samples, composition, pixels) and find the mode composition
            num_samples, num_classes, width, height = dataset.labels.shape
            print(f"  Processing 4D labels for {model_type} model ({num_samples} samples)...")
            labels_reshaped = dataset.labels.view(num_samples, num_classes, -1)

            # find the most repeated composition for each tile
            labels_reduced = []
            for i in range(num_samples):
                # Use mean instead of mode for speed (much faster than torch.unique)
                # This gives us the average composition across the tile
                mean_composition = labels_reshaped[i].mean(dim=1)
                labels_reduced.append(mean_composition)

                if (i + 1) % 1000 == 0:
                    print(f"    Processed {i+1}/{num_samples} samples...")

            labels_reduced = torch.stack(labels_reduced)
            print(f"  Completed processing labels.")
        elif dataset.labels.ndim == 2:
            # Labels are already 2D (old dataset format), use directly
            print(f"  Detected 2D labels for {model_type} model (old dataset format)...")
            labels_reduced = dataset.labels
        else:
            raise ValueError(f"Unexpected label dimensions for {model_type}: {dataset.labels.shape}")
    else:
        labels_reduced = dataset.labels
        # Ensure labels_reduced is a tensor
        if not isinstance(labels_reduced, torch.Tensor):
            if isinstance(labels_reduced, list):
                labels_reduced = torch.stack(labels_reduced) if labels_reduced else torch.empty(0, 0)
            else:
                labels_reduced = torch.tensor(labels_reduced)

        # Check if labels are 4D (spatial labels) and reduce to 2D
        if labels_reduced.ndim == 4:
            # Labels are (samples, num_classes, height, width)
            # Reduce to (samples, num_classes) by taking mean across spatial dimensions
            print(f"  Detected 4D labels, reducing to 2D for {model_type} model...")
            # Process in batches to avoid memory issues
            batch_size = 1000
            num_samples = labels_reduced.shape[0]
            reduced_labels_list = []

            for i in range(0, num_samples, batch_size):
                end_idx = min(i + batch_size, num_samples)
                batch = labels_reduced[i:end_idx]
                reduced_batch = batch.mean(dim=(2, 3))
                reduced_labels_list.append(reduced_batch)

                if (i + batch_size) % 5000 == 0:
                    print(f"    Processed {min(i + batch_size, num_samples)}/{num_samples} samples...")

            labels_reduced = torch.cat(reduced_labels_list, dim=0)
            del reduced_labels_list
            torch.cuda.empty_cache()

        elif labels_reduced.ndim == 3:
            # Labels are (samples, num_classes, length) - for 1D CNN
            labels_reduced = labels_reduced.mean(dim=2)
        elif labels_reduced.ndim != 2:
            raise ValueError(f"Unexpected label dimensions: {labels_reduced.shape}")

    # combine file names and labels for unique composition checking
    combined = torch.cat((file_name_tensor, labels_reduced), dim=1)
    unique_combinations, unique_ids = torch.unique(combined, dim=0, return_inverse=True)

    data_df = pd.DataFrame({
        'tile_index': range(len(dataset)),
        'composition_label': unique_ids
    })

    # Check class distribution
    class_counts = data_df['composition_label'].value_counts()
    total_samples = len(data_df)
    num_classes = len(class_counts)

    print(f"\nDataset split info:")
    print(f"  Total samples: {total_samples}")
    print(f"  Unique compositions: {num_classes}")
    print(f"  Train ratio: {train_ratio}")
    print(f"  Expected train samples: {int(total_samples * train_ratio)}")
    print(f"  Expected val samples: {int(total_samples * (1 - train_ratio))}")

    # Check if dataset is too small for stratified split
    min_samples_per_class = 2
    val_size = int(total_samples * (1 - train_ratio))

    if val_size < num_classes * min_samples_per_class:
        print(f"\n WARNING: Dataset too small for stratified split!")
        print(f"  Validation set ({val_size} samples) < Required ({num_classes * min_samples_per_class} samples)")
        print(f"  Performing random split instead...")

        train_indices, val_indices = train_test_split(
            data_df['tile_index'].values,
            test_size=1 - train_ratio,
            random_state=42,
        )

        train_indices, val_indices = list(train_indices), list(val_indices)
        train_dataset = Subset(dataset, train_indices)
        val_dataset = Subset(dataset, val_indices)

        if check_stratification:
            check_stratified_split(train_dataset, val_dataset, dataset.file_names, unique_ids)

        return train_dataset, val_dataset

    rare_classes = class_counts[class_counts < 2].index.tolist()

    if rare_classes:
        # Handle rare classes (with only 1 sample)
        # Separate rare samples from common samples
        rare_mask = data_df['composition_label'].isin(rare_classes)
        rare_indices = data_df[rare_mask]['tile_index'].values
        common_df = data_df[~rare_mask]

        if len(common_df) > 0:
            # Perform stratified split on common classes
            train_indices, val_indices = train_test_split(
                common_df['tile_index'].values,
                stratify=common_df['composition_label'],
                test_size=1 - train_ratio,
                random_state=42,
            )

            # Add rare samples to training set
            train_indices = np.concatenate([train_indices, rare_indices])

            print(f"Warning: {len(rare_classes)} rare composition(s) with only 1 sample added to training set")
        else:
            # All classes are rare, do random split
            print("Warning: All classes have only 1 sample, performing random split")
            train_indices, val_indices = train_test_split(
                data_df['tile_index'].values,
                test_size=1 - train_ratio,
                random_state=42,
            )
    else:
        # Normal stratified split
        train_indices, val_indices = train_test_split(
            data_df['tile_index'].values,
            stratify=data_df['composition_label'],
            test_size=1 - train_ratio,
            random_state=42,
        )

    train_indices, val_indices = list(train_indices), list(val_indices)
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    if check_stratification:
        check_stratified_split(train_dataset, val_dataset, dataset.file_names, unique_ids)

    return train_dataset, val_dataset


def check_stratified_split(train_dataset, val_dataset, file_names, unique_ids):
    """
    Count and print the number of samples for each unique label per file name in both train and validation sets,
    along with the percentage of samples in the validation set.

    :param train_dataset: Subset of HSITileDataset for the training set
    :param val_dataset: Subset of HSITileDataset for the validation set
    :param file_names: Original list of file names from the complete dataset
    :param unique_ids: Unique IDs for each label combination
    """

    def count_samples(dataset):
        """
        Helper function to count samples for each unique label in a dataset subset.
        """
        label_counts = defaultdict(lambda: defaultdict(int))

        for idx in dataset.indices:
            file_name = file_names[idx]
            label_id = unique_ids[idx].item()  # Retrieve unique ID for the label
            label_counts[file_name][label_id] += 1  # Increment count for this label within the file name

        return label_counts

    # get sample counts for train and validation sets
    train_label_counts = count_samples(train_dataset)
    val_label_counts = count_samples(val_dataset)

    # print results side-by-side with percentage calculation
    print("Sample counts per unique label per file name (Train vs Validation):\n")
    for file_name in set(train_label_counts.keys()).union(val_label_counts.keys()):
        print(f"File Name: {file_name}")
        unique_labels = set(train_label_counts[file_name].keys()).union(val_label_counts[file_name].keys())

        for label_id in unique_labels:
            train_count = train_label_counts[file_name].get(label_id, 0)
            val_count = val_label_counts[file_name].get(label_id, 0)
            total_count = train_count + val_count
            val_percentage = (val_count / total_count * 100) if total_count > 0 else 0
            print(
                f"  Label ID: {label_id}, Train Count: {train_count}, Validation Count: {val_count}, Validation %: {val_percentage:.2f}%")
        print()


class HSITileDataset(Dataset):
    def __init__(self, tiles, labels, coords, file_names, rgb_images, masks, tiled_images, class_names,
                 training_mode=False, preprocessing=None, convert_to_rgb=False, rgb_bands=(70, 50, 20)):
        """
        Custom dataset for HSI tile data with optional preprocessing.

        :param tiles: Tensor containing tile data (num_tiles, channels, tile_size, tile_size)
        :param labels: Tensor containing labels for each tile (num_tiles, num_classes)
        :param coords: Tensor containing tile coordinates (num_tiles, 4)
        :param file_names: Array containing file names for each tile (num_tiles,)
        :param rgb_images: Dictionary mapping file names to their corresponding RGB image arrays
        :param masks: Dictionary mapping file names to their corresponding mask arrays
        :param tiled_images: Dictionary mapping file names to their corresponding tiled images
        :param class_names: List of class names corresponding to label compositions
        :param training_mode: Boolean indicating if the dataset is used for training; if True, skips returning rgb/tiled images
        :param preprocessing: Optional preprocessing transformations to apply to the tiles
        :param convert_to_rgb: If True, converts hyperspectral tiles to RGB using specified bands
        :param rgb_bands: Tuple of 3 band indices to use for RGB conversion (R, G, B)
        """
        if preprocessing:
            print("Applying preprocessing during dataset initialization...")
            tiles = torch.stack([torch.tensor(preprocessing(image=tile.numpy())['image']) for tile in tiles])

        self.tiles = tiles
        self.labels = labels
        self.coords = coords
        self.file_names = file_names
        self.rgb_images = rgb_images
        self.tiled_images = tiled_images
        self.masks = masks
        self.class_names = class_names
        self.training_mode = training_mode
        self.convert_to_rgb = convert_to_rgb
        self.rgb_bands = rgb_bands

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        """
        Retrieve the tile, label, and coordinates for a given index, optionally excluding RGB and tiled images.

        :param idx: Index of the tile to retrieve
        :return: Tuple containing (tile, label, coordinates) if training_mode=True,
                 otherwise (tile, label, coordinates, file_name, RGB image, tiled image)
        """
        tile = self.tiles[idx]
        label = self.labels[idx]

        # Convert to RGB if requested
        if hasattr(self, 'convert_to_rgb') and self.convert_to_rgb:
            # tile shape: (channels, height, width) or (1, channels, height, width)
            if tile.dim() == 4:
                tile = tile.squeeze(0)  # Remove batch dimension if present

            # Extract RGB bands
            r_band = tile[self.rgb_bands[0], :, :]
            g_band = tile[self.rgb_bands[1], :, :]
            b_band = tile[self.rgb_bands[2], :, :]

            # Stack to create RGB image: (3, height, width)
            tile = torch.stack([r_band, g_band, b_band], dim=0)

            # Normalize to [0, 1] range
            tile = (tile - tile.min()) / (tile.max() - tile.min() + 1e-8)

        # Convert to fusion format if requested (spectral channels, RGB depth, H, W)
        elif hasattr(self, 'convert_to_fusion') and self.convert_to_fusion:
            # tile shape: (1, channels, height, width) or (channels, height, width)
            if tile.dim() == 4:
                tile = tile.squeeze(0)  # Remove batch dimension if present

            # tile now: (channels, height, width) = (224, H, W)
            channels, height, width = tile.shape

            # Extract RGB bands - use indexing instead of creating new tensors
            r_idx, g_idx, b_idx = self.rgb_bands

            # Normalize RGB bands to [0, 1] in-place to save memory
            r_band = tile[r_idx:r_idx+1, :, :]  # (1, H, W)
            g_band = tile[g_idx:g_idx+1, :, :]  # (1, H, W)
            b_band = tile[b_idx:b_idx+1, :, :]  # (1, H, W)

            r_band = (r_band - r_band.min()) / (r_band.max() - r_band.min() + 1e-8)
            g_band = (g_band - g_band.min()) / (g_band.max() - g_band.min() + 1e-8)
            b_band = (b_band - b_band.min()) / (b_band.max() - b_band.min() + 1e-8)

            # Create RGB depth dimension by stacking: (3, H, W)
            rgb_stack = torch.cat([r_band, g_band, b_band], dim=0)

            # Expand spectral dimension to match RGB: (channels, 3, H, W)
            # Use unsqueeze + expand (view-based, no memory copy) instead of repeat
            tile = tile.unsqueeze(1).expand(channels, 3, height, width).contiguous()

            # Multiply spectral with RGB to create fusion
            # This creates (224, 3, H, W) where each spectral band is modulated by RGB
            tile = tile * rgb_stack.unsqueeze(0)

        if self.training_mode:
            return tile, label
        else:
            coord = self.coords[idx]
            file_name = self.file_names[idx]

            return tile, label, coord, file_name

    def get_rgb_images(self):
        return self.rgb_images

    def get_tiled_images(self):
        return self.tiled_images

    def get_class_names(self):
        return self.class_names

    def get_mask(self):
        return self.masks


class IndexedHSIDataset(Dataset):
    """
    Dataset that loads tiles lazily from individual files using an index.
    This avoids loading all tiles into memory at once.
    """

    def __init__(self, index_path, training_mode=True):
        """
        Args:
            index_path: Path to the index file (dataset_index.pt)
            training_mode: If True, returns only (tile, label) for training
        """
        import pathlib
        self.index_path = pathlib.Path(index_path)
        self.dataset_dir = self.index_path.parent

        # Load index
        index_data = torch.load(self.index_path, weights_only=False)
        self.tile_index = index_data['tile_index']
        self.class_names = index_data['class_names']
        self.num_samples = index_data['num_samples']
        self.extraction_mode = index_data['extraction_mode']
        self.tile_size = index_data.get('tile_size', None)
        self.use_aggregation = index_data.get('use_aggregation', False)
        self.dtype = index_data.get('dtype', 'float32')
        self.has_visualization = index_data.get('has_visualization', False)

        self.training_mode = training_mode

        # Simple cache: store recently accessed tiles
        from collections import OrderedDict
        self.cache = OrderedDict()
        self.cache_size = 1000  # Keep last 1000 tiles in memory

        logger.info(f"IndexedHSIDataset loaded from {index_path} with {self.num_samples} samples (lazy loading enabled)")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """Load tile from disk if not in cache"""
        # Check cache first
        if idx in self.cache:
            tile_data = self.cache[idx]
            # Move to end (most recently used)
            self.cache.move_to_end(idx)
        else:
            # Load from disk
            tile_info = self.tile_index[idx]
            tile_path = self.dataset_dir / tile_info['path']
            tile_data = torch.load(tile_path, weights_only=False)

            # Add to cache
            self.cache[idx] = tile_data

            # Evict oldest if cache is full
            if len(self.cache) > self.cache_size:
                self.cache.popitem(last=False)

        tile = tile_data['tile']
        label = tile_data['label']

        if self.training_mode:
            return tile, label
        else:
            coord = tile_data['coord']
            file_name = tile_data['file_name']
            return tile, label, coord, file_name

    def get_class_names(self):
        return self.class_names

    def clear_cache(self):
        """Clear the tile cache to free memory"""
        self.cache.clear()
        torch.cuda.empty_cache()


def create_hsi_dataset_from_csv(csv_file: str, hsimage_folder: str | None, dataset_path: str,
                                extraction_mode: str = 'raw',
                                tile_size: tuple = (64, 64), stride: int = 64, use_aggregation: bool = False,
                                preprocessing: list = None,
                                rgb_folder: str = None, mask_folder: str = None, hsimage_folder_prefix: str = None,
                                use_float16: bool = True, save_visualization: bool = False,
                                use_incremental_save: bool = True):
    """
    Creates and saves an HSI dataset from annotation masks, either using extraction modes of 'smart_tiling', 'smart_patching' or 'raw'.

    Prerequisites:
      - Each `.hsimage` file should have an associated `.png` RGB image and `_mask.png` segmentation mask
        in the same directory, with matching filenames.
      - Mask image must have void (0,0,0), background (255,0,0), and fabric (255,255,255) regions

    Parameters:
        csv_file (str): Path to a CSV file with filenames and class compositions.
        hsimage_folder (str): Directory containing `.hsimage` files, `.png` RGB images, and `_mask.png` segmentation masks.
        dataset_path (str): Path where the processed dataset will be saved.
        extraction_mode (str): Mode for tile extraction (e.g., 'smart_tiling') or 'raw' for non-tiled.
        tile_size (tuple): Size of each tile (height, width) when tiling is used.
        stride (int): Stride for tile extraction, applicable in tiling mode.
        use_aggregation (bool): If True, aggregates patches for CNN1D models.
        preprocessing (list): List of preprocessing transformations to apply to the hyperspectral image.
        rgb_folder (str): Optional, Path to the folder containing RGB images, default to hsimage_folder
        mask_folder (str): Optional, Path to the folder containing `.png` segmentation masks, default to hsimage_folder.
        use_float16 (bool): If True, use float16 instead of float32 to reduce memory usage. Default True.
        save_visualization (bool): If True, save RGB/tiled/mask images. Default False for training datasets.
        use_incremental_save (bool): If True, save tiles incrementally to avoid memory issues. Default True.

    Returns:
        HSITileDataset: The processed dataset with HSI tiles or raw samples, labels, and coordinates.
    """
    if rgb_folder is None:
        rgb_folder = hsimage_folder
    if mask_folder is None:
        mask_folder = hsimage_folder

    # Determine dtype based on use_float16 parameter
    dtype = np.float16 if use_float16 else np.float32
    torch_dtype = torch.float16 if use_float16 else torch.float32

    df = pd.read_csv(csv_file, dtype={'file_name': str})

    # Only create visualization mappings if requested
    rgb_images_mapping = {} if save_visualization else None
    tiled_images_mapping = {} if save_visualization else None
    rgb_images_mask = {} if save_visualization else None

    class_names = df.columns[1:].tolist()
    num_classes = len(class_names)
    scan_data = ScanData()

    # For incremental save mode, create directory structure
    if use_incremental_save and extraction_mode != 'raw':
        import pathlib
        dataset_dir = pathlib.Path(dataset_path).parent / pathlib.Path(dataset_path).stem
        tiles_dir = dataset_dir / 'tiles'
        tiles_dir.mkdir(parents=True, exist_ok=True)

        # Track tile metadata for index
        tile_index = []
        total_tiles = 0
    else:
        # Legacy mode: accumulate in lists
        tiles_list, coords_list, labels_list, file_names_list = [], [], [], []

    for _, row in df.iterrows():
        file_name = row['file_name']
        class_composition = row[class_names].values.astype(np.float32)

        if extraction_mode != 'raw':
            print(f"Processing: {file_name} (extraction_mode={extraction_mode}, tile_size={tile_size}, stride={stride}, aggregation={use_aggregation})")
        else:
            print(f"Processing: {file_name} (extraction_mode={extraction_mode}, aggregation={use_aggregation})")

        scan_data.load(os.path.join(hsimage_folder, f'{file_name}.hsimage'))
        ffc_image = apply_flatfield_correction(scan_data.get_raw(), scan_data.get_whiteref(),
                                               scan_data.get_darkref())

        # Convert to float16 immediately to save memory
        ffc_image = ffc_image.astype(dtype)

        # apply preprocessing steps
        if preprocessing:
            for transform in preprocessing:
                # ensure ffc_image shape matches expected input for transformations
                ffc_tensor = torch.tensor(ffc_image, dtype=torch_dtype).permute(2, 0, 1)  # shape (C, H, W)
                try:
                    ffc_tensor = transform(ffc_tensor)
                except Exception as e:
                    print(f"Direct transformation failed: {e}. Trying named argument approach.")
                    ffc_tensor = transform(image=ffc_tensor.numpy())['image']
                    ffc_tensor = torch.tensor(ffc_tensor, dtype=torch_dtype)
                ffc_image = ffc_tensor.permute(1, 2, 0).numpy()  # back to (H, W, C)

        # load RGB image and mask
        rgb_image = cv2.imread(os.path.join(rgb_folder, f'{file_name}.png'), cv2.IMREAD_COLOR)
        mask = cv2.imread(os.path.join(mask_folder, f'{file_name}_mask.png'), cv2.IMREAD_COLOR)

        # crop the railing edge from width, crop the height by 64 multiplier
        crop_height = (ffc_image.shape[0] // 64) * 64
        ffc_image_cropped = ffc_image[:crop_height, 64:-64]
        rgb_image_cropped = rgb_image[:crop_height, 64:-64]
        mask_cropped = mask[:crop_height, 64:-64]

        # create label matrix with float16/float32
        label_matrix = np.zeros((mask_cropped.shape[0], mask_cropped.shape[1], num_classes), dtype=dtype)

        # Mark different regions based on mask colors
        # Black pixels (0,0,0) represent background
        # White pixels (255,255,255) represent fabric
        label_matrix[(mask_cropped == [0, 0, 0]).all(axis=-1), 1] = 1  # mark black as background (index 1)
        label_matrix[(mask_cropped == [255, 255, 255]).all(axis=-1)] = class_composition  # mark white as fabric

        # tiling (used for training) or raw (used for inference) extraction mode
        if extraction_mode != 'raw':
            tile_extractor = HSITileExtractor(mask_cropped, tile_size, extraction_mode, stride)
            tile_coords = tile_extractor.extract_tiles()
            tiles_ffc = [ffc_image_cropped[c[1]:c[3], c[0]:c[2]] for c in tile_coords]
            tiles_labels = [label_matrix[c[1]:c[3], c[0]:c[2]] for c in tile_coords]

            # Only create tiled visualization if requested
            if save_visualization:
                tiled_image = visualize_extracted_tiles(rgb_image_cropped, tile_coords, file_name, extraction_mode)
                tiled_images_mapping[file_name] = tiled_image

            if use_aggregation:
                # specific structure for 1D cnn ==> shape is (bc,c,vector_length) (bc,1,224) spectral is transformed into input vector
                tiles_ffc = [np.mean(tile, axis=(0, 1)) for tile in tiles_ffc]
                tiles_labels = [np.mean(tile, axis=(0, 1)) for tile in tiles_labels]

            # Save tiles incrementally or accumulate in list
            if use_incremental_save:
                # Save each tile to individual file
                for i, (tile_ffc, tile_label, coord) in enumerate(zip(tiles_ffc, tiles_labels, tile_coords)):
                    if use_aggregation:
                        tile_tensor = torch.tensor(tile_ffc, dtype=torch_dtype).unsqueeze(0)  # (1, C)
                        label_tensor = torch.tensor(tile_label, dtype=torch_dtype)  # (num_classes,)
                    else:
                        # Per-tile归一化已禁用 - 保持原始反射率值
                        # 原因：旧模型是用未归一化数据训练的，需要保持一致
                        # 如果要启用归一化，需要重新训练模型
                        # tile_min = tile_ffc.min()
                        # tile_max = tile_ffc.max()
                        # if tile_max > tile_min:
                        #     tile_ffc = (tile_ffc - tile_min) / (tile_max - tile_min)
                        
                        tile_tensor = torch.tensor(tile_ffc, dtype=torch_dtype).permute(2, 0, 1)  # (C, H, W)
                        label_tensor = torch.tensor(tile_label, dtype=torch_dtype).permute(2, 0, 1)  # (num_classes, H, W)

                    coord_tensor = torch.tensor(coord, dtype=torch.int32)

                    # Save to individual file
                    tile_filename = f'{file_name}_tile_{total_tiles + i}.pt'
                    tile_path = tiles_dir / tile_filename
                    torch.save({
                        'tile': tile_tensor,
                        'label': label_tensor,
                        'coord': coord_tensor,
                        'file_name': file_name
                    }, tile_path)

                    # Add to index
                    tile_index.append({
                        'path': f'tiles/{tile_filename}',
                        'file_name': file_name
                    })

                    # Delete to free memory
                    del tile_tensor, label_tensor, coord_tensor

                total_tiles += len(tiles_ffc)
                # Delete intermediate data
                del tiles_ffc, tiles_labels
            else:
                # Legacy mode: accumulate in lists
                if use_aggregation:
                    tiles_tensor = torch.tensor(np.array(tiles_ffc), dtype=torch_dtype).unsqueeze(1)
                    labels_tensor = torch.tensor(np.array(tiles_labels), dtype=torch_dtype)
                else:
                    # 关键修复：添加 per-tile min-max 归一化
                    # 对每个 tile 进行归一化
                    tiles_ffc_normalized = []
                    for tile in tiles_ffc:
                        tile_min = tile.min()
                        tile_max = tile.max()
                        if tile_max > tile_min:  # 避免除以0
                            tile = (tile - tile_min) / (tile_max - tile_min)
                        tiles_ffc_normalized.append(tile)
                    
                    tiles_tensor = torch.tensor(np.array(tiles_ffc_normalized), dtype=torch_dtype).permute(0, 3, 1, 2)
                    labels_tensor = torch.tensor(np.array(tiles_labels), dtype=torch_dtype).permute(0, 3, 1, 2)

                coords_tensor = torch.tensor(np.array(tile_coords), dtype=torch.int32)
                tiles_list.append(tiles_tensor)
                labels_list.append(labels_tensor)
                coords_list.append(coords_tensor)
                file_names_list.extend([file_name] * len(tiles_tensor))

                # Delete intermediate data to free memory
                del tiles_ffc, tiles_labels, tiles_tensor, labels_tensor, coords_tensor

        else:
            # raw mode (no tiling) used for inference
            tile_coords = [[0, 0, ffc_image_cropped.shape[1], ffc_image_cropped.shape[0]]]

            # Only create tiled visualization if requested
            if save_visualization:
                tiled_image = visualize_extracted_tiles(rgb_image_cropped, tile_coords, file_name, 'raw')
                tiled_images_mapping[file_name] = tiled_image

            ffc_tensor = torch.tensor(ffc_image_cropped, dtype=torch_dtype).permute(2, 0, 1)
            label_tensor = torch.tensor(label_matrix, dtype=torch_dtype).permute(2, 0, 1)
            coord_tensor = torch.tensor(tile_coords[0], dtype=torch.int32)

            # Raw mode always uses list accumulation (not incremental save)
            tiles_list.append(ffc_tensor)
            labels_list.append(label_tensor)
            coords_list.append(coord_tensor)
            file_names_list.append(file_name)

            # Delete intermediate data to free memory
            del ffc_tensor, label_tensor, coord_tensor

        # Store visualization data only if requested
        if save_visualization:
            rgb_images_mapping[file_name] = rgb_image_cropped
            rgb_images_mask[file_name] = mask_cropped

        # Delete large intermediate variables to free memory
        del ffc_image, ffc_image_cropped, rgb_image_cropped, mask_cropped, label_matrix
        torch.cuda.empty_cache()

    # Create and save dataset based on mode
    if use_incremental_save and extraction_mode != 'raw':
        # Save index file
        index_data = {
            'tile_index': tile_index,
            'class_names': class_names,
            'num_samples': total_tiles,
            'extraction_mode': extraction_mode,
            'tile_size': tile_size,
            'use_aggregation': use_aggregation,
            'dtype': 'float16' if use_float16 else 'float32',
            'has_visualization': save_visualization
        }
        index_path = dataset_dir / 'dataset_index.pt'
        torch.save(index_data, index_path)
        print(f"Incremental dataset saved to {dataset_dir} with {total_tiles} tiles.")
        print(f"Index file: {index_path}")

        # Merge all tile files - load all at once then stack (faster than batch concat)
        print(f"Merging {total_tiles} tile files into single dataset...")
        all_tiles = []
        all_labels = []
        all_coords = []
        all_file_names = []

        for i, tile_info in enumerate(tile_index):
            tile_path = dataset_dir / tile_info['path']
            tile_data = torch.load(tile_path, weights_only=False)

            all_tiles.append(tile_data['tile'])
            all_labels.append(tile_data['label'])
            all_coords.append(tile_data['coord'])
            all_file_names.append(tile_data['file_name'])

            if (i + 1) % 1000 == 0:
                print(f"Loaded {i+1}/{total_tiles} tiles...")

        print("Stacking tensors...")
        tiles_tensor = torch.stack(all_tiles, dim=0)
        labels_tensor = torch.stack(all_labels, dim=0)
        coords_tensor = torch.stack(all_coords, dim=0)
        file_names_array = np.array(all_file_names)

        # Free memory
        del all_tiles, all_labels, all_coords, all_file_names
        torch.cuda.empty_cache()

        print("Creating dataset object...")
        dataset = HSITileDataset(
            tiles_tensor, labels_tensor, coords_tensor, file_names_array,
            rgb_images=rgb_images_mapping, masks=rgb_images_mask,
            tiled_images=tiled_images_mapping, class_names=class_names,
            training_mode=True
        )

        # Save final dataset
        print(f"Saving dataset to {dataset_path}...")
        torch.save(dataset, dataset_path)
        print(f"Dataset saved with {len(dataset)} samples.")

        # Clean up temporary files
        print(f"Cleaning up temporary tile files...")
        import shutil
        shutil.rmtree(tiles_dir)
        index_path.unlink()
        # Try to remove the dataset_dir if it's empty
        try:
            dataset_dir.rmdir()
        except:
            pass
        print(f"Temporary files deleted.")

        if use_float16:
            print(f"Using float16 (memory reduced by ~50% compared to float32)")
        if not save_visualization:
            print(f"Visualization data not saved (significant memory savings)")

        return dataset
    else:
        # Legacy mode: concatenate and create dataset
        if extraction_mode != 'raw':
            tiles_tensor = torch.cat(tiles_list, dim=0)
            labels_tensor = torch.cat(labels_list, dim=0)
            coords_tensor = torch.cat(coords_list, dim=0)

            # Delete lists to free memory
            del tiles_list, labels_list, coords_list
        else:
            # used lists instead of torch.stack to include images from multiple sizes
            tiles_tensor = tiles_list
            labels_tensor = labels_list
            coords_tensor = coords_list

        file_names_array = np.array(file_names_list)

        # create and save the dataset
        dataset = HSITileDataset(tiles_tensor, labels_tensor, coords_tensor, file_names_array,
                                 rgb_images_mapping, rgb_images_mask, tiled_images_mapping, class_names,
                                 training_mode=False if extraction_mode == 'raw' else True)
        torch.save(dataset, dataset_path)
        print(f"HSITileDataset saved to {dataset_path} with {len(tiles_tensor)} samples.")

        # Log memory savings info
        if use_float16:
            print(f"Using float16 (memory reduced by ~50% compared to float32)")
        if not save_visualization:
            print(f"Visualization data not saved (significant memory savings)")

        return dataset


def load_hsi_dataset(dataset_path, lazy_loading=False):
    """
    Load an HSI tiled dataset.

    :param dataset_path: Path to the saved dataset (.pt or .npz file)
    :param lazy_loading: Ignored, kept for backward compatibility
    :return: An instance of HSITileDataset
    """
    dataset = torch.load(dataset_path, weights_only=False)
    if not isinstance(dataset, HSITileDataset):
        raise TypeError(f"The loaded object is not of type 'HSITileDataset', but {type(dataset)}")
    logger.info(f"HSITileDataset loaded from {dataset_path} with {len(dataset)} samples.")
    return dataset


def load_prediction_dict(path):
    """
    Load the prediction dictionary from the specified file path.

    :param path: The file path from which to load the dictionary.
    :return: The loaded prediction dictionary.
    """
    try:
        prediction_dict = torch.load(path, weights_only=False)
        logger.info(f"Prediction cube loaded from {path}")
        return prediction_dict
    except Exception as e:
        logger.error(f"Failed to load prediction dictionary from {path}: {e}")
        raise
