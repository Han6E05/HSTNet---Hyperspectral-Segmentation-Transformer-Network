"""
Hyperspectral Image Dataset Creation Pipeline

This script automates the creation of hyperspectral image (HSI) datasets from provided CSV files that list the HSI files and
their metadata. It supports the generation of datasets in various forms: tiled datasets for convolutional neural networks (CNNs)
and raw datasets for general use.

The pipeline processes the listed HSI files in three different configurations:
1. Smart Tiling: Extracts uniform tiles from HSI files for training UNet architectures, using non-overlapping tiles of specified dimensions.
2. Smart Patching: Extracts small, densely overlapped patches from HSI files for training 1D-CNN models, intended to capture fine-grained spectral information.
3. Raw Extraction: Copies raw HSI data into a dataset format without any tiling or patching, suitable for test scenarios or applications requiring full image data.

Functions:
- create_hsi_dataset_from_csv: Creates a dataset from HSI files listed in a CSV, with configurable extraction modes and preprocessing options.

Key Parameters for `create_hsi_dataset_from_csv`:
- csv_file: Path to the CSV file listing HSI images and metadata.
- hsimage_folder: Directory containing the HSI files referenced in the csv_file.
- extraction_mode: Method of data extraction ('smart_tiling', 'smart_patching', or 'raw').
- tile_size: Dimensions of the tiles or patches to be extracted (applicable in tiling and patching modes).
- stride: Overlap between consecutive tiles or patches (applicable in tiling and patching modes).
- use_aggregation: Whether to aggregate spectral data (applicable in patching mode).
- preprocessing: List of preprocessing operations to apply to each image before extraction.
- rgb_folder: Specify rgb image folder, default to same as hsimage_folder
- mask_folder: Specify mask image folder, default to same as hsimage_folder
- hsimage_folder_prefix: folder of folders containing hsimage files

Usage:
The script is executed as the main module and processes all listed configurations sequentially. Adjustments to the dataset creation parameters should be made in accordance with the specific requirements of the planned usage of the datasets.

Dependencies:
- elements.load_data: Module that includes functions for loading and creating HSI datasets.
- elements.preprocess: Module containing preprocessing classes and functions specific to hyperspectral data.
- pathlib: for finding current file path

Example:
Running this script will create three different datasets as specified in the CSV files and parameters set in the calls to `create_hsi_dataset_from_csv`.

Author:
- [Milad Isakhani Zakaria]
- modified by [Felix Yang]
"""

from pathlib import Path
# working dir
def _get_working_dir():
    return Path(__file__).resolve().parent

# configure logging
from elements.utils import LoggerSingleton
LoggerSingleton.setup_logger(_get_working_dir())
logger = LoggerSingleton.get_logger()

from elements.load_data import load_hsi_dataset, create_hsi_dataset_from_csv
from elements.preprocess import HyperHuePreprocessor, SpectralNorm

# executiona
if __name__ == '__main__':

    size = {
        'smart_tiling': 64,
        'smart_patching': 5,
        'raw': None
    }
    aggregation = {
        'smart_tiling': False,
        'smart_patching': True,
        'raw': False
    }

    BASE_DIR = Path(__file__).resolve().parent
    csv_folder = Path('/home/student/S2/HIT/dataset')
    
    # For SSFTT training: use smart_tiling with 64x64 tiles
    filename = 'test_set'
    mode = 'smart_tiling' # 'raw', 'smart_patching', or 'smart_tiling'

    create_hsi_dataset_from_csv(
        csv_file= csv_folder / (filename + '.csv'),                          # csv file containing file names, fabric compositions and hsimage locations
        hsimage_folder= '/media/public_shared/Projecten/extern/HIT/Tianhan/data/',# [UNUSED] substituted by rgb_folder, mask_folder and hsimage_folder_prefix (folder of hsimage files)
        dataset_path= csv_folder / (filename + '.npz'),                      # datapath to save the dataset
        extraction_mode= mode,                                             # extraction mode: smart_tiling, smart_patching, raw
        tile_size= (size[mode], size[mode]),                               # tile size for smart_tiling and smart_patching modes
        stride= size[mode],                                                # stride for smart_tiling and smart_patching modes
        use_aggregation= aggregation[mode],                                # True for smart_patching otherwise False
        preprocessing= [],                                                 # preprocessing list either empty or HyperHuePreprocessor
        use_float16= True,                                                 # Use float16 to reduce memory usage by ~50%
        save_visualization= False,                                         # Don't save RGB/tiled/mask images for training datasets
        use_incremental_save= True)                                        # Save tiles incrementally to avoid memory issues
    #     rgb_folder= BASE_DIR / 'dataset/example',                          # specify rgb image folder, default to same as hsimage_folder
    #     mask_folder= BASE_DIR / 'dataset/example',                         # specify mask image folder, default to same as hsimage_folder
    #     hsimage_folder_prefix='/media/public_shared/Projecten/extern/HIT/Tianhan/data/',   # folder of folders containing hsimage files
    # )
