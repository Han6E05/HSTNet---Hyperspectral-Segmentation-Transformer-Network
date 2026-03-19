from pathlib import Path

import numpy as np
import cv2
import albumentations
import torch
import torch.utils
import torch.utils.data
import numexpr as ne
from elements.visualize import visualize_image_and_mask

def _get_working_dir():
    return Path(__file__).resolve().parent

data_path = BASE_DIR = Path(__file__).resolve().parent

# configure logging
from elements.utils import LoggerSingleton
LoggerSingleton.setup_logger(_get_working_dir())
logger = LoggerSingleton.get_logger()


def compute_ffc_statistics(ffc_data):
    """
    compute mean, median, 25th percentile, and 75th percentile for a flat-field corrected image.

    :param ffc_data: input ffc image (3d array: height x width x channels)
    :return: tuple containing mean, median, 25th percentile, and 75th percentile for each channel
    """
    mean = np.mean(ffc_data, axis=(0, 1))
    median = np.median(ffc_data, axis=(0, 1))
    p25 = np.percentile(ffc_data, 25, axis=(0, 1))
    p75 = np.percentile(ffc_data, 75, axis=(0, 1))

    return mean, median, p25, p75

def apply_flatfield_correction(raw_data, white_frame, dark_frame):
    """
    Applies flat-field correction to a hyperspectral image cube, handling any zero or near-zero divisor values.

    :param raw_data: Raw hyperspectral image data
    :param white_frame: White reference frame
    :param dark_frame: Dark reference frame
    :return: Flat-field corrected HSI image
    """
    raw_data = raw_data.astype(np.float32)  # convert to float32 for consistent calculations
    white_frame = white_frame.astype(np.float32)
    dark_frame = dark_frame.astype(np.float32)

    dark_frame_mean = np.median(dark_frame, axis=0)
    bright_frame_mean = np.median(white_frame, axis=0)

    divisor = bright_frame_mean - dark_frame_mean
    divisor[divisor == 0] = 1e-5  # avoid division by zero

    ffc_img = ne.evaluate("(raw_data - dark_frame_mean) / divisor")    # apply flat-field correction with numexpr
    ffc_img = ne.evaluate("where(ffc_img < 0, 0, where(ffc_img > 1, 1, ffc_img))")  # In-place clipping

    return ffc_img

def convert_hsi_to_rgb(hsi_image, red_bin=(71, 85), green_bin=(99, 119), blue_bin=(147, 189)):
    """
    Converts an HSI image to RGB by averaging specific wavelength bins for red, green, and blue channels.

    :param hsi_image: Input HSI image
    :param red_bin: Tuple specifying the bin range for the red channel
    :param green_bin: Tuple specifying the bin range for the green channel
    :param blue_bin: Tuple specifying the bin range for the blue channel
    :return: RGB image in RGB order (not BGR)
    """
    hsi_image = hsi_image*255
    return np.stack([
        np.mean(hsi_image[..., red_bin[0]:red_bin[1]], axis=-1),
        np.mean(hsi_image[..., green_bin[0]:green_bin[1]], axis=-1),
        np.mean(hsi_image[..., blue_bin[0]:blue_bin[1]], axis=-1)
    ], axis=-1).astype(np.uint8)

class HyperHuePreprocessor:
    """
    Preprocessor for extracting Hyper-Hue and Saturation.
    Converts hyperspectral data (C, H, W) into a representation focusing on hyper-hue and saturation.
    """

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply the Hyper-Hue preprocessing to the input tensor.

        :param image: Input tensor of shape (C, H, W).
        :return: Preprocessed tensor of shape ((C - 1) + 1, H, W), where
                 (C - 1) corresponds to Hyper-Hue channels and the +1 channel is Saturation.
        """
        # Ensure input is 3D: (C, H, W)
        if image.ndim != 3:
            raise ValueError(f"Expected input with 3 dimensions (C, H, W), got {image.ndim} dimensions.")

        # apply the HyperHue computation
        hh, s, _ = hc2hhsi_torch(image)

        # concatenate hyper-hue and saturation along the channel dimension
        return torch.cat([hh, s.unsqueeze(0)], dim=0)

def hc2hhsi_torch(hc: torch.Tensor):
    dims = hc.shape[0]

    # saturation
    s = hc.max(dim=0)[0] - hc.min(dim=0)[0]

    # intensity
    i = 1 / dims * torch.sum(hc, dim=0)

    # hues
    hh = hc2hh_torch(hc)

    return hh, s, i

def hc2hh_torch(hc: torch.Tensor):
    # calculate the components c
    dims = hc.shape[0]
    rows = hc.shape[1]
    cols = hc.shape[2]

    # chromacities
    c = torch.zeros(size=(dims-1, rows, cols), dtype=torch.float32, device=hc.device)
    for i in range(dims - 1):
        nonZeroEle = dims - i # nonZeroEle is the number of non-zero elements of the base unit vector u1, u2, ...
        c[i, :, :] = ((nonZeroEle - 1) ** 0.5 / nonZeroEle ** 0.5) * hc[i, :, :] \
                     - (1 / ((nonZeroEle - 1) ** 0.5 * nonZeroEle ** 0.5)) * torch.sum(hc[i+1:dims, :, :], dim=0)

    # hues
    hh = torch.atan2(c[1:, ...], c[:-1, ...])
    return hh

class SpectralNorm(albumentations.BasicTransform):
    """
    Albumentations wrapper for the spectral_norm preprocessing method (used in hyper-spectral imaging)
    """

    def __init__(self, always_apply=True, p=1.0):
        super(SpectralNorm, self).__init__(always_apply=always_apply, p=p)

    @property
    def targets(self):
        return {"image": self.apply}

    def apply(self, array, **params):
        return spectral_norm(array)

    def get_transform_init_args_names(self):
        return ""

    def get_params_dependent_on_targets(self, params):
        return {}

def spectral_norm(input_image: np.ndarray) -> np.ndarray:
    """
    Normalize the spectral data for hyperspectral imaging.

    For data of shape:
    - (1, 224): Normalizes each spectral band vector individually (for CNN1D).
    - (224, H, W): Normalizes each pixel across the 224 spectral bands (for CNN3D).

    :param input_image: Input data, either (1, 224) or (224, H, W)
    :return: Spectrally normalized data.
    """
    array_copy = input_image.copy()

    if array_copy.ndim == 2:  # Case: (1, 224) when spectral bands are transformed into vectors for cnn1d
        norm = np.linalg.norm(array_copy, axis=1, keepdims=True)  # Shape: (1, 1)
        array_copy[norm == 0, :] = 1 / array_copy.shape[1]
        norm[norm == 0] = 1
        array_copy = array_copy / norm

    elif array_copy.ndim == 3:  # Case: (224, H, W) when hs image is tiled for unet or cnn3d
        norm = np.linalg.norm(array_copy, axis=0, keepdims=True)
        array_copy[:, norm[0] == 0] = 1 / array_copy.shape[0]
        norm[norm == 0] = 1
        array_copy = array_copy / norm

    else:
        raise ValueError(f"Unsupported input shape: {array_copy.shape}. Expected (1, 224) or (224, H, W).")

    return array_copy


def generate_binary_fabric_mask(image, file_name, visualize):
    """
    generate a fabric mask by converting to grayscale, applying gaussian blur, and using morphological operations.

    :param image: input rgb fabric image
    :param file_name: file name for display purposes
    :param visualize: whether to visualize the original and masked image
    :return: normalized binary mask (values 0 or 1)
    """
    # adjust contrast and brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=4, beta=10)

    # convert to grayscale and apply gaussian blur
    grayscale_image = cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2GRAY)
    blurred_image = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

    # apply otsu's threshold to create a binary mask
    _, mask = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # use morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=7)

    normalized_mask = (cleaned_mask // 255).astype(np.uint8)

    if visualize:
        visualize_image_and_mask(image, normalized_mask, file_name)

    return normalized_mask

def crop(image, region):
    """
    crop the image based on the provided region (x, y, width, height).

    :param image: input image to crop
    :param region: tuple specifying (x, y, width, height)
    :return: cropped image
    """
    x, y, w, h = region
    return image[y:y + h, x:x + w]

def extract_random_patches(mask, patch_size, num_patches, target_color):
    """
    Extract random patches from an image based on mask and target color.
    """
    coords = []
    mask_target = cv2.inRange(mask, target_color, target_color)

    while len(coords) < num_patches:
        y = np.random.randint(0, mask.shape[0] - patch_size[0])
        x = np.random.randint(0, mask.shape[1] - patch_size[1])
        patch = mask_target[y:y + patch_size[0], x:x + patch_size[1]]

        if cv2.countNonZero(patch) == np.prod(patch_size):  # All pixels match
            coords.append((x, y, x + patch_size[1], y + patch_size[0]))

    return coords