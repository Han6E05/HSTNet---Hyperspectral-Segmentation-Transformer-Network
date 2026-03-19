import torch.utils
import torch.utils.data
import os
import torch

# configure logging
from elements.utils import LoggerSingleton
logger = LoggerSingleton.get_logger()

def save_model(model, directory, filename='model.npz'):
    """
    Save the model state to a specified directory with a given filename.

    :param model: The model whose state dict is to be saved.
    :param directory: The directory where the model will be saved.
    :param filename: The filename for the saved model state.
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = os.path.join(directory, filename)
        torch.save(model.state_dict(), file_path)
    except Exception as e:
        logger.error(f"Failed to save the model at {file_path}: {e}")
        raise



