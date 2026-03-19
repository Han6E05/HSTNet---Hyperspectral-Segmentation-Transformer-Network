import torch
import torch.utils
import torch.utils.data
NoneType = type(None)

# configure logging
from elements.utils import LoggerSingleton
logger = LoggerSingleton.get_logger()

def create_prediction_dict(prediction_dict):
    """
    Create an output dictionary based on the processed prediction data.

    :param prediction_dict: Dictionary containing all prediction and metadata.
    :return: A dictionary organized by file names with respective mappings and labels.
    """
    output_dict = {
        'class_names': prediction_dict['class_names'],
        'file_names': list(prediction_dict['data'].keys())
    }

    # Handle case where input_image_mapping is None (no RGB images saved)
    if prediction_dict['input_image_mapping'] is not None:
        file_names_to_process = prediction_dict['input_image_mapping'].keys()
    else:
        file_names_to_process = prediction_dict['data'].keys()

    for file_name in file_names_to_process:
        file_data = prediction_dict['data'][file_name]
        output_dict[file_name] = {
            'input_image_mapping': prediction_dict['input_image_mapping'][file_name] if prediction_dict['input_image_mapping'] is not None else None,
            'output_image_mapping': prediction_dict['output_image_mapping'][file_name],
            'input_image_mask': prediction_dict['input_image_mask'][file_name] if prediction_dict['input_image_mask'] is not None else None,
            'labels_composition': file_data['labels_composition'],
            'predicted_composition': file_data['predicted_composition']
        }
        
        # Add uncertainty data if available
        if 'uncertainty' in file_data:
            output_dict[file_name]['uncertainty'] = file_data['uncertainty']
        if 'epistemic_uncertainty' in file_data:
            output_dict[file_name]['epistemic_uncertainty'] = file_data['epistemic_uncertainty']
        if 'aleatoric_uncertainty' in file_data:
            output_dict[file_name]['aleatoric_uncertainty'] = file_data['aleatoric_uncertainty']
        if 'per_class_uncertainty' in file_data:
            output_dict[file_name]['per_class_uncertainty'] = file_data['per_class_uncertainty']

    return output_dict


def create_results_dict(dataset):
    """
    Initialize the prediction dictionary with dataset metadata and placeholders for predictions.

    :param dataset: Dataset object containing the data and metadata.
    :return: A dictionary prepared to store prediction outputs and metadata.
    """
    return {
        'input_image_mapping': dataset.get_rgb_images(),
        'input_image_mask': dataset.get_mask(),
        'output_image_mapping': {},
        'label_image_mapping': {},
        'class_names': dataset.get_class_names(),
        'data': {}
    }


def save_prediction_dict(prediction_dict, path):
    """
    Save the prediction dictionary to the specified file path.

    :param prediction_dict: The dictionary containing prediction results and metadata.
    :param path: The file path where the dictionary should be saved.
    """
    torch.save(prediction_dict, path)
    logger.info(f"Prediction cube saved to {path}")