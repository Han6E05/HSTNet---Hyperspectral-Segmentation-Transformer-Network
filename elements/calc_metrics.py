import numpy as np
import os
NoneType = type(None)
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

# configure logging
from elements.utils import LoggerSingleton
logger = LoggerSingleton.get_logger()

def calculate_metrics(results_dict,saved_dir, config):
    """
    calculate metrics for each file in the prediction dictionary.

    :param results_dict: dictionary containing predictions, labels, and metadata
    :param saved_dir: directory to save the predictions

    :return: updated dictionary with calculated metrics
    """
    class_names = results_dict['class_names']
    num_classes = len(class_names)
    
    # Background index (index 1 is background, index 0 is void/unused)
    background_index = 1
    
    for file_name, data in results_dict['data'].items():
        output_image = results_dict['output_image_mapping'][file_name]
        label_image = results_dict['label_image_mapping'][file_name]

        # Calculate composition over ALL pixels (including background)
        # This gives us the true distribution: background + fabrics = 100%
        data['predicted_composition'] = np.mean(output_image, axis=(0, 1))
        data['labels_composition'] = np.mean(label_image, axis=(0, 1))  # Use mean instead of median for consistency

        # Calculate fabric-only composition (excluding background)
        # This is what users care about: "what percentage of each fabric in the fabric regions"
        pred_fabric_sum = data['predicted_composition'][2:].sum()  # Sum of all fabric classes (skip void and background)
        true_fabric_sum = data['labels_composition'][2:].sum()
        
        # Normalize fabric compositions to 100% (only if there are fabrics)
        if pred_fabric_sum > 0:
            data['predicted_fabric_composition'] = data['predicted_composition'][2:] / pred_fabric_sum
        else:
            data['predicted_fabric_composition'] = np.zeros(num_classes - 2)
            
        if true_fabric_sum > 0:
            data['labels_fabric_composition'] = data['labels_composition'][2:] / true_fabric_sum
        else:
            data['labels_fabric_composition'] = np.zeros(num_classes - 2)

        # Calculate metrics using FULL composition (including background)
        # This evaluates the model's overall performance including background detection
        cosine_sim_full = cosine_similarity([data['labels_composition']], [data['predicted_composition']])[0][0]
        mse_full = mean_squared_error(data['labels_composition'], data['predicted_composition'])
        mae_full = mean_absolute_error(data['labels_composition'], data['predicted_composition'])

        # Calculate metrics using FABRIC-ONLY composition (excluding background)
        # This evaluates the model's performance on fabric classification only
        cosine_sim_fabric = cosine_similarity([data['labels_fabric_composition']], [data['predicted_fabric_composition']])[0][0]
        mse_fabric = mean_squared_error(data['labels_fabric_composition'], data['predicted_fabric_composition'])
        mae_fabric = mean_absolute_error(data['labels_fabric_composition'], data['predicted_fabric_composition'])

        # Store both sets of metrics
        data['cosine_sim'] = cosine_sim_full
        data['mse'] = mse_full
        data['mae'] = mae_full
        data['cosine_sim_fabric'] = cosine_sim_fabric
        data['mse_fabric'] = mse_fabric
        data['mae_fabric'] = mae_fabric

    # display metrics for each file
    logger.info(f"{'file name':<15} {'cosine (full)':<20} {'mae (full)':<15} {'mse (full)':<15} {'cosine (fabric)':<20} {'mae (fabric)':<15} {'mse (fabric)':<15}")
    logger.info("-" * 120)
    
    # Calculate mean metrics
    all_cosine_sims = []
    all_maes = []
    all_mses = []
    all_cosine_sims_fabric = []
    all_maes_fabric = []
    all_mses_fabric = []
    
    for file_name, data in results_dict['data'].items():
        logger.info(f"{file_name:<15} {data['cosine_sim']:<20.4f} {data['mae']:<15.4f} {data['mse']:<15.4f} {data['cosine_sim_fabric']:<20.4f} {data['mae_fabric']:<15.4f} {data['mse_fabric']:<15.4f}")
        all_cosine_sims.append(data['cosine_sim'])
        all_maes.append(data['mae'])
        all_mses.append(data['mse'])
        all_cosine_sims_fabric.append(data['cosine_sim_fabric'])
        all_maes_fabric.append(data['mae_fabric'])
        all_mses_fabric.append(data['mse_fabric'])
    
    # Log mean metrics
    mean_cosine_sim = np.mean(all_cosine_sims)
    mean_mae = np.mean(all_maes)
    mean_mse = np.mean(all_mses)
    mean_cosine_sim_fabric = np.mean(all_cosine_sims_fabric)
    mean_mae_fabric = np.mean(all_maes_fabric)
    mean_mse_fabric = np.mean(all_mses_fabric)
    
    logger.info("-" * 120)
    logger.info(f"{'MEAN':<15} {mean_cosine_sim:<20.4f} {mean_mae:<15.4f} {mean_mse:<15.4f} {mean_cosine_sim_fabric:<20.4f} {mean_mae_fabric:<15.4f} {mean_mse_fabric:<15.4f}")
    logger.info("=" * 120)

    # display the average predicted composition for each sample (file name)
    class_names = results_dict['class_names']
    fabric_names = class_names[2:]  # Skip void (0) and background (1)
    
    logger.info("\naverage predicted and true FABRIC composition per sample (background excluded):")
    header = f"{'file name':<20} {' '.join([f'{name:<15}' for name in fabric_names])}"
    logger.info(header)
    logger.info("-" * len(header))

    for file_name, data in results_dict['data'].items():
        # Display fabric-only composition (normalized to 100%)
        avg_comp_pred = data['predicted_fabric_composition']
        comp_str_pred = " ".join([f"{value * 100:.1f}%".ljust(15) for value in avg_comp_pred])
        logger.info(f"{file_name}-pred".ljust(20) + comp_str_pred)

        avg_comp_true = data['labels_fabric_composition']
        comp_str_true = " ".join([f"{value * 100:.1f}%".ljust(15) for value in avg_comp_true])
        logger.info(f"{file_name}-true".ljust(20) + comp_str_true)
 

    # write results
    results_file_path = os.path.join(saved_dir,'log',config.experiment, 'results.txt')
    with open(results_file_path, 'w') as f:
        f.write(f"{'file name':<15} {'cosine (full)':<20} {'mae (full)':<15} {'mse (full)':<15} {'cosine (fabric)':<20} {'mae (fabric)':<15} {'mse (fabric)':<15}\n")
        f.write("-" * 120 + "\n")
        for file_name, data in results_dict['data'].items():
            f.write(f"{file_name:<15} {data['cosine_sim']:<20.4f} {data['mae']:<15.4f} {data['mse']:<15.4f} {data['cosine_sim_fabric']:<20.4f} {data['mae_fabric']:<15.4f} {data['mse_fabric']:<15.4f}\n")
        
        # Write mean metrics
        f.write("-" * 120 + "\n")
        f.write(f"{'MEAN':<15} {mean_cosine_sim:<20.4f} {mean_mae:<15.4f} {mean_mse:<15.4f} {mean_cosine_sim_fabric:<20.4f} {mean_mae_fabric:<15.4f} {mean_mse_fabric:<15.4f}\n")
        f.write("=" * 120 + "\n")

        f.write("\naverage predicted and true FABRIC composition per sample (background excluded):\n")
        header = f"{'file name':<20} {' '.join([f'{name:<15}' for name in fabric_names])}"
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        for file_name, data in results_dict['data'].items():
            avg_comp_pred = data['predicted_fabric_composition']
            comp_str_pred = " ".join([f"{value * 100:.1f}%".ljust(15) for value in avg_comp_pred])
            f.write(f"{file_name}-pred".ljust(20) + comp_str_pred + "\n")

            avg_comp_true = data['labels_fabric_composition']
            comp_str_true = " ".join([f"{value * 100:.1f}%".ljust(15) for value in avg_comp_true])
            f.write(f"{file_name}-true".ljust(20) + comp_str_true + "\n")

    return results_dict

