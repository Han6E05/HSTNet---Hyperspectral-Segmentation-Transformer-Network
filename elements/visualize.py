import numpy as np
NoneType = type(None)
from elements.common.visualize import launch_tb
import os
from torch.utils.tensorboard import SummaryWriter
import cv2
import matplotlib.pyplot as plt
from matplotlib.table import Table


def plot_spectral_data(ffc_data, title):
    """
    plot flat-field correction data across wavelengths for one or more samples.

    :param ffc_data: ffc data array with shape (num_samples, 1, 224) or (224,)
    :param title: title for the plot
    """
    wavelengths = np.linspace(935.61, 1720.23, 224)
    plt.figure(figsize=(10, 6))

    # plot multiple samples or a single sample
    if ffc_data.ndim == 3:
        for i, sample in enumerate(ffc_data[:, 0]):
            plt.plot(wavelengths, sample, label=f'patch {i}')
        plt.legend()
    else:
        plt.plot(wavelengths, ffc_data)

    plt.title(title)
    plt.xlabel('wavelength (nm)')
    plt.ylabel('ffc reflectance')
    plt.xlim(950, 1700)
    plt.ylim(0, 1.2)
    plt.tight_layout()
    plt.show()

def visualize_image_and_mask(image, mask, title):
    """
    Visualize the original image and its corresponding mask side by side.

    Args:
        image (numpy.ndarray): The original image loaded using OpenCV.
        mask (numpy.ndarray): The generated mask image.
        title (str): A title for the visualization, typically the base name of the processed file.
    """
    plt.figure(figsize=(12, 6))

    # display the original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f'Original Image: {title}')
    plt.axis('off')

    # display the mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask)
    plt.title(f'Mask: {title}')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def create_tb(experiment_dir: str, tb_name: str = "tensorboard", delete_previous: bool = True, start_tb: bool = True) -> SummaryWriter:
    """
    Create a TensorBoard SummaryWriter object, saving logs in the specified experiment directory.

    :param experiment_dir: Directory where TensorBoard logs will be saved.
    :param tb_name: Subdirectory for this specific TensorBoard run.
    :param delete_previous: Should the previous writer with the same name be deleted or should this value be appended?
    :param start_tb: Should a TensorBoard be started with this SummaryWriter.
    :return: The SummaryWriter object

    :example:

    True
    """
    # full path for TensorBoard logs
    tb_path = os.path.join(experiment_dir, tb_name)

    # optionally delete the previous log directory
    if delete_previous and os.path.exists(tb_path):
        import shutil
        shutil.rmtree(tb_path)

    # create the SummaryWriter at the specified path
    writer = SummaryWriter(log_dir=tb_path)

    # optionally start TensorBoard
    if start_tb:
        launch_tb(tb_path)

    return writer

def show_loss_tb(loss, epoch, writer: SummaryWriter, name: str = "loss"):
    """
    Show loss on a TensorBoard.

    :param loss: Current loss
    :param epoch: Tensorboard step
    :param name: Name of the metric
    :param writer: Writer to use

    """
    writer.add_scalar(name, loss, epoch)
    writer.flush()

def display_output_mapping(output_dict, output_dir):
    """
    Display the segmentation output for each file in the test dataset using specified classes for RGB channels.
    Also displays uncertainty maps if available.

    :param output_dict: dictionary containing predictions, labels, and output mappings
    :param output_dir: Directory where output will be saved
    """
    class_names = output_dict['class_names']
    file_names = output_dict['file_names']
    
    # Check if uncertainty data is available
    has_uncertainty = False
    if len(file_names) > 0:
        first_file = file_names[0]
        if first_file in output_dict and 'uncertainty' in output_dict[first_file]:
            has_uncertainty = True
            print(f"Uncertainty data detected - will generate uncertainty visualizations")

    for file_name in file_names:
        all_true = output_dict[file_name]['labels_composition']
        top_indexes = np.argsort(all_true)[::-1]  # Sort indices by descending values
        red_idx = top_indexes[0]
        green_idx = top_indexes[1]
        blue_idx = top_indexes[2]

        converted_rgb_image = output_dict[file_name]['input_image_mapping']
        predicted_segmentation_mask = (output_dict[file_name]['output_image_mapping'][:, :, [red_idx, green_idx, blue_idx]]*255).astype(np.uint8)
        ground_truth_mask = output_dict[file_name]['input_image_mask']

        fabrics = [class_names[red_idx], class_names[green_idx], class_names[blue_idx]]
        true = [all_true[red_idx], all_true[green_idx], all_true[blue_idx]]

        # Only process ground truth mask if it exists (not None)
        if ground_truth_mask is not None:
            # Make a copy to avoid modifying the original
            ground_truth_mask = ground_truth_mask.copy()
            
            # assign true values to fabrics on the mask for visualization purposes
            red_mask = (ground_truth_mask[:, :, 0] == 255) & (ground_truth_mask[:, :, 1] == 0) & (ground_truth_mask[:, :, 2] == 0)
            ground_truth_mask[red_mask] = [0, 0, 0]

            # set (255, 255, 255) pixels to scaled true values
            white_mask = (ground_truth_mask[:, :, 0] == 255) & (ground_truth_mask[:, :, 1] == 255) & (ground_truth_mask[:, :, 2] == 255)
            true_scaled = (np.array([all_true[red_idx], all_true[green_idx], all_true[blue_idx]]) * 255).astype(np.int32)
            ground_truth_mask[white_mask] = true_scaled

        all_pred = output_dict[file_name]['predicted_composition']
        pred = np.round([all_pred[red_idx], all_pred[green_idx], all_pred[blue_idx]],2)
        red_info = (fabrics[0],true[0],pred[0])
        green_info = (fabrics[1],true[1],pred[1])
        blue_info = (fabrics[2],true[2],pred[2])
        fig = visualize_segmentation_results(converted_rgb_image=converted_rgb_image,ground_truth_segmentation=ground_truth_mask,
                                     predicted_segmentation=predicted_segmentation_mask,main_title=f'{file_name}',
                                     red_info=red_info,green_info=green_info,blue_info=blue_info)

        # Save the plot
        save_path = os.path.join(output_dir, f"{file_name}_results.png")
        fig.savefig(save_path)
        plt.close(fig)  # Close the plot to free up memory
        
        # Generate uncertainty visualization if available
        if has_uncertainty and 'uncertainty' in output_dict[file_name]:
            uncertainty_save_path = os.path.join(output_dir, f"{file_name}_uncertainty.png")
            visualize_uncertainty_map(
                uncertainty_data=output_dict[file_name],
                output_path=uncertainty_save_path,
                class_names=class_names
            )
    
    # Generate uncertainty summary if available
    if has_uncertainty:
        summary_path = os.path.join(output_dir, "uncertainty_summary.png")
        # Need to reconstruct results_dict format for summary
        results_dict_for_summary = {'data': {}}
        for file_name in file_names:
            if 'uncertainty' in output_dict[file_name]:
                results_dict_for_summary['data'][file_name] = {
                    'uncertainty': output_dict[file_name]['uncertainty'],
                    'epistemic_uncertainty': output_dict[file_name].get('epistemic_uncertainty'),
                    'aleatoric_uncertainty': output_dict[file_name].get('aleatoric_uncertainty')
                }
        create_uncertainty_summary(results_dict_for_summary, summary_path)


def create_output_mapping(results_dict, enable_background:bool = False):
    """
    create segmentation output for each rgb image in the test dataset based on model predictions.

    :param results_dict: dictionary containing predictions, labels, and coordinates
    :return: updated prediction dictionary with segmentation output mappings
    """
    input_image_mapping = results_dict['input_image_mapping']
    class_names = results_dict['class_names']
    num_classes = len(class_names)

    for file_name in results_dict['data'].keys():
        # Get image dimensions from RGB mapping if available, otherwise infer from coordinates
        if input_image_mapping is not None and file_name in input_image_mapping:
            rgb_img = input_image_mapping[file_name]
            height, width, _ = rgb_img.shape
        else:
            # Infer dimensions from coordinates
            data = results_dict['data'][file_name]
            coords = data['coords']
            max_x = max(coord[2] for coord in coords)  # x2
            max_y = max(coord[3] for coord in coords)  # y2
            height, width = max_y, max_x
        
        results_dict['output_image_mapping'][file_name] = np.zeros((height, width, num_classes), dtype=np.float32)
        results_dict['label_image_mapping'][file_name] = np.zeros((height, width, num_classes), dtype=np.float32)
        
        # For overlapping tiles, we need to track how many times each pixel is covered
        results_dict['output_count_mapping'] = results_dict.get('output_count_mapping', {})
        results_dict['output_count_mapping'][file_name] = np.zeros((height, width), dtype=np.float32)

    # fill the output mapping based on predictions and coordinates
    for file_name, data in results_dict['data'].items():

        if not enable_background and input_image_mapping is not None and file_name in input_image_mapping:
            # Use simple thresholding to create masks for background to avoid displaying prediction for background
            rgb_img = input_image_mapping[file_name]
            gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
            #threshold = int(np.percentile(gray, 20))
            threshold = 60
            mask = gray > threshold
            #_, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            data['preds'][0] *= mask[..., None]

        for i, coords in enumerate(data['coords']):
            x1, y1, x2, y2 = coords
            
            # Accumulate predictions and count overlaps
            results_dict['output_image_mapping'][file_name][y1:y2, x1:x2, :] += data['preds'][i]
            results_dict['label_image_mapping'][file_name][y1:y2, x1:x2, :] += data['labels'][i]
            results_dict['output_count_mapping'][file_name][y1:y2, x1:x2] += 1

    # Average overlapping predictions
    for file_name in results_dict['data'].keys():
        count_map = results_dict['output_count_mapping'][file_name]
        # Avoid division by zero
        count_map_safe = np.where(count_map > 0, count_map, 1)
        
        # Average the predictions where tiles overlap
        results_dict['output_image_mapping'][file_name] /= count_map_safe[..., None]
        results_dict['label_image_mapping'][file_name] /= count_map_safe[..., None]

        # Fill all uncovered pixels (zeros) with the background vector [0, 1, 0, ..., 0]
        # This ensures that regions not covered by any tile are marked as background
        background_vector = np.zeros(num_classes, dtype=np.float32)
        background_vector[1] = 1  # index 1 is background
        zero_mask = np.all(results_dict['output_image_mapping'][file_name] == 0, axis=-1)
        results_dict['output_image_mapping'][file_name][zero_mask] = background_vector
        results_dict['label_image_mapping'][file_name][zero_mask] = background_vector

    return results_dict

def visualize_segmentation_results(converted_rgb_image, ground_truth_segmentation, predicted_segmentation, main_title, red_info, green_info, blue_info):
    """
    Modern, clean visualization of segmentation results with improved aesthetics.
    
    :param converted_rgb_image: The RGB-converted HSI image to display (can be None).
    :param ground_truth_segmentation: The ground truth segmentation to display (can be None).
    :param predicted_segmentation: The predicted segmentation to display.
    :param main_title: Main title displayed above the images.
    :param red_info: Tuple containing (fabric name, true label, prediction) for the red channel.
    :param green_info: Tuple containing (fabric name, true label, prediction) for the green channel.
    :param blue_info: Tuple containing (fabric name, true label, prediction) for the blue channel.
    :return: matplotlib figure object
    """
    # Determine how many subplots we need
    num_plots = sum([converted_rgb_image is not None, ground_truth_segmentation is not None, True])
    
    # Create figure with modern styling
    fig = plt.figure(figsize=(7 * num_plots, 8), facecolor='white')
    gs = fig.add_gridspec(4, num_plots, height_ratios=[0.5, 6, 0.3, 1.5], hspace=0.35, wspace=0.15)
    
    # Main title with modern font
    fig.suptitle(main_title, fontsize=20, fontweight='bold', y=0.98)
    
    plot_idx = 0
    
    # Display images with clean borders
    if converted_rgb_image is not None:
        ax = fig.add_subplot(gs[1, plot_idx])
        ax.imshow(converted_rgb_image)
        ax.set_title('RGB Image', fontsize=14, pad=10, fontweight='600')
        ax.axis('off')
        plot_idx += 1
    
    if ground_truth_segmentation is not None:
        ax = fig.add_subplot(gs[1, plot_idx])
        ax.imshow(ground_truth_segmentation)
        ax.set_title('Ground Truth', fontsize=14, pad=10, fontweight='600')
        ax.axis('off')
        plot_idx += 1
    
    # Predicted segmentation (always shown)
    ax = fig.add_subplot(gs[1, plot_idx])
    ax.imshow(predicted_segmentation)
    ax.set_title('Prediction', fontsize=14, pad=10, fontweight='600')
    ax.axis('off')
    
    # Create modern legend table spanning all columns
    ax_legend = fig.add_subplot(gs[3, :])
    ax_legend.axis('off')
    
    # Prepare table data with better formatting (use text instead of emoji)
    table_data = [
        ['Channel', 'Fabric', 'Ground Truth', 'Prediction', 'Error'],
        ['Red', red_info[0], f'{red_info[1]:.1%}', f'{red_info[2]:.1%}', f'{abs(red_info[1]-red_info[2]):.1%}'],
        ['Green', green_info[0], f'{green_info[1]:.1%}', f'{green_info[2]:.1%}', f'{abs(green_info[1]-green_info[2]):.1%}'],
        ['Blue', blue_info[0], f'{blue_info[1]:.1%}', f'{blue_info[2]:.1%}', f'{abs(blue_info[1]-blue_info[2]):.1%}']
    ]
    
    # Create table with modern styling
    table = ax_legend.table(cellText=table_data, cellLoc='center', loc='center',
                           bbox=[0.1, 0, 0.8, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Header row styling
    for i in range(5):
        cell = table[(0, i)]
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(weight='bold', color='white')
    
    # Data rows styling with color indicators
    row_colors = ['#ffcccc', '#ccffcc', '#ccccff']  # Light red, green, blue
    for i in range(1, 4):
        for j in range(5):
            cell = table[(i, j)]
            if j == 0:  # Channel column - use color indicator
                cell.set_facecolor(row_colors[i-1])
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#ffffff')
            cell.set_edgecolor('#bdc3c7')
            cell.set_linewidth(1)
            
            # Bold the fabric names
            if j == 1:
                cell.set_text_props(weight='bold')
            
            # Color code errors (red if > 5%, green if < 2%)
            if j == 4:
                error_val = abs(red_info[1]-red_info[2]) if i == 1 else (abs(green_info[1]-green_info[2]) if i == 2 else abs(blue_info[1]-blue_info[2]))
                if error_val > 0.05:
                    cell.set_text_props(color='#e74c3c', weight='bold')
                elif error_val < 0.02:
                    cell.set_text_props(color='#27ae60', weight='bold')
    
    return fig

def visualize_extracted_tiles(image, coords, file_name, tile_extraction_mode):
    """
    visualize the extracted patches on a copy of the input image with a dynamically generated title.
    :param image: input fabric image
    :param coords: list of coordinates of patches
    :param file_name: name of the image file for logging
    :param tile_extraction_mode: 'fabric_grid' or 'fabric_random' search method used to extract patches
    """
    tiled_image = image.copy()

    for i, (x1, y1, x2, y2) in enumerate(coords):
        cv2.rectangle(tiled_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # display the image with the patches
    #plt.figure(figsize=(10, 10))
    #plt.imshow(tiled_image)
    #plt.title(f'{len(coords)} tiles extracted using {tile_extraction_mode} from {file_name}')
    #plt.axis('off')
    #plt.show()
    return tiled_image




def visualize_uncertainty_map(uncertainty_data: dict, output_path: str, class_names: list = None):
    """
    Visualize uncertainty maps for predictions.
    
    Args:
        uncertainty_data: Dictionary containing uncertainty information
        output_path: Path to save visualization
        class_names: List of class names for labeling
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    # Extract uncertainty metrics
    total_unc = uncertainty_data.get('uncertainty', None)
    epistemic_unc = uncertainty_data.get('epistemic_uncertainty', None)
    aleatoric_unc = uncertainty_data.get('aleatoric_uncertainty', None)
    per_class_unc = uncertainty_data.get('per_class_uncertainty', None)
    
    if total_unc is None:
        print("No uncertainty data available")
        return
    
    # Check if uncertainty is a list of tiles or a single array
    if isinstance(total_unc, list):
        # Multiple tiles - compute mean uncertainty across all tiles
        total_unc = np.mean([np.mean(tile) for tile in total_unc])
        if epistemic_unc is not None and isinstance(epistemic_unc, list):
            epistemic_unc = np.mean([np.mean(tile) for tile in epistemic_unc])
        if aleatoric_unc is not None and isinstance(aleatoric_unc, list):
            aleatoric_unc = np.mean([np.mean(tile) for tile in aleatoric_unc])
        
        # Create a simple bar chart instead of spatial map
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        metrics = ['Total\nUncertainty']
        values = [total_unc]
        colors = ['steelblue']
        
        if epistemic_unc is not None:
            metrics.append('Epistemic\n(Model)')
            values.append(epistemic_unc)
            colors.append('coral')
        
        if aleatoric_unc is not None:
            metrics.append('Aleatoric\n(Data)')
            values.append(aleatoric_unc)
            colors.append('lightgreen')
        
        bars = ax.bar(metrics, values, color=colors, edgecolor='black', alpha=0.7)
        ax.set_ylabel('Uncertainty', fontsize=12)
        ax.set_title('Average Uncertainty Metrics', fontsize=14, fontweight='bold')
        ax.set_ylim(0, max(values) * 1.2)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.4f}',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved uncertainty bar chart to: {output_path}")
        return
    
    # Single array - create spatial heatmap
    # Squeeze dimensions if needed
    if total_unc.ndim > 2:
        total_unc = total_unc.squeeze()
    if epistemic_unc is not None and epistemic_unc.ndim > 2:
        epistemic_unc = epistemic_unc.squeeze()
    if aleatoric_unc is not None and aleatoric_unc.ndim > 2:
        aleatoric_unc = aleatoric_unc.squeeze()
    
    # Create figure
    num_plots = 3 if epistemic_unc is not None and aleatoric_unc is not None else 1
    fig, axes = plt.subplots(1, num_plots, figsize=(6 * num_plots, 5))
    
    if num_plots == 1:
        axes = [axes]
    
    # Determine vmax based on actual data range
    vmax = min(max(2.0, np.max(total_unc)), 2.0)  # Cap at 2.0 for direct sum formula
    
    # Plot total uncertainty
    im1 = axes[0].imshow(total_unc, cmap='jet', vmin=0, vmax=vmax)
    axes[0].set_title('Total Uncertainty')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
    
    if num_plots == 3:
        # Plot epistemic uncertainty
        im2 = axes[1].imshow(epistemic_unc, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title('Epistemic Uncertainty\n(Model Uncertainty)')
        axes[1].axis('off')
        plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Plot aleatoric uncertainty
        im3 = axes[2].imshow(aleatoric_unc, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('Aleatoric Uncertainty\n(Data Uncertainty)')
        axes[2].axis('off')
        plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved uncertainty visualization to: {output_path}")


def create_uncertainty_summary(results_dict: dict, output_path: str):
    """
    Create a summary visualization of uncertainty across all samples.
    
    Args:
        results_dict: Results dictionary with uncertainty data
        output_path: Path to save summary
    """
    import matplotlib.pyplot as plt
    
    all_uncertainties = []
    all_epistemic = []
    all_aleatoric = []
    sample_names = []
    
    for file_name, data in results_dict['data'].items():
        if 'uncertainty' in data:
            unc = data['uncertainty']
            if isinstance(unc, np.ndarray):
                mean_unc = np.mean(unc)
                all_uncertainties.append(mean_unc)
                sample_names.append(file_name)
                
                if 'epistemic_uncertainty' in data:
                    all_epistemic.append(np.mean(data['epistemic_uncertainty']))
                if 'aleatoric_uncertainty' in data:
                    all_aleatoric.append(np.mean(data['aleatoric_uncertainty']))
    
    if len(all_uncertainties) == 0:
        print("No uncertainty data to summarize")
        return
    
    # Create summary plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram of total uncertainty
    axes[0, 0].hist(all_uncertainties, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(all_uncertainties), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(all_uncertainties):.3f}')
    axes[0, 0].set_xlabel('Total Uncertainty')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Total Uncertainty')
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.3)
    
    # 2. Epistemic vs Aleatoric
    if len(all_epistemic) > 0 and len(all_aleatoric) > 0:
        axes[0, 1].scatter(all_epistemic, all_aleatoric, alpha=0.6, c=all_uncertainties, 
                          cmap='jet', s=50)
        axes[0, 1].set_xlabel('Epistemic Uncertainty')
        axes[0, 1].set_ylabel('Aleatoric Uncertainty')
        axes[0, 1].set_title('Epistemic vs Aleatoric Uncertainty')
        axes[0, 1].grid(alpha=0.3)
        plt.colorbar(axes[0, 1].collections[0], ax=axes[0, 1], label='Total Uncertainty')
    
    # 3. Top uncertain samples
    top_n = min(20, len(all_uncertainties))
    sorted_indices = np.argsort(all_uncertainties)[-top_n:]
    top_samples = [sample_names[i] for i in sorted_indices]
    top_values = [all_uncertainties[i] for i in sorted_indices]
    
    axes[1, 0].barh(range(top_n), top_values, color='coral', edgecolor='black')
    axes[1, 0].set_yticks(range(top_n))
    axes[1, 0].set_yticklabels([s[:20] for s in top_samples], fontsize=8)
    axes[1, 0].set_xlabel('Uncertainty')
    axes[1, 0].set_title(f'Top {top_n} Most Uncertain Samples')
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # 4. Statistics summary
    stats_text = f"""
    Uncertainty Statistics:
    
    Total Samples: {len(all_uncertainties)}
    
    Mean Uncertainty: {np.mean(all_uncertainties):.4f}
    Std Uncertainty: {np.std(all_uncertainties):.4f}
    Min Uncertainty: {np.min(all_uncertainties):.4f}
    Max Uncertainty: {np.max(all_uncertainties):.4f}
    
    High Confidence (< 0.3): {np.sum(np.array(all_uncertainties) < 0.3)} ({100*np.sum(np.array(all_uncertainties) < 0.3)/len(all_uncertainties):.1f}%)
    Medium Confidence (0.3-0.5): {np.sum((np.array(all_uncertainties) >= 0.3) & (np.array(all_uncertainties) < 0.5))} ({100*np.sum((np.array(all_uncertainties) >= 0.3) & (np.array(all_uncertainties) < 0.5))/len(all_uncertainties):.1f}%)
    Low Confidence (> 0.5): {np.sum(np.array(all_uncertainties) >= 0.5)} ({100*np.sum(np.array(all_uncertainties) >= 0.5)/len(all_uncertainties):.1f}%)
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=10, verticalalignment='center',
                   family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved uncertainty summary to: {output_path}")
    print(stats_text)
