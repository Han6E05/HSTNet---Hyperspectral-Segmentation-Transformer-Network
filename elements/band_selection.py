"""
Applying Band Selection

This script performs band selection using PCA, LDA, or Autoencoder using the training dataset,
then store the selected band in a .json file for the main pipeline to use.

Functions:
- Select_PCA: Select bands from the given dataset with the PCA method, then return the index of selected bands and truncated dataset.
- Select_LDA: Select bands from the given dataset with the LDA method, then return the index of selected bands and truncated dataset.
- Select_Autoencoder: Select bands using an autoencoder-based reconstruction error method.
- plot_selection: plot the selected bands and save the figure.
- plot_main: plot the base figure using information of the dataset.

Usage:
To run the script, modify the parameters.
- dataset_path: the location of the training dataset npz file.
- method: a list containing 'PCA', 'LDA', or/and 'Autoencoder'.
- channels: a list containing numbers of bands want to be selected.
- cut: a list containing number of bands that want to be cut off.
- windowed: a list containing True or/and False.
Make sure the dataset is properly placed in the correct directory.

Dependencies:
- numpy: For numerical operations.
- torch: For manipulating tensors in the dataset and training autoencoder.
- scikit-learn: For PCA and LDA implementations.
- matplotlib: For plotting bands.

Authors:
- [Felix Yang]
- Some source code from [Jarno Cremer]
"""


import os
import time
import torch
import warnings
import numpy as np
import json
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from sklearn.decomposition import PCA
from pathlib import Path
import matplotlib.pyplot as plt

def _get_working_dir():
    return Path(__file__).resolve().parent

# configure logging
from elements.utils import LoggerSingleton
LoggerSingleton.setup_logger(_get_working_dir())
logger = LoggerSingleton.get_logger()

from elements.load_data import load_hsi_dataset
from torch.utils.data import DataLoader


def load_dataset_subset(dataset_path, max_samples=5000):
    """
    Load only a subset of the dataset to save memory.
    Returns tiles and labels averaged over spatial dimensions.
    """
    print(f"Loading dataset subset (max {max_samples} samples)...")
    dataset = load_hsi_dataset(dataset_path=dataset_path)
    
    # Immediately subsample to reduce memory
    total_samples = len(dataset.tiles)
    if total_samples > max_samples:
        print(f"  Sampling {max_samples} out of {total_samples} samples...")
        indices = torch.randperm(total_samples)[:max_samples]
        tiles = dataset.tiles[indices]
        labels = dataset.labels[indices] if hasattr(dataset, 'labels') else None
    else:
        tiles = dataset.tiles
        labels = dataset.labels if hasattr(dataset, 'labels') else None
    
    # Average over spatial dimensions immediately
    tiles_avg = tiles.mean(dim=(2, 3))  # (N, C)
    if labels is not None:
        labels_avg = labels.mean(axis=(2, 3))  # (N, num_classes)
    else:
        labels_avg = None
    
    # Free original data
    del dataset, tiles, labels
    import gc
    gc.collect()
    
    print(f"  Loaded tiles shape: {tiles_avg.shape}")
    return tiles_avg, labels_avg


def Select_PCA(npz, num_channels=40, cutoff_edge_size=0, windowed=False, max_samples=5000):
    """
    Select bands using PCA.
    
    Args:
        npz: Path to dataset
        num_channels: Number of bands to select
        cutoff_edge_size: Number of edge bands to exclude
        windowed: Whether to use windowed selection
        max_samples: Maximum number of samples to use for PCA (to save memory)
    """
    # Load only subset of data
    tiles_reshaped, _ = load_dataset_subset(npz, max_samples=max_samples)

    # Move tensor to CPU only for PCA computation
    tiles_np = tiles_reshaped.cpu().numpy()
    if cutoff_edge_size > 0:
        tiles_np = tiles_np[:, cutoff_edge_size:-cutoff_edge_size]
        print(f"After cutoff, shape: {tiles_np.shape}")

    # Apply PCA to select top bands
    from sklearn.decomposition import PCA
    print(f"Running PCA with {num_channels} components...")
    pca = PCA(n_components=num_channels)
    pca.fit(tiles_np)

    # Get absolute PCA loadings (importance of each band)
    importance_scores = np.abs(pca.components_).sum(axis=0)  # Sum across all components
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")

    # Skip correlation matrix computation to save memory
    # correlation_matrix = np.corrcoef(tiles_np, rowvar=False)
    # plot_correlation_matrix(correlation_matrix)

    # Initialize list for selected bands
    selected_bands = []

    # First band: pick the most important band
    sorted_indices = np.argsort(-importance_scores)  # Sort in descending order of importance

    if windowed:
        selected_bands = []
        index = 0
        from math import floor, ceil
        num_bands = len(importance_scores)
        bands_available = (num_bands - cutoff_edge_size*2)
        window_size = floor( bands_available / (num_channels+10) )
        num_window = ceil(bands_available/window_size)
        selected_bucket = [0]*num_window
        while len(selected_bands) < num_channels:
            b = sorted_indices[index]
            if selected_bucket[b//window_size] == 0:
                selected_bucket[b // window_size] = 1
                selected_bands.append(b)
            index += 1

    else:
        selected_bands = sorted_indices[0:num_channels]

    """selected_bands.append(sorted_indices[0])

    # Iteratively select the next band, avoiding high redundancy
    for _ in range(1, num_channels):
        max_score = -np.inf
        max_score_band = None

        for i in sorted_indices:
            if i not in selected_bands:
                # Calculate the redundancy with already selected bands
                redundancy_penalty = 0
                for j in selected_bands:
                    redundancy_penalty += correlation_matrix[i, j] ** 2  # Penalize high correlation

                # Penalize based on redundancy (lower redundancy is better)
                score = importance_scores[i] - redundancy_penalty

                if score > max_score:
                    max_score = score
                    max_score_band = i

        selected_bands.append(max_score_band)

    selected_bands = torch.tensor(selected_bands, dtype=torch.long)"""

    #plot_pca_band_importance(importance_scores, selected_bands)
    selected_bands = [x+cutoff_edge_size for x in selected_bands]
    print(f"PCA bands selected: {selected_bands}")

    # Return selected bands only (don't modify dataset to save memory)
    return selected_bands, None

def Select_Autoencoder(npz, num_channels=40, cutoff_edge_size=0, windowed=False, 
                       epochs=50, batch_size=256, learning_rate=0.001, bottleneck_ratio=0.1,
                       max_samples=5000):
    """
    Select bands using autoencoder-based reconstruction error method.
    
    The autoencoder learns to compress and reconstruct spectral data. Bands with higher
    reconstruction error are considered more informative and are selected.
    
    Args:
        npz: Path to the dataset npz file
        num_channels: Number of bands to select
        cutoff_edge_size: Number of edge bands to cut off from both sides
        windowed: If True, ensure selected bands are distributed across the spectrum
        epochs: Number of training epochs for the autoencoder
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        bottleneck_ratio: Ratio of bottleneck size to input size (smaller = more compression)
        max_samples: Maximum number of samples to use for training (to save memory)
    
    Returns:
        selected_bands: List of selected band indices
        dataset: Dataset with selected bands
    """
    # Load only subset of data
    X, _ = load_dataset_subset(npz, max_samples=max_samples)
    
    # Apply cutoff if specified
    if cutoff_edge_size > 0:
        X = X[:, cutoff_edge_size:-cutoff_edge_size]
    
    n_bands = X.shape[1]
    bottleneck_size = max(int(n_bands * bottleneck_ratio), num_channels)
    
    print(f"Training autoencoder: {n_bands} bands -> {bottleneck_size} bottleneck -> {n_bands} bands")
    print(f"Data shape after spatial averaging: {X.shape}")
    
    # Define autoencoder architecture
    class SpectralAutoencoder(torch.nn.Module):
        def __init__(self, input_size, bottleneck_size):
            super(SpectralAutoencoder, self).__init__()
            hidden_size = (input_size + bottleneck_size) // 2
            
            # Encoder
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.Linear(hidden_size, bottleneck_size),
                torch.nn.ReLU()
            )
            
            # Decoder
            self.decoder = torch.nn.Sequential(
                torch.nn.Linear(bottleneck_size, hidden_size),
                torch.nn.ReLU(),
                torch.nn.BatchNorm1d(hidden_size),
                torch.nn.Linear(hidden_size, input_size)
            )
        
        def forward(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SpectralAutoencoder(n_bands, bottleneck_size).to(device)
    criterion = torch.nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Prepare data loader
    X_tensor = X.to(device)
    dataset_ae = torch.utils.data.TensorDataset(X_tensor)
    dataloader = DataLoader(dataset_ae, batch_size=batch_size, shuffle=True)
    
    # Training loop
    print("Training autoencoder...")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            batch_data = batch[0]
            
            optimizer.zero_grad()
            reconstructed = model(batch_data)
            loss = criterion(reconstructed, batch_data).mean()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    
    # Calculate reconstruction error per band
    print("Calculating band importance...")
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        # Calculate MSE per band across all samples
        band_errors = torch.mean((X_tensor - reconstructed) ** 2, dim=0).cpu().numpy()
    
    # Select bands with highest reconstruction error (most informative)
    sorted_indices = np.argsort(-band_errors)  # Sort in descending order
    
    if windowed:
        selected_bands = []
        index = 0
        from math import floor, ceil
        bands_available = n_bands
        window_size = floor(bands_available / (num_channels + 10))
        num_window = ceil(bands_available / window_size)
        selected_bucket = [0] * num_window
        
        while len(selected_bands) < num_channels:
            b = sorted_indices[index]
            if selected_bucket[b // window_size] == 0:
                selected_bucket[b // window_size] = 1
                selected_bands.append(b)
            index += 1
    else:
        selected_bands = sorted_indices[:num_channels].tolist()
    
    # Adjust indices if edges were cut
    if cutoff_edge_size > 0:
        selected_bands = [x + cutoff_edge_size for x in selected_bands]
    
    print(f"Autoencoder bands selected: {selected_bands}")
    print(f"Top 5 band reconstruction errors: {sorted(band_errors, reverse=True)[:5]}")
    
    # Return selected bands only (don't modify dataset to save memory)
    return selected_bands, None

def Select_LDA(npz, num_channels=40, cutoff_edge_size=20, windowed=False):
    dataset = load_hsi_dataset(dataset_path=npz)

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    #import matplotlib.pyplot as plt

    # --- 1. Prepare data ---
    tiles = dataset.tiles.clone()
    class_names = dataset.class_names
    labels = dataset.labels

    print(f"dataset.tiles shape = {dataset.tiles.shape}")
    print(f"dataset.class_names shape = {len(class_names)}")
    print(f"dataset.labels shape = {dataset.labels.shape}")
    
    # Average over spatial dimensions first
    tiles_averaged = dataset.tiles.mean(dim=(2, 3))  # (N, C)
    labels = dataset.labels.numpy()  # (N, num_classes, H, W)
    labels_averaged = labels.mean(axis=(2, 3))  # (N, num_classes)
    
    class_counts = np.sum(labels_averaged, axis=0)
    for i, count in enumerate(class_counts):
        print(f"Class {i}: {int(count)} samples")

    # --- Step 1: Prepare data ---
    if cutoff_edge_size > 0:
        X = tiles_averaged[:, cutoff_edge_size:-cutoff_edge_size].numpy()
    else:
        X = tiles_averaged.numpy()  # (N, C)

    y = []
    composition_to_class = {}
    class_id = 0
    for comp in labels_averaged:
        key = tuple(round(v, 5) for v in comp)
        if key not in composition_to_class:
            composition_to_class[key] = class_id
            class_id += 1
        y.append(composition_to_class[key])

    # Optional: remove 'void' class (index 0)
    #valid_mask = y != 0
    #X = X[valid_mask]
    #y = y[valid_mask]

    from collections import Counter
    counts = Counter(y)
    print("Class sample counts:", counts)
    print("Unique classes found:", len(counts))
    for i, count in counts.items():
        print(f"{i}: {count}")
    n_classes = len(counts)

    n_features = X.shape[1]
    n_components = min(n_features, n_classes - 1)
    # --- Step 2: Run LDA with 'eigen' solver to access explained variance ---
    lda = LinearDiscriminantAnalysis(n_components=n_components, solver='eigen')
    X_lda = lda.fit_transform(X, y)

    # --- Step 3: Explained variance ratio ---
    print("Explained variance ratio per component:")
    for i, ratio in enumerate(lda.explained_variance_ratio_):
        print(f"  Component {i + 1}: {ratio:.4f}")

    # Optional: plot it
    #plt.figure(figsize=(6, 3))
    #plt.plot(lda.explained_variance_ratio_, marker='o')
    #plt.title("LDA Explained Variance Ratio")
    #plt.xlabel("Component")
    #plt.ylabel("Variance Ratio")
    #plt.grid(True)
    #plt.tight_layout()
    #plt.show()

    # --- Step 4: Use lda.coef_ to rank original bands ---
    # coef_: shape (n_classes, n_bands)
    band_importance = np.mean(np.abs(lda.coef_), axis=0)  # (n_bands,)

    # Sort and select top bands
    sorted_indices = np.argsort(band_importance)[::-1].copy()

    if windowed:
        top_band_indices = []
        index = 0
        from math import floor, ceil
        num_bands = len(band_importance)
        bands_available = (num_bands - cutoff_edge_size * 2)
        window_size = floor(bands_available / (num_channels + 10))
        num_window = ceil(bands_available / window_size)
        selected_bucket = [0] * num_window
        while len(top_band_indices) < num_channels:
            b = sorted_indices[index]
            if selected_bucket[b // window_size] == 0:
                selected_bucket[b // window_size] = 1
                top_band_indices.append(b)
            index += 1

    else:
        top_band_indices = sorted_indices[:num_channels]

    if cutoff_edge_size > 0:
        top_band_indices = [x+cutoff_edge_size for x in top_band_indices]


    print(f"\nTop {num_channels} bands (most discriminative) selected by LDA:", top_band_indices)

    # Optional: plot band importances
    #plt.figure(figsize=(10, 3))
    #plt.bar(np.arange(len(band_importance)), band_importance)
    #plt.scatter(top_band_indices, band_importance[top_band_indices], color='red', label='Top Bands')
    #plt.title("Band Importance from LDA Coefficients")
    #plt.xlabel("Band Index")
    #plt.ylabel("Importance")
    #plt.legend()
    #plt.tight_layout()
    #plt.show()

    # Select the LDA-reduced channels while keeping tensor format
    #dataset.tiles = tiles[:, :, top_band_indices]  # Keep (295730, 1, num_channels)
    new_tiles = []
    if type(dataset.tiles) is list:
        for sample in dataset.tiles:
            sample = sample[top_band_indices, :, :]
            new_tiles.append(sample)
            dataset.tiles = new_tiles
    else:
        dataset.tiles = dataset.tiles[:, :, top_band_indices]

    return top_band_indices, dataset

def plot_selection(ax, handles, labels, bands, title):
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    for line in ax.lines:
        ax1.plot(line.get_xdata(), line.get_ydata(), color=line.get_color())
    for ind in bands:
        ax1.axvline(x=(1700 - 900) / 223 * ind + 900)
    # additional plot A...
    ax1.legend(handles, labels)
    ax1.set_xlabel(ax.get_xlabel())
    ax1.set_ylabel(ax.get_ylabel())
    ax1.set_title(title)
    plt.show()
    fig1.savefig(title+'.png', dpi=300, bbox_inches="tight")

def plot_main(dataset_path, max_samples=10000):
    """
    Plot average spectral signatures for each class.
    
    Args:
        dataset_path: Path to dataset
        max_samples: Maximum samples to use (to save memory)
    """
    # Load subset of data
    tiles_per_tile, labels_per_tile = load_dataset_subset(dataset_path, max_samples=max_samples)
    
    # Convert to numpy
    data_per_tile = tiles_per_tile.numpy()  # Shape: (N, channels)
    labels_per_tile = labels_per_tile  # Already numpy, shape: (N, num_classes)
    
    # Load class names
    dataset = load_hsi_dataset(dataset_path=dataset_path)
    class_names = dataset.class_names
    del dataset  # Free memory
    import gc
    gc.collect()
    
    composition_to_band_intensity = {}
    for i in range(len(labels_per_tile)):
        comp = labels_per_tile[i]
        
        # Find the dominant class (highest proportion)
        key = ""
        max_val = comp.max()
        if max_val > 0.5:  # Only consider if class proportion > 50%
            dominant_class_idx = comp.argmax()
            key = dataset.class_names[dominant_class_idx]
        
        if key == "":
            continue

        if key not in composition_to_band_intensity:
            print(key)
            composition_to_band_intensity[key] = {'avg': np.zeros(data_per_tile.shape[1]), 'cnt': 0}
        composition_to_band_intensity[key]['avg'] += data_per_tile[i]
        composition_to_band_intensity[key]['cnt'] += 1

    for item in composition_to_band_intensity:
        composition_to_band_intensity[item]['avg'] = composition_to_band_intensity[item]['avg'] / \
                                                     composition_to_band_intensity[item]['cnt']
        x = composition_to_band_intensity[item]['avg']
        x = (x - x.min()) / (x.max() - x.min())
        composition_to_band_intensity[item]['avg'] = x



    plt.rcParams.update({
        "font.size": 24,
        "axes.labelsize": 26,
        "axes.titlesize": 30,
        "xtick.labelsize": 22,
        "ytick.labelsize": 22,
        "legend.fontsize": 18,
    })
    plt.tight_layout()

    # Get actual number of bands from data
    num_bands = data_per_tile.shape[1]
    x = np.linspace(900, 1700, num_bands)

    # plot base figure
    fig, ax = plt.subplots(figsize=(10, 7))
    for item in composition_to_band_intensity:
        lb = item
        ax.plot(x, composition_to_band_intensity[item]['avg'], label=lb, alpha=0.8)

    ax.set_xlabel("bandwidths (nm)")
    ax.set_ylabel("normalized intensity")
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    plt.show()
    fig.savefig('base.png', dpi=300, bbox_inches='tight')
    return ax, handles, labels

if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent
    dataset_path = Path('/home/student/S2/HIT/dataset/train_set.npz')

    ax, handles, labels = plot_main(dataset_path)

    selection_dict = {}

    methods = ['PCA', 'LDA', 'Autoencoder']
    channels = [100]
    cut = [0]
    windowed = [True]
    max_samples = 5000  # Use subset of data to save memory

    for method in methods:
        for c in channels:
            for cut_size in cut:
                for window in windowed:
                    iscut = 'NOCUT' if cut_size==0 else 'CUT'
                    iswindowed = 'WINDOW' if window else 'WINDOWLESS'
                    title = method + '_' + str(c) + '_' + iscut + '_' + iswindowed
                    
                    print(f"\n{'='*60}")
                    print(f"Running: {title}")
                    print(f"{'='*60}")
                    
                    if method == 'PCA':
                        bands, _ = Select_PCA(dataset_path, c, cutoff_edge_size=cut_size, 
                                             windowed=window, max_samples=max_samples)
                    elif method == 'LDA':
                        bands, _ = Select_LDA(dataset_path, c, cutoff_edge_size=cut_size, 
                                             windowed=window)
                    elif method == 'Autoencoder':
                        bands, _ = Select_Autoencoder(dataset_path, c, cutoff_edge_size=cut_size, 
                                                     windowed=window, max_samples=max_samples)
                    else:
                        continue

                    plot_selection(ax, handles, labels, bands, title)
                    selection_dict[title] = [int(x) for x in bands]

    print(selection_dict)
    # write dict to file
    with open("band_selection_info.json", "w") as f:
        json.dump(selection_dict, f, indent=2)

    # Additionally, you can reprint the selected indices and plot the result using the following code.
    """
    import json
    with open("band_selection_info.json", "r") as f:
        data = json.load(f)
        for key, value in data.items():
            plot_selection(ax, handles, labels, value, key)
            print(key, value)
    """