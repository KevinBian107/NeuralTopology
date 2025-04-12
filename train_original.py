"""
Topology of Deep Neural Networks - Reproduction Code
Based on the paper by Naitzat, Zhitnikov, and Lim (2020)
https://arxiv.org/pdf/2004.06093
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # For headless environments
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import gudhi as gd
import networkx as nx

from ripser import ripser
from persim import plot_diagrams

import yaml

np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#----------------------------------------------------------------#
# Data Generation According to  https://arxiv.org/pdf/2004.06093 #
#----------------------------------------------------------------#

def generate_dataset_D1(num_samples=8000):
    """
    Generate Dataset D-I: 2D manifold with nine green disks in a larger disk with nine holes
    """
    # Parameters for the dataset
    grid_size = 100
    disk_radius = 7
    large_disk_radius = 40
    
    # Centers of small disks (9 disks in a 3x3 grid)
    centers_x = np.array([-20, 0, 20, -20, 0, 20, -20, 0, 20])
    centers_y = np.array([20, 20, 20, 0, 0, 0, -20, -20, -20])
    
    # Create a grid of points
    x = np.linspace(-large_disk_radius, large_disk_radius, grid_size)
    y = np.linspace(-large_disk_radius, large_disk_radius, grid_size)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack([xx.ravel(), yy.ravel()])
    
    # Determine points inside the large disk
    distances_large = np.sqrt(points[:, 0]**2 + points[:, 1]**2)
    large_disk_mask = distances_large <= large_disk_radius
    
    # Determine points inside any of the small disks
    small_disks_mask = np.zeros_like(large_disk_mask, dtype=bool)
    for i in range(len(centers_x)):
        distances = np.sqrt((points[:, 0] - centers_x[i])**2 + (points[:, 1] - centers_y[i])**2)
        small_disks_mask = np.logical_or(small_disks_mask, distances <= disk_radius)
    
    # Points in Mb (red): inside large disk but outside small disks
    mask_b = np.logical_and(large_disk_mask, ~small_disks_mask)
    # Points in Ma (green): inside small disks
    mask_a = small_disks_mask
    
    # Sample points from each class
    points_a = points[mask_a]
    points_b = points[mask_b]
    
    # Randomly select points if we have more than we need
    if len(points_a) > num_samples // 2:
        indices_a = np.random.choice(len(points_a), num_samples // 2, replace=False)
        points_a = points_a[indices_a]
    
    if len(points_b) > num_samples // 2:
        indices_b = np.random.choice(len(points_b), num_samples // 2, replace=False)
        points_b = points_b[indices_b]
    
    # Create labels (0 for class a, 1 for class b)
    labels_a = np.zeros(len(points_a))
    labels_b = np.ones(len(points_b))
    
    # Combine data
    X = np.vstack([points_a, points_b])
    y = np.hstack([labels_a, labels_b])
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

def generate_dataset_D2(num_samples=8000):
    """
    Generate Dataset D-II: 3D manifold with nine pairs of interlocking tori
    """
    def generate_torus_points(R, r, n_theta, n_phi, center=np.array([0, 0, 0]), rotation=None):
        """Generate points on a torus with major radius R and minor radius r"""
        theta = np.linspace(0, 2*np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        theta, phi = np.meshgrid(theta, phi)
        
        # Cartesian coordinates
        x = (R + r * np.cos(phi)) * np.cos(theta)
        y = (R + r * np.cos(phi)) * np.sin(theta)
        z = r * np.sin(phi)
        
        points = np.column_stack([x.flatten(), y.flatten(), z.flatten()])
        
        # Apply rotation if specified
        if rotation is not None:
            rotation_matrix = rotation
            points = np.dot(points, rotation_matrix.T)
        
        # Apply translation
        points = points + center
        
        return points
    
    # Parameters for tori
    R, r = 5, 1  # Major and minor radius
    n_theta, n_phi = 20, 20  # Number of points around each circle
    
    # Centers for the nine pairs of tori
    grid_positions = np.array([
        [-20, -20, 0], [0, -20, 0], [20, -20, 0],
        [-20, 0, 0], [0, 0, 0], [20, 0, 0],
        [-20, 20, 0], [0, 20, 0], [20, 20, 0]
    ])
    
    # Generate points for each pair of interlocking tori
    points_a = []  # Green tori
    points_b = []  # Red tori
    
    for center in grid_positions:
        # Create rotation matrices for the two tori
        # One torus is in the xy-plane, the other is in the xz-plane
        rotation_a = np.eye(3)  # Identity (no rotation)
        rotation_b = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0]
        ])  # Interlock
        
        # Generate points for the two tori
        torus_a_points = generate_torus_points(R, r, n_theta, n_phi, center, rotation_a)
        torus_b_points = generate_torus_points(R, r, n_theta, n_phi, center, rotation_b)
        
        points_a.append(torus_a_points)
        points_b.append(torus_b_points)
    
    # Combine all points
    points_a = np.vstack(points_a)
    points_b = np.vstack(points_b)
    
    # Sample points if we have more than we need
    if len(points_a) > num_samples // 2:
        indices_a = np.random.choice(len(points_a), num_samples // 2, replace=False)
        points_a = points_a[indices_a]
    
    if len(points_b) > num_samples // 2:
        indices_b = np.random.choice(len(points_b), num_samples // 2, replace=False)
        points_b = points_b[indices_b]
    
    # Create labels (0 for class a, 1 for class b)
    labels_a = np.zeros(len(points_a))
    labels_b = np.ones(len(points_b))
    
    # Combine data
    X = np.vstack([points_a, points_b])
    y = np.hstack([labels_a, labels_b])
    
    # Shuffle the data
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

def generate_dataset_D3(num_samples=8000):
    """
    Generate Dataset D-III: 3D manifold with nine units of concentric spheres
    """
    def generate_sphere_points(radius, n_points, center=np.array([0, 0, 0])):
        """Generate approximately n_points on a sphere of given radius"""
        # Fibonacci sphere method
        indices = np.arange(0, n_points, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/n_points)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = radius * np.cos(theta) * np.sin(phi)
        y = radius * np.sin(theta) * np.sin(phi)
        z = radius * np.cos(phi)
        
        points = np.column_stack([x, y, z])
        points = points + center
        return points
    
    # Radii
    outer_radius, mid_radius, inner_radius = 8, 5, 2
    n_points_per_sphere = num_samples // 27  # (9 units * 3 spheres each = 27)
    
    # Centers for the nine units
    grid_positions = np.array([
        [-25, -25, 0], [0, -25, 0], [25, -25, 0],
        [-25, 0, 0],  [0, 0, 0],   [25, 0, 0],
        [-25, 25, 0], [0, 25, 0],  [25, 25, 0]
    ])
    
    # Generate points
    points_a = []  # Green (middle spheres)
    points_b = []  # Red (outer shells + inner balls)
    
    for center in grid_positions:
        outer_sphere = generate_sphere_points(outer_radius, n_points_per_sphere, center)
        middle_sphere = generate_sphere_points(mid_radius, n_points_per_sphere, center)
        inner_sphere = generate_sphere_points(inner_radius, n_points_per_sphere, center)
        
        # middle sphere is green => class a
        points_a.append(middle_sphere)
        # outer + inner = class b
        points_b.append(outer_sphere)
        points_b.append(inner_sphere)
    
    # Combine
    points_a = np.vstack(points_a)
    points_b = np.vstack(points_b)
    
    # Sample if more than needed
    if len(points_a) > num_samples // 2:
        indices_a = np.random.choice(len(points_a), num_samples // 2, replace=False)
        points_a = points_a[indices_a]
    
    if len(points_b) > num_samples // 2:
        indices_b = np.random.choice(len(points_b), num_samples // 2, replace=False)
        points_b = points_b[indices_b]
    
    labels_a = np.zeros(len(points_a))
    labels_b = np.ones(len(points_b))
    
    X = np.vstack([points_a, points_b])
    y = np.hstack([labels_a, labels_b])
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    return X, y

def visualize_dataset(X, y, dataset_name, dim=3):
    """Quick scatterplot for the dataset"""
    plt.figure(figsize=(10, 8))
    
    if dim == 2:
        plt.scatter(X[y==0, 0], X[y==0, 1], alpha=0.5, label='Class a (Ma)')
        plt.scatter(X[y==1, 0], X[y==1, 1], alpha=0.5, label='Class b (Mb)')
        plt.title(f'Dataset {dataset_name}')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend()
        
    elif dim == 3:
        ax = plt.axes(projection='3d')
        ax.scatter(X[y==0, 0], X[y==0, 1], X[y==0, 2], alpha=0.5, label='Class a (Ma)')
        ax.scatter(X[y==1, 0], X[y==1, 1], X[y==1, 2], alpha=0.5, label='Class b (Mb)')
        ax.set_title(f'Dataset {dataset_name}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.legend()
    
    plt.savefig(f"dataset_{dataset_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

#---------------------------#
# Architecture              #
#---------------------------#

class NeuralNetwork(nn.Module):
    """Basic MLP with a chosen activation function"""
    
    def __init__(self, input_dim, hidden_dims, output_dim, activation='relu'):
        super(NeuralNetwork, self).__init__()
        
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation function: {activation}")
        
        self.layers = nn.ModuleList()
        
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        
        for i in range(len(hidden_dims) - 1):
            self.layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        
        self.layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x):
        self.intermediate_activations = []
        
        x = self.layers[0](x)
        x = self.activation(x)
        self.intermediate_activations.append(x.detach())
        
        for i in range(1, len(self.layers) - 1):
            x = self.layers[i](x)
            x = self.activation(x)
            self.intermediate_activations.append(x.detach())
        
        x = self.layers[-1](x)
        x = self.output_activation(x)
        return x
    
    def get_layer_activations(self, data_loader):
        """
        Returns a list of numpy arrays containing the activations from each layer
        for all samples in the given data_loader.
        """
        self.eval()
        activations = [[] for _ in range(len(self.layers))]
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                _ = self.forward(inputs)
                
                for i, layer_activation in enumerate(self.intermediate_activations):
                    activations[i].append(layer_activation.cpu().numpy())
        
        # If a class has zero samples, each list in activations[i] could be empty,
        # leading to an error on np.vstack. So check here:
        final_activations = []
        for layer_act_list in activations:
            if len(layer_act_list) == 0:
                # Return an empty array or skip
                final_activations.append(np.array([]))
            else:
                final_activations.append(np.vstack(layer_act_list))
        
        return final_activations

def train_network(model, train_loader, val_loader, criterion, optimizer, num_epochs=100, patience=10):
    """Basic training loop with early stopping"""
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets.unsqueeze(1).float())
                val_loss += loss.item() * inputs.size(0)
                
                # Accuracy
                predicted = (outputs > 0.5).float()
                total += targets.size(0)
                correct += (predicted.view(-1) == targets).sum().item()
        
        val_loss = val_loss / len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch {epoch}/{num_epochs} | '
                  f'Train Loss: {train_loss:.4f} | '
                  f'Val Loss: {val_loss:.4f} | '
                  f'Val Accuracy: {val_accuracy:.2f}%')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
    
    return train_losses, val_losses

#---------------------------#
# Topology Analysis         #
#---------------------------#

def compute_graph_geodesic_distance(X, k=15):
    """
    Construct a k-NN graph and compute geodesic distance.
    """
    from sklearn.neighbors import NearestNeighbors
    
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
    distances, indices = nbrs.kneighbors(X)
    
    n_samples = X.shape[0]
    graph = np.zeros((n_samples, n_samples))
    
    # Fill adjacency
    for i in range(n_samples):
        for j_idx, j in enumerate(indices[i]):
            graph[i, j] = distances[i, j_idx]
            graph[j, i] = distances[i, j_idx]
    
    # Turn into NetworkX graph
    G = nx.from_numpy_array(graph)
    
    # Remove zero-weight edges (self-loops)
    for u, v, d in list(G.edges(data=True)):
        if d['weight'] == 0:
            G.remove_edge(u, v)
    
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(G))
    
    dist_matrix = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        for j in range(n_samples):
            if j in path_lengths[i]:
                dist_matrix[i, j] = path_lengths[i][j]
            else:
                dist_matrix[i, j] = float('inf')
    
    return dist_matrix

def compute_vietoris_rips_complex(X, k=15, max_dim=2):
    """
    Compute persistent homology from geodesic distances.
    """
    dist_matrix = compute_graph_geodesic_distance(X, k)
    max_finite = np.max(dist_matrix[np.isfinite(dist_matrix)])
    dist_matrix[np.isinf(dist_matrix)] = max_finite * 2
    
    results = ripser(dist_matrix, maxdim=max_dim, distance_matrix=True)
    return results

def compute_betti_numbers(ph_results, threshold=2.5):
    """
    Count how many features persist past 'threshold' in each dimension.
    """
    betti_numbers = []
    
    for dim, diagram in enumerate(ph_results['dgms']):
        count = np.sum((diagram[:, 0] <= threshold) & (diagram[:, 1] > threshold))
        betti_numbers.append(count)
    
    return betti_numbers

def perform_topology_analysis(network, dataloader, class_idx=0, k=15, threshold=2.5, max_dim=2):
    """
    1) Extract data for a particular class.
    2) Get layer activations.
    3) Compute persistent homology and Betti numbers.
    """
    all_inputs = []
    all_labels = []
    for inputs, labels in dataloader:
        all_inputs.append(inputs.numpy())
        all_labels.append(labels.numpy())
    if len(all_inputs) == 0:
        print("Dataloader is empty! Skipping topology analysis.")
        return []
    
    all_inputs = np.vstack(all_inputs)
    all_labels = np.hstack(all_labels)
    
    class_data = all_inputs[all_labels == class_idx]
    
    # ----------------- GUARD CLAUSE FIX -----------------
    if len(class_data) == 0:
        print(f"No samples found for class {class_idx}. Skipping topology analysis.")
        return []
    
    class_data_tensor = torch.tensor(class_data, dtype=torch.float32).to(device)
    class_labels_tensor = torch.zeros(len(class_data), dtype=torch.long).to(device)
    class_dataset = TensorDataset(class_data_tensor, class_labels_tensor)
    class_dataloader = DataLoader(class_dataset, batch_size=64, shuffle=False)
    
    layer_activations = network.get_layer_activations(class_dataloader)
    
    # Compute Betti for input data
    input_ph_results = compute_vietoris_rips_complex(class_data, k=k, max_dim=max_dim)
    input_betti = compute_betti_numbers(input_ph_results, threshold=threshold)
    
    layer_betti = [input_betti]
    
    # Layer-wise
    for layer_idx, activations in enumerate(layer_activations):
        if activations.size == 0:
            # If empty, skip
            print(f"Layer {layer_idx+1} had no activations for class {class_idx}. Skipping.")
            layer_betti.append([0]*(max_dim+1))
            continue
        
        print(f"Computing topology for layer {layer_idx+1}/{len(layer_activations)}...")
        ph_results = compute_vietoris_rips_complex(activations, k=k, max_dim=max_dim)
        betti = compute_betti_numbers(ph_results, threshold=threshold)
        layer_betti.append(betti)
    
    return np.array(layer_betti)

def visualize_topology_change(layer_betti, activation_name, dataset_name, class_name="Ma"):
    """
    Plots Betti numbers per layer to see how topology changes.
    """
    layer_betti = np.array(layer_betti)
    if layer_betti.size == 0:
        # If no data, skip
        print(f"No Betti data to plot for {dataset_name}, {class_name}.")
        return
    
    num_layers = layer_betti.shape[0]
    max_dim = layer_betti.shape[1] - 1
    
    layer_labels = ['input'] + [f'layer {i+1}' for i in range(num_layers-1)]
    
    plt.figure(figsize=(12, 6))
    
    for dim in range(max_dim+1):
        plt.plot(layer_betti[:, dim], marker='o', label=f'β{dim}')
    
    total = np.sum(layer_betti, axis=1)
    plt.plot(total, marker='s', linestyle='--', color='black', label='Total')
    
    plt.title(f'Topology Across Layers ({activation_name}, {dataset_name}, {class_name})')
    plt.xlabel('Layer')
    plt.ylabel('Betti Number')
    plt.xticks(range(num_layers), layer_labels, rotation=45)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"topology_change_{activation_name}_{dataset_name}_{class_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

#------------------------#
# Generate + Train + TDA #
#------------------------#

if __name__ == "__main__":
    with open("configs/configs_original.yml", "r") as f:
        config = yaml.safe_load(f)

    results_dir = config["results_dir"]
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    learning_rate = config["learning_rate"]
    validation_split = config["validation_split"]
    test_split = config["test_split"]
    activations = config["activations"]
    architectures = config["architectures"]

    os.makedirs(results_dir, exist_ok=True)
    os.chdir(results_dir)
    
    datasets = {
        'D-I': generate_dataset_D1(),
        'D-II': generate_dataset_D2(),
        'D-III': generate_dataset_D3()
    }
    
    for name, (X, y) in datasets.items():
        dim = X.shape[1]
        visualize_dataset(X, y, name, dim=dim)
        print(f"Generated dataset {name} with {X.shape[0]} samples in {dim}D")
    
    for dataset_name, (X, y) in datasets.items():
        print(f"\nProcessing dataset {dataset_name}...")
        
        # Stratified train/val/test splits
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y, test_size=test_split, stratify=y, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=validation_split / (1.0 - test_split),
            stratify=y_trainval,
            random_state=42
        )
        
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                      torch.tensor(y_train, dtype=torch.long))
        val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                    torch.tensor(y_val, dtype=torch.long))
        test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                     torch.tensor(y_test, dtype=torch.long))
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        for activation in activations:
            print(f"\nTraining with {activation} activation...")
            
            net = NeuralNetwork(
                input_dim=architectures[dataset_name]['input_dim'],
                hidden_dims=architectures[dataset_name]['hidden_dims'],
                output_dim=architectures[dataset_name]['output_dim'],
                activation=activation
            ).to(device)
            
            criterion = nn.BCELoss()
            optimizer = optim.Adam(net.parameters(), lr=learning_rate)
            
            train_losses, val_losses = train_network(
                model=net,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                num_epochs=num_epochs,
                patience=20
            )
            
            # Evaluate on test
            net.eval()
            test_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = net(inputs)
                    loss = criterion(outputs, targets.unsqueeze(1).float())
                    
                    test_loss += loss.item() * inputs.size(0)
                    predicted = (outputs > 0.5).float()
                    total += targets.size(0)
                    correct += (predicted.view(-1) == targets).sum().item()
            
            test_loss /= len(test_loader.dataset)
            test_accuracy = 100 * correct / total
            
            print(f'Test Loss: {test_loss:.4f} | Test Accuracy: {test_accuracy:.2f}%')
            
            torch.save(net.state_dict(), f"{dataset_name}_{activation}_model.pth")
            
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, label='Training Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title(f'Loss Curves ({dataset_name}, {activation})')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.savefig(f"loss_curves_{dataset_name}_{activation}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Topology analysis
            print("\nAnalyzing topology for class Ma (label 0)...")
            k_value = 14 if dataset_name == 'D-I' else 35
            threshold_value = 2.5
            
            ma_betti = perform_topology_analysis(
                network=net,
                dataloader=test_loader,
                class_idx=0,
                k=k_value,
                threshold=threshold_value,
                max_dim=X.shape[1]
            )
            visualize_topology_change(ma_betti, activation, dataset_name, class_name="Ma")
            
            print("\nAnalyzing topology for class Mb (label 1)...")
            mb_betti = perform_topology_analysis(
                network=net,
                dataloader=test_loader,
                class_idx=1,
                k=k_value,
                threshold=threshold_value,
                max_dim=X.shape[1]
            )
            visualize_topology_change(mb_betti, activation, dataset_name, class_name="Mb")
    
    print("\nAnalysis complete.")