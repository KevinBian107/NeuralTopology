import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # For headless environments
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

import gudhi  # for AlphaComplex
from tqdm import tqdm


##############################################################################
#                 SUBSAMPLING + GUDHI + PCA FOR ALPHA COMPLEX                #
##############################################################################

def subsample_data(X, max_points=20, seed=42):
    """
    If X has more than `max_points` rows, randomly choose `max_points`.
    Otherwise, return X as-is.
    """
    if len(X) > max_points:
        np.random.seed(seed)
        indices = np.random.choice(len(X), size=max_points, replace=False)
        return X[indices]
    return X

def compute_alpha_complex_diagram(X, pca_components=10, max_dim=2):
    """
    1) Optionally reduce dimension of X via PCA (if needed),
    2) Build an AlphaComplex from GUDHI,
    3) Compute persistence intervals,
    4) Return them in a dictionary that mimics ripser's {'dgms': [...]} format.
    """
    if X.shape[1] > pca_components:
        pca = PCA(n_components=pca_components)
        X_reduced = pca.fit_transform(X)
    else:
        X_reduced = X
    
    alpha_complex = gudhi.AlphaComplex(points=X_reduced)
    simplex_tree = alpha_complex.create_simplex_tree()
    simplex_tree.persistence()  # up to full dimension by default

    diag_by_dim = [[] for _ in range(max_dim + 1)]
    for (simplex, (death_time, dim)) in simplex_tree.persistence():
        if dim <= max_dim:
            birth_time = simplex_tree.filtration(simplex)
            diag_by_dim[dim].append((birth_time, death_time))
    
    dgms = []
    for intervals in diag_by_dim:
        if len(intervals) == 0:
            dgms.append(np.empty((0, 2)))
        else:
            dgms.append(np.array(intervals))
    
    return {"dgms": dgms}

def compute_betti_numbers(ph_results, threshold=2.5):
    betti_numbers = []
    for dim, diagram in enumerate(ph_results["dgms"]):
        count = np.sum((diagram[:, 0] <= threshold) & (diagram[:, 1] > threshold))
        betti_numbers.append(count)
    return betti_numbers


##############################################################################
#                           NEURAL NETWORK CODE                              #
##############################################################################

class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim=1, activation='relu'):
        super().__init__()
        
        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'leaky_relu':
            self.act = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        self.layers = nn.ModuleList(layers)
        self.output_activation = nn.Sigmoid()
        
    def forward(self, x):
        self.intermediate_activations = []
        out = x
        
        for i in range(len(self.layers) - 1):
            out = self.layers[i](out)
            out = self.act(out)
            self.intermediate_activations.append(out.detach())
        
        out = self.layers[-1](out)
        out = self.output_activation(out)
        return out
    
    def get_layer_activations(self, data_loader, device='cpu'):
        self.eval()
        acts = [[] for _ in range(len(self.layers))]
        
        with torch.no_grad():
            for inputs, _ in data_loader:
                inputs = inputs.to(device)
                _ = self.forward(inputs)
                for i, layer_act_tensor in enumerate(self.intermediate_activations):
                    acts[i].append(layer_act_tensor.cpu().numpy())
        
        final_acts = []
        for layer_list in acts:
            if len(layer_list) == 0:
                final_acts.append(np.array([]))
            else:
                final_acts.append(np.vstack(layer_list))
        return final_acts

def train_network(model, train_loader, val_loader, epochs=10, lr=1e-3, patience=3, device='cpu'):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    train_losses, val_losses = [], []
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        
        train_pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}] Train", leave=False)
        for inputs, labels in train_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * inputs.size(0)
        
        train_loss_avg = total_train_loss / len(train_loader.dataset)
        train_losses.append(train_loss_avg)
        
        model.eval()
        total_val_loss = 0.0
        correct = 0
        total = 0
        
        val_pbar = tqdm(val_loader, desc=f"[Epoch {epoch+1}/{epochs}] Val  ", leave=False)
        for inputs, labels in val_pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            total_val_loss += loss.item() * inputs.size(0)
            
            preds = (outputs > 0.5).float()
            correct += (preds.view(-1) == labels).sum().item()
            total += labels.size(0)
        
        val_loss_avg = total_val_loss / len(val_loader.dataset)
        val_accuracy = 100.0 * correct / total
        val_losses.append(val_loss_avg)
        
        print(f"[Epoch {epoch+1}/{epochs}] "
              f"Train Loss: {train_loss_avg:.4f} | "
              f"Val Loss: {val_loss_avg:.4f} | "
              f"Val Acc: {val_accuracy:.2f}%")
        
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print("Early stopping triggered!")
                break
    
    return train_losses, val_losses


##############################################################################
#              PERFORM TOPOLOGY ANALYSIS WITH SUBSAMPLING                    #
##############################################################################

def perform_topology_analysis(
    network,
    data_loader,
    class_label=0,
    pca_components=10,
    threshold=2.5,
    max_dim=2,
    device='cpu',
    max_points_input=2000,
    max_points_activations=2000
):
    """
    We now do SUBSAMPLING for both the input data (class_x) and each layer's activations
    to reduce TDA cost significantly.

    1) Gather all data from data_loader,
    2) Filter by `class_label`,
    3) Subsample => compute alpha complex => input_betti
    4) Pass data through net => layer activations => subsample => alpha complex => betti
    5) Return array shape (#layers+1, #dimensions+1).
    """
    all_x, all_y = [], []
    for inputs, labels in data_loader:
        all_x.append(inputs.numpy())
        all_y.append(labels.numpy())
    if len(all_x) == 0:
        print("Data loader is empty; skipping.")
        return []
    
    all_x = np.vstack(all_x)
    all_y = np.hstack(all_y)
    
    # Filter by class
    class_x = all_x[all_y == class_label]
    if len(class_x) == 0:
        print(f"No samples found for class {class_label}. Skipping.")
        return []
    
    # A) TDA on raw input
    class_x = subsample_data(class_x, max_points=max_points_input, seed=42)
    ph_input = compute_alpha_complex_diagram(class_x, pca_components=pca_components, max_dim=max_dim)
    input_betti = compute_betti_numbers(ph_input, threshold=threshold)
    
    # B) Get layer activations (full data first)
    class_tensor_x = torch.tensor(all_x[all_y == class_label], dtype=torch.float32, device=device)
    class_tensor_y = torch.zeros(len(class_x), dtype=torch.long, device=device)
    # ^ careful: we used 'class_x' for partial data, but let's re-check lengths
    class_dataset = TensorDataset(class_tensor_x, class_tensor_y)
    class_loader_2 = DataLoader(class_dataset, batch_size=64, shuffle=False)
    
    layer_acts = network.get_layer_activations(class_loader_2, device=device)
    
    # C) For each layer, subsample and compute alpha complex
    layer_betti = [input_betti]
    for idx, A in enumerate(layer_acts):
        if A.size == 0:
            print(f"Layer {idx+1} has no samples for class {class_label}.")
            layer_betti.append([0]*(max_dim+1))
            continue
        
        # Subsample the activations
        A_sub = subsample_data(A, max_points=max_points_activations, seed=42)
        
        ph_layer = compute_alpha_complex_diagram(A_sub, pca_components=pca_components, max_dim=max_dim)
        bnums = compute_betti_numbers(ph_layer, threshold=threshold)
        layer_betti.append(bnums)
    
    return np.array(layer_betti)

def plot_betti(layer_betti, title_str="Topology change", save_name="topology.png"):
    if layer_betti is None or len(layer_betti) == 0:
        print("No Betti data to plot.")
        return
    
    layer_betti = np.array(layer_betti)
    num_layers = layer_betti.shape[0]
    dims = layer_betti.shape[1]
    
    plt.figure(figsize=(10, 6))
    x_labels = ["Input"] + [f"Layer {i}" for i in range(1, num_layers)]
    
    for d in range(dims):
        plt.plot(layer_betti[:, d], marker='o', label=f"β{d}")
    
    total = np.sum(layer_betti, axis=1)
    plt.plot(total, marker='s', linestyle='--', color='black', label='Total')
    
    plt.title(title_str)
    plt.xlabel("Layer")
    plt.ylabel("Betti Number")
    plt.xticks(range(num_layers), x_labels, rotation=45)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_name, dpi=300)
    plt.close()

##############################################################################
#                                 MAIN                                       #
##############################################################################

def main():
    import sys
    print(f"Running on Python {sys.version}")
    
    BATCH_SIZE = 128
    EPOCHS = 5
    LR = 1e-3
    PATIENCE = 3
    ACTIVATION = 'relu'
    
    PCA_COMPONENTS = 10
    THRESHOLD = 2.5
    MAX_DIM = 2
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # We'll subsample to these many points for TDA:
    MAX_POINTS_INPUT = 2000
    MAX_POINTS_ACTIVATIONS = 2000
    
    # Save results
    os.makedirs("results_mnist", exist_ok=True)
    
    # 1) MNIST 0/1
    transform = transforms.Compose([transforms.ToTensor()])
    train_data_full = datasets.MNIST(root=".", train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=".", train=False, download=True, transform=transform)
    
    def filter_01(dataset):
        idxs = [i for i, (img, label) in enumerate(dataset) if label in [0, 1]]
        return idxs
    
    train_idxs = filter_01(train_data_full)
    test_idxs = filter_01(test_data)
    
    train_data_01 = torch.utils.data.Subset(train_data_full, train_idxs)
    test_data_01 = torch.utils.data.Subset(test_data, test_idxs)
    
    def create_Xy(subset):
        images, labels = [], []
        for i in range(len(subset)):
            img, label = subset[i]
            img_np = img.numpy().flatten()
            images.append(img_np)
            labels.append(label)
        X = np.array(images)
        y = np.array(labels)
        return X, y
    
    X_train_all, y_train_all = create_Xy(train_data_01)
    X_test, y_test = create_Xy(test_data_01)
    
    # 2) Split train->train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_all, y_train_all, test_size=0.2, random_state=42, stratify=y_train_all
    )
    
    # 3) Dataloaders
    train_tensor = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                 torch.tensor(y_train, dtype=torch.long))
    val_tensor = TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                               torch.tensor(y_val, dtype=torch.long))
    test_tensor = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                                torch.tensor(y_test, dtype=torch.long))
    
    train_loader = DataLoader(train_tensor, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_tensor, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_tensor, batch_size=BATCH_SIZE, shuffle=False)
    
    # 4) Build net
    input_dim = 28*28
    hidden_dims = [128, 64]
    net = NeuralNetwork(input_dim, hidden_dims, output_dim=1, activation=ACTIVATION).to(DEVICE)
    
    # 5) Train
    print(f"Training on MNIST 0/1. Using PCA + AlphaComplex + Subsampling for TDA.")
    train_losses, val_losses = train_network(
        model=net,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        lr=LR,
        patience=PATIENCE,
        device=DEVICE
    )
    
    # Evaluate
    net.eval()
    criterion = nn.BCELoss()
    test_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = net(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
            test_loss += loss.item() * inputs.size(0)
            
            preds = (outputs > 0.5).float()
            correct += (preds.view(-1) == labels).sum().item()
            total += labels.size(0)
    
    test_loss /= len(test_loader.dataset)
    test_accuracy = 100.0 * correct / total
    print(f"[Test] Loss: {test_loss:.4f} | Accuracy: {test_accuracy:.2f}%")
    
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.title("Training Curves (MNIST 0 vs. 1)")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results_mnist/loss_curves.png", dpi=300)
    plt.close()
    
    # 6) Topology with subsampling
    print("\n=== Topology Analysis (Class 0) ===")
    betti_0 = perform_topology_analysis(
        network=net,
        data_loader=test_loader,
        class_label=0,
        pca_components=PCA_COMPONENTS,
        threshold=THRESHOLD,
        max_dim=MAX_DIM,
        device=DEVICE,
        max_points_input=MAX_POINTS_INPUT,
        max_points_activations=MAX_POINTS_ACTIVATIONS
    )
    plot_betti(
        betti_0,
        title_str="Topology Through Layers (Class 0, PCA+Alpha+Subsample)",
        save_name="results_mnist/topology_class0.png"
    )
    
    print("\n=== Topology Analysis (Class 1) ===")
    betti_1 = perform_topology_analysis(
        network=net,
        data_loader=test_loader,
        class_label=1,
        pca_components=PCA_COMPONENTS,
        threshold=THRESHOLD,
        max_dim=MAX_DIM,
        device=DEVICE,
        max_points_input=MAX_POINTS_INPUT,
        max_points_activations=MAX_POINTS_ACTIVATIONS
    )
    plot_betti(
        betti_1,
        title_str="Topology Through Layers (Class 1, PCA+Alpha+Subsample)",
        save_name="results_mnist/topology_class1.png"
    )
    
    print("Done! Check 'results_mnist/' for outputs.")


if __name__ == "__main__":
    main()
