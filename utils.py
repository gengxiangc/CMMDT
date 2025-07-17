from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Subset

class CMMDataDataset_OneCircle(torch.utils.data.Dataset):
    """
    Dataset class for single circle CMM measurements.
    
    This class handles the loading and preprocessing of CMM measurement data for a single circle.
    It performs circle fitting, residual calculation, and feature standardization.
    
    Args:
        data_path (str): Path to the data directory containing measurement files
        summary_df (pd.DataFrame): DataFrame containing file names and metadata
        radius (float, optional): Expected radius of the circle. Defaults to 50mm
        standardize (bool, optional): Whether to standardize features. Defaults to True
    """
    
    def __init__(self, data_path, summary_df, radius=50, standardize=True):
        self.data_path = data_path
        self.summary = summary_df
        self.radius = radius
        self.standardize = standardize

    def __len__(self):
        return len(self.summary)

    def standardize_features(self, data):

        if data.size == 0:
            raise ValueError("Input data cannot be empty")
            
        if not np.all(np.isfinite(data)):
            raise ValueError("Input data contains invalid values")
            
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True) + 1e-6
        return (data - mean) / std

    def fit_circle(self, x, y):
        A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
        b = x**2 + y**2
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        x_center, y_center = params[:2]
        return x_center, y_center, self.radius

    def __getitem__(self, idx):
        file_name = self.summary.iloc[idx]["file_name"]
        data = pd.read_csv(os.path.join(self.data_path, file_name))

        x = data["X"].values
        y = data["Y"].values

        x_R, y_R = x, y
        x_center_R, y_center_R, radius_R = self.fit_circle(x_R, y_R)

        radius_R = 50

        angles_R = np.arctan2(y_R - y_center_R, x_R - x_center_R)

        x_fitted_R = x_center_R + radius_R * np.cos(angles_R)
        y_fitted_R = y_center_R + radius_R * np.sin(angles_R)

        residuals_x = x_R - x_fitted_R
        residuals_y = y_R - y_fitted_R
        norm_x_R = (x_fitted_R - x_center_R)
        norm_y_R = (y_fitted_R - y_center_R)
        norm_mag_R = np.sqrt(norm_x_R**2 + norm_y_R**2) + 1e-6
        norm_x_R /= norm_mag_R
        norm_y_R /= norm_mag_R

        residuals_n = (x_R - x_fitted_R) * norm_x_R + (y_R - y_fitted_R) * norm_y_R

        input_data = np.vstack([
            x,
            y,
            residuals_n,
            residuals_x,
            residuals_y
        ]).astype(np.float32)

        if self.standardize:
            input_data = self.standardize_features(input_data)

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        return input_tensor


class CMMDataDataset_TwoCircles(torch.utils.data.Dataset):
    def __init__(self, data_path, summary_df, radius=50, standardize=True):
        self.data_path = data_path
        self.summary = summary_df
        self.radius = radius
        self.standardize = standardize

    def __len__(self):
        return len(self.summary)

    def standardize_features(self, data):
        mean = np.mean(data, axis=1, keepdims=True)
        std = np.std(data, axis=1, keepdims=True) + 1e-6
        return (data - mean) / std

    def fit_circle(self, x, y):
        A = np.column_stack([2 * x, 2 * y, np.ones(len(x))])
        b = x**2 + y**2
        params = np.linalg.lstsq(A, b, rcond=None)[0]
        x_center, y_center = params[:2]
        return x_center, y_center, self.radius

    def __getitem__(self, idx):
        file_name = self.summary.iloc[idx]["file_name"]
        data = pd.read_csv(os.path.join(self.data_path, file_name))

        x = data["X"].values
        y = data["Y"].values

        n_half = len(x) // 2
        x_R, y_R = x[:n_half], y[:n_half]
        x_06R, y_06R = x[n_half:], y[n_half:]

        x_center_R, y_center_R, radius_R = self.fit_circle(x_R, y_R)
        x_center_06R, y_center_06R, radius_06R = self.fit_circle(x_06R, y_06R)

        radius_R = 50
        radius_06R = 30

        angles_R = np.arctan2(y_R - y_center_R, x_R - x_center_R)
        angles_06R = np.arctan2(y_06R - y_center_06R, x_06R - x_center_06R)

        x_fitted_R = x_center_R + radius_R * np.cos(angles_R)
        y_fitted_R = y_center_R + radius_R * np.sin(angles_R)
        x_fitted_06R = x_center_06R + radius_06R * np.cos(angles_06R)
        y_fitted_06R = y_center_06R + radius_06R * np.sin(angles_06R)

        residuals_x = np.concatenate([x_R - x_fitted_R, x_06R - x_fitted_06R])
        residuals_y = np.concatenate([y_R - y_fitted_R, y_06R - y_fitted_06R])

        norm_x_R = (x_fitted_R - x_center_R)
        norm_y_R = (y_fitted_R - y_center_R)
        norm_mag_R = np.sqrt(norm_x_R**2 + norm_y_R**2) + 1e-6
        norm_x_R /= norm_mag_R
        norm_y_R /= norm_mag_R

        norm_x_06R = (x_fitted_06R - x_center_06R)
        norm_y_06R = (y_fitted_06R - y_center_06R)
        norm_mag_06R = np.sqrt(norm_x_06R**2 + norm_y_06R**2) + 1e-6
        norm_x_06R /= norm_mag_06R
        norm_y_06R /= norm_mag_06R

        residuals_n = np.concatenate([
            (x_R - x_fitted_R) * norm_x_R + (y_R - y_fitted_R) * norm_y_R,
            (x_06R - x_fitted_06R) * norm_x_06R + (y_06R - y_fitted_06R) * norm_y_06R
        ])

        input_data = np.vstack([
            x,
            y,
            residuals_n,
            residuals_x,
            residuals_y
        ]).astype(np.float32)

        if self.standardize:
            input_data = self.standardize_features(input_data)

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        return input_tensor

from sklearn.preprocessing import StandardScaler

class PairedDataset(torch.utils.data.Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def prepare_data(data_path, args):

    label = args.target
    task = args.task
    num_test = args.num_test
    num_train = args.num_train

    summary = pd.read_csv(os.path.join(data_path, "summary.csv"))
    label_scaler = StandardScaler()
    # 提取标签并标准化
    if label == "label":
        y_scaled = summary[label].values.reshape(-1, 1)
    else:
        y_all = summary[label].values.reshape(-1, 1) * 100000

        y_scaled = label_scaler.fit_transform(y_all).astype(np.float32)

    # 构建输入特征
    if task == 'OneCircle':
        feature_dataset = CMMDataDataset_OneCircle(data_path, summary_df=summary)
    if task == 'TwoCircles':
        feature_dataset = CMMDataDataset_TwoCircles(data_path, summary_df=summary)
    features = [feature_dataset[i] for i in range(len(feature_dataset))]
    features = torch.stack(features)
    labels = torch.tensor(y_scaled).squeeze(1)

    dataset = PairedDataset(features, labels)

    total_len = len(dataset)

    rng = np.random.default_rng(seed=42)
    all_indices = rng.permutation(total_len).tolist()

    test_indices = all_indices[-num_test:]
    train_indices = all_indices[:num_train]

    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, test_set, label_scaler

def standardize_data(data):
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-6
    return (data - mean) / std

def fit_circle(x, y):
    def cost(params):
        x0, y0, r = params
        return np.sum((np.sqrt((x - x0)**2 + (y - y0)**2) - r)**2)
    x_mean, y_mean = np.mean(x), np.mean(y)
    r_mean = np.mean(np.sqrt((x - x_mean)**2 + (y - y_mean)**2))
    return x_mean, y_mean, r_mean


def save_results_r(model, optimizer, scheduler, train_loader, test_loader,
                 epoch_losses, metrics, label_scaler, args, output_path, device):
    """
    Unified function to save all model results and data
    """
    os.makedirs(output_path, exist_ok=True)

    # 1. Save model state
    model_save_path = os.path.join(output_path, "model_state.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': args.epochs,
        'model_config': {
            'model_type': args.model,
            'target': args.target,
            'task': args.task,
            'in_channels': 5,
            'out_dim': 1,
            'task_type': 'regression',
            'conv_channels': (32, 32, 32) if args.model == 'Transformer' else None,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs
        }
    }, model_save_path)
    print(f"Model state saved to {model_save_path}")

    # 2. Save training and test data with predictions
    def process_data_loader(loader, label_scaler):
        all_inputs = []
        all_true_labels = []
        all_predicted_labels = []

        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                labels = labels.to(device).float().view(-1, 1)
                outputs = model(inputs)

                # Inverse transform the predictions and labels
                outputs = label_scaler.inverse_transform(outputs.cpu().reshape(-1, 1)).flatten()
                labels = label_scaler.inverse_transform(labels.cpu().reshape(-1, 1)).flatten()

                all_inputs.append(inputs.cpu().numpy())
                all_true_labels.append(labels)
                all_predicted_labels.append(outputs)

        return {
            'inputs': np.concatenate(all_inputs),
            'true_labels': np.concatenate(all_true_labels),
            'predicted_labels': np.concatenate(all_predicted_labels)
        }

    # Process and save training data
    train_results = process_data_loader(train_loader, label_scaler)
    test_results = process_data_loader(test_loader, label_scaler)

    # Save all results in a single file
    results = {
        'train': {
            'data': train_results,
            'metrics': {
                'losses': epoch_losses
            }
        },
        'test': {
            'data': test_results,
            'metrics': metrics
        },
        'model_config': {
            'model_type': args.model,
            'target': args.target,
            'task': args.task,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs
        },
        'label_scaler': {
            'mean_': label_scaler.mean_,
            'scale_': label_scaler.scale_
        }
    }

    # Save all results in a single file
    results_path = os.path.join(output_path, "results.pt")
    torch.save(results, results_path)
    print(f"All results saved to {results_path}")

    # Save training metrics for easy plotting
    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, len(epoch_losses) + 1)),
        "Loss": epoch_losses
    })
    metrics_df.to_csv(os.path.join(output_path, "training_metrics.csv"), index=False)
    print("Training metrics saved for plotting")

def save_results_cl(model, optimizer, scheduler, train_loader, test_loader,
                 epoch_losses, train_accuracies, test_accuracies, metrics,
                 args, output_path, device):

    os.makedirs(output_path, exist_ok=True)

    # 1. Save model state
    model_save_path = os.path.join(output_path, "model_state.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': args.epochs,
        'model_config': {
            'model_type': args.model,
            'target': args.target,
            'task': args.task,
            'in_channels': 5,
            'out_dim': 4,
            'dropout_rate': 0.1,
            'task_type': 'classification',
            'conv_channels': (32, 32, 32) if args.model == 'Transformer' else None,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs
        }
    }, model_save_path)
    print(f"Model state saved to {model_save_path}")

    # 2. Save all data and results in a structured format
    def process_data_loader(loader):
        all_inputs = []
        all_true_labels = []
        all_predicted_labels = []
        all_probabilities = []

        model.eval()
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)

                all_inputs.append(inputs.cpu().numpy())
                all_true_labels.append(labels.numpy())
                all_predicted_labels.append(predicted.cpu().numpy())
                all_probabilities.append(probabilities.cpu().numpy())

        return {
            'inputs': np.concatenate(all_inputs),
            'true_labels': np.concatenate(all_true_labels),
            'predicted_labels': np.concatenate(all_predicted_labels),
            'probabilities': np.concatenate(all_probabilities)
        }

    # Process and save training data
    train_results = process_data_loader(train_loader)
    test_results = process_data_loader(test_loader)

    # Save all results in a single file
    results = {
        'train': {
            'data': train_results,
            'metrics': {
                'losses': epoch_losses,
                'accuracies': train_accuracies
            }
        },
        'test': {
            'data': test_results,
            'metrics': {
                'accuracies': test_accuracies,
                'final_metrics': metrics
            }
        },
        'model_config': {
            'model_type': args.model,
            'target': args.target,
            'task': args.task,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'epochs': args.epochs
        }
    }

    # Save all results in a single file
    results_path = os.path.join(output_path, "results.pt")
    torch.save(results, results_path)
    print(f"All results saved to {results_path}")

    # Save training metrics for easy plotting
    metrics_df = pd.DataFrame({
        "Epoch": list(range(1, len(epoch_losses) + 1)),
        "Loss": epoch_losses,
        "Train Accuracy": train_accuracies,
        "Test Accuracy": test_accuracies
    })
    metrics_df.to_csv(os.path.join(output_path, "training_metrics.csv"), index=False)
    print("Training metrics saved for plotting")