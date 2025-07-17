"""
Data-driven error source identification for metrology digital twins: example on Virtual CMM
This script belongs to the EPM ViDit project 22DIT01, task 2.1.6.
Author: Gengxiang CHEN, Charyar Mehdi-Souzani
USPN, gengxiang.chen@univ-paris13.fr

"""

from model import *
from utils import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os
import torch
from utils import save_results_r

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join("data")

    print("data_path:: ", data_path)
    output_path = os.path.join("output", f"{args.task}", f"model_output_{args.target}_{args.model}")

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    train_set, test_set, label_scaler = prepare_data(data_path, args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = CMMTransformer(in_channels=5,
                           out_dim=1,
                           task_type='regression').to(device)

    criterion = nn.MSELoss()

    # Optimizer configuration with L2 regularization
    # Learning rate: 0.001 for initial training
    # Weight decay: 1e-4 for regularization strength
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    time_start = time.perf_counter()
    time_step = time.perf_counter()


    ########################################################################
    print("Start training the model ...")

    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    epoch_losses = []

    for epoch in range(args.epochs):
        epoch_loss = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        time_step_end = time.perf_counter()
        Time = time_step_end - time_step
        avg_loss = epoch_loss / len(train_loader)
        epoch_losses.append(avg_loss)

        test_loss = 0
        test_mae = 0

        model.eval()
        with torch.no_grad():
            all_preds = []
            all_targets = []
            for inputs, targets in test_loader:
                inputs = inputs.to(device)
                targets = targets.to(device).float().view(-1, 1)

                outputs = model(inputs)

                loss = criterion(outputs, targets)
                test_loss += loss.item()

                outputs = label_scaler.inverse_transform(outputs.cpu().reshape(-1, 1)).flatten()
                targets = label_scaler.inverse_transform(targets.cpu().reshape(-1, 1)).flatten()

                all_preds.append(torch.Tensor(outputs))
                all_targets.append(torch.Tensor(targets))
                # all_inputs.append(inputs.cpu())

            y_pred = torch.cat(all_preds).squeeze().numpy()
            y_true = torch.cat(all_targets).squeeze().numpy()

            test_loss = test_loss / len(test_loader)
            test_mae = mean_absolute_error(y_true, y_pred)

            model.train()

            print("\nSample predictions on test set:")
            print("True : ", np.round(y_true[:5], 4))
            print("Pred : ", np.round(y_pred[:5], 4))
        
        print(f"Epoch {epoch + 1}/{args.epochs}, Loss: {avg_loss:.4f}, Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}, Time: {Time:.3f}" )
        scheduler.step()
        time_step = time.perf_counter()

    ########################################################################

    print("Start testing the model ...")
    metrics, features, labels, predictions = evaluate_model(model, test_loader, label_scaler, device=device)

    if args.save == 1:
        print("Saving results ...")
        save_results_r(model, optimizer, scheduler, train_loader, test_loader,
                    epoch_losses, metrics, label_scaler, args, output_path, device)

        print("All results saved successfully!")

    return metrics

def evaluate_model(model, test_loader, label_scaler, device=None):

    model.eval()
    all_preds = []
    all_targets = []
    all_inputs = []

    with torch.no_grad():

        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device).float().view(-1, 1)

            outputs = model(inputs)

            outputs = label_scaler.inverse_transform(outputs.cpu().reshape(-1, 1)).flatten()
            targets = label_scaler.inverse_transform(targets.cpu().reshape(-1, 1)).flatten()

            all_preds.append(torch.Tensor(outputs))
            all_targets.append(torch.Tensor(targets))
            all_inputs.append(inputs.cpu())

    y_pred = torch.cat(all_preds).squeeze().numpy()
    y_true = torch.cat(all_targets).squeeze().numpy()

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nTest MSE: {mse:.5f}")
    print(f"Test MAE: {mae:.5f}")
    print(f"Test R² Score: {r2:.5f}")

    return {
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }, torch.cat(all_inputs), torch.cat(all_targets), torch.cat(all_preds)


import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CMM Digital Twin')

    parser.add_argument('--model', type=str, default='Transformer') # 'Transformer'
    parser.add_argument('--target', type=str, default='squareness')  # squareness,scale_x,scale_y,classification
    parser.add_argument('--batch_size', type=int, default=50, help='Batch size')
    parser.add_argument('--num_train', type=int, default=600, help='Number of training samples')
    parser.add_argument('--num_test', type=int, default=200, help='Number of test samples')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--save_name', type=str, default='CMM-', help='Base name for saving results')
    parser.add_argument('--task', type=str, default='OneCircle', help='TwoCircles or OneCircle')
    parser.add_argument('--save', type=int, default=1, help='Save everything or not')

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n============================")
    if torch.cuda.is_available():
        print("torch.cuda.get_device_name(0): " + str(torch.cuda.get_device_name(0)))
    print("=============================\n")

    print(f" Target: {args.target} | Model: {args.model} | Task: {args.task} ")

    main(args)




