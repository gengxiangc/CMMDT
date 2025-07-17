"""
Data-driven error source identification for metrology digital twins: example on Virtual CMM
This script belongs to the EPM ViDit project 22DIT01, task 2.1.6.
Author: Gengxiang CHEN, USPN, gengxiang.chen@univ-paris13.fr

"""

from torch.utils.data import Dataset, DataLoader, random_split
import torch
from model import *
from utils import *
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import wandb
import time
import os
import torch
from utils import save_results_cl

def main(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join("data")

    num_classes = 4
    print("data_path:: ", data_path)
    output_path = os.path.join("output", f"{args.task}", f"model_output_{args.target}_{args.model}")

    os.makedirs(data_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    train_set, test_set, label_scaler = prepare_data(data_path, args)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    model = CMMTransformer(in_channels=5,
                           out_dim=4,
                           dropout_rate=0.1,
                           conv_channels=(32, 32, 32),
                           task_type='classification').to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)

    time_start = time.perf_counter()
    time_step = time.perf_counter()

    ########################################################################
    print("Start training the model ...")

    model.train()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    epoch_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(args.epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        time_step_end = time.perf_counter()
        Time = time_step_end - time_step
        avg_loss = epoch_loss / len(train_loader)
        train_accuracy = 100 * correct / total
        epoch_losses.append(avg_loss)
        train_accuracies.append(train_accuracy)
        
        # Calculate test metrics
        test_loss = 0
        test_correct = 0
        test_total = 0
        model.eval()
        class_correct = [0] * num_classes
        class_total = [0] * num_classes

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()

                for i in range(labels.size(0)):
                    label = labels[i].item()
                    pred = predicted[i].item()
                    if label == pred:
                        class_correct[label] += 1
                    class_total[label] += 1

        test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        model.train()
        
        print(f"Epoch {epoch + 1}/{args.epochs}, "
              f"Loss: {avg_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, "
              f"Test Loss: {test_loss:.4f}, "
              f"Test Acc: {test_accuracy:.2f}%, "
              f"Time: {Time:.3f}")

        scheduler.step()
        time_step = time.perf_counter()

        print("Per-class accuracy:")
        for i in range(num_classes):
            acc = 100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            print(f"  Class {i} Accuracy: {acc:.2f}%")

    ########################################################################
    print("Start testing the model ...")
    metrics, features, labels, predictions = evaluate_model(model, test_loader, device=device)

    if args.save == 1:
        print("Saving results ...")
        save_results_cl(model, optimizer, scheduler, train_loader, test_loader,
                    epoch_losses, train_accuracies, test_accuracies, metrics,
                    args, output_path, device)
    
        print("All results saved successfully!")

    return metrics

def evaluate_model(model, test_loader, criterion=nn.CrossEntropyLoss(), device=None):
    model.eval()
    all_preds = []
    all_targets = []
    all_inputs = []
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.append(predicted.cpu())
            all_targets.append(labels.cpu())
            all_inputs.append(inputs.cpu())

    test_loss = test_loss / len(test_loader)
    test_accuracy = 100 * correct / total

    print(f"\nTest Loss: {test_loss:.5f}")
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    return {
        "Test Loss": test_loss,
        "Test Accuracy": test_accuracy
    }, torch.cat(all_inputs), torch.cat(all_targets), torch.cat(all_preds)

import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='CMM Digital Twin')

    parser.add_argument('--model', type=str, default='Transformer') #  'CNN' or  'Transformer'
    parser.add_argument('--target', type=str, default='label')  # squareness, scale_x, scale_y, label
    parser.add_argument('--batch_size', type=int, default=20, help='Batch size')
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




