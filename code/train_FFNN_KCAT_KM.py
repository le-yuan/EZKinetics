#!/usr/bin/python
# coding: utf-8
# Date: 2026-03-04

import os
import copy
import pickle
import torch
import argparse
import pandas as pd
import numpy as np
from os.path import join
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def get_arguments():
    parser = argparse.ArgumentParser(
        description="FFNN training using embeddings calculated from different protein and compound language models."
    )
    parser.add_argument(
        "--protein_model",
        type=str,
        required=True,
        help="This is the pre-trained protein language model",
    )
    parser.add_argument(
        "--substrate_model",
        type=str,
        required=True,
        help="This is the pre-trained compound language model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch size for training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.2,
        help="Dropout rate",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        nargs="+",
        default=[1024, 512, 256],
        help="Hidden layer sizes, e.g. --hidden_dims 1024 512 256",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience",
    )

    args = parser.parse_args()
    return args

# Load protein embeddings
def load_protein_embeddings(protein_path) :
    try :
        # protein_embedding = torch.load(protein_path)
        protein_embedding = torch.load(protein_path, weights_only=False)
        # print("Protein embeddings loaded successfully")
        return protein_embedding
    except Exception as e :
        print(f"An error occurred: {e}")

# Load substrate embeddings
def load_substrate_embeddings(substrate_path) :
    try :
        with open(substrate_path, "rb") as infile :
            substrate_embedding = pickle.load(infile)
            # print("Substrate embeddings loaded successfully")
            return substrate_embedding
    except Exception as e :
        print(f"An error occurred: {e}")

# Load all of the protein and substrate embeddings
def load_embeddings(datasets_dir, training_dir, protein_model, substrate_model, reduce_to_1D=True):
    protein_embeddings = dict()
    substrate_embeddings = dict()
    protein_files = os.listdir(join(datasets_dir, training_dir, "embeddings", "Protein", protein_model))
    substrate_files = os.listdir(join(datasets_dir, training_dir, "embeddings", "SMILES", substrate_model))

    for i in range(len(protein_files)):
        protein_path = os.path.join(datasets_dir, training_dir, "embeddings", "Protein", protein_model, f"Protein_embeddings_V{i+1}.pt")
        protein_embedding = load_protein_embeddings(protein_path)
        for key, value in protein_embedding.items():
            if reduce_to_1D:
                protein_embeddings[key] = value.mean(0)
            else:
                protein_embeddings[key] = value
    print("The length of protein embeddings:", len(protein_embeddings))

    for k in range(len(substrate_files)):
        substrate_path = os.path.join(datasets_dir, training_dir, "embeddings", "SMILES", substrate_model, f"SMILES_repr_{k+1}.pkl")
        substrate_embedding = load_substrate_embeddings(substrate_path)
        for key, value in substrate_embedding.items():
            if "UniMol" == substrate_model:
                substrate_embeddings[key] = value  # already 1D: shape (d,)
            elif reduce_to_1D:
                substrate_embeddings[key] = value.squeeze(0).mean(0).detach().numpy()
            else:
                substrate_embeddings[key] = value
    print("The length of substrate embeddings:", len(substrate_embeddings))

    return protein_embeddings, substrate_embeddings

def global_average_pooling(embedding):
    return embedding.mean(0)

def prepare_dataset(data, protein_embeddings, substrate_embeddings):
    features = []
    labels = []
    protein_sequences = []
    substrate_smiles = []
    for idx in range(len(data)):
        protein_seq = data.iloc[idx]['Sequence']
        substrate_smile = data.iloc[idx]['SMILES']
        ph = data.iloc[idx]["PH"]
        temp = data.iloc[idx]["Temperature"]
        label = data.iloc[idx]['Log10_value']
        protein_embedding = protein_embeddings.get(protein_seq[:1022])
        substrate_embedding = substrate_embeddings.get(substrate_smile)
        if protein_embedding is None or substrate_embedding is None:
            continue

        if protein_embedding.ndim == 1:
            protein_pooled = protein_embedding
        else:
            protein_pooled = global_average_pooling(protein_embedding)

        if substrate_embedding.ndim == 1:
            substrate_pooled = substrate_embedding
        else:
            substrate_pooled = global_average_pooling(substrate_embedding.squeeze(0)).detach().numpy()

        combined_embedding = np.concatenate((protein_pooled, substrate_pooled, np.array([ph, temp])))
        features.append(combined_embedding)
        labels.append(label)
        protein_sequences.append(protein_seq)
        substrate_smiles.append(substrate_smile)

    return np.array(features), np.array(labels), protein_sequences, substrate_smiles

class RegressionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class FFNNRegressor(nn.Module):
    def __init__(self, input_dim, hidden_dims=[1024, 512, 256], dropout=0.2):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for X_batch, y_batch in dataloader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def evaluate_loss(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            running_loss += loss.item() * X_batch.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def predict(model, dataloader, device):
    model.eval()
    preds = []
    targets = []

    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)

            preds.append(outputs.cpu().numpy())
            targets.append(y_batch.numpy())

    preds = np.vstack(preds).flatten()
    targets = np.vstack(targets).flatten()
    return preds, targets

def train_ffnn(datasets_dir, training_dir, protein_model, substrate_model, model_name,
               batch_size=128, epochs=200, lr=1e-3, dropout=0.2,
               hidden_dims=[1024, 512, 256], patience=20):

    print("Loading full dataset from ../data/kinetics_data/data_KCATKM.csv ...")
    data_df = pd.read_csv("../data/kinetics_data/data_KCATKM.csv")
    print("Total data:", len(data_df))

    train_data, test_val_data = train_test_split(
        data_df,
        test_size=0.2,
        random_state=42,
        shuffle=True
    )
    val_data, test_data = train_test_split(
        test_val_data,
        test_size=0.5,
        random_state=42,
        shuffle=True
    )

    print("Train data size:", len(train_data))
    print("Validation data size:", len(val_data))
    print("Test data size:", len(test_data))

    protein_embeddings, substrate_embeddings = load_embeddings(
        datasets_dir, training_dir, protein_model, substrate_model, reduce_to_1D=True
    )
    print("Finish protein and substrate embeddings...")

    X_train, y_train, _, _ = prepare_dataset(train_data, protein_embeddings, substrate_embeddings)
    X_val, y_val, _, _ = prepare_dataset(val_data, protein_embeddings, substrate_embeddings)
    X_test, y_test, protein_sequences_test, substrate_smiles_test = prepare_dataset(
        test_data, protein_embeddings, substrate_embeddings
    )

    print("Prepared train samples:", len(X_train))
    print("Prepared val samples:", len(X_val))
    print("Prepared test samples:", len(X_test))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = RegressionDataset(X_train, y_train)
    val_dataset = RegressionDataset(X_val, y_val)
    test_dataset = RegressionDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    input_dim = X_train.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FFNNRegressor(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        dropout=dropout
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    print("Device:", device)
    print("Input dimension:", input_dim)
    print("Model:", model)
    print(f"batch_size = {batch_size}")
    print(f"epochs = {epochs}")
    print(f"learning rate = {lr}")
    print(f"dropout = {dropout}")
    print(f"hidden_dims = {hidden_dims}")
    print(f"patience = {patience}")

    best_val_loss = float("inf")
    best_model_state = None
    early_stop_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loss(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch+1}/{epochs}] | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    model.load_state_dict(best_model_state)
    print("Finish model training...")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")

    model_save_path = join(
        "../model",
        training_dir,
        model_name,
        protein_model + "&" + substrate_model,
        f"lr_{lr}_bs_{batch_size}_dropout_{dropout}"
    )
    os.makedirs(model_save_path, exist_ok=True)

    torch.save(model.state_dict(), join(model_save_path, f"FFNN_model_{training_dir}.pt"))
    with open(join(model_save_path, f"scaler_{training_dir}.pkl"), "wb") as file:
        pickle.dump(scaler, file)

    y_pred, y_true = predict(model, test_loader, device)

    test_data_with_predictions = pd.DataFrame({
        'protein_sequence': protein_sequences_test,
        'substrate_SMILES': substrate_smiles_test,
        'y_test': y_true,
        'y_pred': y_pred
    })
    test_data_save_path = join(model_save_path, "test_data_predictions.csv")
    test_data_with_predictions.to_csv(test_data_save_path, index=False)

    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    metrics_df = pd.DataFrame([{
        "protein_model": protein_model,
        "substrate_model": substrate_model,
        "batch_size": batch_size,
        "lr": lr,
        "dropout": dropout,
        "test_mse": mse,
        "test_r2": r2,
        "test_pearson": pearson_corr
    }])
    metrics_df.to_csv(join(model_save_path, "metrics.csv"), index=False)

    print(f"Test MSE: {mse}")
    print(f"Test R²: {r2}")
    print(f"Pearson correlation coefficient: {pearson_corr}")

def main():
    args = get_arguments()
    datasets_dir = "../data"
    training_dir = "KCAT_KM"
    protein_model = args.protein_model
    substrate_model = args.substrate_model
    model_name = "FFNN"

    print("Training model for:", training_dir)
    print(model_name)
    print(protein_model + "&" + substrate_model)

    train_ffnn(
        datasets_dir=datasets_dir,
        training_dir=training_dir,
        protein_model=protein_model,
        substrate_model=substrate_model,
        model_name=model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        hidden_dims=args.hidden_dims,
        patience=args.patience,
    )


if __name__ == '__main__':
    main()


