#!/usr/bin/python
# coding: utf-8
# Date: 2025-12-04

import os
import pickle
import torch
import argparse
import pandas as pd
import numpy as np
from os.path import join
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import RandomizedSearchCV


def get_arguments() :
    parser = argparse.ArgumentParser(description="Model training using embeddings calculated from different protein and compound language models.")
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

    args = parser.parse_args()
    return args

# Load train data and test data
def load_data(datasets_dir, training_dir) :
    train_data = pd.read_csv(join(datasets_dir, training_dir, "train_test", 'train.csv'))
    test_data = pd.read_csv(join(datasets_dir, training_dir, "train_test", 'test.csv'))
    print("This is for:", training_dir)
    print("Total data:", len(train_data) + len(test_data))
    print("Train data size:", len(train_data))
    print("Test data size:", len(test_data))
    return train_data, test_data

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

# Prepare the dataset
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

def train_ridge(datasets_dir, training_dir, protein_model, substrate_model, model_name):
    train_data, test_data = load_data(datasets_dir, training_dir)
    print("Finish data loading...")

    protein_embeddings, substrate_embeddings = load_embeddings(
        datasets_dir, training_dir, protein_model, substrate_model, reduce_to_1D=True
    )
    print("Finish protein and substrate embeddings...")

    X_train, y_train, _, _ = prepare_dataset(train_data, protein_embeddings, substrate_embeddings)
    X_test, y_test, protein_sequences_test, substrate_smiles_test = prepare_dataset(
        test_data, protein_embeddings, substrate_embeddings
    )

    print("Training feature shape:", X_train.shape)
    print("Test feature shape:", X_test.shape)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    param_dist = {
        "alpha": np.logspace(-3, 3, 100),
        "fit_intercept": [True, False],
        "solver": ["auto", "svd", "cholesky", "lsqr", "sag", "sparse_cg"]
    }

    model = Ridge()

    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        n_jobs=-1,
        random_state=42,
        scoring="r2"
    )

    random_search.fit(X_train, y_train)
    print("Best parameters:", random_search.best_params_)

    best_model = random_search.best_estimator_
    model = best_model
    print("Finish model training...")

    model_save_path = join("../model", training_dir, model_name, protein_model + "&" + substrate_model)
    os.makedirs(model_save_path, exist_ok=True)

    with open(join(model_save_path, f"Ridge_model_{training_dir}.pkl"), "wb") as file:
        pickle.dump(model, file)

    with open(join(model_save_path, f"scaler_{training_dir}.pkl"), "wb") as file:
        pickle.dump(scaler, file)

    y_pred = model.predict(X_test)

    test_data_with_predictions = pd.DataFrame({
        "protein_sequence": protein_sequences_test,
        "substrate_SMILES": substrate_smiles_test,
        "y_test": y_test,
        "y_pred": y_pred
    })
    test_data_save_path = join(model_save_path, "test_data_predictions.csv")
    test_data_with_predictions.to_csv(test_data_save_path, index=False)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    pearson_corr, _ = pearsonr(y_test, y_pred)

    print(f"Test MSE: {mse}")
    print(f"Test R²: {r2}")
    print(f"Pearson correlation coefficient: {pearson_corr}")


def main():
    args = get_arguments()
    datasets_dir = "../data"
    training_dir = "KCAT_KM"
    protein_model = args.protein_model
    substrate_model = args.substrate_model
    model_name = "Ridge"

    print("Training model for:", training_dir)
    print(model_name)
    print(protein_model + "&" + substrate_model)

    train_ridge(datasets_dir, training_dir, protein_model, substrate_model, model_name)


if __name__ == '__main__':
    main()


