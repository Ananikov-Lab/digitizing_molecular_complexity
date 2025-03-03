import os
import pickle
from pprint import pprint

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import umap
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
from tqdm import tqdm
from utils import get_predictor


RDLogger.DisableLog("rdApp.*")


def read_mols():
    data_path = "data/mols_clean_pubchem.txt"
    mols = []
    with open(data_path, "r") as f:
        for line in tqdm(f, total=75000):
            line = line.strip().split("\t")[-1]
            mols.append(line)
    return mols


def compute_and_save_embeddings(
    smiles_list,
    complexity_list,
    method="tsne",
    n_components=2,
    perplexity=30,
    n_neighbors=15,
    min_dist=0.1,
):
    """
    Computes and saves embeddings and molecular complexity to files.

    Parameters:
        smiles_list (list of str): List of SMILES strings.
        complexity_list (list of float): List of molecular complexity values.
        method (str): Method for dimensionality reduction ('tsne' or 'umap'). Default is 'tsne'.
        n_components (int): Number of dimensions for the reduced embedding. Default is 2.
        perplexity (int): Perplexity parameter for t-SNE. Default is 30.
        n_neighbors (int): Number of neighbors for UMAP. Default is 15.
        min_dist (float): Minimum distance parameter for UMAP. Default is 0.1.
    """
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fingerprints.append(fp)
        else:
            fingerprints.append(None)

    valid_indices = [i for i, fp in enumerate(fingerprints) if fp is not None]
    fingerprints = [fingerprints[i] for i in valid_indices]
    complexity_list = [complexity_list[i] for i in valid_indices]
    smiles_list = [smiles_list[i] for i in valid_indices]

    fingerprints_array = np.array(fingerprints)

    if method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=perplexity)
    elif method == "umap":
        reducer = umap.UMAP(
            n_components=n_components, n_neighbors=n_neighbors, min_dist=min_dist
        )
    else:
        raise ValueError("Invalid method. Choose 'tsne' or 'umap'.")

    embeddings = reducer.fit_transform(fingerprints_array)

    with open("data/pubchem/embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    with open("data/pubchem/smiles_list.pkl", "wb") as f:
        pickle.dump(smiles_list, f)
    with open("data/pubchem/complexity_list.pkl", "wb") as f:
        pickle.dump(complexity_list, f)


def fix_seed(seed=42):
    np.random.seed(seed)
    import random

    random.seed(seed)
    return seed


def load_precomputed_data():
    with open("data/pubchem/embeddings.pkl", "rb") as f:
        embeddings = pickle.load(f)
    with open("data/pubchem/smiles_list.pkl", "rb") as f:
        smiles_list = pickle.load(f)
    with open("data/pubchem/complexity_list.pkl", "rb") as f:
        complexity_list = pickle.load(f)

    elts_to_keep = set(["C", "O", "N", "Cl", "Br", "F", "I", "P", "S", "Si", "B"])
    filtered_indices = []
    for i, smiles in enumerate(smiles_list):
        if "." in smiles:
            continue
        mol = Chem.MolFromSmiles(smiles)
        elts = set([atom.GetSymbol() for atom in mol.GetAtoms()])
        if elts.issubset(elts_to_keep):
            # or more than 2 molecules
            filtered_indices.append(i)
    embeddings = embeddings[filtered_indices]
    smiles_list = [smiles_list[i] for i in filtered_indices]
    complexity_list = [complexity_list[i] for i in filtered_indices]
    return embeddings, smiles_list, complexity_list


def plot_precomputed_embeddings(method="tsne"):
    embeddings, smiles_list, complexity_list = load_precomputed_data()
    mpl.rcParams.update({"ytick.direction": "out"})

    dbscan = DBSCAN(eps=1, min_samples=40)
    clusters = dbscan.fit_predict(embeddings)

    unique_labels = set(clusters)
    selected_indices = []
    for label in unique_labels:
        if label == -1:
            continue  # Skip noise points
        cluster_indices = np.where(clusters == label)[0]
        cluster_center = np.mean(embeddings[cluster_indices], axis=0)
        distances = np.linalg.norm(embeddings[cluster_indices] - cluster_center, axis=1)
        selected_indices.append(cluster_indices[np.argmin(distances)])

    selected_coordinates = embeddings[selected_indices]

    plt.figure(figsize=(5, 3))
    scatter = plt.scatter(
        embeddings[:, 0],
        embeddings[:, 1],
        c=complexity_list,
        cmap="jet",
        alpha=0.6,
        s=1,
    )
    plt.colorbar(scatter, label="Molecular Complexity")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(f"plots/pubchem_{method}_perplexity_with_crosses.pdf", dpi=600)

    coordinates_to_keep = np.array(
        [[24.733444, 63.494503], [-4.220219, -81.40841], [15.310571, 0.4385271]]
    )
    closest_indices = []
    for coord in coordinates_to_keep:
        distances = np.linalg.norm(embeddings - coord, axis=1)
        closest_indices.append(np.argmin(distances))

    selected_coordinates = embeddings[closest_indices]
    selected_indices = closest_indices

    top_5_smiles_per_cluster = {}
    for idx in selected_indices:
        distances = np.linalg.norm(embeddings - embeddings[idx], axis=1)
        closest_indices = np.argsort(distances)[:5]
        top_5_smiles_per_cluster[idx] = [
            (smiles_list[i], complexity_list[i]) for i in closest_indices
        ]

    plt.scatter(
        selected_coordinates[:, 0],
        selected_coordinates[:, 1],
        color="k",
        marker="x",
        s=10,
        alpha=1,
    )

    for idx, coord in zip(selected_indices, selected_coordinates):
        print(
            f"Index: {idx}, SMILES: {smiles_list[idx]}, Coordinates: {coord}, Complexity: {complexity_list[idx]}"
        )
        pprint(top_5_smiles_per_cluster[idx])
        print()
    plt.tight_layout()
    plt.savefig(f"plots/pubchem_{method}_perplexity_with_crosses_and_top5.png", dpi=600)
    plt.show()


if __name__ == "__main__":
    fix_seed()
    mols = read_mols()[:100]
    if not os.path.exists("data/pubchem/embeddings.pkl"):
        predictor = get_predictor()
        mc_list = predictor.predict(mols)
        compute_and_save_embeddings(mols, mc_list, method="tsne", perplexity=30)
    else:
        plot_precomputed_embeddings(method="tsne")
