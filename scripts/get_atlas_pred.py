import json

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdChemReactions
from utils import get_predictor

from mc.utils import valid_mol

RDLogger.DisableLog("rdApp.*")


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def process_reaction(reaction: str):
    rxn = rdChemReactions.ReactionFromSmarts(reaction)
    reacts = rxn.GetReactants()
    prods = rxn.GetProducts()

    reacts_smiles = [Chem.MolToSmiles(react) for react in reacts]
    prods_smiles = [Chem.MolToSmiles(prod) for prod in prods]

    return reacts_smiles, prods_smiles


def label(cmpds_list, predictor, reduce="sum"):
    splitted = [[c for c in cmpd.split(".") if valid_mol(c)] for cmpd in cmpds_list]
    ids = [[i for _ in range(len(cmpd))] for i, cmpd in enumerate(splitted)]
    cmpds = flatten(splitted)
    ids = flatten(ids)
    mc = predictor.predict(cmpds)
    mc_per_cmpd = [[] for i in range(max(ids) + 1)]
    for idx, i in enumerate(ids):
        mc_per_cmpd[i].append(mc[idx])
    if reduce is not None:
        if reduce == "sum":
            f = lambda x: sum(x)
        elif reduce == "max":
            f = lambda x: max(x) if x else 0
        elif reduce == "mean":
            f = lambda x: sum(x) / len(x)
        else:
            raise ValueError("Invalid value for reduce")
        mc_per_cmpd = [f(x) for x in mc_per_cmpd]
    return mc_per_cmpd


def main():
    with open("data/atlas/rxnclass2name.json", "r") as f:
        rxnclass2name = json.load(f)
    df = pd.read_csv("data/atlas/schneider50k.tsv", sep="\t", index_col=0)
    ft_10k_fps = np.load("data/atlas/fps_ft_10k.npz")["fps"]
    df["rxn_category"] = df.rxn_class.apply(lambda x: ".".join(x.split(".")[:2]))
    df["rxn_superclass"] = df.rxn_class.apply(lambda x: x.split(".")[0])

    rxns = df["rxn"].tolist()
    df["reacts"] = df["rxn"].apply(lambda s: s.split(">>")[0])
    df["prods"] = df["rxn"].apply(lambda s: s.split(">>")[1])
    reacts = df["reacts"].tolist()
    prods = df["prods"].tolist()

    predictor = get_predictor()
    labels_reacts = label(reacts, predictor, "max")
    labels_prods = label(prods, predictor, "max")
    df["mc_delta"] = np.array(labels_prods) - np.array(labels_reacts)
    df.to_csv("data/atlas/schneider50k_mc.csv", index=False)


if __name__ == "__main__":
    main()
