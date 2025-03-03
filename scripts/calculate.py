import json
from argparse import ArgumentParser
from utils import get_predictor


def main():
    parser = ArgumentParser()
    parser.add_argument("--txt_with_smiles", type=str, required=True)
    args = parser.parse_args()

    with open(args.txt_with_smiles, "r") as f:
        smiles_list = f.readlines()
    smiles_list = [smiles.strip() for smiles in smiles_list]

    predictor = get_predictor()
    mc_list = predictor.predict(smiles_list)
    mc_dict = {smiles: mc for smiles, mc in zip(smiles_list, mc_list)}

    with open("example_mc.json", "w") as f:
        json.dump(mc_dict, f)


if __name__ == "__main__":
    main()
