# Code for the paper "Digitization of molecular complexity with machine learning"

This repository contains the code for the paper "Digitization of molecular complexity with machine learning".

## Installation
After cloning the repository, install conda environment with all the dependencies.
```bash
conda env create -f environment.yml
conda activate mc
pip install -e .
```
To use the code you will also need to download the zip file containing all the relevant data from [here](https://drive.google.com/file/d/1vlHBtQqh2-CMav8x4edBPzyUXkq3BhMe/view?usp=sharing) and unzip it in the project's folder.
```bash
unzip data.zip
rm data.zip
```

## Usage

To calculate the molecular complexity of a list of smiles, run the following command:
```bash
python scripts/calculate.py --txt_with_smiles data/example_smiles.txt
```
Where `data/example_smiles.txt` is a text file with one SMILES string per line. The output will be saved in `example_mc.json` file.

To plot the results from the paper, run the scripts from `scripts` folder that begin with `plot_`:
```bash
python scripts/plot_fda.py
python scripts/plot_synthesis.py
```



