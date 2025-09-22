# MCLSR

This repository contains a PyTorch implementation of the paper:
**Multi-level Contrastive Learning Framework for Sequential Recommendation** <br>
*Ziyang Wang, Huoyu Liu, Wei Wei, Yue Hu, Xian-Ling Mao, Shaojian He, Rui Fang, Dangyang chen* <br>
**CIKM 2022**

**[arXiv paper link](https://arxiv.org/abs/2208.13007)**

The official source code for this paper was not made publicly available. This implementation was written from scratch based on the paper's description to provide a working and reproducible benchmark for the research community.

## Installation

### Using uv (Recommended)

1. Create and activate a virtual environment:
   ```bash
   uv venv --python 3.12
   source ./.venv/bin/activate
   ```

2. Install dependencies:

   **For development**
   ```bash
   uv sync --all-extras --frozen
   ```

   **For production**
   ```bash
   uv sync --frozen
   ```

### Using pip

1. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source ./.venv/bin/activate
   ```

2. Install dependencies:

   **For development:**
   ```bash
   pip install -e ".[dev]"
   ```

   **For production:**
   ```bash
   pip install -e .
   ```

## Preparing datasets
All datasets used in our experiments are available for download from our cloud storage.

- [Google drive](https://drive.google.com/drive/folders/1dt6eojPi5UILKG5KMVQi-jP5QG1t-HJx?usp=sharing)

After downloading, place the raw files in a folder of your choice. Example structure we recommend:
```
./data/
└── Clothing/
    └── data.csv
```

To prepare the raw data for training, run the Jupyter notebooks provided in the [notebooks](./notebooks) directory. The notebooks will generate the required `.txt` data splits.

**Important:** Before running a notebook, make sure that the file paths inside it point to your downloaded raw data and the desired output location.

## Model training
To train a model, simply run the following from the root directory:
```shell
python src/train_mclsr.py

python src/train_sasrec.py
```
