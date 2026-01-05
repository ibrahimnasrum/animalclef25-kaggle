# AnimalCLEF25 — Animal Identification (MegaDescriptor-L-384)
**Deep Feature Extraction • Cosine Similarity Retrieval • Thresholding for New Individual • Kaggle Submission**

This repository contains our **AnimalCLEF 2025** solution for **animal identification** using a **pretrained MegaDescriptor-L-384** model.  
The approach follows a **retrieval / re-identification pipeline**:

> **Query image → deep embedding → compare with database embeddings (cosine similarity) → pick top-1 identity → apply threshold → generate submission CSV**

---

## Project Overview
### What this project does
- Loads **AnimalCLEF2025** dataset (split into **database** and **query** sets)
- Extracts deep embeddings using **MegaDescriptor-L-384** (via `timm`)
- Computes **cosine similarity** between query and database embeddings
- Predicts identity using the **highest similarity match (Top-1)**
- Applies a threshold to handle uncertain matches:
  - **> 0.6** → accept prediction
  - **< 0.6** → mark as `new_individual`
- Exports `sample_submission_v2.csv` for Kaggle submission
- Includes visualization (dataset grid + similarity score distribution)

---

## Method (From the code)
### 1) Dataset Loading
We use `wildlife_datasets.datasets.AnimalCLEF2025`:
- `dataset_database`: all images labeled as **database**
- `dataset_query`: all images labeled as **query**

A standard ImageNet normalization pipeline is applied:
- Resize: **384×384**
- Normalize: mean `(0.485, 0.456, 0.406)` / std `(0.229, 0.224, 0.225)`

### 2) Feature Extraction (MegaDescriptor)
We load the pretrained model:
```python
name = "hf-hub:BVRA/MegaDescriptor-L-384"
model = timm.create_model(name, num_classes=0, pretrained=True)
```
Then extract embeddings using DeepFeatures:

```python
extractor = DeepFeatures(model, device="cuda", batch_size=32, num_workers=0)
features_database = extractor(dataset_database)
features_query = extractor(dataset_query)
```

### 3) Similarity Search (Cosine Similarity)

We compute cosine similarity and pick the top match:
```python
similarity = CosineSimilarity()(features_query, features_database)
pred_idx = similarity.argsort(axis=1)[:, -1]
pred_scores = similarity[range(n_query), pred_idx]
```

### 4) Thresholding
We mark uncertain predictions as a new individual:
```python
threshold = 0.6
new_individual = "new_individual"
```
5) Submission File
A Kaggle-ready CSV is generated:
```python
create_sample_submission(dataset_query, predictions, file_name="sample_submission_v2.csv")
```

## Files in This Repository

- Animal_Classification_Final_V2.ipynb — main notebook (data loading, inference, submission export)

- Report_AnimalCLEF25 by Bad Genius - Deep Learning .pdf — project report / methodology write-up

- Notes_on_the_data — dataset download note (Kaggle competition link)

## Dataset Setup (Kaggle)

1. Download the **AnimalCLEF 2025** dataset from Kaggle (competition dataset).
2. Extract/unzip it and place it in a folder like this:

```text
data/animal-clef-2025/
```
3. In the notebook, update the dataset root path:
 ```python
 root = r"data/animal-clef-2025"
 ```
_The notebook uses AnimalCLEF2025(root, ...), so root must point to the dataset root folder that matches the structure expected by the wildlife_datasets library._


## Installation
### Recommended Environment
- Python 3.9+
- CUDA GPU recommended for faster embedding extraction

### Install dependencies

pip install numpy pandas matplotlib
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm
pip install wildlife-datasets wildlife-tools

If you are on CPU-only machine, install CPU PyTorch instead.

## How to Run (Step-by-step)

**Option A — Run the Notebook (Recommended)**
1. Open Animal_Classification_Final_V2.ipynb
2. Update the dataset root path:

  root = r"YOUR_DATASET_PATH"

4. Run all cells from top to bottom

Output file will be created:
- sample_submission_v2.csv (or sample_submission_v2.csv depending on your filename)

**Option B — Export to Script (Optional)**

You can also export notebook to .py and run:
 ```python
python Animal_Classification_Final_V2.py
 ```

## Credits / Team
BAD GENIUS — AnimalCLEF25 (Deep Learning / Wildlife Re-ID approach)

**Section: 1**
Group Members:
1. Muhammad Hafidzuddin
2. Muhairis Bin Azman
3. Ibrahim Bin Nasrum
4. Shareen Arawie
