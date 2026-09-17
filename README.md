## Dataset

The complete dataset is **large** and is therefore hosted externally on Google Drive rather than included in this repository.

### Dataset & Resources

The complete project data and related resources are available through the following Google Drive:

**[Google Drive — BELT Dataset & Resources](https://drive.google.com/drive/folders/14829etkP5N7kePzCbHAFkZmZaHt4j3XK?usp=sharing)**

The Drive contains the following directories:

```text
Google Drive/
├── data/
├── data_image/
├── augmented/
├── augmented_bagging/
├── augmented_belt/
├── augmented_v2/
├── checkpoints/
├── plots/
├── notebooks/
├── Documents/
├── old/
└── ...
```

### Required Data

To run the BELT pipeline, you primarily need to download:

* **`data/`** — processed datasets, including training, development, and test files.
* **`data_image/`** — corresponding image data required for the multimodal pipeline.

The other directories contain additional project resources such as:

* `augmented/`, `augmented_bagging/`, `augmented_belt/`, `augmented_v2/` — augmented datasets and experiment-specific data.
* `checkpoints/` — saved model checkpoints.
* `plots/` — generated plots and experiment results.
* `notebooks/` — Jupyter notebooks used during development and experimentation.
* `Documents/` — supporting project documents.
* `old/` — previous experiments and results.

These additional resources are **optional** and can be accessed if you want to inspect the experiments, intermediate results, notebooks, checkpoints, or previous results.

### Required Directory Structure

After downloading the required folders, your local data directory should contain:

```text
data_dir/
├── data/
│   └── processed/
│       ├── inform_train.csv
│       ├── inform_dev.csv
│       └── inform_test.csv
│
└── data_image/
    └── ...
```

> **Note:** The exact image directory structure should be preserved when downloading `data_image`, as the paths referenced by the dataset files must remain valid.

---

## Running the Pipeline

Once the required `data` and `data_image` directories are available locally:

```bash
python main.py --data-dir "C:/path/to/data" --base "C:/path/to/data_image"
```

Replace the paths with the corresponding locations on your system.

### Quick Start

1. Clone this repository.
2. Install the required dependencies.
3. Open the **Google Drive — BELT Dataset & Resources** link.
4. Download the **`data/`** and **`data_image/`** directories.
5. Preserve their directory structure.
6. Run the pipeline using `main.py`.

The remaining folders in the Google Drive are available for **optional inspection and reproducibility of previous experiments and results**.
