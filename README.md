# Hotel cancellation prediction

Run commands from the project root.

## 1. Install dependencies

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name hotel-cancellation --display-name "Python (hotel-cancellation)"
```

## 2. Download the dataset

[Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

```bash
mkdir -p data/raw
curl -fL "https://www.kaggle.com/api/v1/datasets/download/jessemostipak/hotel-booking-demand" -o data/raw/hotel-booking-demand.zip
unzip -n data/raw/hotel-booking-demand.zip -d data/raw
```

## 3. Open the notebook

```bash
python -m jupyterlab notebook/project.ipynb
```

Run all notebook cells from top to bottom: data → training → evaluation.

Select the **Python (hotel-cancellation)** kernel. In PyCharm, use
`.venv/bin/python` as the project and notebook interpreter.
The first notebook cell adds `src` to the import path, so no package installation
is needed. Python 3.10 or newer is required.

## 4. Review and sync the notebook

`notebook/project.ipynb` is paired with `notebook/project.py` using Jupytext.
The `.py` file contains every code cell and Markdown cell (as comments), separated
by `# %%` markers. Use it for line comments in GitHub pull requests. Outputs and
plots remain in the `.ipynb` file.

After editing either file, save and close the editor, then run:

```bash
python -m jupytext --sync notebook/project.ipynb
```

Sync uses the most recently modified file for cell contents and keeps notebook
outputs where possible. Sync before switching which file you edit; editing both
independently before syncing can overwrite changes. Reopen and rerun affected
cells after code changes, because retained outputs may be outdated.

Commit both files together. When Jupytext is loaded by JupyterLab, saving the
notebook also updates its text pair; restart JupyterLab after installing it.
In PyCharm, use the sync command explicitly.

- `0_preprocessing.py`: load data, prepare features, define transformations.
- `1_modeling.py`: placeholder for the model pipeline.
- `2_evaluation.py`: placeholder for metrics.

