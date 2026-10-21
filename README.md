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
is needed. The current report was checked with Python 3.11.

To execute the entire report from a fresh kernel and save outputs:

```bash
.venv/bin/python scripts/execute_notebook.py
```

This runs the full small-grid searches on the dataset; it can take several minutes.
The script checkpoints the notebook after each executed cell. After a failed run,
fix the error and start again from the first cell before treating outputs as final.

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

If another environment is active, use `.venv/bin/python -m jupytext` instead.

## 5. Study protocol and checks

The report is a retrospective comparison, not a validated booking-time system.
It retains the original grouped 60/20/20 split and four model families. All CV
and tuning share three group-isolated folds. ROC AUC selects ranking models;
the validation F1 threshold is only a classification reference. Contact-budget
tables show recall versus workload without assuming a hotel precision target.
The additional time-ordered check is exploratory and does not change the main model.

- `src/hotel_cancellation/0_preprocessing.py`: feature preparation, common group
  keys, outer split, grouped CV and fitted transformations.
- `src/hotel_cancellation/1_modeling.py`: baseline pipelines, parameter summaries
  and the RF complexity comparison.
- `src/hotel_cancellation/2_evaluation.py`: metrics, group-bootstrap intervals,
  contact workloads and diagnostic plots.
- `src/hotel_cancellation/plot_style.py`: shared preprocessing palette, saturated
  colours, transparent backgrounds, typography and axes styling for all figures.

The Training section includes readable search grids, an RF depth/leaf-size
comparison and train-only CV scores. Evaluation lists selected configurations
alongside CV and separate validation ROC AUC. Baseline threshold metrics are
explicitly labelled at 0.5.

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The notebook includes the figures, references and result tables needed for review.
Local `reports/` exports and `models/` artifacts are ignored by Git. Include `src/`
with the notebook when submitting; do not include the raw dataset.

## 6. Copy figures into slides

All embedded plots have a transparent canvas. Each displayed figure is also
exported to `reports/figures/` as a transparent PNG (220 dpi) and SVG. Copy the
embedded image or insert the exported file into your slides. Dark labels and
saturated colours are designed for light notebook and slide backgrounds. Colour inside a heatmap, tree node or
decision region represents chart content and remains visible.
