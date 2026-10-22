# Hotel cancellation prediction

Compare four classification models for hotel booking cancellations.
Results, plots and explanations are in `notebook/project.ipynb`.

## Setup

Use Python 3.11 and run commands from the project root.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --sys-prefix --name hotel-cancellation --display-name "Python (hotel-cancellation)"
```

Download [Hotel Booking Demand](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)
and place `hotel_bookings.csv` in `data/raw/`.

## Run

```bash
python -m jupyterlab notebook/project.ipynb
```

Select the `Python (hotel-cancellation)` kernel and run all cells in order.
In PyCharm, use `.venv/bin/python`. A full run takes several minutes.

To run the notebook and save outputs from the terminal:

```bash
python scripts/execute_notebook.py
```

## Files

- `notebook/project.ipynb`: report and results.
- `notebook/project.py`: paired text version for code review.
- `src/hotel_cancellation/`: preprocessing, models and evaluation.
- `reports/figures/`: exported plots.
- `models/`: saved models.

After editing either notebook file, sync the pair and commit both:

```bash
python -m jupytext --sync notebook/project.ipynb
```

Sync before switching files; avoid editing both independently.

Run tests with:

```bash
python -m unittest discover -s tests -v
```

Submit the notebook with `src/`; leave out the raw data and generated artifacts.
