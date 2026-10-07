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

- `0_preprocessing.py`: load data, prepare features, define transformations.
- `1_modeling.py`: placeholder for the model pipeline.
- `2_evaluation.py`: placeholder for metrics.

