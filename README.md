# Heart Disease Prediction - ML Project 1

EPFL Machine Learning Course (CS-433)

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/superflash41/epfl-ml-project-1
cd epfl-ml-project-1
```

2. Create and activate a virtual environment (optional but recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install required packages:

```bash
pip install -r requirements.txt
```

4. Download the dataset:

- Place the CSV files (x_train.csv, y_train.csv, x_test.csv) in the `data/` directory

## Running the Code

1. To train the model and generate predictions:

```bash
python src/run.py
```

2. The script will:

- Preprocess the data
- Train the model
- Generate predictions
- Output performance metrics

## Project Description

This project aims to predict heart disease risk using health survey data from the Behavioral Risk Factor Surveillance System (BRFSS). We implement logistic regression and apply it to predict myocardial infarction risk based on lifestyle and health factors.
