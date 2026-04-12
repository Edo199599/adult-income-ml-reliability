# ML Reliability Mini-System (Adult Income)

## Overview
This is a binary classification project based on realistic tabular data.
The goal is to predict whether a person earns more than 50K per year.

The main purpose of this project is not to build the best possible model.
It is meant to show a clean and reliable machine learning workflow: proper data split, leakage-safe preprocessing, baseline comparison, cross-validation, light hyperparameter tuning, and final evaluation on a separate test set.

## Dataset
Chosen dataset: UCI Adult / Census Income.

I selected this dataset because it is a good first example of binary classification on tabular data with both numerical and categorical features.
It is simple enough to stay manageable, but realistic enough to build a proper pipeline and practice model evaluation.

## Project goals
This project is meant to show:
- correct train/test split
- preprocessing applied only on training data through a pipeline
- comparison against a dummy baseline
- cross-validation on the training set
- final evaluation on a separate test set
- one sanity test

## Project structure
- `src/prepare_data.py`: raw data loading and basic cleaning
- `src/train.py`: training pipeline and model selection
- `src/eval.py`: final evaluation and baseline comparison
- `src/config.py`: project constants
- `tests/test_sanity.py`: simple sanity test

## Scope
Version 1 includes:
- Logistic Regression
- ColumnTransformer
- OneHotEncoder and StandardScaler
- Dummy baseline
- StratifiedKFold
- light GridSearchCV
- final test report
