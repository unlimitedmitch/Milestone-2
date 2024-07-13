# Principal Component Analysis (PCA) on Cancer Dataset

This repository contains a Python script that performs Principal Component Analysis (PCA) on the Breast Cancer Wisconsin (Diagnostic) dataset from `sklearn.datasets`. The script also includes an optional step to implement logistic regression for prediction using the reduced dataset.

## Prerequisites

Before running the code, ensure that you have the following dependencies installed:

- Python (version 3.6 or higher)
- NumPy
- Scikit-learn
- Matplotlib

You can install the required packages using pip:

```
pip install numpy scikit-learn matplotlib
```

## Usage

1. Clone or download this repository to your local machine.
2. Navigate to the repository directory.
3. Run the Python script:

```
python pca_cancer_dataset.py
```

## Code Explanation

The script performs the following tasks:

1. **PCA Implementation**:
   - Loads the Breast Cancer Wisconsin (Diagnostic) dataset from `sklearn.datasets`.
   - Creates a PCA instance and fits the data.
   - Prints the explained variance ratio for each principal component.
   - Plots the cumulative explained variance ratio to help determine the number of principal components to retain.

2. **Dimensionality Reduction**:
   - Creates a new PCA instance with `n_components=2` to reduce the dataset to 2 principal components.
   - Fits and transforms the data using the new PCA instance.
   - Prints the shape of the reduced dataset.

3. **Logistic Regression for Prediction**:
   - Splits the reduced dataset into training and testing sets.
   - Creates a logistic regression model.
   - Trains the model on the training data.
   - Makes predictions on the test data.
   - Calculates the accuracy score of the predictions.

## Output

The script will output the following:

- Explained variance ratio for each principal component.
- A plot showing the cumulative explained variance ratio versus the number of principal components.
- The shape of the reduced dataset (2 principal components).
- The accuracy score of the logistic regression model (if the optional step is included).

## Notes

- The script uses the Breast Cancer Wisconsin (Diagnostic) dataset from `sklearn.datasets`. However, you can modify the code to use other cancer datasets available in `sklearn.datasets` by changing the `load_breast_cancer()` function call.
- The step for logistic regression is included for demonstration purposes. You can remove or modify this step based on your requirements.
- The random state for the train-test split is set to 42 for reproducibility. You can change this value or remove it if desired.