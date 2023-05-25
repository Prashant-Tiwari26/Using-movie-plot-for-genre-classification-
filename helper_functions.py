from sklearn.metrics import *
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def performance(model, x_test, y_test):
    """
    Calculates and displays the performance metrics of a trained model.

    Parameters:
    -----------
    model : object
        The trained machine learning model.

    x_test : array-like of shape (n_samples, n_features)
        The input test data.

    y_test : array-like of shape (n_samples,)
        The target test data.

    Returns:
    --------
    None

    Prints:
    -------
    Model Performance:
        Classification report containing precision, recall, F1-score, and support for each class.
    Accuracy:
        The accuracy of the model on the test data.
    Confusion Matrix:
        A plot of the confusion matrix, showing the true and predicted labels for the test data.

    Example:
    --------
    >>> performance(model, x_test, y_test)
    """

    preds = model.predict(x_test)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    print("                 Model Performance")
    print(report)
    print(f"Accuracy = {round(accuracy*100, 2)}%")
    matrix = confusion_matrix(y_test, preds)
    matrix_disp = ConfusionMatrixDisplay(matrix)
    matrix_disp.plot(cmap='Blues')
    plt.show()