# Python function used in the project

from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support,
                             confusion_matrix,
                             ConfusionMatrixDisplay)
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


def calculate_metrics(y_true, y_pred):
    """
    Args:
    ----
    y_pred - Pred labels
    y_true - True labels
    
    Return:
    ----
    Return a dictionary with the model metrics of Accuracy, Precision, Recall and F1-score
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")

    return {"accuracy":accuracy,
           "precision":precision,
           "f1-score":f1,
           "recall":recall}


def get_model_preds(model, dataset, y_true):
    """
    Recibes a model and a validation or test dataset and return the predictions of the model
    
    Args:
    ----
    model (Keras Model) - Model used to make predictions
    dataset (Dataset) - val/test dataset to evaluate the model
    y_true (array) - True labels to calculate the model results
    
    Returns:
    --------
    pred_probs (array) - Prediction probabilities of the model on the dataset
    preds (array) - Predicted labels of the model on the dataset
    results (dict) - The model results (accuracy, precision, recall, f1-score)
    """
    pred_probs = model.predict(dataset) # Prediction probabilities
    preds = tf.squeeze(tf.round(pred_probs)) # Predicted labels
    results = calculate_metrics(y_true, preds) # Results dictionary
    return pred_probs, preds, results


def results_table(names, *args):
    """
    Model results updatable table funcion. Receives model results as arguments
    in dictionary form and returns a dataframe to display and compare them
    
    Args:
    -----
    model_names (list): List with the names of the models to display in the table (Same amount than *args)
    *args (dict): As many dictionaries of results as models to evaluate (Same amount than model_names)
    """
    results_dict = {}
    for index, arg in enumerate(args):
        results_dict[names[index]] = arg

    return pd.DataFrame(results_dict).T


def make_cm(y_true, y_pred, title):
    """
    Function to make a simple custom confusion matrix
    
    Args:
    -----
    y_true (array) - True labels
    y_pred (array) - Predicted labels
    title (str) - Title for the confusion matrix
    """
    # Text font
    font = {'family' : 'sans-serif',
    'weight' : 'normal',
    'size'   : 14}
    
    cm = confusion_matrix(y_true, y_pred) # Confusion matrix
    cmp = ConfusionMatrixDisplay(cm) # CM to display
    fig, ax = plt.subplots(figsize=(7,7)) # Fig size
    plt.rc('font', **font) # Font sizes
    cmp.plot(ax=ax, cmap=plt.cm.Blues) # Plot CM
    cmp.ax_.set_title(title, fontsize=24) # Setting title

