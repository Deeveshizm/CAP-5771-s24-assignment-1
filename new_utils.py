"""
   Use to create your own functions for reuse 
   across the assignment

   Inside part_1_template_solution.py, 
  
     import new_utils
  
    or
  
     import new_utils as nu
"""
import pickle
import numpy as np
from typing import Type, Dict
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    cross_validate,
    KFold,
)

def check_flt_and_std(X):
    is_fl = np.issubdtype(X.dtype, np.floating)
    is_std =  np.all((X >= 0) & (X <= 1))
    return is_fl, is_std
def scale_data(X):
    is_flt, is_std = check_flt_and_std(X)
    print(f'''Data is of type Floating point: {is_flt}
Data is b/w 0 and 1: {is_std}''')
    if is_flt==False | is_std==False:
        print('Fixing data ...')
        if not is_flt:
          x = X.astype(float)
        if not is_std:
          means = np.mean(X, axis=0)  
          stds = np.std(X, axis=0)
          x = (X-means)/stds
    else:
       x = X
    return x

def intcheck(y):
   if not np.issubdtype(y.dtype, np.integer):
      print('y is not integer')
      print('fixing that...')
      y = y.astype(int)
      return y
   else:
      print('y is of integer type')
      return y
    

def classifier_with_cv(
    Xtrain: NDArray[np.floating],
    ytrain: NDArray[np.int32],
    clf: BaseEstimator,
    cv: KFold = KFold,
):
    """
    Train a simple classifier using k-vold cross-validation.

    Parameters:
        - X: Features dataset.
        - y: Labels.
        - cv_class: The cross-validation class to use.
        - estimator_class: The training classifier class to use.
        - n_splits: Number of splits for cross-validation.
        - print_results: Whether to print the results.

    Returns:
        - A dictionary with mean and std of accuracy and fit time.
    """
    scores = cross_validate(clf, Xtrain, ytrain, cv=cv)
    return scores


def cv_result_dict(cv_dict: Dict):
    score = {}
    for key, array in cv_dict.items():
        if key=='fit_time':
           score[f'mean_{key}'] = array.mean()
           score[f'std_{key}'] = array.std()
        if key=='test_score':
           score[f'mean_accuracy'] = array.mean()
           score[f'std_accuracy'] = array.std()   
    return score


        ### accuracy calculated out of confusion matrix 
def calculate_accuracy(confusion_matrix):
    """
    Calculate accuracy from a confusion matrix.

    Parameters:
        confusion_matrix: 2D numpy array representing the confusion matrix.

    Returns:
        Accuracy computed from the confusion matrix.
    """
    # Calculate accuracy from confusion matrix
    TP = confusion_matrix[1, 1]  # True Positives
    TN = confusion_matrix[0, 0]  # True Negatives
    total_samples = confusion_matrix.sum()  # Total Samples

    accuracy = (TP + TN) / total_samples
    return accuracy

def compute_accuracies(confusion_matrix_train_orig, confusion_matrix_test_orig, confusion_matrix_train_best, confusion_matrix_test_best):
    """
    Compute accuracies for each confusion matrix.

    Parameters:
        confusion_matrix_train_orig: Confusion matrix for training data with original estimator.
        confusion_matrix_test_orig: Confusion matrix for testing data with original estimator.
        confusion_matrix_train_best: Confusion matrix for training data with best estimator.
        confusion_matrix_test_best: Confusion matrix for testing data with best estimator.

    Returns:
        A dictionary containing accuracies for each confusion matrix.
    """
    accuracies = {}

    # Calculate accuracy for each confusion matrix
    accuracies["accuracy_orig_full_training"] = calculate_accuracy(confusion_matrix_train_orig)
    accuracies["accuracy_orig_full_testing"] = calculate_accuracy(confusion_matrix_test_orig)
    accuracies["accuracy_best_full_training"] = calculate_accuracy(confusion_matrix_train_best)
    accuracies["accuracy_best_full_testing"] = calculate_accuracy(confusion_matrix_test_best)

    return accuracies