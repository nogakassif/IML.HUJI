from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    indices = np.array_split(np.arange(X.shape[0]), cv, axis=0)
    val_loss = 0
    train_loss = 0
    for i in range(cv):
        y_train = np.delete(y,indices[i])
        x_val = (X[indices[i][0]:indices[i][-1]])
        if X.shape[1] == 1:
            x_val = x_val.flatten()
            x_train = np.delete(X, indices[i])
        else:
            x_val = x_val.reshape(indices[i][-1] - indices[i][0], X.shape[1])
            x_train = np.delete(X, indices[i], axis=0)

        y_val = y[indices[i][0]:indices[i][-1]]
        y_pred_val = estimator.fit(x_train,y_train).predict(x_val)
        y_pred_train = estimator.fit(x_train,y_train).predict(x_train)
        val_loss += scoring(y_val,y_pred_val)
        train_loss += scoring(y_train,y_pred_train)
    return (train_loss/cv), (val_loss/cv)
