from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso
from utils import *

pio.templates.default = "simple_white"
pio.renderers.default = "browser"

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    X = np.random.uniform(low=-1.2,high=2,size=n_samples)
    eps = np.random.normal(0, noise, n_samples)
    f= lambda x: (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    y_noiseless = f(X)
    y = y_noiseless + eps
    x_train, y_train, x_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), (2/3))

    x_train_graph = np.array(x_train).flatten().tolist()
    y_train_graph = np.array(y_train).flatten().tolist()
    x_test_graph = np.array(x_test).flatten().tolist()
    y_test_graph = np.array(y_test).flatten().tolist()

    colors = np.array(["greenyellow", "darkred", "blue"])
    go.Figure(
        [go.Scatter(x=x_train_graph, y=y_train_graph, mode="markers",name="$train$",
                    marker=dict(color=colors[0])),
         go.Scatter(x=X, y=y_noiseless, mode="markers",
                    name="$data without noise$",
                    marker=dict(color=colors[1])),
         go.Scatter(x=x_test_graph, y=y_test_graph, mode="markers",name="$test$",
                    marker=dict(color=colors[2]))]).update_layout(
        title=rf"$\textbf{{Dataset with noise = {noise}, num of samples = {n_samples}}}$",
        font=dict(size=24), xaxis_title = "$\\text{x}$",
        yaxis_title="$\\text{f(x)}$", height=800).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    average_train = []
    validation_errors = []
    for k in range(11):
        estimator = PolynomialFitting(k)
        train_loss, val_loss = cross_validate(estimator,x_train.to_numpy(),y_train.to_numpy(),mean_square_error)
        average_train.append(train_loss)
        validation_errors.append(val_loss)

    k_star = int(np.argmin(validation_errors))

    k = list(range(11))
    go.Figure(
        [go.Scatter(x=k, y=average_train, mode="markers + lines",name="$average train$",
                    marker=dict(color=colors[0])),
         go.Scatter(x=k, y=validation_errors, mode="markers + lines",name="$validation errors$",
                    marker=dict(color=colors[1]))])\
        .update_layout(
        title=rf"$\textbf{{(2) Train and Validation Loss as Function of K, With Number of Samples = {n_samples} and Noise = {noise}}}$",
        font=dict(size=24),
        xaxis_title = "$\\text{K = polynomial degree}$",
        yaxis_title = "$\\text{losses}$").show()


    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_estimator = PolynomialFitting(k_star).fit(x_train.to_numpy().flatten(), y_train.to_numpy().flatten())
    test_error = best_estimator.loss(x_test.to_numpy().flatten(), y_test.to_numpy().flatten())
    print(f"\nFor data with {n_samples} samples, and noise = {noise}:\nBest k is: {k_star}, test error is: {test_error}.")



def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    colors = np.array(["greenyellow", "darkred"])

    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    indices = np.random.choice(X.shape[0], n_samples)
    x_train = X[indices]
    y_train = y[indices]
    x_test = np.delete(X, indices, axis=0)
    y_test = np.delete(y, indices, axis=0)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    k_space = np.linspace(0, 3, n_evaluations)
    ridge_average_train = []
    ridge_validation_errors = []
    lasso_average_train = []
    lasso_validation_errors = []
    for k in k_space:
        ridge_est = RidgeRegression(k)
        lasso_est = Lasso(k)
        ridge_train_loss, ridge_val_loss = cross_validate(ridge_est, x_train,
                                              y_train,
                                              mean_square_error)
        ridge_average_train.append(ridge_train_loss)
        ridge_validation_errors.append(ridge_val_loss)
        lasso_train_loss, lasso_val_loss = cross_validate(lasso_est, x_train,
                                              y_train,
                                              mean_square_error)
        lasso_average_train.append(lasso_train_loss)
        lasso_validation_errors.append(lasso_val_loss)

    go.Figure([go.Scatter(x=k_space, y=ridge_average_train,
                          marker=dict(color=colors[0]),
                          mode='markers+lines', name="$Training Error$"),
               go.Scatter(x=k_space, y=ridge_validation_errors,
                          marker=dict(color=colors[1]),
                          mode='markers+lines', name="$Ridge Errors$"), ],
              layout=go.Layout(title=r"Ridge Errors",
                               yaxis_title="errors",
                               xaxis_title="lamda"))

    go.Figure([go.Scatter(x=k_space, y=lasso_average_train,
                          marker=dict(color=colors[0]),
                          mode='markers+lines', name="$Training Error$"),
               go.Scatter(x=k_space, y=lasso_validation_errors,
                          marker=dict(color=colors[1]),
                          mode='markers+lines', name="$Lasso Errors$"), ],
              layout=go.Layout(title=r"Lasso Errors",
                               yaxis_title="errors",
                               xaxis_title="lamda"))


    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    best_for_ridge = k_space[int(np.argmin(ridge_validation_errors))]
    best_for_lasso = k_space[int(np.argmin(lasso_validation_errors))]

    best_ridge_est = RidgeRegression(best_for_ridge).fit(x_train, y_train)
    best_lasso_est = Lasso(best_for_lasso).fit(x_train, y_train)
    linear_est = LinearRegression().fit(x_train, y_train)

    ridge_err = mean_square_error(best_ridge_est.predict(x_test),y_test)
    lasso_err = mean_square_error(best_lasso_est.predict(x_test), y_test)
    linear_err = mean_square_error(linear_est.predict(x_test),y_test)

    print(f"Ridge MSE = {ridge_err} with regularization parameter = {best_for_ridge}")
    print(f"Lasso MSE = {lasso_err} with regularization parameter = {best_for_lasso}")
    print(f"Least Squares Regression = {linear_err}")

if __name__ == '__main__':
    np.random.seed(0)
    # select_polynomial_degree()
    # select_polynomial_degree(noise=0)
    # select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()