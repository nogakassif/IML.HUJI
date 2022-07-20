from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test
from utils import *

pio.templates.default = "simple_white"
pio.renderers.default = "browser"


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """

    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = []
    weights = []

    def callback(class_name: GradientDescent, weight: np.ndarray,
                 value: np.ndarray):
        values.append(value)
        weights.append(weight)
        return

    return callback, values, weights


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    ##### L1 #####
    l1_losses = []
    for e in etas:
        fixed = FixedLR(e)
        l1_model = L1(init.copy())
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        GradientDescent(learning_rate=fixed, callback=callback_l1).fit(
            l1_model, None, None)
        l1_losses.append(values_l1[-1])

        # Question 1:
        plot_descent_path(L1, np.array(weights_l1),
                          f"The descent path for eta = {e} with L1 norm")

        # Question 3:
        x = np.linspace(1, len(values_l1) + 1, len(values_l1))
        go.Figure([go.Scatter(x=x, y=values_l1, mode='markers',
                              marker=dict(color="darkorchid"))]).update_layout(
            title=f"The convergence rate for eta = {e} with L1 norm",
            xaxis_title="num of iteration", yaxis_title="value of the norm"
        )

    ##### L2 #####
    l2_losses = []
    for e in etas:
        fixed = FixedLR(e)
        l2_model = L2(init.copy())
        callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
        GradientDescent(learning_rate=fixed, callback=callback_l2).fit(
            l2_model, None, None)
        l2_losses.append(values_l2[-1])

        # Question 1:
        plot_descent_path(L2, np.array(weights_l2),
                          f"The descent path for eta = {e} with L2 norm")

        # Question 3:
        go.Figure([go.Scatter(x=x, y=values_l2, mode='markers',
                              marker=dict(
                                  color="darkorchid"))]).update_layout(
            title=f"The convergence rate for eta = {e} with L2 norm",
            xaxis_title="num of iteration", yaxis_title="value of the norm"
        )

    # Question 4:
    print("min loss for L1 = ", min(l1_losses))
    print("min loss for L2 = ", min(l2_losses))


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    norms = []
    iter = []
    weights_095 = None

    for g in gammas:
        exp = ExponentialLR(eta, g)
        l1_model = L1(init.copy())
        callback_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        GradientDescent(learning_rate=exp, callback=callback_l1).fit(l1_model,
                                                                     None,
                                                                     None)
        norms.append(values_l1)
        iter.append(len(values_l1))

        if g == 0.95:
            weights_095 = np.array(weights_l1)

    # Plot algorithm's convergence for the different values of gamma
    colors = ["yellow", "greenyellow", "aquamarine", "turquoise"]
    fig = go.Figure()
    for i, g in enumerate(gammas):
        x = np.linspace(1, iter[i] + 1, iter[i])
        fig.add_traces([go.Scatter(x=x, y=norms[i], name=f"{g}",
                                   mode='markers + lines',
                                   line=dict(color=colors[i], width=0.05))])
    fig.update_layout(
        title="algorithm's convergence for the different values of gamma",
        xaxis_title="num of iteration", yaxis_title="value of the norm")
    # fig.show()

    # Question 6:
    min_norms = []
    for i in range(4):
        min_norms.append(min(norms[i]))
    print("\nThe min norm achieved using the exponential decay is ",
          min(min_norms))

    # Plot descent path for gamma=0.95
    exp = ExponentialLR(eta, 0.95)
    l2_model = L2(init.copy())
    callback_l2, values_l2, weights_l2 = get_gd_state_recorder_callback()
    GradientDescent(learning_rate=exp, callback=callback_l2).fit(l2_model,
                                                                 None, None)

    plot_descent_path(L1, weights_095, "gamma = 0.95 with L1 norm")
    plot_descent_path(L2, np.array(weights_l2), "gamma = 0.95 with L2 norm")


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def fit_logistic_regression():
    from utils import custom

    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    logistic = LogisticRegression(include_intercept=True,
                                  solver=GradientDescent(
                                      learning_rate=FixedLR(1e-4),
                                      max_iter=20000))
    logistic.fit(X_train.to_numpy(), y_train.to_numpy())

    from sklearn.metrics import roc_curve, auc
    c = [custom[0], custom[-1]]
    fpr, tpr, thresholds = roc_curve(y_train, logistic.predict_proba(
        X_train.to_numpy()))
    go.Figure(
        data=[go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                         line=dict(color="black", dash='dash'),
                         name="Random Class Assignment"),
              go.Scatter(x=fpr, y=tpr, mode='markers+lines', text=thresholds,
                         name="", showlegend=False, marker_size=5,
                         marker_color=c[1][1],
                         hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: %{x:.3f}<br>TPR: %{y:.3f}")],
        layout=go.Layout(
            title=rf"$\text{{ROC Curve Of Fitted Model - AUC}}={auc(fpr, tpr):.6f}$",
            xaxis=dict(title=r"$\text{False Positive Rate (FPR)}$"),
            yaxis=dict(title=r"$\text{True Positive Rate (TPR)}$")))

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter

    for model in ["l1", "l2"]:
        lambdas = [0.01, 0.001, 0.002, 0.005, 0.02, 0.05, 1]
        train_scores = []
        validation_scores = []
        for lam in lambdas:
            train, validation = cross_validate(
                LogisticRegression(penalty=model, lam=lam),
                                   X_train.to_numpy(),
                y_train.values.flatten(), misclassification_error, 5)
            train_scores.append(train)
            validation_scores.append(validation)

        best_lamda = int(np.argmin(np.array(validation_scores)))
        print(f"The best lambda for {model} is: {lambdas[best_lamda]}")
        logistic_regression2 = LogisticRegression(penalty=model, lam=best_lamda).fit(X_train.to_numpy(), y_train.values)
        test_error = logistic_regression2.loss(X_test.to_numpy(), y_test.values)
        print(f"The model {model} test error is: {test_error}")

if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
