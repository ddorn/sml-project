# %%

from itertools import product
import numpy as np
import plotly
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

import rich
from rich.table import Table
from rich.console import Console


def show(images, *titles, rows=3, cols=4, format: str=None, **kwargs):
    """
    Show a grid of images with titles.

    Params:
        images: a 3d image array (image, height, width)
        titles: one or more arrays of titles, each of the same length as images
        rows: number of rows in the grid
        cols: number of columns in the grid
        format: format string for the titles
        kwargs: passed to Figure.update_layout, can be used to set the title etc.
    """

    if rows * cols < len(images):
        subset = np.random.choice(len(images), rows * cols)
        images = images[subset]
        titles = [ts[subset] for ts in titles]
        print(titles, subset)
    if titles is not None:
        if format:
            titles = [format.format(*t) for t in zip(*titles)]
        else:
            titles = [" ".join(map(str, t)) for t in zip(*titles)]

    fig = px.imshow(images.reshape(-1, 8, 8), facet_col=0, facet_col_wrap=cols)
    # Add titles
    for i, title in enumerate(titles):
        r = i // cols
        c = i % cols
        # Plotly indexes annotations for rows in reverse order 🤷‍♂
        index = (rows - r - 1) * cols + c
        fig.layout.annotations[index].text = title
    fig.update_layout(**kwargs)
    fig.show()


def train_test_split_anomaly(X, is_normal,
                             test_normal_size=0.4,
                             train_anomaly_size=0.0,
                             print_report=False,
                             random_state=None,
                             ):
    """
    Split the dataset into train and test sets, with the test set containing
    a mix of normal and anomaly data points.

    Params:
        X: the features
        is_normal: a boolean array indicating which data points are normal
        test_normal_size: the proportion of normal data points to put in the test set
        train_anomaly_size: the proportion of anomaly data points to put in the training set, relative to the size of the training set
        print_report: whether to print a report of the sizes of dataset split
        random_state: passed to train_test_split

    Returns:
        X_train, X_test, y_train, y_test: the normal labels are +1 and anomaly are -1.
    """
    assert 0 <= test_normal_size <= 1
    assert 0 <= train_anomaly_size <= 1

    X_normal = X[is_normal]
    X_anomaly = X[~is_normal]

    # Compute the sizes of the 4 sets (normal and anomaly × training and testing)
    n_normal = len(X_normal)
    n_anomaly = len(X_anomaly)
    n_test_normal = int(n_normal * test_normal_size)
    n_train_normal = n_normal - n_test_normal
    # train_anomaly_size * train_size = n_train_anomaly
    # train_size = n_train_normal + n_train_anomaly
    # train_anomaly_size * (n_train_normal + n_train_anomaly) = n_train_anomaly
    # train_anomaly_size * n_train_normal + train_anomaly_size * n_train_anomaly = n_train_anomaly
    # train_anomaly_size * n_train_normal = n_train_anomaly - train_anomaly_size * n_train_anomaly
    # train_anomaly_size * n_train_normal = n_train_anomaly * (1 - train_anomaly_size)
    n_train_anomaly = train_anomaly_size * n_train_normal / (1 - train_anomaly_size)
    n_train_anomaly = int(n_train_anomaly)
    n_train = n_train_normal + n_train_anomaly
    n_test_anomaly = n_anomaly - n_train_anomaly
    # n_test_anomaly = n_test_normal
    n_test = n_test_normal + n_test_anomaly

    if print_report:
        report = Table(
            "Dataset split", "Normal", "Anomaly", "Total",
            title="Dataset split",
            box=rich.box.SIMPLE,
            safe_box=True)
        report.add_row("Train", str(n_train_normal), str(n_train_anomaly), str(n_train))
        report.add_row("Test", str(n_test_normal), str(n_test_anomaly), str(n_test))
        report.add_row("Total", str(n_normal), str(n_anomaly), str(n_normal + n_anomaly))
        # console = Console(force_terminal=True)
        # console.print(report)
        rich.print(report)


    # Split the normal into training and testing sets
    X_normal_train, X_normal_test = train_test_split(X_normal, train_size=n_train_normal, random_state=random_state)

    # Split the anomaly into training and testing sets
    if n_train_anomaly == 0:
        X_anomaly_train = np.empty((0, *X.shape[1:]))
        X_anomaly_test = X_anomaly
    else:
        X_anomaly_train, X_anomaly_test = train_test_split(X_anomaly, train_size=n_train_anomaly, test_size=n_test_anomaly, random_state=random_state)

    # Combine the normal and anomaly training sets
    X_train = np.concatenate([X_normal_train, X_anomaly_train])
    y_train = np.concatenate([
        np.ones(len(X_normal_train)),
        -np.ones(len(X_anomaly_train))
    ])

    # Combine the normal and anomaly testing sets
    X_test = np.concatenate([X_normal_test, X_anomaly_test])
    y_test = np.concatenate([
        np.ones(len(X_normal_test)),
        -np.ones(len(X_anomaly_test))
    ])

    # Sanity check the sizes
    assert len(X_normal_train) == n_train_normal, f"{len(X_normal_train)} != {n_train_normal}"
    assert len(X_normal_test) == n_test_normal, f"{len(X_normal_test)} != {n_test_normal}"
    assert len(X_anomaly_train) == n_train_anomaly, f"{len(X_anomaly_train)} != {n_train_anomaly}"
    assert len(X_anomaly_test) == n_test_anomaly, f"{len(X_anomaly_test)} != {n_test_anomaly}"
    assert len(X_train) == n_train, f"{len(X_train)} != {n_train}"
    assert len(X_test) == n_test, f"{len(X_test)} != {n_test}"
    assert len(X_train) == len(y_train), f"{len(X_train)} != {len(y_train)}"
    assert len(X_test) == len(y_test), f"{len(X_test)} != {len(y_test)}"

    return X_train, X_test, y_train, y_test


def plot_confusion(y_true, y_pred, **kwargs):
    """
    Plot a confusion matrix for the given true and predicted labels.

    Params:
        y_true: the true labels
        y_pred: the predicted labels
        kwargs: passed to Figure.update_layout
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)


    text = [[
        f"{value:.0%}<br>({n=:.0f})"
          for n, value in row
        ] for row in np.stack([cm, cm_norm], axis=2)
    ]
    fig = ff.create_annotated_heatmap(
        z=cm_norm,
        x=["Predicted anomaly", "Predicted normal"],
        y=["Actual anomaly", "Actual normal"],
        annotation_text=text,
    )
    kwargs.setdefault("title", "Confusion matrix")
    fig.update_layout(**kwargs)
    fig.show()

def plot_confusion_3d(matrix, x, x_title: str = None,
                      lines: list[str] = None,
                      **plot_kwargs):
    """
    Plot the evolution of the confusion matrix over a parameter.

    Params:
        matrix: a 3d array of shape (lines?, n, 2, 2) where n is the number of values of the parameter.
            The confusion matrix should be normalized with respect to the true labels (normalise="true").
        param: the values of the parameter, of shape (n,)
        param_name: the name of the parameter, used in the axis labels
    """
    names = [
        "True anomaly", "False normal",
        "False anomaly", "True normal"
    ]

    if lines is None:
        lines = ["trace"]
        assert len(matrix.shape) == 3
        matrix = matrix[np.newaxis, ...]

    assert matrix.shape == (len(lines), len(x), 2, 2)

    # Make a 2x2 plot, for each metric
    fig = make_subplots(rows=2, cols=2, subplot_titles=names)

    if len(lines) > 1:
        colors = plotly.colors.DEFAULT_PLOTLY_COLORS.copy()
    else:
        colors = [None]

    for plot_idx, name in enumerate(names):
        row = plot_idx // 2
        col = plot_idx % 2
        for l, line in enumerate(lines):
            scatter = go.Scatter(
                x=x,
                y=matrix[l, :, row, col] * 100,
                mode="lines",
                name=line,
                legendgroup=f"group{l}",
                showlegend=plot_idx == 0 and len(lines) > 1,
                line=dict(color=colors[l % len(colors)]),
            )
            fig.add_trace(scatter, row=row + 1, col=col + 1)

    if x_title is not None:
        fig.update_xaxes(title_text=x_title, row=2)
    fig.update_yaxes(title_text="Percentage", col=1)
    fig.update_layout(**plot_kwargs)

    # Set limits for the y-axis to 0-100
    # fig.update_yaxes(range=[0, 100])

    # Remove the names of the traces from the legend
    # fig.update_layout(showlegend=False)

    fig.show()


def collect_y_preds(X_train, X_test, **kwargs_lists: list):
    """
    Train a OSVM model for each combination of argument in kwargs_lists,
    and return the predictions in a multidimensional array.

    Params:
        X_train: the training set
        X_test: the testing set
        kwargs_lists: each other keyword argument is a list of values to try and is passed to the OSVM constructor.
    Returns:
        y_preds: a multidimensional array of shape (len(kwargs_lists[0]), len(kwargs_lists[1]), ..., len(X_test))
    """

    keys = list(kwargs_lists.keys())
    values = list(kwargs_lists.values())
    all_combinations = product(*values)
    y_preds = []
    for combination in all_combinations:
        kwargs = dict(zip(keys, combination))
        osvm = OneClassSVM(**kwargs)
        osvm.fit(X_train)
        y_pred = osvm.predict(X_test)
        y_preds.append(y_pred)
    # turn the list of y_preds into a an array, and reshape it according to kwargs_lists
    shape = [len(values) for values in values]
    y_preds = np.array(y_preds).reshape(*shape, -1)
    return y_preds


# Some tests
if __name__ == "__main__":
    x = y = np.random.rand(100, 64)
    preds = collect_y_preds(x, y, nu=[0.1, 0.2, 0.3], kernel=["linear", "rbf"])
    assert preds.shape[:-1] == (3, 2)



if __name__ == "__main__":
    print("Tests passed!")