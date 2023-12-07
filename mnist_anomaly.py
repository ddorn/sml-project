# %% All the imports
from dataclasses import dataclass
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

import plotly.graph_objects as go
import plotly.express as px

from utils import *

# %% Load the a simple dataset consisting of 8x8 images of digits
X, y = datasets.load_digits(return_X_y=True)
# X, y = datasets.load_iris(return_X_y=True)
# X, y = datasets.load_wine(return_X_y=True)

# Show what the dataset looks like
show(X, y, title="Some digits")

# %%

NORMAL_CLASS = 0
TRAIN_ANOMALY_SIZE = 0.1

# Split the dataset between the normal and anomaly
is_normal = (y == NORMAL_CLASS)
is_normal = (y < 2)

X_train, X_test, y_train, y_test = train_test_split_anomaly(
    X, is_normal,
    test_normal_size=0.2,
    train_anomaly_size=TRAIN_ANOMALY_SIZE,
    balanced_test_set=False,
    random_state=42,
    print_report=True,
)

show(X_train, y_train, title="Some digits from the training set")
show(X_test, y_test, title="Some digits from the testing set")

# %% Normalize the features based on mean and std of train set.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

show(X_train_scaled, y_train, title="Some digits from the training set (normalized)")
show(X_test_scaled, y_test, title="Some digits from the testing set (normalized)")

# %% Train a MLP model

hiddens = np.arange(1, 64)
aurocs = []
thresholds = []

for hidden in hiddens:

    mlp = MLPRegressor(hidden_layer_sizes=(hidden,),
                        activation="logistic",
                        max_iter=1000,
                        )
    mlp.fit(X_train_scaled, X_train_scaled)

    X_test_through_mlp = mlp.predict(X_test_scaled)

    distance = np.mean((X_test_scaled - X_test_through_mlp) ** 2, axis=1)
    probas = np.exp(-distance)

    # compute best threshold
    fpr, tpr, thres = roc_curve(y_test, probas)
    best_threshold = thres[np.argmax(tpr - fpr)]
    thresholds.append(best_threshold)

    auroc = roc_auc_score(y_test, probas)
    aurocs.append(auroc)


# Plot
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=hiddens,
    y=aurocs,
    # Plot the threshold too
    text=thresholds,
    hovertemplate="Number of hidden units: %{x}<br>AUROC: %{y:.2f}<br>Threshold: %{text:.2f}",
    mode="lines",
    name="MLP"
))
fig.update_layout(title="AUROC for MLP")
fig.update_xaxes(title_text="Number of hidden units")
fig.update_yaxes(title_text="AUROC")
fig.show()

# %% Train a MLP model

hidden = 6
mlp = MLPRegressor(hidden_layer_sizes=(hidden,),
                    activation="logistic",
                    max_iter=1000)
mlp.fit(X_train_scaled, X_train_scaled)
X_test_through_mlp = mlp.predict(X_test_scaled)

distance = np.mean((X_test_scaled - X_test_through_mlp) ** 2, axis=1)
probas = np.exp(-distance)


# %% Visualize the reconstruction for the test set
random_idx = np.random.randint(min(len(X_test), len(X_train)), size=4)
inputs = X_test_scaled[random_idx]
outputs = mlp.predict(inputs)

# Unscaled inputs and outputs
inputs = scaler.inverse_transform(inputs)
outputs = scaler.inverse_transform(outputs)

# Put inputs and outputs side by side
inputs = np.concatenate([inputs, outputs], axis=0)

show(inputs, rows=2,
      zmin=0, zmax=16,
      title="Some digits from the testing set")


# %% ROC curve
ROC = 1
if ROC:
    fpr, tpr, thresholds = roc_curve(y_test, probas, drop_intermediate=False)
else:
    fpr, tpr, thresholds = precision_recall_curve(y_test, probas)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=fpr,
    y=tpr,
    # Plot the threshold too
    text=thresholds,
    hovertemplate="False positive rate: %{x:.2f}<br>True positive rate: %{y:.2f}<br>Threshold: %{text:.2f}"
        if ROC else "Precision: %{x:.2f}<br>Recall: %{y:.2f}<br>Threshold: %{text:.2f}",
    mode="lines",
    name="MLP"
))
# fig.update_layout(title="ROC curve for MLP with 6 hidden units, normal digits 0 and 1")
# Rmove margins
FOR_SLIDES = dict(
    margin=dict(l=0, r=0, b=0, t=0),
    height=400, width=400,
    font=dict(size=15),
)
fig.update_layout(**FOR_SLIDES)
if ROC:
    fig.update_xaxes(title_text="False positive rate")
    fig.update_yaxes(title_text="True positive rate")
else:
    fig.update_xaxes(title_text="Precision")
    fig.update_yaxes(title_text="Recall")
fig.show()

fig.write_image("images/roc_nn_6.png", scale=4)

# %% Show tpr for different thresholds

with_FPR = 1
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=thresholds.clip(0, 1),
    y=tpr,
    mode="lines",
    name="TPR" if ROC else "Precision"
))
if with_FPR:
    # Add the fpr
    fig.add_trace(go.Scatter(
        x=thresholds.clip(0, 1),
        y=fpr,
        mode="lines",
        name="FPR" if ROC else "Recall"
    ))
# Add a line for the best threshold for 90% recall
target_recall = 0.9
target_color = "mediumorchid"
fig.add_shape(
    type="line",
    x0=0,
    y0=target_recall,
    x1=1,
    y1=target_recall,
    line=dict(
        color=target_color,
        width=2,
        dash="dot",
    ),
)
fig.add_annotation(
    x=0.18,
    y=target_recall,
    text=f"Target recall: {target_recall:.0%}",
    showarrow=False,
    yshift=-10,
    font=dict(size=15, color=target_color),
)
if ROC:
    min_threshold = thresholds[np.argmax(tpr >= target_recall)]
else:
    min_threshold = thresholds[np.argmin(tpr >= target_recall)]
fig.add_shape(
    type="line",
    x0=min_threshold,
    y0=0,
    x1=min_threshold,
    y1=1,
    line=dict(
        color=target_color,
        width=2,
        dash="dot",
    ),
)
# No margins
kwargs = FOR_SLIDES.copy()
if with_FPR:
    kwargs["width"] = 500
fig.update_layout(**kwargs)
fig.update_xaxes(title_text="Threshold", range=[0, 1.015])
if ROC:
    fig.update_yaxes(title_text="True positive rate" if not with_FPR else "Rate")
else:
    fig.update_yaxes(title_text="Precision" if not with_FPR else "Rate")
fig.show()

if with_FPR:
    fig.write_image("images/tpr-with-threshold-and-fpr.png", scale=4)
else:
    fig.write_image("images/tpr-with-threshold.png", scale=4)



# %% Datasets

@dataclass
class Dataset:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray

def all_datasets(
    test_normal_size=0.2,
    train_anomaly_size=0,
    balanced_test_set=False,
    random_state=42,
    max_size=5_000,
):
    def mk_dataset(name, X, normal):
        return Dataset(
            name,
            *train_test_split_anomaly(
                X[:max_size], normal[:max_size],
                test_normal_size=test_normal_size,
                train_anomaly_size=train_anomaly_size,
                balanced_test_set=balanced_test_set,
                random_state=random_state,
                print_report=True,
            )
        )

    digits_X, digits_y = datasets.load_digits(return_X_y=True)

    yield mk_dataset("Digits 0", digits_X, digits_y == 0)
    yield mk_dataset("Digits 1", digits_X, digits_y == 1)
    yield mk_dataset("Digits 0 and 1", digits_X, digits_y < 2)

    # Credit card fraud dataset
    creditcard_X, creditcard_y = datasets.fetch_openml(
        data_id=42175, return_X_y=True, as_frame=False
    )
    creditcard_y = creditcard_y.astype(int)
    yield mk_dataset("Credit card fraud", creditcard_X, creditcard_y == 0)

    # NSL-KDD dataset
    nslkdd_X, nslkdd_y = datasets.fetch_openml(
        data_id=42193, return_X_y=True, as_frame=False
    )
    nslkdd_y = nslkdd_y.astype(int)
    yield mk_dataset("NSL-KDD", nslkdd_X, nslkdd_y == 0)

    # Cifar10 dataset
    cifar10_X, cifar10_y = datasets.fetch_openml(
        data_id=40926, return_X_y=True, as_frame=False
    )
    cifar10_y = cifar10_y.astype(int)
    yield mk_dataset("Cifar10 (automobile)", cifar10_X, cifar10_y == 1)
    yield mk_dataset("Cifar10 (automobile and airplane)", cifar10_X, cifar10_y < 2)

all_datasets = list(all_datasets())

# %% Classifiers

def all_classifiers():
    for hidden in [4, 8, 16, 32]:
        yield f"MLP {hidden} hidden", MLPRegressor(hidden_layer_sizes=(hidden,), activation="logistic", max_iter=10_000)
    for kernel in ["linear", "poly", "rbf", "sigmoid"]:
        yield f"SVM {kernel}", OneClassSVM(kernel=kernel, nu=0.3)

# %% Train all the classifiers on all the datasets

results = []
for dataset in all_datasets:
    for name, clf in all_classifiers():
        print(f"Training {name} on {dataset.name}")
        clf.fit(dataset.X_train, dataset.X_train)
        results.append((dataset.name, name, clf))

# %% Test all the classifiers on all the datasets, computing the best threshold for each,
# using both F1 and sufficient recall of 90% as metrics.

target_recall = 0.9
test_results = []

for dataset_name, name, clf in results:
    print(f"Testing {name} on {dataset_name}")

    dataset = next(d for d in all_datasets if d.name == dataset_name)

    if isinstance(clf, MLPRegressor):
        X_test_through_clf = clf.predict(dataset.X_test)
        distance = np.mean((dataset.X_test - X_test_through_clf) ** 2, axis=1)
        # probas = np.exp(-distance)
        probas = distance
    elif isinstance(clf, OneClassSVM):
        probas = -clf.score_samples(dataset.X_test)
    else:
        raise Exception(f"Unknown classifier: {clf}")

    # compute best threshold
    prec, rec, thres = precision_recall_curve(-dataset.y_test, probas, drop_intermediate=False)
    f1 = 2 * prec * rec / (prec + rec)
    # Replace nan with 0
    f1 = np.nan_to_num(f1)
    best_threshold = thres[np.argmax(f1)]
    # print(best_threshold, f1)

    # compute threshold for 90% recall
    min_threshold = thres[np.argmin(rec >= (target_recall))]
    # print(min_threshold)

    # compute f1 score in both cases
    y_pred = (probas >= best_threshold).astype(int)
    f1 = f1_score(-dataset.y_test, y_pred * 2 - 1)
    y_pred = (probas >= min_threshold).astype(int)
    f1_90 = f1_score(-dataset.y_test, y_pred * 2 - 1)
    # print(f"f1: {f1:.4f}, f1_90: {f1_90:.4f}")
    # results.append((dataset.name, name, f1, f1_90, clf))
    test_results.append((dataset.name, name, f1, f1_90))

# %% Save the results
import pickle
with open("results.pkl", "wb") as f:
    pickle.dump(results, f)

# %% Load the results
import pickle
with open("results.pkl", "rb") as f:
    results = pickle.load(f)

# %% Plot the results
# x: which dataset
# y: f1_90 / f1 for each classifier
# color: classifier

RELATIVE_F1 = 0

from collections import defaultdict


per_classifier = defaultdict(lambda: ([], [], []))
for dataset_name, classifier_name, f1, f1_90 in test_results:
    if "Credit" in dataset_name:
        continue
    if dataset_name == "Cifar10 (automobile)":
        dataset_name = "Cifar10 (auto)"
    if dataset_name == "Cifar10 (automobile and airplane)":
        dataset_name = "Cifar10 (auto&plane)"
    classifier_name = classifier_name.replace("MLP ", "NN ")


    per_classifier[classifier_name][0].append(dataset_name)
    per_classifier[classifier_name][1].append(f1)
    per_classifier[classifier_name][2].append(f1_90)


fig = go.Figure()
for classifier_name, (dataset_names, f1s, f1s_90) in per_classifier.items():
    f1s = np.array(f1s)
    f1s_90 = np.array(f1s_90)
    if classifier_name.startswith("NN"):
        legendgroup = "NN"
        marker = "cross"
    else:
        legendgroup = "SVM"
        marker = "x"

    fig.add_trace(go.Scatter(
        x=dataset_names,
        y=f1s_90 / f1s if RELATIVE_F1 else f1s_90,
        # Use % for the y axis
        mode="markers",
        name=classifier_name,
        legendgroup=legendgroup,
        marker=dict(symbol=marker, size=10, opacity=0.8),
    ))
# fig.update_layout(title=f"F1 score for {target_recall:.0%} recall / F1 score")
# fig.update_xaxes(title_text="Dataset")
fig.update_yaxes(title_text="F1 / Best possible F1" if RELATIVE_F1 else f"F1 score for {target_recall:.0%} recall",
                 tickformat=".0%" if RELATIVE_F1 else ".2f")
kwargs = FOR_SLIDES.copy()
kwargs["width"] = 600
fig.update_layout(**kwargs)
fig.show()

if RELATIVE_F1:
    fig.write_image("images/big-plot.png", scale=3)
else:
    fig.write_image("images/big-plot-f1s.png", scale=3)


# %%

# Train the OSVM model
osvm = OneClassSVM(nu=0.3)
osvm.fit(X_train_scaled)

# Predict the labels for the testing set
y_pred = osvm.predict(X_test_scaled)

print(classification_report(y_test, y_pred))
plot_confusion(y_test, y_pred)


# %% Show some of the predictions
# True label in brackets. (1 = normal, -1 = anomaly)
show(X_test, y_pred, y_test,
     format="Pred {} ({:.0f})",
     title="Some predictions")




# %% Show the influence of the nu parameter on the confusion matrix
nus = np.linspace(0.025, 0.8, 40)

# Calculate the confusion matrix for each nu value
cms = []
f1s = []
for nu in nus:
    osvm = OneClassSVM(nu=nu)
    osvm.fit(X_train_scaled)
    y_pred = osvm.predict(X_test_scaled)
    cms.append(confusion_matrix(y_test, y_pred, normalize="true"))
    f1s.append(f1_score(y_test, y_pred))

# Stack the all the confusion matrices into a 3D array
cms = np.array(cms)
f1s = np.array(f1s)

# Plot the confusion matrices
plot_confusion_3d(cms, nus, x_title="nu",
        title="Confusion matrix for different nu values in OSVM"
                    f'<br><span style="font-size: 14px">Normal digit {NORMAL_CLASS}; train anomaly size: {TRAIN_ANOMALY_SIZE:.0%}</span>')

# Plot the f1 score
fig = px.line(x=nus, y=f1s)
fig.update_layout(title="F1 score for different nu values in OSVM"
                  f'<br><span style="font-size: 14px">Normal digit {NORMAL_CLASS}; train anomaly size: {TRAIN_ANOMALY_SIZE:.0%}</span>')
fig.update_xaxes(title_text="nu")
fig.update_yaxes(title_text="F1 score")
fig.show()




# %% Show confusion for different kernels and nu
nus = np.linspace(0.025, 0.9, 20)
kernels = ["linear", "poly", "rbf", "sigmoid"]

# Calculate the confusion matrix for each kernel
preds = collect_y_preds(X_train_scaled, X_test_scaled, kernel=kernels, nu=nus)
cms = np.apply_along_axis(
    lambda y_preds: confusion_matrix(y_test, y_preds, normalize="true"),
    axis=-1, arr=preds
)
f1s = np.apply_along_axis(
    lambda y_preds: f1_score(y_test, y_preds),
    axis=-1, arr=preds
)

# Plot the confusion matrices
plot_confusion_3d(cms, nus, x_title="nu", lines=kernels,
        title="Confusion matrix for different kernels in OSVM"
                    f'<br><span style="font-size: 14px">{NORMAL_CLASS=}; {TRAIN_ANOMALY_SIZE=:.0%}</span>')

# Plot the f1 score
fig = go.Figure()
for kernel, f1 in zip(kernels, f1s):
    fig.add_trace(go.Scatter(
        x=nus,
        y=f1,
        mode="lines",
        name=f"{kernel}"
    ))
fig.update_layout(title="F1 score for different kernels in OSVM"
                  f'<br><span style="font-size: 14px">{NORMAL_CLASS=}; {TRAIN_ANOMALY_SIZE=:.0%}</span>')
fig.update_xaxes(title_text="nu")
fig.update_yaxes(title_text="F1 score")
fig.show()

"""Note: the f1 score is very sensitive to class imbalance.
In practice we see two different plots when test set is balanced or not.
Probably worth talking about it.
https://en.wikipedia.org/wiki/F-score#Dependence_of_the_F-score_on_class_imbalance"""


# %%
# Plot precision-recall curve
# cm: (kernel, nu, class, pred)

# Calculate the precision and recall for each kernel
true_positives = cms[..., 1, 1]
false_positives = cms[..., 0, 1]
false_negatives = cms[..., 1, 0]

precisions = true_positives / (true_positives + false_positives)
recalls = true_positives / (true_positives + false_negatives)

fig = go.Figure()
for kernel, precision, recall in zip(kernels, precisions, recalls):
    fig.add_trace(go.Scatter(
        x=recall,
        y=precision,
        mode="lines",
        name=f"{kernel}"
    ))
fig.update_layout(title="Precision-recall curve for different kernels in OSVM"
                    f'<br><span style="font-size: 14px">{NORMAL_CLASS=}; {TRAIN_ANOMALY_SIZE=:.0%}</span>')
fig.update_xaxes(title_text="Recall")
fig.update_yaxes(title_text="Precision")
fig.show()

print("Note: this is not really a precision-recall curve, "
      "since the nu parameter is not the decision threshold. ")


# %% A precission-recall curve for a given nu&kernel

nu = 0.1
kernel = "linear"
# kernel = "poly"
degree = 3
# kernel = "rbf"

osvm = OneClassSVM(nu=nu, kernel=kernel, degree=degree)
osvm.fit(X_train_scaled)
y_pred = osvm.score_samples(X_test_scaled)

precision, recall, _ = precision_recall_curve(y_test, y_pred)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=recall,
    y=precision,
    mode="lines",
    name=f"{kernel}"
))
fig.update_layout(title="Precision-recall curve for different kernels in OSVM"
                    f'<br><span style="font-size: 14px">{NORMAL_CLASS=}; {TRAIN_ANOMALY_SIZE=:.0%}</span>')
fig.update_xaxes(title_text="Recall")
fig.update_yaxes(title_text="Precision")
fig.show()





# %% Show the f1 score for each digit, for different nu values
nus = np.linspace(0.025, 0.8, 40)
digits = np.arange(10)

# Calculate the f1 score for each digit and nu value
f1s = []

for digit in digits:
    X_train, X_test, y_train, y_test = train_test_split_anomaly(
        X, y == digit,
        test_normal_size=0.1,
        train_anomaly_size=TRAIN_ANOMALY_SIZE,
        random_state=42,
        print_report=False
    )

    # Normalize the features based on mean and std of train set.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the OSVM model for each nu value
    f1s_digit = []
    for nu in nus:
        osvm = OneClassSVM(nu=nu)
        osvm.fit(X_train_scaled)
        y_pred = osvm.predict(X_test_scaled)
        f1s_digit.append(f1_score(y_test, y_pred))

    f1s.append(f1s_digit)

# Plot the f1 score for each digit
fig = go.Figure()
for digit, f1s_digit in zip(digits, f1s):
    fig.add_trace(go.Scatter(
        x=nus,
        y=f1s_digit,
        mode="lines",
        name=f"Digit {digit}"
    ))
fig.update_layout(title="F1 score for different nu values in OSVM"
                  f'<br><span style="font-size: 14px">Train anomaly size: {TRAIN_ANOMALY_SIZE:.0%}</span>')
fig.update_xaxes(title_text="nu")
fig.update_yaxes(title_text="F1 score")
fig.show()

# %%
