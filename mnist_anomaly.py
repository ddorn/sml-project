# %% All the imports
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import plotly.graph_objects as go
import plotly.express as px

from utils import *

# %% Load the a simple dataset consisting of 8x8 images of digits
X, y = datasets.load_digits(return_X_y=True)

# Show what the dataset looks like
show(X, y, title="Some digits")

# %%

NORMAL_CLASS = 0
TRAIN_ANOMALY_SIZE = 0.1

# Split the dataset between the normal and anomaly
is_normal = (y == NORMAL_CLASS)

X_train, X_test, y_train, y_test = train_test_split_anomaly(
    X, is_normal,
    test_normal_size=0.1, train_anomaly_size=TRAIN_ANOMALY_SIZE,
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

# %%

# Train the OSVM model
osvm = OneClassSVM(nu=0.05)
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
nus = np.linspace(0.025, 0.8, 20)
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

print("""Warning: the f1 score is very sensitive to class imbalance.
Currently the test dataset is not balanced at all.
https://en.wikipedia.org/wiki/F-score#Dependence_of_the_F-score_on_class_imbalance""")





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
