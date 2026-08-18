# ============================================================
# LINEAR REGRESSION
# Closed-Form Normal Equation vs Gradient Descent
#
# Images are stored in ONE separate folder
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv(
    r"C:/Users/navya/OneDrive/Desktop/Machine L/Placement_prediction/dataset/final_preprocess_M2.csv"
)

# Extract all columns except last column
X = data.iloc[:, :-1].values

# Extract last column as target
y = data.iloc[:, -1].values


# ============================================================
# 2. CHECK DATA FOR NaN AND INFINITY
# ============================================================

print("\n========== DATA CHECK ==========")

print("X shape:", X.shape)
print("y shape:", y.shape)

print("NaN values in X:", np.isnan(X).sum())
print("NaN values in y:", np.isnan(y).sum())

print("Infinity values in X:", np.isinf(X).sum())
print("Infinity values in y:", np.isinf(y).sum())


# Remove rows containing NaN or infinity
valid_rows = (
    np.isfinite(X).all(axis=1)
    & np.isfinite(y)
)

X = X[valid_rows]
y = y[valid_rows]

print("\nAfter cleaning:")
print("X shape:", X.shape)
print("y shape:", y.shape)


# ============================================================
# 3. CREATE IMAGE OUTPUT FOLDER
# ============================================================

IMAGE_FOLDER = (
    "C:/Users/navya/OneDrive/Desktop/Machine L/"
    "Placement_prediction/outputs/"
    "Linear_Regression_CFNE_GD_Compare_M2"
)

os.makedirs(IMAGE_FOLDER, exist_ok=True)

print("\nImage output folder:")
print(IMAGE_FOLDER)


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 5. FEATURE SCALING
#    Important for Gradient Descent
# ============================================================

x_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)

X_test_scaled = x_scaler.transform(X_test)


# ============================================================
# 6. TARGET SCALING
#    Important for stable Gradient Descent
# ============================================================

y_scaler = StandardScaler()

y_train_scaled = y_scaler.fit_transform(
    y_train.reshape(-1, 1)
).ravel()

y_test_scaled = y_scaler.transform(
    y_test.reshape(-1, 1)
).ravel()


# ============================================================
# 7. CLOSED FORM SOLUTION
#    NORMAL EQUATION
# ============================================================

# Add bias column

X_train_bias = np.c_[
    np.ones((X_train.shape[0], 1)),
    X_train
]

X_test_bias = np.c_[
    np.ones((X_test.shape[0], 1)),
    X_test
]


# ============================================================
# Normal Equation
#
# theta = (X^T X)^(-1) X^T y
#
# pinv() is safer than inv()
# ============================================================

theta = np.linalg.pinv(
    X_train_bias.T.dot(X_train_bias)
).dot(
    X_train_bias.T
).dot(
    y_train
)


# ============================================================
# Normal Equation Prediction
# ============================================================

pred_normal = X_test_bias.dot(theta)


# ============================================================
# Normal Equation Metrics
# ============================================================

mse_normal = mean_squared_error(
    y_test,
    pred_normal
)

r2_normal = r2_score(
    y_test,
    pred_normal
)


print("\n------ Closed Form Normal Equation ------")

print("Coefficients:")
print(theta)

print("MSE:", mse_normal)

print("R2 Score:", r2_normal)


# ============================================================
# 8. GRADIENT DESCENT
# ============================================================

# Add bias column to scaled X

X_train_gd = np.c_[
    np.ones((X_train_scaled.shape[0], 1)),
    X_train_scaled
]

X_test_gd = np.c_[
    np.ones((X_test_scaled.shape[0], 1)),
    X_test_scaled
]


# Number of training examples

m = len(y_train_scaled)


# Initialize parameters

theta_gd = np.zeros(
    X_train_gd.shape[1]
)


# ============================================================
# LEARNING RATE
# ============================================================

learning_rate = 0.01

epochs = 1000


# ============================================================
# 9. STORE LOSS FOR EACH EPOCH
# ============================================================

loss_history = []


# ============================================================
# 10. GRADIENT DESCENT ITERATIONS
# ============================================================

for epoch in range(epochs):

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predictions_scaled = X_train_gd.dot(theta_gd)


    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    errors = predictions_scaled - y_train_scaled


    # --------------------------------------------------------
    # Gradient
    # --------------------------------------------------------

    gradients = (
        (2 / m)
        * X_train_gd.T.dot(errors)
    )


    # --------------------------------------------------------
    # Update parameters
    # --------------------------------------------------------

    theta_gd -= (
        learning_rate * gradients
    )


    # --------------------------------------------------------
    # Calculate training MSE
    # --------------------------------------------------------

    loss = np.mean(
        errors ** 2
    )


    # --------------------------------------------------------
    # Check for NaN / Infinity
    # --------------------------------------------------------

    if not np.isfinite(loss):

        print("\nERROR: Gradient Descent diverged!")
        print("Epoch:", epoch)
        print("Loss:", loss)

        raise ValueError(
            "Gradient Descent produced NaN or Infinity. "
            "Check learning rate or input data."
        )


    loss_history.append(loss)


# ============================================================
# 11. GRADIENT DESCENT PREDICTION
# ============================================================

# Prediction in scaled target space

pred_gd_scaled = X_test_gd.dot(
    theta_gd
)


# ============================================================
# Convert predictions back to original target scale
# ============================================================

pred_gd = y_scaler.inverse_transform(
    pred_gd_scaled.reshape(-1, 1)
).ravel()


# ============================================================
# FINAL NaN CHECK
# ============================================================

print("\n========== PREDICTION CHECK ==========")

print(
    "NaN in Normal Equation predictions:",
    np.isnan(pred_normal).sum()
)

print(
    "NaN in Gradient Descent predictions:",
    np.isnan(pred_gd).sum()
)

print(
    "Infinity in Normal Equation predictions:",
    np.isinf(pred_normal).sum()
)

print(
    "Infinity in Gradient Descent predictions:",
    np.isinf(pred_gd).sum()
)


# ============================================================
# 12. GRADIENT DESCENT METRICS
# ============================================================

mse_gd = mean_squared_error(
    y_test,
    pred_gd
)

r2_gd = r2_score(
    y_test,
    pred_gd
)


print("\n------ Gradient Descent ------")

print("Coefficients:")
print(theta_gd)

print("MSE:", mse_gd)

print("R2 Score:", r2_gd)


# ============================================================
# 13. COMPARISON
# ============================================================

print("\n=========== Comparison ===========")

print("\nNormal Equation")

print("MSE =", mse_normal)

print("R2 =", r2_normal)


print("\nGradient Descent")

print("MSE =", mse_gd)

print("R2 =", r2_gd)


# ============================================================
# IMAGE 1
# ACTUAL VS PREDICTED VALUES
# ============================================================

plt.figure(figsize=(8, 6))


plt.scatter(
    y_test,
    pred_normal,
    alpha=0.5,
    label="Normal Equation"
)


plt.scatter(
    y_test,
    pred_gd,
    alpha=0.5,
    label="Gradient Descent"
)


# Perfect prediction line

minimum = min(
    y_test.min(),
    pred_normal.min(),
    pred_gd.min()
)

maximum = max(
    y_test.max(),
    pred_normal.max(),
    pred_gd.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)


plt.xlabel("Actual Values")

plt.ylabel("Predicted Values")

plt.title(
    "Actual vs Predicted Values"
)

plt.legend()

plt.grid(True)

plt.tight_layout()


image1 = os.path.join(
    IMAGE_FOLDER,
    "actual_vs_predicted.png"
)


plt.savefig(
    image1,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nImage saved:")
print(image1)


# ============================================================
# IMAGE 2
# RESIDUAL COMPARISON
# ============================================================

# Residual = Actual - Predicted

normal_residuals = (
    y_test - pred_normal
)


gd_residuals = (
    y_test - pred_gd
)


plt.figure(figsize=(9, 6))


plt.scatter(
    pred_normal,
    normal_residuals,
    alpha=0.5,
    label="Normal Equation"
)


plt.scatter(
    pred_gd,
    gd_residuals,
    alpha=0.5,
    label="Gradient Descent"
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted Values"
)


plt.ylabel(
    "Residuals"
)


plt.title(
    "Residual Comparison"
)


plt.legend()

plt.grid(True)

plt.tight_layout()


image2 = os.path.join(
    IMAGE_FOLDER,
    "residual_comparison.png"
)


plt.savefig(
    image2,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Image saved:")
print(image2)


# ============================================================
# IMAGE 3
# GRADIENT DESCENT LOSS CURVE
# ============================================================

plt.figure(figsize=(9, 6))


plt.plot(
    range(1, epochs + 1),
    loss_history
)


plt.xlabel(
    "Epoch"
)


plt.ylabel(
    "Mean Squared Error"
)


plt.title(
    "Gradient Descent Convergence"
)


plt.grid(True)

plt.tight_layout()


image3 = os.path.join(
    IMAGE_FOLDER,
    "gradient_descent_loss.png"
)


plt.savefig(
    image3,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("Image saved:")
print(image3)


# ============================================================
# 14. SAVE IMAGE INFORMATION
# ============================================================

image_info = pd.DataFrame({

    "Image": [
        "actual_vs_predicted.png",
        "residual_comparison.png",
        "gradient_descent_loss.png"
    ],

    "Description": [
        "Actual values versus predictions from both methods",
        "Residual comparison between Normal Equation and Gradient Descent",
        "MSE loss across Gradient Descent epochs"
    ]

})


image_info.to_csv(
    os.path.join(
        IMAGE_FOLDER,
        "image_information.csv"
    ),
    index=False
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("\n==========================================")

print("PROCESS COMPLETED SUCCESSFULLY")

print("==========================================")


print("\nAll images are stored in ONE folder:")

print(IMAGE_FOLDER)


print("\nGenerated images:")

print("1. actual_vs_predicted.png")

print("2. residual_comparison.png")

print("3. gradient_descent_loss.png")


print("\nOriginal dataset was NOT modified.")