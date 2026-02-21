# comparison.py
"""
Random Forest Classifier is and ensemble learning method that combines miltiple decision
trees to imporove classification accuracy and reduce overfitting. Each tree is trained
on a random subset of the data and features, and the final prediction is made by majority
vote of all the trees. I chose Random Forest because it has a cool name, and because it
can capture complex non-linear relationships that a simple from-scratch logistic regression
might miss. It often performs very well on structured datasets like the Wisconsin Breast 
Cancer dataset."""

import torch
from sklearn.ensemble import RandomForestClassifier
from binary_classification import load_data, train, predict, accuracy

# Load normalized data from binary_classification.py
X_train, X_test, y_train, y_test, feature_names = load_data()

# ==============================
# Part 1: From-Scratch Model
# ==============================
# Train using your implemented model
w, b, losses = train(X_train, y_train, alpha=0.01, n_epochs=100, verbose=False)
y_pred_scratch = predict(X_test, w, b)
acc_scratch = accuracy(y_test, y_pred_scratch)
print(f"From-scratch model test accuracy: {acc_scratch:.4f}")

# ==============================
# Part 2: Random Forest Model
# ==============================
# Convert torch tensors to numpy for sklearn
X_train_np = X_train.numpy()
X_test_np = X_test.numpy()
y_train_np = y_train.numpy()
y_test_np = y_test.numpy()

# Initialize and train Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_np, y_train_np)

# Predict and compute accuracy
y_pred_rf = rf_model.predict(X_test_np)
acc_rf = (y_pred_rf == y_test_np).mean()
print(f"Random Forest test accuracy: {acc_rf:.4f}")

# ==============================
# Comparison Comment
# ==============================
"""
In this comparison, the Random Forest model achieved slightly lower accuracy than 
the from-scratch linear model. This is, perhaps, expected because the Wisconsin 
Breast Cancer dataset is relatively simple and may be well-suited to a linear decision
boundary. However, the Random Forest's performance is still quite good, and it has
the advantage of being able to capture non-linear relationships and interactions between
features, whereas our linear model is limited to a single linear decision boundary."""