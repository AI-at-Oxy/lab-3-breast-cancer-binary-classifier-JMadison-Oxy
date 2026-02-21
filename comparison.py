# comparison.py
"""
Random Forest Classifier is and ensemble learning method that combines miltiple decision
trees to imporove classification accuracy and reduce overfitting. Each tree is trained
on a random subset of the data and features, and the final prediction is made by majority
vote of all the trees. I chose Random Forest because it has a cool name, and because it
can capture complex non-linear relationships that a simple from-scratch logistic regression
might miss. It often performs very well on structured datasets like the Wisconsin Breast 
Cancer dataset."""

from binary_classification import load_data, MyLogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
x, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# -------------------------------
# From-scratch model (assume already implemented)
# -------------------------------
scratch_model = MyLogisticRegression(lr=0.01, epochs=1000)
scratch_model.fit(X_train, y_train)
y_pred_scratch = scratch_model.predict(X_test)
accuracy_scratch = accuracy_score(y_test, y_pred_scratch)
print(f"From-scratch model test accuracy: {accuracy_scratch:.4f}")

# -------------------------------
# Scikit-learn Random Forest model
# -------------------------------
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest test accuracy: {accuracy_rf:.4f}")

# -------------------------------
# Comparison comment
# -------------------------------
"""
In this comparison, the Random Forest classifier achieved higher test accuracy than the 
from-scratch logistic regression model. This is likely because Random Forest can capture 
non-linear interactions between features, whereas logistic regression assumes a linear decision 
boundary. Ensemble methods like Random Forest also reduce variance and overfitting, which is 
especially helpful on smaller datasets like this one."""