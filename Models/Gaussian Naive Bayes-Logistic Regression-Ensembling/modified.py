import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

# Generate sample dataset
X, y = make_circles(n_samples=1000, noise=0.1, random_state=42)

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Naive Bayes Classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Train Logistic Regression Classifier
lr_model = LogisticRegression(C=1.0)
lr_model.fit(X_train, y_train)

# Make Predictions
nb_predictions = nb_model.predict(X_test)
lr_predictions = lr_model.predict(X_test)

# Calculate individual accuracies
nb_accuracy = accuracy_score(y_test, nb_predictions)
lr_accuracy = accuracy_score(y_test, lr_predictions)

# Ensemble: Combine predictions using majority voting
ensemble_predictions = []
for nb_pred, lr_pred in zip(nb_predictions, lr_predictions):
    # Majority voting
    ensemble_predictions.append(1 if nb_pred + lr_pred > 1 else 0)

ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)

# Print accuracies
print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
print(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")

# Visualization
h = 0.01
x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict probabilities for decision boundaries
nb_Z = nb_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
lr_Z = lr_model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
nb_probs = nb_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
lr_probs = lr_model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
ensemble_probs = (nb_probs + lr_probs) / 2
ensemble_Z = (ensemble_probs > 0.5).astype(int).reshape(xx.shape)

# Plot decision boundaries
plt.figure(figsize=(18, 6))

# Naive Bayes
plt.subplot(1, 3, 1)
plt.title("Naive Bayes Decision Boundary")
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', marker='o', alpha=0.5)
plt.contourf(xx, yy, nb_Z, alpha=0.3, cmap='viridis')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')

# Logistic Regression
plt.subplot(1, 3, 2)
plt.title("Logistic Regression Decision Boundary")
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', marker='o', alpha=0.5)
plt.contourf(xx, yy, lr_Z, alpha=0.3, cmap='viridis')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')

# Ensemble Model
plt.subplot(1, 3, 3)
plt.title("Ensemble Model Decision Boundary")
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', marker='o', alpha=0.5)
plt.contourf(xx, yy, ensemble_Z, alpha=0.3, cmap='viridis')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')

plt.tight_layout()
plt.show()

# Print accuracies
print(f"Naive Bayes Accuracy: {nb_accuracy * 100:.2f}%")
print(f"Logistic Regression Accuracy: {lr_accuracy * 100:.2f}%")
print(f"Ensemble Model Accuracy: {ensemble_accuracy * 100:.2f}%")