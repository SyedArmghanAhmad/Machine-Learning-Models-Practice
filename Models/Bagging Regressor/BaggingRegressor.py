import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

# Sample data: Calorie Intake, Workout Duration (in minutes)
clients_data = np.array([
    [2000, 60], [2500, 45], [1800, 75], [2200, 50], [2100, 62],
    [2300, 70], [1900, 55], [2000, 65], [2400, 80], [2100, 40],
    [2200, 50], [2500, 60], [1800, 65], [2300, 55], [2400, 70],
    [2000, 50], [2100, 60], [1900, 60], [2500, 55], [2300, 60],
    [2200, 65], [1800, 75], [2100, 70], [2000, 45], [2200, 80],
    [2300, 85], [2400, 40], [2500, 60], [2000, 55], [2100, 50],
    [1900, 65], [2400, 50], [2300, 75], [2500, 80], [2200, 60],
    [2100, 70], [2400, 55], [2300, 60], [2200, 80], [2500, 45],
    [1800, 60], [2100, 50], [1900, 60], [2000, 65], [2200, 55]
])

# Corresponding weight loss values (in kg)
weight_loss = np.array([
    3, 2, 4, 3, 3.5, 4.5, 3.7, 4.2, 4.8, 3.9, 4.1, 4.0, 3.6, 4.3, 3.8,
    3.2, 3.7, 3.9, 2.8, 4.2, 3.3, 4.1, 4.4, 3.6, 3.9, 3.0, 4.0, 3.7, 3.8,
    4.1, 4.2, 3.5, 3.8, 3.6, 4.4, 4.0, 3.7, 3.5, 3.9, 4.0, 3.2, 3.8, 3.5, 4.7, 2.6
])

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(clients_data, weight_loss, test_size=0.25, random_state=42)

# Base estimator: DecisionTreeRegressor
base_estimator = DecisionTreeRegressor(random_state=42)

# Bagging Regressor Model
model = BaggingRegressor(estimator=base_estimator, random_state=42)

# Hyperparameter grid for tuning
param_grid = {
    'estimator__max_depth': [3, 5, 7, 10],  # Max depth of the base decision trees
    'n_estimators': [10, 50, 100],  # Number of base estimators (trees) in Bagging
    'max_samples': [0.5, 0.7, 1.0],  # Fraction of samples to train each base estimator
    'max_features': [0.5, 0.7, 1.0]  # Fraction of features to train each base estimator
}

# Apply GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Best parameters found by GridSearchCV
best_params = grid_search.best_params_

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred = best_model.predict(X_test)

# Calculate Mean Squared Error
mse_after_tuning = mean_squared_error(y_test, y_pred)

# MSE before hyperparameter tuning (with the base model)
model.fit(X_train, y_train)
y_pred_before_tuning = model.predict(X_test)
mse_before_tuning = mean_squared_error(y_test, y_pred_before_tuning)

# Display the best parameters from GridSearchCV
print("\nBest Parameters from GridSearchCV:")
print("="*50)
print(f"{'estimator__max_depth':<25}{'max_features':<15}{'max_samples':<15}{'n_estimators'}")
print("="*50)
best_params = grid_search.best_params_
print(f"{best_params['estimator__max_depth']:<25}{best_params['max_features']:<15}{best_params['max_samples']:<15}{best_params['n_estimators']}")
print("="*50)

# Display the true vs predicted weight loss in a clean, aligned format
print("\nTrue vs Predicted Weight Loss (kg):")
print("="*50)
print(f"{'True Weight Loss (kg)':<25}{'Predicted Weight Loss (kg)'}")
print("="*50)

# Loop through the values and print in a consistent format
for true_val, pred_val in zip(y_test, y_pred):
    print(f"{true_val:<25.2f}{pred_val:.2f}")

print("="*50)

# Mean Squared Error calculation (Before and After tuning)
print(f"Mean Squared Error before tuning: {mse_before_tuning:.2f}")
print(f"Mean Squared Error after tuning: {mse_after_tuning:.2f}")
print(f"Improvement in MSE: {mse_before_tuning - mse_after_tuning:.2f}")



# Save the results in a CSV file
import pandas as pd
results_df = pd.DataFrame({'True Weight Loss': y_test, 'Predicted Weight Loss': y_pred})
results_df.to_csv(r"E:\Machine Learning\Models\Bagging Regressor\output\results.csv", index=False)

# Plotting a single decision tree from the best model
plt.figure(figsize=(12, 8))
plt.title('Base Decision Tree from Bagging (General View)')
plot_tree(best_model.estimators_[0], filled=True, rounded=True, feature_names=["Calorie Intake", "Workout Duration"])
plt.tight_layout()
plt.savefig(r"E:\Machine Learning\Models\Bagging Regressor\output\Base_Decision_Tree_from_Bagging.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()

# Scatter plot: True vs. Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', edgecolors='black', alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)  # Ideal line
plt.title('True vs Predicted Weight Loss')
plt.xlabel('True Weight Loss (kg)', fontsize=12)
plt.ylabel('Predicted Weight Loss (kg)', fontsize=12)
plt.grid(True)
plt.tight_layout()

# Saving the plot
plt.savefig(r"E:\Machine Learning\Models\Bagging Regressor\output\True_vs_Predicted_Weight_Loss.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()
