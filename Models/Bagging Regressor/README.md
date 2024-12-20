
# Bagging Regressor with Decision Tree Base Estimator

This project demonstrates the use of **Bagging Regressor** with a **Decision Tree Regressor** as the base estimator to predict weight loss based on calorie intake and workout duration. The model is optimized using **GridSearchCV** for hyperparameter tuning to improve the prediction accuracy. Below is a detailed explanation of the code, the concepts used, and the reasoning behind every part of the implementation.

## Table of Contents

1. [Introduction](#introduction)
2. [Key Concepts](#key-concepts)
3. [Data](#data)
4. [Model Explanation](#model-explanation)
5. [Model Evaluation](#model-evaluation)
6. [Plots and Results](#plots-and-results)
7. [Conclusion](#conclusion)

---

## Introduction

This project uses **Bagging Regressor** for weight loss prediction based on two input features: **calorie intake** and **workout duration**. The model's performance is first evaluated with default parameters, followed by optimization using **GridSearchCV** to improve the accuracy. The results are stored in a CSV file and visualized with plots.

---

## Key Concepts

1. **Bagging Regressor**:
   - **Bagging** stands for Bootstrap Aggregating. It is an ensemble technique where multiple models (base learners) are trained on different random subsets of the data. Bagging helps reduce overfitting and improves the generalization of models.
   - The Bagging Regressor in this case uses a **Decision Tree Regressor** as the base estimator, meaning the ensemble model consists of several decision trees.

2. **Decision Tree Regressor**:
   - A decision tree splits the data based on feature values into smaller sets, aiming to predict a target value (in this case, weight loss) based on these splits. Decision Trees are simple yet powerful algorithms, but they can overfit. Bagging helps mitigate this issue.

3. **GridSearchCV**:
   - **GridSearchCV** is a method for tuning hyperparameters. It performs an exhaustive search over a specified parameter grid, evaluating the model with each combination of parameters. The best combination is chosen based on cross-validation results, which ensures better generalization on unseen data.

4. **Mean Squared Error (MSE)**:
   - MSE is used to evaluate the model's performance by comparing the predicted values with the actual values. Lower MSE indicates better performance.

---

## Data

The data consists of two features:

- **Calorie Intake** (in kcal)
- **Workout Duration** (in minutes)

Each record corresponds to a client and their **weight loss** in kilograms after following a certain workout and calorie intake regime.

Here is an excerpt of the data:

```python
[2000, 60], [2500, 45], [1800, 75], ...
```

The target variable is **weight_loss**, representing the amount of weight lost by each client.

---

## Model Explanation

1. **Data Splitting**:
   The data is split into training and testing sets using `train_test_split`. The training set is used to train the model, while the testing set is used to evaluate its performance.

   ```python
   X_train, X_test, y_train, y_test = train_test_split(clients_data, weight_loss, test_size=0.25, random_state=42)
   ```

   - `test_size=0.25`: This means 25% of the data is used for testing, and 75% is used for training.
   - `random_state=42`: Ensures that the split is reproducible.

2. **Bagging Regressor with Decision Tree**:
   A **Decision Tree Regressor** is used as the base model for Bagging. This tree model learns the relationship between the features and the target (weight loss). The ensemble of trees (bagging) helps reduce variance and improve the model’s stability.

   ```python
   base_estimator = DecisionTreeRegressor(random_state=42)
   model = BaggingRegressor(estimator=base_estimator, random_state=42)
   ```

3. **Hyperparameter Tuning with GridSearchCV**:
   We define a hyperparameter grid to search through and optimize the model:

   ```python
   param_grid = {
       'estimator__max_depth': [3, 5, 7, 10],
       'n_estimators': [10, 50, 100],
       'max_samples': [0.5, 0.7, 1.0],
       'max_features': [0.5, 0.7, 1.0]
   }
   ```

   - `estimator__max_depth`: The maximum depth of the individual decision trees. A lower depth prevents overfitting.
   - `n_estimators`: The number of decision trees in the ensemble. More trees generally improve performance but increase computation time.
   - `max_samples`: The fraction of samples used to train each tree. Using 1.0 means all samples are used for each tree, and lower values introduce more randomness.
   - `max_features`: The fraction of features considered when splitting a node. Lower values make each tree more diverse.

4. **Fitting the Model**:
   After optimizing the hyperparameters with `GridSearchCV`, the model is fit to the training data, and predictions are made on the test data.

   ```python
   grid_search.fit(X_train, y_train)
   best_model = grid_search.best_estimator_
   y_pred = best_model.predict(X_test)
   ```

---

## Model Evaluation

1. **Mean Squared Error (MSE)** is calculated before and after tuning to evaluate the improvement in the model's performance:

   ```python
   mse_after_tuning = mean_squared_error(y_test, y_pred)
   mse_before_tuning = mean_squared_error(y_test, y_pred_before_tuning)
   ```

2. **Results are printed** to show the best parameters and a comparison between true and predicted weight loss:

   ```python
   print(f"Mean Squared Error before tuning: {mse_before_tuning:.2f}")
   print(f"Mean Squared Error after tuning: {mse_after_tuning:.2f}")
   ```

---

## Plots and Results

1. **Decision Tree Visualization**:
   We visualize the first decision tree from the bagging model using `plot_tree`:

   ```python
   plot_tree(best_model.estimators_[0], filled=True, rounded=True)
   ```

   This shows how the first decision tree in the ensemble splits the data based on features (calorie intake and workout duration).

2. **True vs Predicted Plot**:
   A scatter plot is created to compare the true weight loss values and the predicted values. The ideal line (red dashed) is added for reference:

   ```python
   plt.scatter(y_test, y_pred, color='blue', edgecolors='black', alpha=0.7)
   plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
   ```

   This plot helps visualize how well the model predicts the weight loss.

---

## Conclusion

In this project:

- We built an ensemble model using **Bagging** with **Decision Trees** to predict weight loss based on calorie intake and workout duration.
- We optimized the model using **GridSearchCV**, which improved the model's accuracy by tuning hyperparameters.
- The model’s performance is evaluated using **Mean Squared Error (MSE)**, and visualizations are provided to compare the true vs predicted values and understand the decision tree model's behavior.

The model can be further improved by incorporating more features, increasing the dataset size, or exploring different base estimators.

---

## Saving Results

The results are saved to a CSV file for further analysis:

```python
results_df.to_csv(r"E:\Machine Learning\Models\Bagging Regressor\output\results.csv", index=False)
```

Additionally, the plots and decision trees are saved as high-resolution images for better presentation and reporting.

---
