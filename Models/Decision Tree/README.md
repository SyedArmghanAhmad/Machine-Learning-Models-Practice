# Decision Tree Regression: Study Hours vs Test Scores

This project demonstrates how a **Decision Tree Regressor** can be used to predict test scores based on the number of hours studied. The implementation includes training a machine learning model, visualizing the decision tree, evaluating its performance using standard metrics, and providing a prediction for a new input.

---

## Problem Statement

Students often wonder how their study hours correlate with their test scores. The goal of this project is to:

1. Model the relationship between hours studied and test scores.
2. Predict the test score for a given number of study hours.
3. Evaluate the accuracy of the model's predictions.

---

## Solution

We use the **Decision Tree Regressor** from the `sklearn` library to build a regression model:

1. **Data Preparation**: Input data consists of study hours and corresponding test scores.
2. **Model Training**: The decision tree regressor is trained on the provided dataset.
3. **Prediction**: The model predicts test scores for new inputs, e.g., 5.5 hours.
4. **Evaluation**: The model's performance is measured using metrics such as:
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - R-squared Score (\(R^2\))
5. **Visualization**:
   - A graphical representation of the decision tree.
   - A scatter plot of actual vs predicted test scores, including the regression line.

---

## Features

- Visual representation of the decision tree.
- Scatter plot with legends for actual and predicted data points.
- Model evaluation metrics to assess accuracy.
- Simple and clean Python implementation.

---

## Installation

 Install the required Python packages:

   ```bash
   pip install numpy matplotlib scikit-learn
   ```

---

## Usage

1. Open the script and modify the `study_hours` and `test_scores` arrays if needed.
2. Run the script:

   ```bash
   python decision_tree_regressor.py
   ```

3. The program will:
   - Train the decision tree regressor on the provided data.
   - Predict test scores for new input (e.g., 5.5 hours of study).
   - Display evaluation metrics.
   - Visualize the decision tree and the regression results.

---

## Outputs

- **Prediction**: Predicted test score for a given number of study hours.
- **Evaluation Metrics**:
  - MAE: Measures average prediction error.
  - MSE: Penalizes larger prediction errors.
  - \(R^2\): Explains how well the model fits the data.
- **Visualizations**:
  - Decision Tree structure.
  - Study Hours vs Test Scores plot.

---

## Example Output

### Metrics

```bash
Mean Absolute Error (MAE): 0.25
Mean Squared Error (MSE): 0.25
R-squared (R2) Score: 1.00
```

### Prediction

```bash
Predicted test score for 5.5 hours of study: 87.50
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Acknowledgments

- [Scikit-learn Documentation](https://scikit-learn.org/stable/documentation.html)
- [NumPy Documentation](https://numpy.org/doc/)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
