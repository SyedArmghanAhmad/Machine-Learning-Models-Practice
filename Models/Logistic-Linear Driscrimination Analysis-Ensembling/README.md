# Ensemble Model for Multiclass Classification using Logistic Regression and Linear Discriminant Analysis

This project demonstrates how to use an ensemble method to classify data into multiple classes using **Logistic Regression (One-vs-Rest)** and **Linear Discriminant Analysis (LDA)** models, combined through a **Voting Classifier**. The dataset used is a synthetically generated dataset with four classes, and the performance of individual models is compared.

## Problem Overview

The goal of this project is to:

1. Generate a synthetic dataset with multiple classes.
2. Apply Logistic Regression (One-vs-Rest) and Linear Discriminant Analysis (LDA) models to classify the data.
3. Combine the models into an ensemble using the Voting Classifier to improve classification accuracy.
4. Visualize and compare the decision boundaries of the models.

## Dataset Generation

The dataset is created using synthetic data with Gaussian distributions for four classes:

1. Class 0: Center at (0, 0)
2. Class 1: Center at (3, 0)
3. Class 2: Center at (6, 0)
4. Class 3: Center at (9, 0) (added after training the initial models)

Each class is generated using random numbers drawn from a Gaussian distribution, and the features are 2D values representing coordinates in a 2-dimensional space.

## Models Used

### 1. **Logistic Regression (One-vs-Rest)**

- **Description:** Logistic Regression is used with a One-vs-Rest (OvR) strategy, which transforms the multiclass problem into multiple binary classification problems. For each class, a separate binary classifier is trained.
- **Implementation:** `LogisticRegression` wrapped in `OneVsRestClassifier`.

### 2. **Linear Discriminant Analysis (LDA)**

- **Description:** LDA is used to find a linear combination of features that best separate the classes. It assumes that data from each class are normally distributed with the same covariance matrix.
- **Implementation:** `LinearDiscriminantAnalysis`.

### 3. **Voting Classifier (Ensemble)**

- **Description:** A Voting Classifier is used to combine the predictions from Logistic Regression and LDA. The ensemble method uses a hard voting strategy, where each model's prediction is given equal weight, and the class with the most votes is selected.
- **Implementation:** `VotingClassifier` with Logistic Regression and LDA as base learners.

## Model Training and Evaluation

### Steps

1. **Data Generation:**
   - Synthetic data is generated with 40 samples per class, with 4 total classes.
   - The data is concatenated and labels are adjusted accordingly.

2. **Training the Models:**
   - The Logistic Regression and LDA models are trained individually on the original dataset (3 classes).
   - After adding a new class, both models are retrained with the new data (4 classes).

3. **Ensemble Model:**
   - The ensemble model is trained using both the Logistic Regression and LDA models.

4. **Prediction and Accuracy Calculation:**
   - Each model (Logistic Regression, LDA, and the Ensemble) predicts the classes on the dataset.
   - Accuracy is calculated for each model to compare their performance.

### Results

- The accuracy for each model is printed in the console.
- The best performing model is identified based on accuracy.

## Visualizing Decision Boundaries

- The decision boundaries for all three models (Logistic Regression, LDA, and Ensemble) are plotted in a 2D space. These visualizations help to understand how each model separates the different classes.
- The decision boundary plots are saved as high-resolution images in the `output` folder for further analysis.

## Code Explanation

1. **Data Generation:** The function `generate_data(n_samples)` creates synthetic data for multiple classes with centers at different points.
2. **Model Creation and Training:**
   - `lr_model = OneVsRestClassifier(LogisticRegression()).fit(X, y)` trains a One-vs-Rest Logistic Regression model.
   - `lda_model = LinearDiscriminantAnalysis().fit(X, y)` trains a Linear Discriminant Analysis model.
   - The new class is generated, and both models are retrained to include this additional class.
3. **Ensemble Model:**
   - `ensemble_model = VotingClassifier(estimators=[('lr', lr_model_4class), ('lda', lda_model_4class)], voting='hard')` creates an ensemble model.
4. **Model Prediction and Accuracy Calculation:**
   - Predictions are made using each model, and the accuracy of each model is calculated using `accuracy_score`.
5. **Visualization:**
   - The decision boundaries of each model are visualized using `matplotlib` and saved as images.

## Conclusion

This project demonstrates the use of ensemble learning by combining Logistic Regression and Linear Discriminant Analysis to classify synthetic data into multiple classes. The decision boundaries of each model are visualized, and their performance is compared using accuracy scores. By combining models, the ensemble method provides an improved classification performance.

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- scikit-learn

To install the required libraries, you can use the following command:

```bash
pip install numpy matplotlib scikit-learn


### Key Points:
1. **Dataset Generation**: Creating synthetic data for classification.
2. **Model Training**: Training Logistic Regression and LDA models.
3. **Ensemble Method**: Combining the models with Voting Classifier for better performance.
4. **Evaluation**: Comparing the performance of each model using accuracy.
5. **Visualization**: Visualizing decision boundaries for better model understanding.


