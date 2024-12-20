# Logistic Regression Model: Book Pages vs Like/Dislike

## Problem Statement

The goal of this project is to predict whether a book will be liked or disliked based on its number of pages. For instance, if Jenny is considering a book with a specific page count, the model should help determine whether she is likely to like it or not. Additionally, we want to evaluate the model's performance by calculating its accuracy.

## Approach and Solution

This project uses Logistic Regression, a classification algorithm, to model the relationship between the number of pages in a book and the likelihood of liking or disliking it. Here's how the problem was solved step-by-step:

### 1. Data Preparation

- The dataset includes:
  - **Features (pages):** Number of pages in the book.
  - **Labels (likes):** Binary outcome (1 = like, 0 = dislike).
- The data was reshaped into a 2D array as required by the scikit-learn library.

### 2. Model Training

- A Logistic Regression model was created using the `LogisticRegression` class from scikit-learn.
- The model was trained on the provided dataset using the `fit()` method.

### 3. Predictions

- To predict Jenny's preference for a book with 260 pages, the model's `predict()` method was used.
- The likelihood of liking the book was visualized using the model's `predict_proba()` method.

### 4. Model Evaluation

- The model's accuracy was calculated using the `accuracy_score()` function from scikit-learn by comparing the predicted labels with the true labels.

### 5. Visualization

- A scatter plot visualizes the dataset points (pages vs. like/dislike).
- The logistic regression curve shows the probability of liking a book as a function of its page count.
- Vertical and horizontal lines indicate the decision boundary and the prediction point for 260 pages.

## Results

- **Prediction:** Jenny will either "Like" or "Not Like" a book with 260 pages based on the model's output.
- **Model Accuracy:** The calculated accuracy score indicates how well the model fits the training data.

## Code Description

### Required Libraries

- `numpy`: For handling numerical data.
- `matplotlib`: For creating plots.
- `sklearn.linear_model.LogisticRegression`: For building the Logistic Regression model.
- `sklearn.metrics.accuracy_score`: For evaluating model performance.

### Key Functions

1. **`LogisticRegression.fit()`**: Trains the model using the dataset.
2. **`LogisticRegression.predict()`**: Predicts the label (like/dislike) for a given number of pages.
3. **`LogisticRegression.predict_proba()`**: Computes the probabilities for each class (like or dislike).
4. **`accuracy_score()`**: Calculates the accuracy of the model by comparing predicted and actual labels.

### Visualization

The output plot includes:

- **Data Points:** Representing the original dataset.
- **Logistic Curve:** Showing the likelihood of liking a book based on its page count.
- **Decision Boundary:** The threshold (0.5) used for classification.
- **Prediction Marker:** Highlighting the page count (260 pages) for Jenny's prediction.

## Outputs

- **Prediction:** Jenny will "Like" or "Not Like" the book.
- **Accuracy:** Displayed as a percentage, showing the model's fit.
- **Visualization:** Saved as an image file (`BookPages vs Like-Dislike-General.jpg`) in the specified output folder.

## Usage

1. Run the script with the provided dataset.
2. Modify the `predicted_book_pages` variable to test predictions for other page counts.
3. Check the generated plot and printed outputs for predictions and model accuracy.

## Example Output

- **Prediction:** "Jenny will Not Like a book of 260 pages."
- **Accuracy:** "Model Accuracy: 88.89%"
- **Plot:** Saved in the output folder as a visual representation of the model and dataset.

---
This project demonstrates how Logistic Regression can be applied to a binary classification problem with a simple and interpretable model.
