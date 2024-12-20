# README: Predicting House Prices Using AdaBoost Regressor

## Problem Statement

In this project, the objective is to predict house prices based on features such as the number of rooms, the age of the house, and their interactions. The challenge lies in building a robust predictive model that can handle moderately noisy data, replicating real-world scenarios where data is not always clean or precise.

### Dataset

We use a **synthetically generated dataset** with the following attributes:

- **num_rooms**: Number of rooms in the house.
- **house_age**: Age of the house in years.
- **interaction_term**: Interaction between the number of rooms and house age.
- **price**: Target variable, representing the price of the house.

The relationship between the features and the price is modeled as:
\[
\text{price} = 50 \times \text{num\_rooms} + 0.5 \times \text{house\_age} + \text{noise}
\]
where `noise` is randomly generated to simulate real-world unpredictability.

### Challenges

1. The dataset includes moderate noise (standard deviation = 40), introducing variability that complicates predictions.
2. Features interact non-linearly, requiring advanced preprocessing and modeling techniques.

## Solution Approach

We employ a machine learning pipeline leveraging the **AdaBoost Regressor** for its capability to reduce bias and variance through ensemble learning.

### Steps in the Solution

1. **Data Generation and Preprocessing**:
   - Synthetic data is generated with 200 samples.
   - An interaction term (`num_rooms * house_age`) is added to capture feature interactions.
   - Features are scaled using `StandardScaler` to standardize the data.
   - Polynomial features of degree 2 are created to account for non-linear relationships.

2. **Train-Test Split**:
   - Data is split into training (80%) and testing (20%) sets to evaluate the model's performance.

3. **Model Training**:
   - A Decision Tree Regressor with a depth of 3 is used as the base learner.
   - AdaBoost Regressor is applied with 200 estimators and a learning rate of 0.4 to iteratively improve predictions.

4. **Model Evaluation**:
   - Predictions are made on the test set.
   - Model performance is measured using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE):
     - **Mean Squared Error (MSE)**: 1858.90
     - **Root Mean Squared Error (RMSE)**: 43.11

5. **Visualization**:
   - Scatter plots of actual vs predicted house prices illustrate the model's accuracy.

## Results

Despite the noise, the AdaBoost Regressor performs well, achieving an RMSE of 43.11, which indicates reasonably accurate predictions for the given synthetic data.

## How to Run the Code

1. Ensure Python is installed along with the required libraries:

   ```bash
   pip install numpy pandas matplotlib scikit-learn
   ```

2. Save the script as a `.py` file and execute it.

   ```bash
   python script_name.py
   ```

3. Observe the output, including the MSE, RMSE, and the visualizations.

## Conclusion

The project demonstrates the strength of ensemble methods like AdaBoost in handling noisy datasets and non-linear relationships. Future steps could involve testing the model on real-world datasets and comparing its performance with other machine learning algorithms.
