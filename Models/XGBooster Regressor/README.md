# House Price Prediction using XGBoost Regressor

This project demonstrates how to predict house prices based on two features: the number of rooms and the age of the house. We use XGBoost, a powerful gradient boosting algorithm, to train a regression model. The goal of the project is to explore how these features influence the price and to evaluate the model's performance using synthetic data.

## Problem Statement

The real estate market often relies on various features, such as the number of rooms in a house and its age, to determine the price. In this project, we aim to build a regression model that predicts house prices using two features:

1. **Number of Rooms**: The total number of rooms in the house.
2. **House Age**: The age of the house in years.

The data used in this project is synthetic, generated randomly to simulate a real-world dataset. The relationship between the features and the target (price) is assumed to be linear, but we use XGBoost to handle more complex relationships that could arise in a real-world scenario.

## Approach

1. **Data Generation**:
   - Synthetic data is generated with 200 samples.
   - `num_rooms` is randomly chosen between 3 and 10.
   - `house_age` is randomly chosen between 1 and 100 years.
   - `price` is calculated using a linear formula involving `num_rooms` and `house_age`, with added noise to simulate real-world data variation.

2. **Model Selection**:
   - We used **XGBoost Regressor**, which is well-suited for regression tasks and can capture non-linear relationships between the features and the target variable.

3. **Model Training**:
   - The dataset is split into training and testing sets (80% for training, 20% for testing).
   - The model is trained using the training data and evaluated on the test data.

4. **Evaluation**:
   - The model’s performance is evaluated using **Mean Squared Error (MSE)** and **Root Mean Squared Error (RMSE)**, which measure the accuracy of the predictions.

5. **Visualization**:
   - Scatter plots are generated to visualize the relationships between the features (`num_rooms` and `house_age`) and the target (`price`).
   - The performance of the model is visualized by comparing actual prices with predicted prices.

## Results

- The model produces reasonable predictions for the house prices.
- The evaluation metrics, MSE and RMSE, provide insights into the prediction accuracy.

## Conclusion

This project shows how XGBoost can be applied to predict house prices based on simple features such as the number of rooms and house age. The approach can be extended to more complex datasets with additional features and used in real estate applications for price estimation.
