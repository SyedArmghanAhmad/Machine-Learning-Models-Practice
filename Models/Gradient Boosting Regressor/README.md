# Gradient Boosting Regressor for House Price Prediction

This project demonstrates the use of the Gradient Boosting Regressor (GBR) algorithm for predicting house prices based on the number of rooms and the age of the house. The dataset is synthetically generated for the purpose of this demonstration. The model is evaluated using metrics like Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

## Project Overview

The goal of this project is to predict the price of a house given its features, specifically the number of rooms and the house's age. The Gradient Boosting Regressor is used to model the relationship between the features and the house prices. The model is trained on a randomly split dataset, and the performance is evaluated on a test set.

## Requirements

- `numpy`: For generating synthetic data and handling numerical operations.
- `pandas`: For data manipulation and creating the DataFrame.
- `matplotlib`: For plotting the data and model results.
- `scikit-learn`: For implementing the Gradient Boosting Regressor and evaluation metrics.

You can install the required packages using the following:

```bash
pip install numpy pandas matplotlib scikit-learn
```

4.The script will generate and save the following plots:

- **Scatter Plots of Features vs Price**: A scatter plot that visualizes the relationship between the number of rooms, house age, and the price.
- **Actual vs Predicted House Prices with GBR**: A plot comparing the predicted house prices to the actual prices on the test set.

The plots will be saved in the `output/` directory.

## Model Evaluation

The performance of the Gradient Boosting Regressor model is evaluated using the following metrics:

- **Mean Squared Error (MSE)**: Measures the average of the squares of the errors between predicted and actual prices.
- **Root Mean Squared Error (RMSE)**: The square root of MSE, providing a more interpretable value in the same units as the target variable (price).

Example output:

``` bash
Mean Squared Error:  2266.70
Root Mean Squared Error:  47.61
```

## Visualizations

The script generates two key visualizations:

1. **Scatter Plots of Features vs Price**: Displays the distribution of the data for the features `num_rooms` and `house_age` against the target variable `price`.
2. **Actual vs Predicted House Prices with GBR**: A scatter plot comparing the predicted house prices against the actual prices, along with a reference line showing perfect predictions.

## Conclusion

This project demonstrates the application of Gradient Boosting Regressor for house price prediction. By splitting the data into training and testing sets, training the model, and evaluating its performance, we can assess the effectiveness of the GBR model for this problem.

Feel free to modify the dataset, adjust model parameters, or implement other regression models to compare the results.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for more information.
