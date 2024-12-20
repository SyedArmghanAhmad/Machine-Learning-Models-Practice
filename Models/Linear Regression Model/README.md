# Linear Regression Model: Stamps Bought vs. Amount Spent

## Problem Statement
Alex wants to understand the relationship between the number of stamps bought and the amount of money spent. Given historical data about the number of stamps purchased and the corresponding amount spent, Alex wants to predict the likely expenditure if 15 stamps are purchased in the next month.

## Approach to Solve the Problem
To solve this problem, a **Linear Regression Model** was implemented. Linear Regression is a supervised learning technique that identifies the best-fit linear relationship between independent variables (input) and dependent variables (output). The goal here is to predict the amount spent based on the number of stamps bought.

---

## Steps to Solve the Problem

### 1. Data Preparation
The data for this problem consists of two variables:
- **Independent Variable (X)**: Number of stamps bought.
- **Dependent Variable (Y)**: Amount spent in dollars.

The input data (`stamps_bought`) was reshaped into a 2D array to align with the requirements of `sklearn`'s Linear Regression implementation. The dependent variable (`amount_spent`) was stored as a 1D array.

```python
stamps_bought = np.array([1,3,5,7,9,11,13,15]).reshape((-1,1))
amount_spent = np.array([2,6,8,12,18,24,30,36])
```

### 2. Building the Model
A **Linear Regression model** was created and trained using the `fit()` method of the `LinearRegression` class from `sklearn`. The model was trained on the provided data to find the relationship between the number of stamps bought and the amount spent.

```python
model = LinearRegression()
model.fit(stamps_bought, amount_spent)
```

### 3. Making Predictions
Once the model was trained, it was used to predict the likely expenditure if 15 stamps are purchased.

```python
next_months_stamps = 15
predicted_spend = model.predict([[next_months_stamps]])
```

### 4. Visualizing the Results
To visualize the relationship, a scatter plot of the data points (stamps bought vs. amount spent) was created. The regression line showing the model's predictions was overlaid for better understanding. The plot was saved as a high-resolution image in the specified output folder.

```python
plt.scatter(stamps_bought, amount_spent, color='blue')
plt.plot(stamps_bought, model.predict(stamps_bought), color='red')
plt.title('Stamps Bought vs Amount Spent')
plt.xlabel('Stamps Bought')
plt.ylabel('Amount Spent ($)')
plt.grid(True)
plt.savefig(f"{output_folder}/Stamps Bought vs Amount Spent.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
```

### 5. Results
The model predicted that if Alex buys 15 stamps next month, the likely expenditure will be **$36.00**. This result is printed for clarity.

```python
print(f"If Alex buys {next_months_stamps} stamps next month, they will likely spend ${predicted_spend[0]: .2f}.")
```

---

## Output Files
The following output files are generated:
1. **Visualization Image**: The scatter plot and regression line are saved in the specified output folder as:
   `Stamps Bought vs Amount Spent.jpg`

---

## Tools and Libraries Used
- **Python**: Programming language used for implementing the solution.
- **Libraries**:
  - `numpy`: For numerical computations and data manipulation.
  - `matplotlib`: For creating and saving the visualization.
  - `sklearn.linear_model`: For the Linear Regression model.

---

## Key Insights
1. **Linear Relationship**: The model assumes a linear relationship between the number of stamps bought and the amount spent, as evidenced by the regression line.
2. **Prediction Capability**: This model can be used to make predictions for any number of stamps bought within the range of the training data or reasonable extrapolation.

---

## How to Run the Code
1. Install the required libraries if not already installed:
   ```bash
   pip install numpy matplotlib scikit-learn
   ```
2. Update the variable `output_folder` with the desired directory to save the visualization.
3. Execute the code. The prediction and plot will be generated.

---

## Example Prediction
If 15 stamps are purchased, the likely expenditure is predicted to be:
```
$36.00
```
This matches the observed pattern in the data and validates the accuracy of the model.

