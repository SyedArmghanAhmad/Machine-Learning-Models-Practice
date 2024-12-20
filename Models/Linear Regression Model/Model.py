import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

#Sample Data
stamps_bought = np.array([1,3,5,7,9,11,13,15]).reshape((-1,1)) # Reshaping the array to make it 2D array as sklearn prefers it in this way.
amount_spent = np.array([2,6,8,12,18,24,30,36])

# Creating a Linear Regression Model
model = LinearRegression()

#Training the model
model.fit(stamps_bought, amount_spent)

#Predictions
next_months_stamps = 15
predicted_spend = model.predict([[next_months_stamps]])

output_folder = r"E:\Machine Learning\Models\Linear Regression Model\output"

#Plotting
plt.scatter(stamps_bought,amount_spent, color = 'blue' )
plt.plot(stamps_bought, model.predict(stamps_bought), color = 'red')
plt.title('Stamps Bought vs Amount Spent')
plt.xlabel('Stamps Bought')
plt.ylabel('Amount Spent ($)')
plt.grid(True)
plt.savefig(f"{output_folder}/Stamps Bought vs Amount Spent.jpg",format='jpg', dpi=300 ,bbox_inches='tight')
plt.tight_layout()
plt.show()

#Displaying Prediction
print(f"If Alex buys {next_months_stamps} stamps next month, he will likely spend ${predicted_spend[0]: .2f}.")
# Display model accuracy (R^2 score)
r2_score = model.score(stamps_bought, amount_spent)
print(f"Model Accuracy (R^2 Score): {r2_score:.2f}")

