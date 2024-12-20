
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

#Seed for reproducibility
np.random.seed(42)

#generate synthetic data
num_samples = 200
num_rooms = np.random.randint(3,10,num_samples)
house_age = np.random.randint(1,100,num_samples)
noise = np.random.normal(0,50,num_samples)

#Assume a linear relation with price = 50 * rooms +0.5*age + noise
price = 50 * num_rooms +0.5 * house_age + noise

# Create a DataFrame
data = pd.DataFrame({'num_rooms': num_rooms, 'house_age': house_age, 'price': price})

output_folder = r"E:\Machine Learning\Models\Gradient Boosting Regressor\output" # use your own here

#Plot
plt.scatter(data['num_rooms'],data['price'], label='Num Rooms vs Price', color ='forestgreen')
plt.scatter(data['house_age'],data['price'], label='Num Rooms vs Price', color ='darkred')
plt.xlabel('Feature Value')
plt.ylabel('Price')
plt.legend()
plt.title('Scatter Plots of Features vs Price')
plt.savefig(f"{output_folder}/Scatter Plots of Features vs Price GBR.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()

#Splitting the data into training and testing sets
X = data[['num_rooms','house_age']]
y = data['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

#initialize and train the model
model_gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.4, max_depth=1, random_state=50)
model_gbr.fit(X_train, y_train)

#Predictions
predictions = model_gbr.predict(X_test)

#Evaluate the model
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse: .2f}")
print(f"Root Mean Squared Error: {rmse: .2f}")

#Plot the predictions
plt.scatter(y_test, predictions, color='orange')
plt.plot([y_test.min(), y_test.max()],[y_test.min(), y_test.max()], 'k--', lw=3)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual Vs Predicted House Prices with AdaBoost')
plt.savefig(f"{output_folder}/Actual Vs Predicted House Prices with GBR.jpg",format='jpg', dpi=300, bbox_inches='tight')
plt.show()


