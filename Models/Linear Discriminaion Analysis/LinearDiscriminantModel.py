import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#Sample Data
#[size_of_Fruit, sweetness_of_Fruit]
fruit_features = np.array([[3,7],[2,8],[3,6],[4,7],[1,4],[2,3],[3,2],[4,3]])
fruit_likes = np.array([1,1,1,1,0,0,0,0]) #1: Like, 0: Dislike

#Creating an LDA Model
model = LinearDiscriminantAnalysis()

#Training the model
model.fit(fruit_features,fruit_likes)

#Prediction
new_fruit = np.array([[2.5,6]]) #[size,sweetness]
predicted_like = model.predict(new_fruit)

output_folder = r"E:\Machine Learning\Models\Linear Discriminaion Analysis\output"
#plotting
plt.scatter(fruit_features[:, 0], fruit_features[:, 1], c=fruit_likes,cmap='viridis',marker='o')
plt.scatter(new_fruit[:,0],new_fruit[:,1], color='darkred',marker='x')
plt.title('Fruits Enjoyment Based on Size and Sweetness')
plt.xlabel('Size')
plt.ylabel('Sweetness')
plt.colorbar(label='Enjoyment Level')
plt.tight_layout()
plt.grid(True)
plt.savefig(f"{output_folder}/Size vs Sweetness General.jpg",format='jpg', dpi=300 ,bbox_inches='tight')
plt.show()

if predicted_like == 1:
    print(f"The new fruit ({new_fruit[0][0]}, {new_fruit[0][1]}) is likely to be liked.")
else:
    print(f"The new fruit ({new_fruit[0][0]}, {new_fruit[0][1]}) is unlikely to be liked.")

#Sarah's Fruit
print(f"Sarah will {'Like'if predicted_like[0]== 1 else 'Not like'} a fruit of size {new_fruit[0,0]} and sweetness {new_fruit[0,1]}.")
