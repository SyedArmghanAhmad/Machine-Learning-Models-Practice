# Documentation

---

## **Problem Statement**

A fruit-selling business wants to predict whether a customer will like or dislike a fruit based on its size and sweetness. Using historical data of customer preferences, a machine learning model should classify new fruits into "liked" or "disliked" categories. Additionally, the business requires a visualization of the decision boundaries and the ability to save the results.

### **Solution**

We solved the problem using **Linear Discriminant Analysis (LDA)**, a supervised machine learning algorithm for classification tasks. The approach involves:

1. Preparing a dataset with features (size and sweetness of fruits) and labels (liked/disliked).
2. Training an LDA model on this dataset.
3. Predicting whether a new fruit (of given size and sweetness) is likely to be liked.
4. Visualizing the data points (fruits) along with the prediction result.
5. Saving the visualization to a specified folder.

---

### Code Explanation

#### **1. Importing Necessary Libraries**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
```

- `numpy`: For handling numerical data, especially arrays.
- `matplotlib.pyplot`: For creating scatter plots and visualizing data.
- `LinearDiscriminantAnalysis`: The classification model used for predictions.

---

#### **2. Input Data**

```python
fruit_features = np.array([[3, 7], [2, 8], [3, 6], [4, 7], [1, 4], [2, 3], [3, 2], [4, 3]])
fruit_likes = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 1: Like, 0: Dislike
```

- `fruit_features`: A 2D array where each row represents a fruit. The columns are:
  - **Column 1**: Size of the fruit.
  - **Column 2**: Sweetness of the fruit.
- `fruit_likes`: A 1D array representing whether a fruit is liked (`1`) or disliked (`0`).

| Fruit | Size | Sweetness | Like/Dislike |
|-------|------|-----------|--------------|
| 1     | 3    | 7         | Like         |
| 2     | 2    | 8         | Like         |
| 3     | 3    | 6         | Like         |
| 4     | 4    | 7         | Like         |
| 5     | 1    | 4         | Dislike      |
| 6     | 2    | 3         | Dislike      |
| 7     | 3    | 2         | Dislike      |
| 8     | 4    | 3         | Dislike      |

---

#### **3. Creating and Training the Model**

```python
model = LinearDiscriminantAnalysis()
model.fit(fruit_features, fruit_likes)
```

- **Model Creation**: LDA reduces dimensionality while maintaining class separability, making it ideal for binary classification tasks.
- **Training**: The model learns the relationship between the features (size and sweetness) and the target labels (`fruit_likes`).

---

#### **4. Making Predictions**

```python
new_fruit = np.array([[2.5, 6]])  # New fruit [size, sweetness]
predicted_like = model.predict(new_fruit)
```

- `new_fruit`: Represents a new fruit with a size of `2.5` and sweetness of `6`.
- `predicted_like`: The model predicts whether this fruit will be liked (`1`) or disliked (`0`).

---

#### **5. Visualizing the Data**

```python
plt.scatter(fruit_features[:, 0], fruit_features[:, 1], c=fruit_likes, cmap='viridis', marker='o')
plt.scatter(new_fruit[:, 0], new_fruit[:, 1], color='darkred', marker='x')
plt.title('Fruits Enjoyment Based on Size and Sweetness')
plt.xlabel('Size')
plt.ylabel('Sweetness')
plt.colorbar(label='Enjoyment Level')
plt.tight_layout()
plt.grid(True)
```

- The first scatter plot displays the historical data:
  - Circles (`marker='o'`) represent fruits, colored based on whether they are liked (`c=fruit_likes` with `viridis` colormap).
- The second scatter plot highlights the new fruit as a red "X" (`marker='x'`).

---

#### **6. Saving the Visualization**

```python
output_folder = r"E:\Machine Learning\Models\Linear Discriminaion Analysis\output"
plt.savefig(f"{output_folder}/Size vs Sweetness General.jpg", format='jpg', dpi=300, bbox_inches='tight')
plt.show()
```

- The plot is saved to the specified directory with high resolution (`dpi=300`).

---

#### **7. Printing Results**

```python
if predicted_like == 1:
    print(f"The new fruit ({new_fruit[0][0]}, {new_fruit[0][1]}) is likely to be liked.")
else:
    print(f"The new fruit ({new_fruit[0][0]}, {new_fruit[0][1]}) is unlikely to be liked.")
```

- This block prints whether the new fruit is likely to be liked or disliked based on the prediction.

---

#### **8. Additional Message for Sarah**

```python
print(f"Sarah will {'Like' if predicted_like[0] == 1 else 'Not like'} a fruit of size {new_fruit[0, 0]} and sweetness {new_fruit[0, 1]}.")
```

- Outputs a personalized message based on the prediction.

---

### Example Output

1. **Visualization**: A scatter plot with data points colored by enjoyment level and the new fruit marked in red.
2. **Predicted Outcome**:

   ```bash
   The new fruit (2.5, 6.0) is likely to be liked.
   Sarah will Like a fruit of size 2.5 and sweetness 6.0.
   ```
