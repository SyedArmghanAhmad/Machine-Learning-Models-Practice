
# Random Forest Classifier for Plant Species Classification

This project demonstrates the use of a **Random Forest Classifier** to classify plant species based on their features, such as leaf size and flower color. The code includes model training, evaluation, and visualization of results and feature importance.

## Problem Statement

The task is to classify plants into two distinct species based on their features:

1. **Leaf Size** (Numerical feature)
2. **Flower Color** (Categorical feature, encoded as a number)

**Objective:** Build a machine learning model to predict the species of a plant using a dataset of labeled examples.

## Dataset

The dataset is a synthetic set of features and labels:

- **Features** (`plants_features`): An array of leaf sizes and flower colors.
- **Labels** (`plants_species`): Binary labels representing two plant species (`0` and `1`).

Example:

| Leaf Size | Flower Color | Species |
|-----------|--------------|---------|
| 3         | 1            | 0       |
| 2         | 2            | 1       |

## Solution Approach

The problem is solved using the **Random Forest Classifier** algorithm, which combines multiple decision trees to make predictions. The steps are:

1. **Data Preparation:**
   - Features and labels are defined as numpy arrays.
   - Data is split into training and testing sets using an 80/20 split.

2. **Model Creation:**
   - A Random Forest Classifier with 10 estimators (trees) is created.

3. **Model Training:**
   - The model is trained on the training dataset using the `.fit()` method.

4. **Evaluation:**
   - Predictions are made on the testing set.
   - Model performance is evaluated using a classification report that includes metrics such as precision, recall, and F1-score.

5. **Visualization:**
   - A scatter plot visualizes the distribution of plant species based on features.
   - A bar plot shows the importance of features in classification.

## How It Works

### Random Forest Classifier

- **Ensemble Learning**: Combines multiple decision trees to improve predictive performance and reduce overfitting.
- **Feature Importance**: Evaluates which features contribute most to classification.

Background processes:

- Each decision tree is trained on a random subset of data and features (bagging).
- Predictions are made by aggregating the outputs of all decision trees (majority voting for classification).

### Code Breakdown

#### Imports

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
```

Required libraries for data manipulation, visualization, and model training.

#### Data Definition

```python
plants_features = np.array([[3,1],[2,2],...])
plants_species = np.array([0, 1, 0, 1, ...])
```

Defines the feature matrix and target labels.

#### Data Splitting

```python
X_Train, X_Test, y_train, y_test = train_test_split(plants_features, plants_species, test_size=0.25, random_state=42)
```

Splits the data into training and testing subsets.

#### Model Creation and Training

```python
model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_Train, y_train)
```

Creates and trains a Random Forest Classifier.

#### Evaluation

```python
y_pred = model.predict(X_Test)
classification_rep = classification_report(y_test, y_pred)
```

Generates predictions and evaluates the model.

#### Visualizations

- **Scatter Plot**: Displays the dataset with different species represented by markers.
- **Feature Importance Plot**: Highlights the contribution of each feature to the classification.

## Output

- **Classification Report:** Includes precision, recall, F1-score, and support for each class.
- **Scatter Plot of Species:** Visualizes plant features.
- **Feature Importance Bar Chart:** Identifies key features for classification.

## Usage

1. Clone this repository.
2. Modify `output_folder` to your desired path for saving visualizations.
3. Run the script using Python 3.x:

   ```bash
   python plant_species_classification.py
   ```

4. View the generated plots in the specified folder.

## Requirements

- Python 3.x
- Numpy
- Matplotlib
- Scikit-learn

## Results

The model provides insight into feature importance and visualizes species classification. It achieves high accuracy by leveraging Random Forest's robust ensemble approach.

## License

This project is licensed under the MIT License. Feel free to use and modify it.
