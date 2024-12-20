# Movie Like Prediction Using Gaussian Naive Bayes

## Overview

This project demonstrates the use of a **Gaussian Naive Bayes** model to predict whether a person (in this case, Jenny) will like or dislike a movie based on its length and genre. The solution also includes visualizing the data and model predictions using **matplotlib** and saving the results as a high-quality image.

The model is trained on a simple dataset where the features are the movie's **length** and **genre**, and the target is whether the person **likes** the movie (1 for "Like", 0 for "Dislike").

## Problem

The task is to predict whether Jenny will like or dislike a movie based on:

1. **Movie Length** (in minutes).
2. **Genre** (represented as a numerical code, where `0` is Action and `1` is Romance).

The model needs to take these two inputs (movie length and genre) and predict the likelihood of Jenny liking the movie.

## Solution

### Steps Taken

1. **Data Preparation:**
   - We started with a small dataset containing movies with their lengths and genre codes, along with a target indicating whether the movie was liked or disliked.
   The dataset consists of:
   - `movie_features`: Movie length (in minutes) and genre code (0 for Action, 1 for Romance).
   - `movie_likes`: Target values representing whether the movie was liked (`1`) or disliked (`0`).

   ```python
   movie_features = np.array([[120, 0],[150, 1],[90, 0],[140, 1],[100, 0],[80, 1],[110, 0],[130, 1]])
   movie_likes = np.array([1,1,0,1,0,1,0,1])  # 1: Like, 0: Dislike
   ```

2. **Model Training:**
   - We used the **Gaussian Naive Bayes** algorithm to train the model. This model is particularly suited for classification tasks where features are assumed to follow a Gaussian (normal) distribution.

   ```python
   model = GaussianNB()
   model.fit(movie_features, movie_likes)
   ```

3. **Making Predictions:**
   - A new movie with length 100 minutes and genre code `1` (Romance) was used as input to predict whether Jenny would like the movie or not.

   ```python
   new_movie = np.array([[100, 1]])  # [movie_length, genre_code]
   predicted_like = model.predict(new_movie)
   ```

4. **Plotting the Results:**
   - We visualized the movie data (movie length vs genre) and marked the prediction of the new movie using **matplotlib**.
   - The plot was saved as a high-resolution image to ensure clarity and ease of presentation.

   ```python
   plt.scatter(movie_features[:, 0], movie_features[:, 1], c=movie_likes, cmap='viridis', marker='o')
   plt.scatter(new_movie[:, 0], new_movie[:, 1], color='darkred', marker='x')
   plt.title('Movie Likes Based on Length and Genre')
   plt.xlabel('Movie Length (min)')
   plt.ylabel('Genre Code')
   plt.colorbar(label='Like/Dislike')
   plt.tight_layout()
   plt.grid(True)
   plt.savefig(f"{output_folder}/MovieLength_vs_Genre.jpg", format='jpg', dpi=300, bbox_inches='tight')
   plt.show()
   ```

5. **Evaluation Metrics:**
   - After training the model, we evaluated its performance using several metrics:
     - **Accuracy**: The percentage of correctly predicted values.
     - **Confusion Matrix**: A matrix to visualize the performance of the classification model.
     - **Precision, Recall, and F1 Score**: Metrics to understand the balance between precision and recall for the given dataset.

   ```python
   accuracy = accuracy_score(movie_likes, y_pred)
   conf_matrix = confusion_matrix(movie_likes, y_pred)
   precision = precision_score(movie_likes, y_pred)
   recall = recall_score(movie_likes, y_pred)
   f1 = f1_score(movie_likes, y_pred)
   ```

6. **Displaying Results:**
   - The final prediction was displayed, showing whether Jenny would "Like" or "Not Like" the movie.
   - The model output is aligned with the prediction made by the Gaussian Naive Bayes classifier.

### Key Insights

- **Data and Features**: The movie's length and genre were sufficient for this simple classification task.
- **Model**: Gaussian Naive Bayes was effective for this problem, as it makes predictions based on conditional probabilities, which is well-suited for scenarios with numeric features.
- **Visualization**: The plot helps in understanding the relationship between the features (length and genre) and the target variable (like/dislike).

## Conclusion

This project demonstrates how to use a **Gaussian Naive Bayes** classifier for a movie like/dislike prediction task based on movie length and genre. The results were visualized using **matplotlib**, and the model's performance was evaluated using key metrics such as accuracy, confusion matrix, precision, recall, and F1 score. The solution is simple yet effective for understanding how basic machine learning models work with small datasets.

## Requirements

To run this code, you need the following Python libraries:

- `numpy`
- `matplotlib`
- `scikit-learn`

You can install the necessary libraries with:

```bash
pip install numpy matplotlib scikit-learn
```
