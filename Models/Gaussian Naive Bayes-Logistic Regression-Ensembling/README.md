# Gaussian Naive Bayes and Logistic Regression Ensemble

This project demonstrates the application of two classification algorithms—**Gaussian Naive Bayes (GNB)** and **Logistic Regression (LR)**—on a synthetic dataset. It also explores the concept of ensembling these models to potentially improve classification performance. The project includes:

- **Data Generation**: Creating a synthetic dataset using `make_circles` from scikit-learn.
- **Model Training**: Training both GNB and LR classifiers on the dataset.
- **Model Evaluation**: Assessing the accuracy of each model.
- **Visualization**: Plotting decision boundaries for each model to visualize their classification regions.
- **Ensemble Method**: Combining the predictions of both models to form an ensemble and evaluating its performance.

## Requirements

Ensure you have the following Python libraries installed:

```bash
pip install numpy matplotlib scikit-learn
```

## Usage

1. **Data Generation**: The `make_circles` function generates a large circle containing a smaller circle in two-dimensional space, with a specified noise level.

2. **Data Splitting**: The dataset is split into training and testing sets using an 80-20 split.

3. **Model Training**:
   - **Gaussian Naive Bayes**: Assumes that the features follow a Gaussian distribution and calculates the probability of each class.
   - **Logistic Regression**: Models the probability of a class using a logistic function.

4. **Model Evaluation**: The accuracy of each model is calculated by comparing the predicted labels to the true labels in the test set.

5. **Visualization**: Decision boundaries for each model are plotted to visualize how each classifier separates the classes.

6. **Ensemble Method**: An ensemble model is created by combining the predictions of both GNB and LR. The final prediction is determined by a majority vote between the two models. The accuracy of the ensemble is then evaluated.

## Output

The script generates the following outputs:

- **Decision Boundary Plots**: Visualizations showing the decision boundaries of GNB, LR, and the ensemble model.
- **Accuracy Scores**: Printed accuracy scores for GNB, LR, and the ensemble model.

## Example Output

```python
Naive Bayes Accuracy: 75.00%
Logistic Regression Accuracy: 46.50%
Ensemble Accuracy: 64.00%
```

*Note: The actual accuracy values may vary depending on the random state and noise level in the dataset.*

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

---

*For more information on Gaussian Naive Bayes and Logistic Regression, refer to the following resources:*

- [Gaussian Naive Bayes Classification with Scikit-Learn](https://garba.org/posts/2022/bayes/)
- [Connecting Naive Bayes and Logistic Regression](https://towardsdatascience.com/connecting-naive-bayes-and-logistic-regression-binary-classification-ce69e527157f)
