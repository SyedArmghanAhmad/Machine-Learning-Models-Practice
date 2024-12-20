import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Sample Data
pages = np.array([100,150,200,250,300,350,400,450,500]).reshape((-1,1)) # Reshaping it into a 2D array as Sklearn prefers it in that way
likes = np.array([0,1,1,1,0,0,0,0,0]) # 1: like, 0: Dislike

#Creating a logistic Regression Model
model = LogisticRegression()

#Training the Model
model.fit(pages,likes)

#Predictions
predicted_book_pages = 500
predicted_Like = model.predict([[predicted_book_pages]])

#Compute Accuracy
predicted_labels = model.predict(pages)
accuracy = accuracy_score(likes, predicted_labels)

output_folder = r"E:\Machine Learning\Models\Logistic Regression\output"
#Plotting
plt.scatter(pages,likes,color='forestgreen')
plt.plot(pages,model.predict_proba(pages)[:,1], color='darkred')
plt.title('Book Pages vs Like/Dislike')
plt.xlabel('Number of Pages')
plt.ylabel('Likelihood of Liking')
plt.axvline(x=predicted_book_pages, color='green', linestyle='--')
plt.axhline(y=0.5,color='grey', linestyle='--')
plt.savefig(f"{output_folder}/BookPages vs Like-Dislike-Disliked.jpg",format='jpg', dpi=300 ,bbox_inches='tight')
plt.show()
plt.grid(True)

# Displaying the Prediction and Accuracy
print(f"Jenny will {'Like' if predicted_Like[0] == 1 else 'Not Like'} a book of {predicted_book_pages} pages.")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

