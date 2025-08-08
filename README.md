# Wisconsin-Breast-Cancer_Prediction
This project uses the K-Nearest Neighbors (KNN) algorithm to classify tumors as malignant or benign using the Wisconsin Breast Cancer Diagnostic Dataset. It includes preprocessing, model training, and performance evaluation using ROC curve and AUC score.

📌 Project Overview
This project walks through the complete pipeline for a binary classification problem using Logistic Regression. It covers data preprocessing, model training, evaluation metrics (confusion matrix, precision, recall, ROC-AUC), and an explanation of the sigmoid function and threshold tuning.

✅ Project Steps
Choose a Binary Classification Dataset
Dataset: [e.g., Breast Cancer, Titanic, Pima Diabetes — replace with yours]
Target variable has two classes (e.g., 0 = Negative, 1 = Positive)
Train/Test Split & Standardization
Dataset is split into training and testing sets
Features are standardized using StandardScaler for better model performance
Fit Logistic Regression Model
A LogisticRegression model from scikit-learn is trained on the dataset
Predicts the probability that a sample belongs to class 1
Evaluate the Model
Confusion Matrix
Precision, Recall, F1-score
ROC Curve and AUC Scor
These metrics provide insights into model performance and class imbalance handling.

Tune Threshold & Explain Sigmoid Function
Default threshold (0.5) is adjusted to observe changes in precision and recall
Sigmoid function outputs probabilities between 0 and 1:
σ(z)= 1/1+e power of −z
 

Threshold tuning helps shift the decision boundary based on problem context (e.g., favoring recall in medical diagnosis)

🛠️ Tech Stack
Python
scikit-learn
Pandas
NumPy
Matplotlib / Seaborn

📈 Example Output
Confusion matrix heatmap
Precision/Recall/F1 scores printed
ROC Curve showing model performance across thresholds

📊 Future Improvements
Add cross-validation
Try other models (SVM, Random Forest)
Use GridSearchCV for hyperparameter tuning
