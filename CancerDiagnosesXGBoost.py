import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier  # Import the SGD model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer  # Import the SimpleImputer
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.svm import SVC
import xgboost as xgb
import seaborn as sns



# Load the data
df = pd.read_csv('CancerData.csv')

# Preprocessing
# For simplicity, let's just fill missing values with the mean of the column
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# Convert categorical variables to numerical if any
# For simplicity, let's just convert 'Gender' to numerical (assuming it's the only categorical variable)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

# Normalize numerical variables
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Split the data into features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the dataset into an optimized data structure called Dmatrix that XGBoost supports
data_dmatrix = xgb.DMatrix(data=X,label=y)

## Commented out the GridSearchCV code because it takes a long time to run, this is how I found the best parameters
## for the XGBoost model.

# # Define the parameter grid
# param_grid = {
#     'max_depth': [3, 4, 5, 6, 7, 8],
#     'learning_rate': [0.01, 0.1, 0.2, 0.3],
#     'n_estimators': [50, 100, 200, 300],
#     'colsample_bytree': [0.3, 0.5, 0.7, 1],
#     'alpha': [0, 1, 10, 50]
# }

# # Create a XGBClassifier object
# xg_clf = xgb.XGBClassifier(objective ='binary:logistic')

# # Create a GridSearchCV object
# grid_search = GridSearchCV(estimator=xg_clf, param_grid=param_grid, cv=3, scoring='accuracy', verbose=2)

# # Fit the GridSearchCV object to the data
# grid_search.fit(X_train, y_train)

# # Get the best parameters
# best_parameters = grid_search.best_params_

# # Print the best parameters
# print(best_parameters)

# Create the XGBoost classifier
xg_clf = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 0.3, learning_rate = 0.3,
                max_depth = 3, alpha = 0, n_estimators = 100)

# Fit the classifier to the training data
xg_clf.fit(X_train, y_train)

# Predict the labels of the test set
preds = xg_clf.predict(X_test)

# Evaluate the accuracy of the prediction
accuracy = float(np.sum(preds==y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

# Print the classification report
print(classification_report(y_test, preds))

# Print the confusion matrix
print(confusion_matrix(y_test, preds))


# Perform k-fold cross-validation
k = 10  # Number of folds
scores = cross_val_score(xg_clf, X, y, cv=k)

# Print the scores
print(f"Cross-Validation Accuracy Scores: {scores}")

# Print the mean score
print(f"Mean Cross-Validation Accuracy: {scores.mean()}")

# Compare with the XGBoost approach
print(f"XGBoost Accuracy: {accuracy}")


# Assuming that 'Diagnosis' is the target variable and 'Gender' is a feature
# Plot the distribution of the target variable
# Men are represented as 0 and women are 1, men are more likely to have cancer
sns.countplot(x='Diagnosis', data=df)
plt.title('Distribution of Diagnosis')
plt.show()

# Plot the correlation matrix
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Plot the confusion matrix for XGBoost classifier
cm = confusion_matrix(y_test, preds)
ConfusionMatrixDisplay(cm, display_labels=xg_clf.classes_).plot()
plt.title('Confusion Matrix for XGBoost Classifier')
plt.show()

# Create the SVM classifier
svm_clf = SVC(kernel='linear')

# Fit the classifier to the training data
svm_clf.fit(X_train, y_train)

# Predict the labels of the test set
svm_preds = svm_clf.predict(X_test)

# Evaluate the accuracy of the prediction
svm_accuracy = float(np.sum(svm_preds==y_test))/y_test.shape[0]
print("SVM accuracy: %f" % (svm_accuracy))

# Print the classification report
print(classification_report(y_test, svm_preds))

# Print the confusion matrix
print(confusion_matrix(y_test, svm_preds))

# Perform k-fold cross-validation
svm_scores = cross_val_score(svm_clf, X, y, cv=k)

# Print the scores
print(f"SVM Cross-Validation Accuracy Scores: {svm_scores}")

# Print the mean score
print(f"SVM Mean Cross-Validation Accuracy: {svm_scores.mean()}")

# Plot the confusion matrix for SVM classifier
svm_cm = confusion_matrix(y_test, svm_preds)
ConfusionMatrixDisplay(svm_cm, display_labels=svm_clf.classes_).plot()
plt.title('Confusion Matrix for SVM Classifier')
plt.show()