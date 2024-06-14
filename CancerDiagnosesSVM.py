import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Filter warnings that are not effecting the model.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load the data
df = pd.read_csv('CancerData.csv')

# Preprocessing
imputer = SimpleImputer(strategy='mean')
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
scaler = MinMaxScaler()
df[df.columns] = scaler.fit_transform(df[df.columns])

# Split the data into features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
y = y.astype(np.int_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Assuming X_train is your training data
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_train)  # Fit the imputer on the training data
X_train = imputer.transform(X_train)  # Transform the training data
X_test = imputer.transform(X_test)  # Transform the testing data

# Determine the ideal number of principal components
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print('The ideal number of principal components is', d)

# Ideal components is all 7 so use all 7 components
pca = PCA(n_components=7)
X_train2 = pca.fit_transform(X_train)


# # Define the parameter grid
# param_grid = {'C': [0.1, 1, 10, 100, 1000], 
#               'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
#               'kernel': ['linear', 'rbf']}

# # Create a GridSearchCV object
# grid = GridSearchCV(svm_clf, param_grid, refit=True, verbose=3)

# # Fit the grid search to the data
# grid.fit(X_train2, y_train)

# # Print the best parameters
# print(grid.best_params_)

# Create the SVM classifier with the best parameters found with the gridsearchCV commented out above.
svm_clf = SVC(C=1000, gamma=0.01, kernel='rbf')

# Fit the SVM classifier to the training data
svm_clf.fit(X_train2, y_train)

# Use the classifier to make predictions
X_test2 = pca.transform(X_test)
predictions = svm_clf.predict(X_test2)

# Evaluate the accuracy of the prediction
accuracy = float(np.sum(predictions==y_test))/y_test.shape[0]
print("Accuracy: %f" % (accuracy))

# Print the classification report
print(classification_report(y_test, predictions))

# Print the confusion matrix
print(confusion_matrix(y_test, predictions))

# Create a pair plot
pc_df = pd.DataFrame(data = X_train2[:,:3], 
                        columns = ['PC1', 'PC2', 'PC3'])

# Add the diagnosis to the DataFrame
pc_df = pd.concat([pc_df, y_train.reset_index(drop=True)], axis = 1)

# Create a pair plot
sns.pairplot(pc_df, hue='Diagnosis')
plt.show()