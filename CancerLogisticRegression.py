import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline

# Filter warnings that are not effecting the model.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Load the data
df = pd.read_csv('CancerData.csv')

# Preprocessing
# Separate numerical and categorical columns
num_cols = df.select_dtypes(include=[np.number]).columns
cat_cols = df.select_dtypes(include=[object]).columns

# Apply SimpleImputer and MinMaxScaler to numerical columns
imputer = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(imputer.fit_transform(df[num_cols]))

# Split the data into features and target
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']
y = y.astype(np.int_)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Ideal components is all 8 so use all 8 components
pca = PCA(n_components=8)
X_train2 = pca.fit_transform(X_train)
X_test2 = pca.transform(X_test)  # Transform the testing data

# Create the Logistic Regression classifier
log_reg = LogisticRegression()

# Fit the Logistic Regression classifier to the training data
log_reg.fit(X_train2, y_train)

# Use the classifier to make predictions
X_test2 = pca.transform(X_test)
predictions = log_reg.predict(X_test2)

# Evaluate the accuracy of the prediction
accuracy = float(np.sum(predictions==y_test))/y_test.shape[0]
print("Accuracy: %f" % (accuracy))

# Print the classification report
print(classification_report(y_test, predictions))

# Print the confusion matrix
print(confusion_matrix(y_test, predictions))

# Convert all columns to numeric
df = df.apply(pd.to_numeric, errors='coerce')

# Compute the correlation matrix
corr_matrix = df.corr()

# Plot the correlation matrix
plt.figure(figsize=(15, 12))  # Increase the size of the figure
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 7})  # Decrease the font size
plt.title('Correlation Matrix')

# Help with the correlation matrix presentation
plt.xticks(fontsize=9)
plt.yticks(fontsize=9)
plt.ylim(len(corr_matrix.columns)-1, -1) \

plt.show() 

# Create a pair plot
pc_df = pd.DataFrame(data = X_train2[:,:3], 
                        columns = ['PC1', 'PC2', 'PC3'])

# Add the diagnosis to the DataFrame
pc_df = pd.concat([pc_df, y_train.reset_index(drop=True)], axis = 1)

# Create a pair plot
sns.pairplot(pc_df, hue='Diagnosis')
plt.show()