# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Step 2: Load the dataset
data = pd.read_csv('Titanic-Dataset.csv')
print("First 10 rows of data:\n", data.head(10))

#checking the size of the data
print("The size of the dataset as (Rows,Columns)::",data.shape)


# Step 3: Data Cleaning

# Checking the  missing values
print("\nMissing values:\n", data.isnull().sum())

# Fill missing 'Age' with median because it helps to have the middle value
data['Age'] = data['Age'].fillna(data['Age'].median())

# # Fill missing 'Embarked' with mode because it helps to have a most common value
data['Embarked'] = data['Embarked'].fillna(data['Embarked'].mode()[0])

# Drop 'Cabin' (too many missing values its better to drop if we don't drop then maybe the model can baised)
data.drop('Cabin', axis=1, inplace=True)

# Drop unnecessary columns (better to remove them because we cannot predict by this columns)
data.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

print(data.isnull().sum()) # Data is clean now

# Step 4: Data Processing

# Encode 'Sex' and 'Embarked'(we Required to encode the data because the machine learning accepts the numerical values
# not a text-based)
le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])        # male = 1, female = 0
data['Embarked'] = le.fit_transform(data['Embarked'])  # Encode C, Q, S into 0,1,2 (example)

print("\nCleaned Data:\n", data.head(10)) # The columns which we had selected for the encode is found to be there or not
#to check it , we use to see the data of the fist 10 rows/columns)


# Step 5: Data Visualization
# 1. Survival counts
sns.countplot(x='Survived', data=data)
plt.title('Survival Counts')
plt.show()

# 2. Survival by Gender
sns.countplot(x='Survived', hue='Sex', data=data)
plt.title('Survival by Gender')
plt.show()

# 3. Age distribution
plt.hist(data['Age'], bins=30, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# 4. Correlation Heatmap
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation')
plt.show()


# Step 6: Machine Learning
# Separate features and target
X = data.drop('Survived', axis=1)
y = data['Survived']


# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)


# Step 7: Model Evaluation
# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Accuracy score
print("Accuracy Score:", accuracy_score(y_test, y_pred))
