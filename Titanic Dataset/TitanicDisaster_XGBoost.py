import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the Titanic training dataset from CSV file
train_data = pd.read_csv(r'C:\Users\mridu\Documents\GitHub Repos\XG-Boost\Titanic Dataset\train.csv')

# Drop unnecessary columns
train_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convert categorical features into numerical format
train_data = pd.get_dummies(train_data, columns=['Sex', 'Embarked'], drop_first=True)

# Display the first few rows of the training dataset
print(train_data.head())

# Assuming the target variable is 'Survived'
X_train = train_data.drop('Survived', axis=1)  # Features for training
y_train = train_data['Survived']                # Target variable for training

# Load the Titanic testing dataset from CSV file
test_data = pd.read_csv(r'C:\Users\mridu\Documents\GitHub Repos\XG-Boost\Titanic Dataset\test.csv')

# Drop unnecessary columns and perform the same preprocessing steps as for the training data
test_data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test_data = pd.get_dummies(test_data, columns=['Sex', 'Embarked'], drop_first=True)

# Assuming the target variable is not available in the test dataset
X_test = test_data  # Features for testing (no target variable)

# Instantiate XGBoost classifier
clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42)

# Train the model
clf.fit(X_train, y_train)

# Predict on test data
y_pred = clf.predict(X_test)

# Save predictions to a CSV file (optional)
predictions = pd.DataFrame({'PassengerId': test_data['PassengerId'], 'Survived': y_pred})
predictions.to_csv('predictions.csv', index=False)
