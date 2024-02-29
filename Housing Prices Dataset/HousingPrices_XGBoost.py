import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the Titanic training dataset from CSV file
train_data = pd.read_csv(r'C:\Users\mridu\Documents\GitHub Repos\XG-Boost\Housing Prices Dataset\train.csv')

# Drop unnecessary columns
train_data.drop(['Id', 'Alley','PoolQC','Fence','MiscFeature','Utilities','Street','Condition1','Condition2','SaleType','SaleCondition','RoofMatl','GarageYrBlt','Heating','HeatingQC','GarageQual','GarageCond'], axis=1, inplace=True)


# Convert categorical features into numerical format
train_data = pd.get_dummies(train_data, columns=['MSZoning', 'LotShape','LandContour','LotConfig','LandSlope','Neighborhood','BldgType','HouseStyle','RoofStyle','GarageType'], drop_first=True)

# Display the first few rows of the training dataset
print(train_data[:12])

epochs =3


#The target Variable is Sale Price
X_train = train_data.drop('SalePrice', axis=1)  # Features for training
y_train = train_data['SalePrice']               # Target variable for training

# Load the Titanic testing dataset from CSV file
test_data = pd.read_csv(r'C:\Users\mridu\Documents\GitHub Repos\XG-Boost\Housing Prices Dataset\test.csv')

# Drop unnecessary columns and perform the same preprocessing steps as for the training data
test_data.drop(['Id', 'Alley','PoolQC','Fence','MiscFeature','Utilities','Street','Condition1','Condition2','SaleType','SaleCondition','RoofMatl','GarageYrBlt','Heating','HeatingQC','GarageQual','GarageCond'], axis=1, inplace=True)
test_data = pd.get_dummies(test_data, columns=['MSZoning', 'LotShape','LandContour','LotConfig','LandSlope','Neighborhood','BldgType','HouseStyle','RoofStyle','GarageType'], drop_first=True)

# Assuming the target variable is not available in the test dataset
X_test = test_data  # Features for testing (no target variable)

# Instantiate XGBoost classifier
clf = xgb.XGBClassifier(objective='binary:logistic', random_state=42,n_estimators=epochs)

# Train the model
clf.fit(X_train, y_train, )


# Predict on test data
y_pred = clf.predict(X_test)

# Save predictions to a CSV file (optional)
predictions = pd.DataFrame({'Id': test_data['Id'], 'SalePrice': y_pred})
predictions.to_csv(r'C:\Users\mridu\Documents\GitHub Repos\XG-Boost\Housing Prices Dataset\predictions.csv', index=False)
