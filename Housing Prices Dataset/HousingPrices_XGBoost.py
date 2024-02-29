import xgboost as xgb
from sklearn.model_selection import train_test_split
import pandas as pd

# Load the Titanic training dataset from CSV file
train_data = pd.read_csv(r'C:\Users\mridu\Documents\GitHub Repos\XG-Boost\Housing Prices Dataset\train.csv')

# Drop unnecessary columns
train_data.drop(['Id', 'Alley','PoolQC','Fence','MiscFeature','Utilities','Street','Condition1','Condition2','SaleType','SaleCondition','RoofMatl','GarageYrBlt','Heating','HeatingQC','GarageQual','GarageCond'], axis=1, inplace=True)

# Convert categorical features into numerical format
train_data = pd.get_dummies(train_data, columns=['MSZoning', 'LotShape','LandContour','LotConfig','LandSlope','Neighborhood','BldgType','HouseStyle','RoofStyle','GarageType','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageFinish','PavedDrive'], drop_first=True)

# Display the first few rows of the training dataset
print(train_data[:12])

# Set the number of boosting rounds
n_estimators = 100

# The target Variable is Sale Price
X_train = train_data.drop('SalePrice', axis=1)  # Features for training
y_train = train_data['SalePrice']               # Target variable for training

# Instantiate XGBoost regressor
regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42, n_estimators=n_estimators, enable_categorical=True)

# Train the model
regressor.fit(X_train, y_train)

# Load the Titanic testing dataset from CSV file
test_data = pd.read_csv(r'C:\Users\mridu\Documents\GitHub Repos\XG-Boost\Housing Prices Dataset\test.csv')

# Drop unnecessary columns and perform the same preprocessing steps as for the training data
test_data.drop(['Id', 'Alley','PoolQC','Fence','MiscFeature','Utilities','Street','Condition1','Condition2','SaleType','SaleCondition','RoofMatl','GarageYrBlt','Heating','HeatingQC','GarageQual','GarageCond'], axis=1, inplace=True)
test_data = pd.get_dummies(test_data, columns=['MSZoning', 'LotShape','LandContour','LotConfig','LandSlope','Neighborhood','BldgType','HouseStyle','RoofStyle','GarageType','Exterior1st','Exterior2nd','MasVnrType','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','CentralAir','Electrical','KitchenQual','Functional','FireplaceQu','GarageFinish','PavedDrive'], drop_first=True)

# Align columns of test_data with train_data
test_data = test_data.reindex(columns=X_train.columns, fill_value=0)

# Predict on test data
y_pred = regressor.predict(test_data)

# Save predictions to a CSV file (optional)
predictions = pd.DataFrame({'SalePrice': y_pred})
predictions.to_csv(r'C:\Users\mridu\Documents\GitHub Repos\XG-Boost\Housing Prices Dataset\predictions.csv', index=False)
