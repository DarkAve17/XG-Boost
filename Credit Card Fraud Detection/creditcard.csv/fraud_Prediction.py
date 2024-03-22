import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
credit_card_data = pd.read_csv(r"C:\Users\mridu\Documents\GitHub Repos\XG-Boost\Credit Card Fraud Detection\creditcard.csv\creditcard.csv")

# Explore the dataset
print(credit_card_data.head())
print(credit_card_data.info())

# Split the data into features (X) and target variable (y)
X = credit_card_data.drop("Class", axis=1)
y = credit_card_data["Class"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions
y_pred = clf.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Create a DataFrame with actual and predicted values
predictions = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

# Save predictions to a CSV file
predictions.to_csv(r"C:\Users\mridu\Documents\GitHub Repos\XG-Boost\Credit Card Fraud Detection\creditcard.csv\fruad_predictions.csv", index=False)