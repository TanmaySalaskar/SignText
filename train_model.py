import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib  # To save the model

# Load dataset
df = pd.read_csv("gesture_data.csv")

# Ensure all rows have exactly 126 features
for i in range(126 - len(df.columns) + 1):
    df[f"LM{len(df.columns) - 1 + i}"] = 0

# Split features and labels
X = df.drop(columns=["Label"]).values
y = df["Label"].values

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Test model accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
joblib.dump(model, "gesture_model.pkl")
print("Model saved as gesture_model.pkl")
