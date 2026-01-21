import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

# 1. Load the Titanic dataset
print("Loading dataset...")
df = pd.read_csv('titanic.csv')

# 2. Perform data preprocessing
print("Preprocessing data...")

# a. Handling missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
df['Fare'] = df['Fare'].fillna(df['Fare'].median())

# b. Feature selection (Select 5 input features + Survived)
# Chosen features: Pclass, Sex, Age, SibSp, Fare
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare']
X = df[features].copy()
y = df['Survived']

# c. Encoding categorical variables
le = LabelEncoder()
X['Sex'] = le.fit_transform(X['Sex']) # Female: 0, Male: 1

# d. Feature scaling (Random Forest doesn't strictly require it, but good practice)
scaler = StandardScaler()
X[['Age', 'Fare']] = scaler.fit_transform(X[['Age', 'Fare']])

# 3. Implement machine learning algorithm (Random Forest)
print("Training Random Forest model...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate the model
print("\nClassification Report:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 5. Save the trained model to disk
print("Saving model and artifacts...")
if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(model, 'model/titanic_survival_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
joblib.dump(le, 'model/label_encoder.pkl')

print("Model saved to model/titanic_survival_model.pkl")

# 6. Demonstrate reloading
print("\nDemonstrating reload and prediction...")
loaded_model = joblib.load('model/titanic_survival_model.pkl')
sample_data = X_test.iloc[:1]
prediction = loaded_model.predict(sample_data)
print(f"Sample Prediction: {'Survived' if prediction[0] == 1 else 'Did Not Survive'}")
