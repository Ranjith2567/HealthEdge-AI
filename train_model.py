import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# 1. Load Dataset
data_path = 'data/diabetes.csv'
if not os.path.exists(data_path):
    print("Error: data/diabetes.csv file-ah kaanom da! Folder-la check pannu.")
    exit()

df = pd.read_csv(data_path)

# 2. Features and Target
# Namma nalaiku pesuna andha 8 features
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Feature Scaling (Professional ML Engineer Standard)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# 6. Accuracy Check
accuracy = model.score(X_test_scaled, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 7. Save Model and Scaler (The "Brain" Files)
if not os.path.exists('models'):
    os.makedirs('models')

joblib.dump(model, 'models/health_edge_model.joblib')
joblib.dump(scaler, 'models/scaler.joblib')

print("Success! Model and Scaler 'models/' folder-la save aayiduchi da!")