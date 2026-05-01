import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define possible values
goals = ['lose_weight', 'gain_muscle', 'maintain']
activities = ['low', 'medium', 'high']
workouts = ['cardio', 'strength', 'mixed', 'rest_light']

# Generate synthetic data
n_samples = 2000

data = {
    'age': np.random.randint(18, 71, n_samples),
    'weight': np.random.uniform(50, 120, n_samples).round(1),
    'height': np.random.uniform(150, 200, n_samples).round(1),
    'goal': [random.choice(goals) for _ in range(n_samples)],
    'activity_level': [random.choice(activities) for _ in range(n_samples)],
}

df = pd.DataFrame(data)

# Define classification logic with more accuracy
def assign_workout(row):
    age, weight, height, goal, activity = row['age'], row['weight'], row['height'], row['goal'], row['activity_level']
    bmi = weight / ((height / 100) ** 2)

    if goal == 'lose_weight':
        if bmi > 25:
            return 'cardio'
        elif activity == 'high':
            return 'mixed'
        else:
            return 'rest_light'
    elif goal == 'gain_muscle':
        if age < 35 and activity in ['medium', 'high']:
            return 'strength'
        elif activity == 'high':
            return 'mixed'
        else:
            return 'rest_light'
    elif goal == 'maintain':
        if activity == 'high':
            return 'mixed'
        elif activity == 'medium':
            return 'strength'
        else:
            return 'cardio'
    return 'rest_light'

df['workout_type'] = df.apply(assign_workout, axis=1)

# Save to CSV
df.to_csv('fitness_dataset.csv', index=False)

print("Dataset created with", len(df), "samples")
print(df.head())
print(df['workout_type'].value_counts())

# Now, preprocessing and modeling
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv('fitness_dataset.csv')

# Encode categorical features
label_encoders = {}
for col in ['goal', 'activity_level', 'workout_type']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Split features and target
X = df[['age', 'weight', 'height', 'goal', 'activity_level']].values
y = df['workout_type'].values

# Scale numerical features
scaler = StandardScaler()
X[:, :3] = scaler.fit_transform(X[:, :3])  # age, weight, height

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define the model
class FitnessRecommender(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FitnessRecommender, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# Model parameters
input_size = X_train.shape[1]
hidden_size = 128
num_classes = len(workouts)

model = FitnessRecommender(input_size, hidden_size, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_logits = model(X_test_tensor)
    y_pred = torch.argmax(y_pred_logits, dim=1).numpy()

accuracy = accuracy_score(y_test, y_pred)
print(f'Test Accuracy: {accuracy:.4f}')
print(classification_report(y_test, y_pred, target_names=workouts))

# Save model and preprocessors
torch.save(model.state_dict(), 'fitness_model.pt')

import joblib
joblib.dump(label_encoders, 'label_encoders.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and preprocessors saved.")