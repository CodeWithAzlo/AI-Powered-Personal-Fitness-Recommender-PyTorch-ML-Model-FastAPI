# Prediction module for the Personal Fitness Recommender
# Loads the trained PyTorch model and preprocessors to make predictions
import torch
import torch.nn as nn
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Define the PyTorch model architecture (must match training)
# Neural network with 3 linear layers, ReLU activations, and dropout for regularization
class FitnessRecommender(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FitnessRecommender, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Input layer
        self.relu = nn.ReLU()  # Activation
        self.dropout = nn.Dropout(0.2)  # Dropout to prevent overfitting
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)  # Hidden layer
        self.relu2 = nn.ReLU()  # Activation
        self.fc3 = nn.Linear(hidden_size // 2, num_classes)  # Output layer

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

# Load the trained model weights
model = FitnessRecommender(5, 128, 4)  # 5 inputs, 128 hidden, 4 outputs
model.load_state_dict(torch.load('fitness_model.pt'))  # Load saved state dict
model.eval()  # Set to evaluation mode (disables dropout)

# Load label encoders and scaler for preprocessing
label_encoders = joblib.load('label_encoders.pkl')  # Encodes categorical to numbers
scaler = joblib.load('scaler.pkl')  # Scales numerical features

# Workout classes in order
workouts = ['cardio', 'strength', 'mixed', 'rest_light']

# Preprocess input data to match training format
def preprocess_input(age, weight, height, goal, activity_level):
    # Encode categorical features to numbers
    goal_encoded = label_encoders['goal'].transform([goal])[0]
    activity_encoded = label_encoders['activity_level'].transform([activity_level])[0]

    # Scale numerical features (age, weight, height) using trained scaler
    numerical = scaler.transform([[age, weight, height]])[0]

    # Combine into feature array: [age_scaled, weight_scaled, height_scaled, goal_encoded, activity_encoded]
    features = np.array([numerical[0], numerical[1], numerical[2], goal_encoded, activity_encoded])
    # Convert to PyTorch tensor and add batch dimension
    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)

# Main prediction function
def predict_workout(age, weight, height, goal, activity_level):
    # Preprocess the input
    input_tensor = preprocess_input(age, weight, height, goal, activity_level)

    # Run inference without gradients
    with torch.no_grad():
        output = model(input_tensor)  # Raw logits from model
        probabilities = torch.softmax(output, dim=1).squeeze()  # Convert to probabilities
        predicted_class = torch.argmax(probabilities).item()  # Get class index
        confidence = probabilities[predicted_class].item()  # Confidence for predicted class

    # Map index to workout name
    workout = workouts[predicted_class]

    # Calculate BMI for explanation
    bmi = weight / ((height / 100) ** 2)
    bmi_category = "underweight" if bmi < 18.5 else "normal" if bmi < 25 else "overweight" if bmi < 30 else "obese"
    # Start professional explanation
    explanation = f"Your profile: Age {age}, {weight}kg ({bmi:.1f} BMI - {bmi_category}), {height}cm height, goal '{goal.replace('_', ' ')}', activity '{activity_level}'. "

    # Add actionable advice based on inputs
    if goal == 'lose_weight':
        if bmi > 25:
            explanation += "To lose weight, focus on cardio exercises like running or cycling to burn calories."
        elif activity_level == 'high':
            explanation += "Combine cardio and strength for sustainable weight loss."
        else:
            explanation += "Start with light cardio and build activity gradually."
    elif goal == 'gain_muscle':
        if age < 35 and activity_level in ['medium', 'high']:
            explanation += "Strength training with weights is ideal for muscle gain at your age and activity level."
        elif activity_level == 'high':
            explanation += "Mix strength and cardio to build muscle while maintaining fitness."
        else:
            explanation += "Begin with light strength exercises and increase intensity as you progress."
    elif goal == 'maintain':
        if activity_level == 'high':
            explanation += "A balanced mix of cardio and strength will help maintain your current fitness."
        elif activity_level == 'medium':
            explanation += "Incorporate strength training to preserve muscle mass."
        else:
            explanation += "Light cardio activities will support weight maintenance."
    else:
        explanation += "Rest or light activities are recommended for recovery."

    # Return result dictionary
    return {
        "recommendation": workout,  # Predicted workout type
        "confidence": round(confidence, 2),  # Confidence score (0-1)
        "explanation": explanation  # Human-readable reason
    }

# Example usage (run this file directly to test)
if __name__ == "__main__":
    result = predict_workout(25, 80, 175, 'lose_weight', 'low')  # Sample input
    print(result)  # Prints the prediction result