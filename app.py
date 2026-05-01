# FastAPI app for Personal Fitness Recommender
# This module provides a REST API to predict workout recommendations based on user inputs
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from predict import predict_workout

# Initialize FastAPI app with title and version
app = FastAPI(title="Personal Fitness Recommender API", version="1.0")

# Define the request model for prediction endpoint
# Uses Pydantic for validation and examples in Swagger UI
class PredictionRequest(BaseModel):
    age: int = Field(default=25, example=25)  # User's age in years
    weight: float = Field(default=80.0, example=80.0)  # Weight in kg
    height: float = Field(default=175.0, example=175.0, description="Height in cm")  # Height in centimeters
    goal: str = Field(default='lose_weight', example='lose_weight')  # Fitness goal: lose_weight, gain_muscle, maintain
    activity_level: str = Field(default='low', example='low')  # Activity level: low, medium, high

    # Validator for goal field to ensure it's one of the allowed values
    @field_validator('goal')
    @classmethod
    def validate_goal(cls, v):
        if v not in ['lose_weight', 'gain_muscle', 'maintain']:
            raise ValueError('Invalid goal')
        return v

    # Validator for activity_level field
    @field_validator('activity_level')
    @classmethod
    def validate_activity(cls, v):
        if v not in ['low', 'medium', 'high']:
            raise ValueError('Invalid activity_level')
        return v

# Prediction endpoint: POST /predict
# Takes user inputs and returns recommendation, confidence, and explanation
@app.post("/predict")
def predict(request: PredictionRequest):
    # Call the prediction function from predict.py
    result = predict_workout(request.age, request.weight, request.height, request.goal, request.activity_level)
    return result

# Root endpoint for API status
@app.get("/")
def read_root():
    return {"message": "Fitness Recommender API"}

# Run the app if executed directly (for development)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)