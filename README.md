# 🏋️ Personal Fitness Recommender

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/yourusername/fitness-recommender?style=social)](https://github.com/yourusername/fitness-recommender)
[![GitHub forks](https://img.shields.io/github/forks/yourusername/fitness-recommender?style=social)](https://github.com/yourusername/fitness-recommender)

An AI-powered fitness recommendation system built with PyTorch and FastAPI. Get personalized workout suggestions based on your profile, with confidence scores and expert explanations.

![Demo](https://via.placeholder.com/800x400?text=AI+Fitness+Recommender+Demo) <!-- Replace with actual screenshot -->

## ✨ Features

- 🤖 **Smart Recommendations**: Predicts optimal workout types (cardio, strength, mixed, rest/light)
- 📊 **Confidence Scoring**: Provides reliability percentage for each recommendation
- 💡 **Detailed Explanations**: Actionable advice based on BMI, age, goals, and activity level
- 🚀 **Production-Ready API**: FastAPI backend with automatic docs and validation
- 📈 **High Accuracy**: 96.5% test accuracy on synthetic dataset
- 🔧 **Easy Deployment**: One-click deploy to Render, Railway, or any cloud platform

## 🏗️ Architecture

```
User Input → Preprocessing → PyTorch Model → Prediction → FastAPI Response
     ↓              ↓              ↓              ↓              ↓
  Age, Weight,   Encoding &     Neural Network  Workout Type +  JSON with
  Height, Goal,  Scaling        (3-layer NN)    Confidence     Explanation
  Activity Level
```

## 📁 Repository Structure

```
fitness-ml-model-pytorch/
├── app.py                      # FastAPI application
├── predict.py                  # Inference functions
├── fitness_recommender.ipynb   # Training notebook (Jupyter)
├── fitness_recommender.py      # Training script (Python)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore rules
```

## 🚀 Quick Start

1. **Clone & Install**
   ```bash
   git clone https://github.com/yourusername/fitness-recommender.git
   cd fitness-recommender
   pip install -r requirements.txt
   ```

2. **Train Model (Optional - Pre-trained available)**
   ```bash
   jupyter notebook fitness_recommender.ipynb
   # Run all cells to train and save model
   ```

3. **Start API**
   ```bash
   uvicorn app:app --reload
   ```

4. **Test API**
   - Open http://127.0.0.1:8000/docs
   - Use Swagger UI to test predictions

## 📦 Installation

### Prerequisites
- Python 3.8+
- PyTorch (CPU or GPU)
- Jupyter Notebook (for training)

### Step-by-Step Setup

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/fitness-recommender.git
   cd fitness-recommender
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Install PyTorch** (if not included)
   ```bash
   # CPU version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

   # GPU version (if you have CUDA)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

## 🎯 Usage

### Training the Model

**Option 1: Jupyter Notebook (Recommended)**
- Open `fitness_recommender.ipynb` in Jupyter Notebook, VS Code, or Google Colab.
- **Free GPU on Colab**: Upload the notebook to [Google Colab](https://colab.research.google.com/) for free GPU acceleration (Runtime > Change runtime type > GPU).
- Run all cells to train the model.

**Option 2: Python Script**
- Run `python fitness_recommender.py` to train and save the model.

Steps:
1. **Dataset Generation**: Creates 2000 synthetic samples with realistic fitness logic
2. **Preprocessing**: Encodes categoricals, scales numerical features
3. **Model Training**: 3-layer neural network trained for 200 epochs
4. **Evaluation**: Achieves 96.5% accuracy on test set
5. **Save Artifacts**: Exports model and preprocessors for production

### Running Predictions

#### Via Python Script
```python
from predict import predict_workout

result = predict_workout(age=25, weight=70, height=175, goal='lose_weight', activity_level='medium')
print(result)
```

#### Via API
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age":25, "weight":70, "height":175, "goal":"lose_weight", "activity_level":"medium"}'
```

## 📚 API Documentation

### Endpoint: `POST /predict`

Predict workout recommendation based on user profile.

**Request Body:**
```json
{
  "age": 25,
  "weight": 70.0,
  "height": 175.0,
  "goal": "lose_weight",
  "activity_level": "medium"
}
```

**Parameters:**
- `age` (int): Age in years (18-70)
- `weight` (float): Weight in kg (40-150)
- `height` (float): Height in cm (140-220)
- `goal` (str): "lose_weight", "gain_muscle", "maintain"
- `activity_level` (str): "low", "medium", "high"

**Response:**
```json
{
  "recommendation": "cardio",
  "confidence": 0.85,
  "explanation": "Your profile: Age 25, 70kg (22.9 BMI - normal), 175cm height, goal 'lose weight', activity 'medium'. Combine cardio and strength for sustainable weight loss."
}
```

**Response Fields:**
- `recommendation` (str): Suggested workout type
- `confidence` (float): Prediction confidence (0-1)
- `explanation` (str): Personalized advice

### Interactive Docs

Visit `http://your-api-url/docs` for Swagger UI testing.

## 🧠 Model Details

### Architecture
- **Type**: Feedforward Neural Network
- **Layers**: 3 Linear layers (5 → 128 → 64 → 4)
- **Activations**: ReLU
- **Regularization**: Dropout (0.2)
- **Output**: Softmax probabilities

### Training
- **Dataset**: 2000 synthetic samples
- **Split**: 80% train, 20% test (stratified)
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- **Epochs**: 200
- **Accuracy**: 96.5%

### Preprocessing
- **Numerical**: StandardScaler (mean=0, std=1)
- **Categorical**: LabelEncoder (goal, activity_level)
- **Features**: Age, Weight, Height, Goal, Activity Level

## 🚀 Deployment

### Local Development
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Cloud Deployment

#### Render
1. Connect GitHub repo
2. Set build command: `pip install -r requirements.txt`
3. Set start command: `uvicorn app:app --host 0.0.0.0 --port $PORT`

#### Railway
1. Import from GitHub
2. Railway auto-detects FastAPI
3. Deploy automatically

#### Heroku
1. Create `Procfile`: `web: uvicorn app:app --host 0.0.0.0 --port $PORT`
2. Deploy via Heroku CLI

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📄 License

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Permissions:**
- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use

**Limitations:**
- ❌ Liability
- ❌ Warranty

**Conditions:**
- 📝 License and copyright notice

## 🔒 .gitignore

The repository includes a comprehensive `.gitignore` file that excludes:

- **Python artifacts**: `__pycache__/`, `*.pyc`, `*.pyo`
- **Environments**: `venv/`, `.env`, virtual environments
- **Model files**: `*.pt`, `*.pkl` (pre-trained models available in releases)
- **Datasets**: `*.csv` (generated synthetically)
- **IDE files**: `.vscode/`, `.idea/`
- **Logs**: `*.log`, temporary files

This keeps the repo clean and focused on source code.

## 🙏 Acknowledgments

- PyTorch for deep learning framework
- FastAPI for modern API development
- Scikit-learn for ML utilities

---

**Built by [CODEWITHAZLO](https://github.com/codewithazlo) - Full Stack + ML Engineer**

Transforming ideas into intelligent solutions. 🚀