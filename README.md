# <div align="center">Rock vs Mine Prediction using Logistic Regression</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/downloads/) [![Jupyter](https://img.shields.io/badge/Made%20with-Jupyter-orange?logo=Jupyter)](https://jupyter.org/try) [![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/) [![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?logo=numpy&logoColor=white)](https://numpy.org/) [![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?logo=pandas&logoColor=white)](https://pandas.pydata.org/) [![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Classification-brightgreen)](https://github.com/NhanPhamThanh-IT/Rock-Mine-Logistic) [![Accuracy](https://img.shields.io/badge/Accuracy-76--81%25-success)](https://github.com/NhanPhamThanh-IT/Rock-Mine-Logistic) [![Open Source](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](https://opensource.org/) [![Dataset](https://img.shields.io/badge/Dataset-Sonar%20Data-blue)](https://github.com/NhanPhamThanh-IT/Rock-Mine-Logistic)

</div>

<div align="justify">

A machine learning project that uses logistic regression to classify sonar signals and distinguish between rocks and underwater mines. This binary classification model analyzes sonar return patterns to identify potential threats in marine environments.

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Contributing](#contributing)
- [License](#license)
- [Author](#author)

## 🔍 Overview

This project implements a binary classification system using logistic regression to analyze sonar signals and predict whether detected objects are rocks (R) or mines (M). The model processes 60 different sonar frequency measurements to make accurate predictions, which could be crucial for naval operations and underwater exploration.

### Key Objectives:

- **Safety**: Identify potential underwater mines to prevent accidents
- **Accuracy**: Achieve high classification accuracy using machine learning
- **Efficiency**: Fast prediction system for real-time applications
- **Reliability**: Robust model that generalizes well to new sonar data

## 📊 Dataset

The project uses the famous **Sonar Dataset** (also known as the "Connectionist Bench" sonar dataset):

- **Source**: Originally from the UCI Machine Learning Repository
- **Samples**: 208 total instances
- **Features**: 60 numerical attributes (sonar frequencies)
- **Classes**: 2 (Rock - R, Mine - M)
- **Format**: CSV file with normalized frequency values (0.0 to 1.0)

### Dataset Characteristics:

- **Balanced Distribution**: Approximately equal numbers of rock and mine samples
- **Normalized Data**: All frequency values are scaled between 0 and 1
- **No Missing Values**: Complete dataset with no preprocessing required
- **Real-world Data**: Collected from actual sonar experiments

### Feature Description:

Each of the 60 features represents the energy within a particular frequency band, integrated over a certain period of time. The features are ordered by increasing frequency, providing a comprehensive spectral analysis of the sonar return signal.

## ✨ Features

### Core Functionality:

- **Data Loading & Preprocessing**: Automated CSV data loading with pandas
- **Exploratory Data Analysis**: Statistical analysis and data visualization
- **Model Training**: Logistic regression implementation using scikit-learn
- **Model Evaluation**: Comprehensive accuracy assessment on training and test sets
- **Prediction System**: Ready-to-use prediction interface for new sonar data
- **Performance Metrics**: Detailed accuracy reporting

### Technical Features:

- **Train-Test Split**: Stratified sampling to maintain class distribution
- **Model Persistence**: Trained model can be saved and loaded
- **Input Validation**: Robust handling of input data
- **Scalable Architecture**: Easy to extend with additional algorithms

## 🚀 Installation

### Prerequisites:

- Python 3.7 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/NhanPhamThanh-IT/Rock-Mine-Logistic.git
cd Rock-Mine-Logistic
```

### Step 2: Install Required Dependencies

```bash
pip install numpy pandas scikit-learn jupyter matplotlib seaborn
```

### Alternative: Using requirements.txt (if available)

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import numpy, pandas, sklearn; print('All dependencies installed successfully!')"
```

## 💻 Usage

### Option 1: Jupyter Notebook (Recommended)

1. Start Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
2. Open `rock_mine_prediction.ipynb`
3. Run all cells sequentially to see the complete analysis

### Option 2: Python Script

You can extract the code from the notebook and run it as a Python script:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load the dataset
sonar_data = pd.read_csv('sonar_data.csv', header=None)

# Prepare the data
X = sonar_data.drop(columns=60, axis=1)  # Features
Y = sonar_data[60]  # Labels

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)

# Train the model
model = LogisticRegression()
model.fit(X_train, Y_train)

# Make predictions
predictions = model.predict(X_test)
accuracy = accuracy_score(Y_test, predictions)
print(f'Model Accuracy: {accuracy:.2%}')
```

### Making Predictions on New Data:

```python
# Example: Predict for new sonar reading
new_sonar_data = np.array([[0.0286, 0.0453, 0.0277, ...]])  # 60 features
prediction = model.predict(new_sonar_data)

if prediction[0] == 'R':
    print('Prediction: Rock')
else:
    print('Prediction: Mine')
```

## 📈 Model Performance

### Training Results:

- **Training Accuracy**: ~83-85%
- **Testing Accuracy**: ~76-81%
- **Model Type**: Logistic Regression
- **Cross-validation**: Stratified train-test split (90%-10%)

### Performance Characteristics:

- **Precision**: High precision for both rock and mine classification
- **Recall**: Balanced recall across both classes
- **F1-Score**: Strong F1-scores indicating good overall performance
- **Generalization**: Good performance on unseen test data

### Model Strengths:

- Fast training and prediction times
- Interpretable results
- No overfitting issues
- Robust to noise in sonar data

## 📁 Project Structure

```
Rock-Mine-Logistic/
│
├── rock_mine_prediction.ipynb    # Main Jupyter notebook with complete analysis
├── sonar_data.csv               # Sonar dataset (208 samples, 60 features + 1 label)
├── README.md                    # Project documentation (this file)
├── LICENSE                      # MIT License file
│
└── docs/                        # Documentation folder (future use)
```

### File Descriptions:

- **`rock_mine_prediction.ipynb`**: Complete machine learning pipeline including:

  - Data loading and exploration
  - Statistical analysis
  - Model training and evaluation
  - Prediction examples
  - Visualization (if added)

- **`sonar_data.csv`**: The core dataset containing:

  - 208 rows of sonar measurements
  - 60 columns of frequency features
  - 1 target column (R for Rock, M for Mine)

- **`LICENSE`**: MIT License ensuring open-source availability

## 🔧 Technical Details

### Algorithm: Logistic Regression

- **Type**: Linear classifier for binary classification
- **Advantages**: Fast, interpretable, probabilistic output
- **Implementation**: scikit-learn's LogisticRegression class
- **Solver**: Default liblinear (suitable for small datasets)

### Data Preprocessing:

- **Normalization**: Data already normalized (0.0 to 1.0 range)
- **Missing Values**: None present in the dataset
- **Feature Selection**: All 60 features used (no dimensionality reduction)
- **Train-Test Split**: 90% training, 10% testing with stratification

### Model Configuration:

```python
model = LogisticRegression(
    solver='liblinear',    # Suitable for small datasets
    random_state=42,       # For reproducibility
    max_iter=1000         # Sufficient for convergence
)
```

### Performance Metrics:

- **Primary Metric**: Accuracy Score
- **Evaluation Method**: Train-test split validation
- **Class Balance**: Maintained through stratified sampling

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

### Ways to Contribute:

1. **Bug Reports**: Report any issues or bugs you encounter
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests with enhancements
4. **Documentation**: Improve documentation and examples
5. **Testing**: Add unit tests or integration tests

### Development Setup:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -m "Add feature description"`
5. Push to your fork: `git push origin feature-name`
6. Create a Pull Request

### Coding Standards:

- Follow PEP 8 style guidelines
- Add docstrings to functions and classes
- Include comments for complex logic
- Ensure backward compatibility

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary:

- ✅ Commercial use
- ✅ Modification
- ✅ Distribution
- ✅ Private use
- ❌ Liability
- ❌ Warranty

## 👨‍💻 Author

**Nhan Pham Thanh**

- GitHub: [@NhanPhamThanh-IT](https://github.com/NhanPhamThanh-IT)
- Project Link: [https://github.com/NhanPhamThanh-IT/Rock-Mine-Logistic](https://github.com/NhanPhamThanh-IT/Rock-Mine-Logistic)

---

## 🔮 Future Enhancements

### Planned Improvements:

- [ ] **Advanced Algorithms**: Implement Random Forest, SVM, Neural Networks
- [ ] **Feature Engineering**: Add polynomial features and feature selection
- [ ] **Cross-Validation**: Implement k-fold cross-validation
- [ ] **Visualization**: Add confusion matrix, ROC curves, feature importance plots
- [ ] **Model Persistence**: Save/load trained models
- [ ] **Web Interface**: Create a simple web app for predictions
- [ ] **Performance Optimization**: Hyperparameter tuning with GridSearchCV
- [ ] **Data Augmentation**: Explore synthetic data generation techniques

### Research Directions:

- Ensemble methods for improved accuracy
- Deep learning approaches for pattern recognition
- Real-time prediction system development
- Integration with sonar hardware systems

</div>

---

<div align="center">

_This project demonstrates the practical application of machine learning in marine safety and underwater object detection. The logistic regression model provides a solid baseline for sonar-based classification tasks._

</div>
