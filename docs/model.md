<div align="justify">

# <div align="center">Logistic Regression Model Documentation</div>

## Table of Contents

- [Overview](#overview)
- [Mathematical Foundation](#mathematical-foundation)
- [Model Architecture](#model-architecture)
- [Implementation Details](#implementation-details)
- [Training Process](#training-process)
- [Model Evaluation](#model-evaluation)
- [Performance Metrics](#performance-metrics)
- [Advantages and Limitations](#advantages-and-limitations)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

This document provides comprehensive documentation for the **Logistic Regression** model used in the Rock vs Mine prediction system. Logistic regression is a statistical method used for binary classification problems, making it ideal for distinguishing between rocks and mines based on sonar signal patterns.

### Why Logistic Regression?

Logistic regression was chosen for this project due to several key advantages:

1. **Binary Classification Excellence**: Perfectly suited for our two-class problem (Rock vs Mine)
2. **Probabilistic Output**: Provides probability estimates for predictions
3. **Interpretability**: Easy to understand and explain model decisions
4. **Computational Efficiency**: Fast training and prediction times
5. **Robust Performance**: Works well with linearly separable data
6. **No Assumptions About Feature Distribution**: Unlike naive Bayes, doesn't assume feature independence

## Mathematical Foundation

### The Logistic Function (Sigmoid)

The core of logistic regression is the **sigmoid function**, which maps any real number to a value between 0 and 1:

```
σ(z) = 1 / (1 + e^(-z))
```

Where:

- `z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ` (linear combination)
- `β₀` = intercept (bias term)
- `β₁, β₂, ..., βₙ` = feature coefficients
- `x₁, x₂, ..., xₙ` = input features (60 sonar frequencies)

### Probability Interpretation

For our binary classification:

- `P(y = Mine | x) = σ(z)` = probability of being a mine
- `P(y = Rock | x) = 1 - σ(z)` = probability of being a rock

### Decision Boundary

The model makes predictions based on a threshold (typically 0.5):

- If `P(y = Mine | x) ≥ 0.5` → Predict "Mine" (M)
- If `P(y = Mine | x) < 0.5` → Predict "Rock" (R)

### Cost Function (Log-Likelihood)

The model is trained by minimizing the **logistic loss** (cross-entropy):

```
J(β) = -1/m * Σ[y*log(h(x)) + (1-y)*log(1-h(x))]
```

Where:

- `m` = number of training samples
- `y` = actual label (1 for Mine, 0 for Rock)
- `h(x) = σ(βᵀx)` = predicted probability

## Model Architecture

### Input Layer

- **Features**: 60 numerical attributes
- **Feature Range**: [0.0, 1.0] (normalized sonar frequencies)
- **Feature Description**: Energy levels at different frequency bands from sonar returns

### Processing

- **Linear Combination**: `z = β₀ + Σ(βᵢ * xᵢ)` for i = 1 to 60
- **Activation**: Sigmoid function applied to linear combination
- **Output**: Probability score between 0 and 1

### Output Layer

- **Binary Classification**: Mine (M) or Rock (R)
- **Prediction Rule**: Threshold-based decision (default: 0.5)

## Implementation Details

### Scikit-Learn Configuration

```python
from sklearn.linear_model import LogisticRegression

# Model initialization with default parameters
model = LogisticRegression()

# Common parameters that can be tuned:
model = LogisticRegression(
    penalty='l2',           # Regularization type (l1, l2, elasticnet, none)
    C=1.0,                  # Inverse regularization strength
    fit_intercept=True,     # Whether to calculate intercept
    class_weight=None,      # Weights associated with classes
    random_state=42,        # Random seed for reproducibility
    solver='lbfgs',         # Optimization algorithm
    max_iter=100,           # Maximum iterations for convergence
    tol=1e-4               # Tolerance for stopping criteria
)
```

### Feature Engineering

In our implementation:

1. **No additional preprocessing required** - data is already normalized
2. **60 features** representing different frequency components
3. **Feature range**: [0.0, 1.0] ensures numerical stability
4. **No feature scaling needed** due to uniform range

### Data Splitting Strategy

```python
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=0.1,      # 10% for testing, 90% for training
    stratify=Y,         # Maintain class distribution
    random_state=1      # Reproducible results
)
```

**Stratification** ensures both training and test sets have similar proportions of rocks and mines.

## Training Process

### 1. Data Preparation

- Load sonar dataset (208 samples, 60 features)
- Separate features (X) from labels (Y)
- Split into training (187 samples) and testing (21 samples)

### 2. Model Training

```python
model = LogisticRegression()
model.fit(X_train, Y_train)
```

### 3. Training Algorithm

The model uses **gradient descent** (or its variants) to minimize the cost function:

1. **Initialize** parameters β randomly
2. **Forward pass**: Calculate predictions using current parameters
3. **Calculate loss**: Compute logistic loss
4. **Backward pass**: Calculate gradients of loss with respect to parameters
5. **Update parameters**: β = β - α \* ∇J(β) where α is learning rate
6. **Repeat** until convergence or maximum iterations reached

### 4. Convergence Criteria

- **Maximum iterations**: Default 100 (usually sufficient)
- **Tolerance**: Default 1e-4 (change in cost function)
- **Gradient norm**: When gradients become very small

## Model Evaluation

### Training Performance

Based on the implementation, the model achieves:

- **Training Accuracy**: Typically 80-85%
- **Convergence**: Usually achieved within 10-20 iterations

### Testing Performance

- **Test Accuracy**: Typically 76-81%
- **Generalization**: Good performance on unseen data

### Performance Analysis

#### Confusion Matrix Components

For binary classification, we can analyze:

| Actual/Predicted | Rock (R) | Mine (M) |
| ---------------- | -------- | -------- |
| **Rock (R)**     | TN       | FP       |
| **Mine (M)**     | FN       | TP       |

Where:

- **TP** (True Positives): Correctly identified mines
- **TN** (True Negatives): Correctly identified rocks
- **FP** (False Positives): Rocks misclassified as mines
- **FN** (False Negatives): Mines misclassified as rocks

#### Key Metrics

1. **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`

   - Overall correctness of the model
   - Target: 75-85% for this dataset

2. **Precision**: `TP / (TP + FP)`

   - Of all predicted mines, how many are actually mines
   - Important for avoiding false alarms

3. **Recall (Sensitivity)**: `TP / (TP + FN)`

   - Of all actual mines, how many were correctly identified
   - Critical for safety (detecting actual mines)

4. **Specificity**: `TN / (TN + FP)`
   - Of all actual rocks, how many were correctly identified
   - Important for operational efficiency

## Performance Metrics

### Expected Performance Range

Based on the sonar dataset characteristics:

| Metric    | Expected Range | Target |
| --------- | -------------- | ------ |
| Accuracy  | 75-85%         | 80%    |
| Precision | 70-85%         | 78%    |
| Recall    | 70-85%         | 78%    |
| F1-Score  | 72-85%         | 78%    |

### Probability Calibration

The model outputs well-calibrated probabilities, meaning:

- Probability 0.8 → About 80% chance of being correct
- Probability 0.6 → About 60% chance of being correct
- This enables confidence-based decision making

### ROC Curve Analysis

A well-performing logistic regression model should have:

- **AUC (Area Under Curve)**: 0.80-0.90
- **Balanced performance** across different threshold values
- **Good separation** between classes

## Advantages and Limitations

### Advantages

1. **Simplicity and Speed**

   - Fast training and prediction
   - Minimal computational requirements
   - Easy to deploy in production

2. **Interpretability**

   - Clear understanding of feature importance through coefficients
   - Probabilistic output for decision confidence
   - No "black box" complexity

3. **Robustness**

   - Less prone to overfitting with proper regularization
   - Stable performance across different datasets
   - Good baseline for comparison with complex models

4. **Probabilistic Output**
   - Provides confidence scores
   - Enables threshold tuning for different use cases
   - Supports risk-based decision making

### Limitations

1. **Linear Decision Boundary**

   - Assumes linear relationship between features and log-odds
   - May struggle with complex, non-linear patterns
   - Limited by linear separability assumption

2. **Feature Independence**

   - Assumes features contribute independently
   - May not capture complex feature interactions
   - Potential for reduced performance with highly correlated features

3. **Sensitivity to Outliers**

   - Extreme values can influence model significantly
   - May require outlier detection and handling
   - Robust preprocessing recommended

4. **Scale Sensitivity**
   - Performance can be affected by feature scaling
   - Regularization parameter tuning important
   - May need feature normalization

## Best Practices

### 1. Data Preprocessing

```python
# Already normalized in our dataset, but generally:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### 2. Cross-Validation

```python
from sklearn.model_selection import cross_val_score

# 5-fold cross-validation
cv_scores = cross_val_score(model, X, Y, cv=5, scoring='accuracy')
print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
```

### 3. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear', 'lbfgs']
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train, Y_train)
```

### 4. Feature Importance Analysis

```python
# Get feature coefficients
feature_importance = abs(model.coef_[0])
important_features = sorted(zip(feature_importance, range(60)), reverse=True)

print("Top 10 most important features:")
for importance, feature_idx in important_features[:10]:
    print(f"Feature {feature_idx}: {importance:.4f}")
```

### 5. Prediction with Confidence

```python
# Get prediction probabilities
probabilities = model.predict_proba(X_test)
predictions = model.predict(X_test)

for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
    confidence = max(prob)
    print(f"Sample {i}: Predicted {pred} with {confidence:.2f} confidence")
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Poor Convergence

**Symptoms**: Model doesn't converge, warnings about max iterations
**Solutions**:

```python
# Increase max iterations
model = LogisticRegression(max_iter=1000)

# Try different solver
model = LogisticRegression(solver='saga', max_iter=1000)

# Scale features if not already done
from sklearn.preprocessing import StandardScaler
```

#### 2. Overfitting

**Symptoms**: High training accuracy, low test accuracy
**Solutions**:

```python
# Increase regularization (decrease C)
model = LogisticRegression(C=0.1)

# Use L1 regularization for feature selection
model = LogisticRegression(penalty='l1', solver='liblinear')
```

#### 3. Class Imbalance

**Symptoms**: Model biased toward majority class
**Solutions**:

```python
# Use class weights
model = LogisticRegression(class_weight='balanced')

# Manual class weights
model = LogisticRegression(class_weight={0: 1, 1: 2})
```

#### 4. Feature Scaling Issues

**Symptoms**: Slow convergence, numerical instability
**Solutions**:

```python
# Standardize features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Performance Optimization

#### 1. Solver Selection

- **lbfgs**: Good for small datasets (like ours)
- **liblinear**: Good for small datasets with L1 regularization
- **saga**: Good for large datasets
- **newton-cg**: Good for L2 regularization

#### 2. Regularization Tuning

```python
# L2 regularization (Ridge) - default
model = LogisticRegression(penalty='l2', C=1.0)

# L1 regularization (Lasso) - for feature selection
model = LogisticRegression(penalty='l1', solver='liblinear', C=1.0)

# Elastic Net - combination of L1 and L2
model = LogisticRegression(penalty='elasticnet', solver='saga',
                          l1_ratio=0.5, C=1.0)
```

#### 3. Threshold Optimization

```python
from sklearn.metrics import precision_recall_curve

# Find optimal threshold for F1 score
precision, recall, thresholds = precision_recall_curve(Y_test, probabilities[:, 1])
f1_scores = 2 * (precision * recall) / (precision + recall)
optimal_threshold = thresholds[np.argmax(f1_scores)]

print(f"Optimal threshold: {optimal_threshold:.3f}")
```

### Model Validation

#### 1. Learning Curves

```python
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

train_sizes, train_scores, val_scores = learning_curve(
    model, X, Y, cv=5, train_sizes=np.linspace(0.1, 1.0, 10)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label='Training')
plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

#### 2. Validation Curves

```python
from sklearn.model_selection import validation_curve

param_range = [0.01, 0.1, 1, 10, 100]
train_scores, val_scores = validation_curve(
    LogisticRegression(), X, Y, param_name='C',
    param_range=param_range, cv=5
)

plt.plot(param_range, train_scores.mean(axis=1), label='Training')
plt.plot(param_range, val_scores.mean(axis=1), label='Validation')
plt.xlabel('C Parameter')
plt.ylabel('Accuracy')
plt.xscale('log')
plt.legend()
plt.show()
```

---

## Conclusion

The logistic regression model provides an excellent foundation for the rock vs mine classification task. Its combination of simplicity, interpretability, and solid performance makes it an ideal choice for this safety-critical application. The model's probabilistic output enables flexible threshold tuning based on operational requirements, whether prioritizing mine detection (high recall) or minimizing false alarms (high precision).

For future improvements, consider:

1. **Feature engineering**: Creating interaction terms or polynomial features
2. **Ensemble methods**: Combining with other algorithms
3. **Advanced regularization**: Exploring elastic net parameters
4. **Threshold optimization**: Tuning for specific operational requirements

The current implementation provides a robust baseline that can be easily understood, maintained, and deployed in production environments.

</div>
