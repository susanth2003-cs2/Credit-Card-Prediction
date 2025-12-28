# Credit-Card-Prediction
💳 Credit Card Default Prediction – Machine Learning Pipeline
This project implements an end-to-end Machine Learning pipeline to predict whether a customer is likely to default on a credit card payment. It covers the complete lifecycle of an ML project, from data preprocessing to model deployment.

📌 Project Overview
Credit card default prediction is a binary classification problem widely used in banking and financial risk analysis. The goal is to identify high-risk customers so that financial institutions can make better lending decisions.

This project demonstrates:

Proper feature engineering & selection
Handling imbalanced datasets
Training and evaluating ML models
Saving and deploying the best model using a pipeline-based approach
🧠 ML Pipeline Architecture
Data Collection
      ↓
Data Cleaning
      ↓
Feature Engineering
  - Handle missing values
  - Categorical → Numerical
  - Variable transformation
  - Outlier treatment
      ↓
Feature Selection
  - Correlation analysis
  - Hypothesis testing
  - Filter methods
      ↓
Data Balancing
      ↓
Feature Scaling
      ↓
Model Training
      ↓
Model Evaluation (AUC-ROC)
      ↓
Hyperparameter Tuning
      ↓
Best Model Selection
      ↓
Model Saving
      ↓
Frontend Integration
      ↓
Cloud Deployment
⚙️ Technologies Used
Programming Language: Python

Libraries:

NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
Framework: Flask (for frontend integration)

Model Storage: Pickle (.pkl)

Deployment: Cloud-based deployment

🧪 Dataset Description
The dataset contains customer credit-related information such as:

Demographic details
Credit limit
Repayment history
Bill amounts
Payment amounts
Target Variable:

0 → No Default
1 → Default
🔧 Feature Engineering
Feature engineering improves model performance by transforming raw data into meaningful features:

Handling missing values
Encoding categorical variables
Scaling numerical features
Outlier detection and treatment
🎯 Feature Selection
Only the most relevant features are selected using:

Correlation analysis
Hypothesis testing
Filter-based methods
This helps in reducing overfitting and improving model efficiency.

⚖️ Data Balancing
Since credit default datasets are usually imbalanced, balancing techniques are applied to ensure fair learning by the model.

📊 Model Training & Evaluation
Multiple ML models are trained and evaluated.

Evaluation Metric Used:

AUC-ROC Curve (preferred for imbalanced classification problems)
The model with the best ROC-AUC score is selected as the final model.

🔍 Hyperparameter Tuning
Hyperparameter tuning is performed to improve model performance and generalization.

💾 Model Saving
The trained model is saved using Pickle for reuse during deployment.

with open('Credit_Card.pkl', 'wb') as f:
    pickle.dump(model, f)
🌐 Frontend & Deployment
A simple Flask-based frontend allows users to input customer details
The trained model predicts default probability
The application is deployed on the cloud for real-time access
📌 Use Cases
Banking risk analysis
Credit approval systems
Financial fraud prevention
🚀 Future Enhancements
Add more advanced models (XGBoost, LightGBM)
Model monitoring and retraining
Improved UI/UX
API-based deployment
⭐ Acknowledgment
This project was built to demonstrate practical implementation of a real-world Machine Learning pipeline in finance.

If you find this project useful, feel free to ⭐ star the repository!

