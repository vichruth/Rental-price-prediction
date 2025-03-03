# Rental Price Prediction

## Overview
This project aims to analyze rental prices using Exploratory Data Analysis (EDA) and train a Machine Learning model to predict rental prices based on various factors such as location, property size, number of bedrooms, and amenities. The model utilizes the XGBoost Regressor for accurate predictions.

## Features
- **Exploratory Data Analysis (EDA):**
  - Distribution of rental prices
  - Correlation analysis between features
  - Feature importance visualization
- **Preprocessing:**
  - Handling missing values
  - Encoding categorical variables
  - Standardizing numerical data
- **Model Training & Evaluation:**
  - Splitting data into train and test sets
  - Training an XGBoost regression model
  - Evaluating using RMSE and R² score

## Installation & Setup
1. **Clone the repository:**
   ```sh
   git clone https://github.com/<YOUR_GITHUB_USERNAME>/<REPO_NAME>.git
   cd <REPO_NAME>
   ```
2. **Create a virtual environment (optional but recommended):**
   ```sh
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate  # For Windows
   ```
3. **Install required dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

## Usage
1. **Run the script:**
   ```sh
   python RentalPricePrediction.py
   ```
2. The script will:
   - Load and preprocess the dataset
   - Perform EDA
   - Train an XGBoost model
   - Evaluate and visualize results

## Dependencies
- Python 3.7+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost

