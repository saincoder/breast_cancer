# Breast Cancer Prediction Web App

This project is a web application developed during my internship at **EcodeCamp**. It aims to predict whether breast cancer is benign or malignant based on various input features. The app is built using Python, Streamlit, and machine learning techniques, offering a user-friendly interface for real-time predictions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Models Used](#models-used)
- [Dataset](#dataset)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **User-Friendly Interface:** Intuitive and easy-to-use interface for inputting patient data.
- **Machine Learning Models:** Implements Logistic Regression, Decision Tree, and Support Vector Machine models for prediction.
- **Real-Time Predictions:** Provides immediate feedback on the input data with predictions clearly displayed.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/breast-cancer-prediction.git
   cd breast-cancer-prediction
   ```

2. **Install the required libraries:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**

   ```bash
   streamlit run app.py
   ```

## Usage

1. Start the Streamlit application.
2. Input the required features in the sidebar form.
3. Click on the **Predict** button to get the prediction result.
4. The app will display whether the cancer is predicted to be benign or malignant.

## Models Used

- **Logistic Regression:** A linear model used for binary classification tasks.
- **Decision Tree Classifier:** A non-linear model that splits the data based on feature importance.
- **Support Vector Machine (SVM):** A powerful model for classification with a clear decision boundary.

## Dataset

The dataset used in this project is the [Breast Cancer Wisconsin (Diagnostic) Data Set](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29). It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass.

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to open an issue or submit a pull request.

## Acknowledgements

This project was developed during my internship at **EcodeCamp**. Special thanks to the mentors and the team for their guidance and support.
