# MU_PDS_01_Noor_10_Diabetes-Regression-Model

# Diabetes Regression Model

This project aims to predict disease progression in diabetes patients using the Diabetes Dataset from Scikit-learn. The goal is to build and evaluate a regression model using various machine learning techniques, including Linear Regression.
The dataset consists of various medical features, and the target is a continuous variable representing disease progression after one year.

## Table of Contents :

1) Project Description
2) Technologies Used
3) Installation Instructions
4) How to Use
5) Model Evaluation
6) Results and Visualizations
7) Contributors
   

## Project Description :

This project uses the Diabetes dataset from Scikit-learn to predict the progression of diabetes disease. The dataset contains features like age, sex, BMI, blood pressure, and other medical attributes.
The target variable is a continuous value that represents the disease progression after one year.

## Key steps in the project:

Load and Explore the Data: Load the diabetes dataset and perform exploratory data analysis (EDA).

Preprocess the Data: Prepare the data for training by scaling the features and splitting into training and testing sets.

Modeling: Train a Linear Regression model and evaluate its performance using metrics like Mean Squared Error (MSE) and R² Score.

Cross-validation: Evaluate the model's generalization ability using 5-fold cross-validation.

Visualization: Visualize the results using Actual vs Predicted and Residual plots to assess model performance.

## Technologies Used

Python 3.x

Scikit-learn: For machine learning and regression modeling.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Matplotlib & Seaborn: For data visualization.

Google Colab: For running the project interactively and for visualizations.

## Installation Instructions :

To run this project locally, you'll need Python 3.x installed on your system. However, since this project is designed to run on Google Colab, there’s no need for local installation of packages if you follow the instructions below.

#### Step 1: Open the Colab Notebook

To start using this project, open the Google Colab notebook diabetes_regression.ipynb. You can either:

i) Clone the GitHub repository and upload the notebook to your own Google Drive, or

ii) Directly open the notebook via this Google Colab link.

#### Step 2: Install Required Libraries

Once the notebook is open in Google Colab, you can install the required libraries by running the following command in a notebook cell (Colab already has most of these installed, but if not, you can install them manually) :

Python code

!pip install scikit-learn pandas numpy matplotlib seaborn

#### Step 3: Running the Notebook

Once all the libraries are installed, simply run the cells in the notebook to:

Load the dataset.

Preprocess the data.

Train the Linear Regression model.

Evaluate the model using metrics like MSE and R² Score.

Visualize the results using Actual vs Predicted and Residuals plots.

#### Step 4: Optional - Download Results

After running the notebook, you can save the results (such as the trained model, evaluation metrics, and plots) locally or on Google Drive.

#### How to Use :

Once you've opened the Colab notebook, the project will guide you through the following steps:

Data Loading and Exploration: The notebook loads the diabetes dataset, performs exploratory data analysis (EDA), and visualizes feature relationships using heatmaps and pairplots.

Data Preprocessing: The dataset is split into training and testing sets, and the features are scaled using StandardScaler.

Model Training: A Linear Regression model is trained on the data.

Evaluation: After training the model, its performance is evaluated using metrics such as Mean Squared Error (MSE) and R² Score.

Visualization: The actual vs predicted values and residuals are visualized to assess the model's performance.

#### Model Evaluation :

The model is evaluated using the following metrics:

I) Mean Squared Error (MSE): Measures the average squared difference between predicted and actual values.

II) R² Score: Represents the proportion of the variance in the target variable that is predictable from the features. A higher R² score indicates a better fit.

Additionally, 5-fold cross-validation is performed to evaluate the model's generalization ability across different data splits.

#### Results and Visualizations :


Actual vs Predicted Plot: A scatter plot comparing the actual disease progression values with the predicted values from the model.

#### Example of output :

1) Evaluation metrics (Mean Squared Error, R² Score)
2) Actual vs Predicted Plot
3) Residual Plot
   
#### Contributors :


Naznin Jahan Noor 




### Notes:


Google Colab: Since we are using Colab, you don’t need to set up a local environment, and everything can run directly in the cloud.

