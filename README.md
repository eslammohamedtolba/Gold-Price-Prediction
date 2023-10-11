# Gold-Price-Prediction
This repository contains a Python script that uses machine learning to predict gold prices. 
The model is based on a dataset provided in the `gld_price_data.csv` file.

## Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.5 or later
- Jupyter Notebook or any Python IDE
- The required Python libraries, which can be installed using `pip`

## Project Overview
The goal of this project is to predict gold prices using a Random Forest Regressor model. 
The following steps are involved in the process:
1. **Data Preprocessing**:
   - Import required modules, including pandas, matplotlib, seaborn, scikit-learn, and more.
   - Load the gold price dataset from `gld_price_data.csv`.
   - Display the first few rows of the dataset and its shape.
   - Check for any missing values in the dataset.
2. **Data Visualization**:
   - Create a correlation matrix and visualize it using a heatmap to identify relationships between variables.
   - Check the distribution of gold prices in the dataset.
3. **Data Splitting**:
   - Split the data into input features (X) and target variable (Y).
   - Further split the data into training and testing sets using a 70-30 split ratio.
4. **Model Training**:
   - Create a Random Forest Regressor model.
   - Train the model using the training data.
5. **Model Evaluation**:
   - Use the trained model to make predictions on both the training and testing data.
   - Calculate the R-squared (R2) score to measure the accuracy of the predictions.
6. **Visualization**:
   - Visualize the difference between predicted and actual values for both the training and testing datasets.

## Model Accuracy
The accuracy of the test predictions is approximately 99%, indicating a strong performance of the model in predicting gold prices.

## Dataset
The dataset used for this project is included in the repository as `gld_price_data.csv`.
It contains historical data on gold prices and various other factors that might affect gold prices.
Feel free to explore the code provided in the repository to see how the gold price prediction model was implemented.

## Contributions

Contributions and suggestions from the community are highly encouraged. 
If you have any ideas for improvements, bug fixes, or new features, please feel free to contribute. 
