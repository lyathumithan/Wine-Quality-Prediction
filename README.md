# Wine Quality Prediction

## Project Description

This project uses Machine Learning to predict wine quality based on its chemical properties.

## Dataset

The dataset used in this project is the Red Wine Quality dataset from the UCI Machine Learning Repository.

It contains 1599 red wine samples with 11 chemical properties and a quality score.

The target variable is `quality`.

## Technologies Used

- Python
- Pandas
- Matplotlib
- Scikit-learn
- VS Code

## Project Workflow

1. Load the Wine Quality dataset
2. Explore and understand the data
3. Perform Exploratory Data Analysis (EDA)
4. Separate features (X) and target (y)
5. Split the data into training and testing sets
6. Train a Random Forest Regression model
7. Make predictions
8. Evaluate the model using MAE and R² Score
9. Analyze feature importance
10. Predict the quality of a new wine

## Results

The Random Forest Regression model achieved:

- Mean Absolute Error (MAE): 0.42
- R² Score: 0.54

The model can predict wine quality with an average error of approximately 0.42 quality points.

## How to Run

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```
4. Run the project:
```bash
python data/main.py
```