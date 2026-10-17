# Machine-Learning-Project

# Crop Yield Prediction Using Machine Learning Regression

## Project Overview

This project develops and compares multiple machine learning regression algorithms
to predict crop yield using agricultural and climatic features.

## Dataset

The dataset contains information about:

- Crop type
- Crop year
- Season
- State
- Cultivated area
- Annual rainfall
- Fertilizer usage
- Pesticide usage
- Average temperature
- Minimum and maximum temperature
- Crop yield

## Problem Statement

Crop yield depends on agricultural and environmental factors such as rainfall,
fertilizer usage, pesticide usage, cultivated area and temperature.

The objective of this project is to predict crop yield and compare ten regression
algorithms using the same train-test split and evaluation metrics.

## Target Variable

`Yield`

## Regression Algorithms

1. Linear Regression
2. Ridge Regression
3. Lasso Regression
4. ElasticNet Regression
5. Polynomial Regression
6. Decision Tree Regressor
7. Random Forest Regressor
8. Gradient Boosting Regressor
9. Support Vector Regressor
10. K-Nearest Neighbors Regressor

## Evaluation Metrics

- R² Score
- RMSE
- MAE
- 5-Fold Cross-Validation R²

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

# Vehicle Size Classification Using Machine Learning

## Project Overview
 
This project develops and compares ten machine learning classification algorithms
to predict a car's size class using its specifications.
 
## Dataset
 
The dataset contains information about:
 
- Make
- Model
- Year
- Engine fuel type
- Engine horsepower
- Engine cylinders
- Transmission type
- Driven wheels
- Number of doors
- Market category
- Vehicle style
- Highway MPG
- City MPG
- Popularity
- MSRP
- Vehicle size
## Problem Statement
 
Vehicle size depends on engine power, drivetrain, body style, fuel economy and brand.
 
The objective of this project is to predict vehicle size and compare ten classification
algorithms using the same train-test split and evaluation metrics.
 
## Target Variable
 
`Vehicle Size` (Compact, Midsize, Large)
 
## Classification Algorithms
 
1. Logistic Regression
2. K-Nearest Neighbors
3. Naive Bayes (Gaussian)
4. Decision Tree Classifier
5. Support Vector Machine (SVC)
6. Random Forest Classifier
7. AdaBoost Classifier
8. Gradient Boosting Classifier
9. Bagging Classifier
10. MLP Classifier
## Evaluation Metrics
 
- Accuracy
- Precision
- Recall
- F1-Score (weighted)
- Confusion Matrix
- ROC-AUC (One-vs-Rest)

## Best Model
 
Gradient Boosting (Accuracy 0.97, Weighted F1 0.97, ROC-AUC 0.995)
 
## Tools Used
 
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

# Amazon Product Segmentation Using Machine Learning Clustering

## Project Overview

This project applies two unsupervised clustering algorithms to group Amazon India
products into segments based on price, discount depth, rating and customer engagement,
and compares them using standard cluster-validation metrics.

## Dataset

The dataset contains information about:

- Product ID, name and category
- Discounted price and actual price
- Discount percentage
- Rating and rating count
- Product description
- User ID, user name, review ID, review title and review content
- Image and product links

The file has 1,465 rows (one per customer review). After keeping one row per product,
1,351 unique products remain.

## Problem Statement

Products on Amazon differ widely in price level, discounting and customer engagement,
and there are no labels that say which products behave alike.

The objective of this project is to discover natural product segments without labels
and to compare K-Means and Agglomerative Hierarchical Clustering using the same
features and validation metrics.

## Target Variable

None (unsupervised). The retail category is not used for fitting; it is used only
afterwards to validate the clusters.

Clustering features (standardised):

- `log_actual_price`
- `discount_percentage_clean`
- `rating_clean`
- `log_rating_count`

## Clustering Algorithms

1. K-Means Clustering
2. Agglomerative Hierarchical Clustering

## Evaluation Metrics

- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Elbow Curve (K-Means) and Dendrogram (Hierarchical)
- PCA 2D and t-SNE cluster visualisation

## Best Model

K-Means with k = 4 (Silhouette 0.2343, Davies-Bouldin 1.2947, Calinski-Harabasz 390.63)

## Tools Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Jupyter Notebook

streamlit app link : https://mlteam9.streamlit.app/
