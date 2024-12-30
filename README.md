# Sentiment Analysis on Reviews

This project performs sentiment analysis on a dataset of product reviews. The analysis involves data preprocessing, feature extraction using TF-IDF, and training multiple machine learning models to classify sentiments into "Happy," "Ok," and "Unhappy."

## Table of Contents
- [Dataset](#dataset)
- [Features](#features)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Models Used](#models-used)
- [Evaluation](#evaluation)
- [Results](#results)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

## Dataset
The dataset contains the following columns:
- `brand`: The brand of the product.
- `manufacturer`: The manufacturer of the product.
- `reviews.didPurchase`: Indicates whether the user purchased the product.
- `reviews.rating`: User ratings (1 to 5).
- `reviews.text`: Text reviews provided by users.

## Features
- Data cleaning and preprocessing.
- Visualization of data distributions (ratings and purchase statuses).
- Text vectorization using TF-IDF:
  - **Word-level features**: Captures individual words' importance.
  - **Character-level features**: Captures n-grams for textual patterns.

## Exploratory Data Analysis (EDA)
- Bar plots for the distribution of ratings.
- Count plots for purchase statuses.
- Word cloud to identify frequent terms in reviews.

## Models Used
The following machine learning models were implemented:
1. **LinearSVC**: Support Vector Classifier with a linear kernel.
2. **SGDClassifier**: Stochastic Gradient Descent-based classifier.
3. **KNeighborsClassifier**: K-Nearest Neighbors for classification.
4. **LogisticRegression**: Logistic regression with balanced class weights.
5. **RandomForestClassifier**: Ensemble-based random forest classifier.
6. **MultinomialNB**: Naive Bayes for multinomial data.
7. **XGBClassifier**: Extreme Gradient Boosting for robust classification.

## Evaluation
- **Accuracy Scores**: Computed for both training and test sets.
- **Classification Reports**: Precision, recall, and F1-scores for each class.
- **Confusion Matrices**: Visualized using heatmaps for true and predicted labels.

## Results
A comparison of test accuracy for different models is visualized in a bar plot. Models like `RandomForestClassifier` and `XGBClassifier` may perform better, depending on the dataset and feature engineering.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/sentiment-analysis.git
   cd sentiment-analysis
