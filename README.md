# HEART-DISEASE
Overview
This guide walks through the process of training Decision Tree and Random Forest classifiers, analyzing overfitting, interpreting feature importance, and evaluating models using cross-validation. The dataset used can be any classification dataset, such as the Heart Disease Dataset.

Requirements
Python 3.x

Libraries: numpy, pandas, matplotlib, seaborn, scikit-learn

Steps
1. Train a Decision Tree Classifier and Visualize the Tree
Load and preprocess the dataset.

Split into training and test sets.

Train a DecisionTreeClassifier using sklearn.

Visualize the tree using plot_tree().

2. Analyze Overfitting and Control Tree Depth
Overfitting occurs when the model memorizes data rather than generalizing.

Experiment with different values of max_depth to control complexity.

Plot train vs. test accuracy to find an optimal depth.

Train a Random Forest and Compare Accuracy
Train a RandomForestClassifier with multiple trees.

Compare test accuracy with Decision Tree results.

Random Forest usually performs better by reducing variance.

4. Interpret Feature Importances
Use feature_importances_ from RandomForestClassifier.

Visualize feature importance using a bar plot.

Important features help in understanding predictive power.

5. Evaluate Using Cross-Validation
Apply cross-validation (CV) to ensure model reliability.

Use cross_val_score() with 5-fold CV.

Compare mean CV accuracy across models.

Dataset
You can use any classification dataset.

Example: Heart Disease Dataset

Available at Kaggle

Run the code
heart-disease.py

Results and Observations
Decision Trees may overfit without depth control.

Random Forest provides higher accuracy and stability.

Feature importance helps interpret the model.

Cross-validation ensures robust evaluation.

References
Scikit-Learn Documentation
Heart Disease Dataset
