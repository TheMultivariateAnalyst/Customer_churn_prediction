# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv('OnlineRetail_processed.csv')

# Split data into training and testing sets
X = df.drop(['CustomerID', 'Churn'], axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Handle missing values using SimpleImputer
imp = SimpleImputer(strategy='mean')
X_train = imp.fit_transform(X_train)
X_test = imp.transform(X_test)

# Train and optimize logistic regression model
logreg = LogisticRegression(max_iter=5000)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
grid_search = GridSearchCV(logreg, param_grid, cv=5)
grid_search.fit(X_train, y_train)
logreg_best = grid_search.best_estimator_
y_pred_logreg = logreg_best.predict(X_test)
logreg_accuracy = accuracy_score(y_test, y_pred_logreg)
print('Logistic Regression Accuracy:', logreg_accuracy)
print(classification_report(y_test, y_pred_logreg))