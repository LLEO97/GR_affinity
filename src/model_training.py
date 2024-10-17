# After data cleaning and feature engineering, train models with the following parameters
# Create a dictionary of models
models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=2023),
        'param_grid': {
            'C': [0.1, 1, 10],
            'solver': ['liblinear', 'saga']
        }
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=2023),
        'param_grid': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 10],
            'min_samples_split': [2, 4, 6]
        }
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=2023),
        'param_grid': {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    },
    'K-Nearest Neighbors': {
        'model': KNeighborsClassifier(),
        'param_grid': {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=2023),
        'param_grid': {
            'n_estimators': [25, 50, 100],
            'learning_rate': [0.01, 0.1, 0.3]
        }
    },
    'Multi-Layer Perceptron': {
        'model': MLPClassifier(random_state=2023),
        'param_grid': {
            'hidden_layer_sizes': [(25, 25), (50, 50), (100, 100)],
            'activation': ['relu', 'tanh', 'logistic'],
            'alpha': [0.0001, 0.001, 0.01, 0.1]
        }
    }
}

results = {}
# Train and evaluate models
for model_name, model_info in models.items():
    print(f'Training {model_name}...')
    model = model_info['model']
    param_grid = model_info['param_grid']
    for input_type, input_data in zip(['all', 'RFE', 'kbest', 'MI', 'PI'], [X_scaled, X_rfecv, X_kbest, X_mi, X_pi]):
        X_train, X_test, y_train, y_test = train_test_split(input_data, y, test_size=0.2, random_state=random_state)
        grid_search = GridSearchCV(model, param_grid, cv=KFold(10, shuffle=True, random_state=random_state))
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        y_pred = best_model.predict(input_data)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        total_accuracy = accuracy_score(y, y_pred)  # Calculate overall accuracy
......
# For the complete code, please contact yongchengl@unr.edu
