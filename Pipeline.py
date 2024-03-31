
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import mlflow
import mlflow.sklearn

import dagshub
# Terminer le run actif s'il y en a un
if mlflow.active_run():
    mlflow.end_run()

# Définir le nom de votre nouvel experiment
experiment_name = "experimentDetec"

# Vérifier si l'expérience existe déjà
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    # Si l'expérience n'existe pas, créer une nouvelle expérience
    mlflow.create_experiment(experiment_name)

# Récupérer l'ID de l'expérience
experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

try:
    # Démarrer un run
    mlflow.start_run(experiment_id=experiment_id)

    # Récupérer les données
    data = pd.read_csv('C:\\Users\\pc\\Desktop\\AI_Night_Challenge\\merged_data.csv')

    # Séparation des fonctionnalités et de la variable cible
    X = data.drop(columns=["label"])
    y = data["label"]

    # Sélectionner uniquement les colonnes numériques
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    X_numeric = X[numeric_cols]

    # Séparation des données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_numeric , y, test_size=0.2, random_state=42)

    # Créer un pipeline avec la normalisation, la sélection de caractéristiques et le modèle
    def create_pipeline(model):
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('selector', SelectKBest(score_func=f_classif)),
            ('model', model)
        ])
        return pipeline

    # Définir les modèles à tester
    models = {
        'SVM': SVC(),
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression()
    }

    # Définir les paramètres pour la sélection de caractéristiques
    selector_params = {'selector__k': [5, 10, 15,20,23]}  # Nombre de caractéristiques à sélectionner

    # Créer un objet de validation croisée
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Pour chaque modèle, évaluez sa performance avec la validation croisée
    for name, model in models.items():
        pipeline = create_pipeline(model)

        # Recherche sur grille avec validation croisée
        with mlflow.start_run(experiment_id=experiment_id) as run:
            grid_search = GridSearchCV(pipeline, param_grid=selector_params, cv=kfold, scoring='accuracy')
            grid_search.fit(X_train, y_train)

            # Obtenir le meilleur score de validation croisée et les meilleurs paramètres
            best_score = grid_search.best_score_
            best_params = grid_search.best_params_

            # Évaluer le modèle sur l'ensemble de test
            y_pred = grid_search.predict(X_test)
            test_accuracy = accuracy_score(y_test, y_pred)

            # Enregistrer les métriques
            mlflow.log_metric("best_score", best_score)
            mlflow.log_metric("test_accuracy", test_accuracy)

            # Enregistrer les paramètres
            mlflow.log_params(best_params)

            # Enregistrer le modèle
            mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

except Exception as e:
    # Terminer le run actif en cas d'erreur
    mlflow.end_run()
    print("An error occurred:", e)
finally:
    # S'assurer que le run est terminé même en cas d'erreur
    mlflow.end_run()
