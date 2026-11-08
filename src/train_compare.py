"""Compare cinq familles de modèles avec recherche d'hyperparamètres."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from src.data_pipeline import X_train, preprocessor, y_train

# L'ordre des classes est fixé pour que les rappels et matrices soient lisibles.
class_labels = ["Dropout", "Enrolled", "Graduate"]

# 1. Définir les familles de modèles et les hyperparamètres à tester.
# Chaque recherche choisira les paramètres qui maximisent le F1 macro.
models_and_parameters = {
    "Régression logistique": (
        LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        ),
        {"model__C": [0.1, 1, 10]},
    ),
    "Forêt aléatoire": (
        RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        {
            "model__n_estimators": [100, 200],
            "model__max_depth": [None, 15],
            "model__min_samples_leaf": [1, 3],
        },
    ),
    "K plus proches voisins": (
        KNeighborsClassifier(),
        {
            "model__n_neighbors": [5, 11, 21],
            "model__weights": ["uniform", "distance"],
        },
    ),
    "SVM à noyau": (
        SVC(
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
        {
            "model__C": [0.1, 1, 10],
            "model__kernel": ["linear", "rbf"],
        },
    ),
    "Arbre de décision": (
        DecisionTreeClassifier(
            class_weight="balanced",
            random_state=42,
        ),
        {
            "model__max_depth": [None, 10, 20],
            "model__min_samples_leaf": [1, 5],
        },
    ),
}

# 2. Les deux validations sont stratifiées à 5 folds :
# l'interne règle les hyperparamètres, l'externe évalue sans biais la comparaison.
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=43)
comparison_rows = []

for model_name, (model, parameter_grid) in models_and_parameters.items():
    print(f"\n{'=' * 60}\n{model_name}")

    all_true_labels = []
    all_predictions = []
    all_probabilities = []
    fold_f1_scores = []
    best_parameters_by_fold = []
    search_seconds = 0.0
    prediction_seconds = 0.0

    # 3. À chaque pli externe, chercher les paramètres sur 4/5 des données,
    # puis produire des prédictions sur le cinquième pli laissé de côté.
    for fold_number, (train_indices, validation_indices) in enumerate(
        outer_cv.split(X_train, y_train),
        start=1,
    ):
        pipeline = Pipeline(
            steps=[
                ("preprocessing", preprocessor),
                ("model", model),
            ]
        )
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=parameter_grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )

        X_fold_train = X_train.iloc[train_indices]
        y_fold_train = y_train.iloc[train_indices]
        X_fold_validation = X_train.iloc[validation_indices]
        y_fold_validation = y_train.iloc[validation_indices]

        search_start = perf_counter()
        search.fit(X_fold_train, y_fold_train)
        search_seconds += perf_counter() - search_start

        prediction_start = perf_counter()
        predictions = search.predict(X_fold_validation)
        probabilities = search.predict_proba(X_fold_validation)
        prediction_seconds += perf_counter() - prediction_start

        all_true_labels.extend(y_fold_validation)
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)
        fold_f1_scores.append(f1_score(
            y_fold_validation,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0,
        ))
        best_parameters_by_fold.append(search.best_params_)

        print(f"Fold {fold_number} - meilleurs paramètres : {search.best_params_}")

    # 4. Résumer les performances obtenues sur toutes les prédictions hors pli.
    macro_f1 = f1_score(
        all_true_labels,
        all_predictions,
        labels=class_labels,
        average="macro",
        zero_division=0,
    )
    recalls = recall_score(
        all_true_labels,
        all_predictions,
        labels=class_labels,
        average=None,
        zero_division=0,
    )
    matrix = confusion_matrix(
        all_true_labels,
        all_predictions,
        labels=class_labels,
    )

    # Score de Brier multiclasses : erreur quadratique moyenne des probabilités.
    # Plus il est bas, mieux les probabilités prédites sont calibrées.
    probability_array = np.asarray(all_probabilities)
    true_array = np.asarray(all_true_labels)
    one_hot_labels = np.column_stack(
        [true_array == label for label in class_labels]
    ).astype(float)
    brier_score = np.mean(np.sum((probability_array - one_hot_labels) ** 2, axis=1))
    accuracy = accuracy_score(all_true_labels, all_predictions)

    print(f"F1 macro (moyenne des 5 folds) : {np.mean(fold_f1_scores):.3f}")
    print(f"F1 macro (toutes prédictions hors pli) : {macro_f1:.3f}")
    print("Rappel par classe :")
    for label, recall in zip(class_labels, recalls):
        print(f"  {label} : {recall:.3f}")
    print("Matrice de confusion (lignes = vraies classes, colonnes = prédites) :")
    print(matrix)
    print(f"Score de Brier multiclasses (plus bas = meilleur) : {brier_score:.3f}")
    print(f"Temps total de recherche et entraînement : {search_seconds:.1f} s")
    print(f"Temps total de prédiction sur les folds : {prediction_seconds:.3f} s")

    # 5. Ajouter une ligne de résultats pour ce modèle au tableau comparatif.
    row = {
        "modele": model_name,
        "f1_macro_moyenne_folds": np.mean(fold_f1_scores),
        "f1_macro_toutes_predictions_hors_pli": macro_f1,
        "accuracy": accuracy,
        "rappel_Dropout": recalls[0],
        "rappel_Enrolled": recalls[1],
        "rappel_Graduate": recalls[2],
        "score_brier_multiclasse": brier_score,
        "temps_recherche_entrainement_secondes": search_seconds,
        "temps_prediction_secondes": prediction_seconds,
        "meilleurs_parametres_par_fold": json.dumps(
            best_parameters_by_fold,
            ensure_ascii=False,
        ),
    }
    for true_index, true_label in enumerate(class_labels):
        for predicted_index, predicted_label in enumerate(class_labels):
            row[
                f"matrice_vrai_{true_label}_predit_{predicted_label}"
            ] = matrix[true_index, predicted_index]
    comparison_rows.append(row)

# Enregistrer le CSV dans outputs/, à la racine du projet.
output_path = Path(__file__).resolve().parent.parent / "outputs" / "model_comparison.csv"
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=comparison_rows[0].keys())
    writer.writeheader()
    writer.writerows(comparison_rows)

print(f"\nTableau comparatif enregistré dans : {output_path}")
