"""Entraîne et évalue le Random Forest final, puis exporte les cas du groupe."""

import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.data_pipeline import X_test, X_train, preprocessor, y_test, y_train


CLASS_LABELS = ["Dropout", "Enrolled", "Graduate"]
# La recherche teste les mêmes paramètres que la comparaison des modèles.
# Le F1 macro donne le même poids à chacune des trois classes.
PARAMETER_GRID = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 15],
    "model__min_samples_leaf": [1, 3],
}


def build_test_instances(model: Pipeline) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sélectionne 15 cas à partager et 3 cas distincts pour la mise au point."""
    probabilities = model.predict_proba(X_test)
    model_classes = model.named_steps["model"].classes_
    # Remet les colonnes de probabilités dans l'ordre explicite de CLASS_LABELS.
    probabilities = probabilities[:, [list(model_classes).index(c) for c in CLASS_LABELS]]
    predictions = model.predict(X_test)
    sorted_probabilities = np.sort(probabilities, axis=1)

    instances = X_test.copy()
    instances.insert(
        0,
        "instance_id",
        [f"test_{row_index:04d}" for row_index in X_test.index],
    )
    instances["true_class"] = y_test
    instances["predicted_class"] = predictions
    for class_index, label in enumerate(CLASS_LABELS):
        instances[f"probability_{label}"] = probabilities[:, class_index]
    instances["confidence"] = sorted_probabilities[:, -1]
    # Un petit écart entre les deux meilleures probabilités signale une
    # prédiction incertaine : le modèle hésite entre deux classes.
    instances["probability_margin"] = (
        sorted_probabilities[:, -1] - sorted_probabilities[:, -2]
    )
    instances["is_correct"] = instances["true_class"] == instances["predicted_class"]
    instances["is_uncertain"] = False

    # Le cas le plus incertain est celui qui a le plus petit écart de probabilités.
    uncertain_index = instances["probability_margin"].idxmin()
    selected_indices = [uncertain_index]

    # Ajoute les erreurs dont les probabilités sont les plus proches, jusqu'à en
    # avoir au moins deux dans le lot partagé.
    incorrect = instances.loc[~instances["is_correct"]].sort_values(
        ["probability_margin", "instance_id"]
    )
    for row_index in incorrect.index:
        if sum(not instances.loc[i, "is_correct"] for i in selected_indices) >= 2:
            break
        if row_index not in selected_indices:
            selected_indices.append(row_index)

    if sum(not instances.loc[i, "is_correct"] for i in selected_indices) < 2:
        raise RuntimeError(
            "Le jeu de test ne fournit pas deux erreurs distinctes à inclure."
        )

    # Complète la sélection à cinq cas par classe réelle, sans doublons.
    for label in CLASS_LABELS:
        class_indices = instances.index[instances["true_class"] == label]
        already_selected = [
            row_index
            for row_index in selected_indices
            if instances.loc[row_index, "true_class"] == label
        ]
        candidates = instances.loc[class_indices].sort_values(
            ["probability_margin", "instance_id"]
        )
        for row_index in candidates.index:
            if len(already_selected) >= 5:
                break
            if row_index not in selected_indices:
                selected_indices.append(row_index)
                already_selected.append(row_index)

    if len(selected_indices) != 15:
        raise RuntimeError(
            f"La sélection principale contient {len(selected_indices)} instances au lieu de 15."
        )

    selected = instances.loc[selected_indices].copy()
    selected.loc[uncertain_index, "is_uncertain"] = True
    if set(selected["true_class"]) != set(CLASS_LABELS):
        raise RuntimeError("Les trois classes ne sont pas représentées dans les 15 cas.")
    if int((~selected["is_correct"]).sum()) < 2:
        raise RuntimeError("La sélection ne contient pas au moins deux erreurs.")

    # Choisit ensuite un cas de chaque classe pour la mise au point, en excluant
    # les 15 cas déjà retenus.
    remaining = instances.drop(index=selected_indices)
    debug_rows = []
    for label in CLASS_LABELS:
        candidates = remaining.loc[remaining["true_class"] == label].sort_values(
            ["probability_margin", "instance_id"]
        )
        if candidates.empty:
            raise RuntimeError(f"Aucune instance distincte disponible pour {label}.")
        debug_rows.append(candidates.iloc[0])
    debugging = pd.DataFrame(debug_rows)

    return selected.reset_index(drop=True), debugging.reset_index(drop=True)


def main() -> None:
    # Le prétraitement et le modèle forment un seul pipeline, pour que le même
    # traitement soit réutilisé lors de l'entraînement et des prédictions.
    pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    cross_validation = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=43,
    )
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=PARAMETER_GRID,
        scoring="f1_macro",
        cv=cross_validation,
        n_jobs=-1,
        refit=True,
    )

    # La validation croisée choisit les paramètres uniquement sur X_train.
    # refit=True réentraîne automatiquement le meilleur pipeline sur tout X_train.
    print("Recherche des paramètres du Random Forest sur X_train...")
    search_start = perf_counter()
    search.fit(X_train, y_train)
    search_seconds = perf_counter() - search_start
    final_model = search.best_estimator_

    # X_test reste indépendant : il est utilisé seulement après le choix des
    # paramètres, pour mesurer les performances finales.
    predictions = final_model.predict(X_test)
    probabilities = final_model.predict_proba(X_test)
    model_classes = final_model.named_steps["model"].classes_
    # Aligne aussi les probabilités sur l'ordre des classes utilisé dans le rapport.
    probabilities = probabilities[:, [list(model_classes).index(c) for c in CLASS_LABELS]]
    recalls = recall_score(
        y_test,
        predictions,
        labels=CLASS_LABELS,
        average=None,
        zero_division=0,
    )
    # Le score de Brier compare les probabilités aux vraies classes encodées
    # en vecteurs indicateurs. Plus le score est bas, meilleures sont les
    # probabilités selon cette mesure.
    one_hot_labels = np.column_stack(
        [np.asarray(y_test) == label for label in CLASS_LABELS]
    ).astype(float)
    brier_score = float(
        np.mean(np.sum((probabilities - one_hot_labels) ** 2, axis=1))
    )
    results = {
        "modele": "Forêt aléatoire",
        "meilleurs_parametres": search.best_params_,
        "f1_macro_validation_croisee_train": float(search.best_score_),
        "accuracy_test": float(accuracy_score(y_test, predictions)),
        "f1_macro_test": float(
            f1_score(
                y_test,
                predictions,
                labels=CLASS_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "rappel_test_par_classe": dict(zip(CLASS_LABELS, recalls.tolist())),
        "score_brier_multiclasse_test": brier_score,
        "matrice_confusion_test": confusion_matrix(
            y_test,
            predictions,
            labels=CLASS_LABELS,
        ).tolist(),
        "rapport_test_par_classe": classification_report(
            y_test,
            predictions,
            labels=CLASS_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "temps_recherche_secondes": search_seconds,
    }

    output_directory = Path(__file__).resolve().parent.parent / "outputs"
    output_directory.mkdir(parents=True, exist_ok=True)

    # Sauvegarde le prétraitement avec le modèle afin de pouvoir prédire plus tard
    # sur des données brutes ayant les colonnes d'origine.
    model_path = output_directory / "model.joblib"
    joblib.dump(final_model, model_path)

    # Enregistre les métriques et les paramètres dans un fichier lisible par machine.
    evaluation_path = output_directory / "evaluation_results.json"
    with evaluation_path.open("w", encoding="utf-8") as results_file:
        json.dump(results, results_file, ensure_ascii=False, indent=2)

    # Exporte séparément les cas à expliquer et les cas réservés à la mise au point.
    selected, debugging = build_test_instances(final_model)
    selected_path = output_directory / "selected_instances.csv"
    debugging_path = output_directory / "debug_instances.csv"
    selected.to_csv(selected_path, index=False, encoding="utf-8-sig")
    debugging.to_csv(debugging_path, index=False, encoding="utf-8-sig")

    print(f"Meilleurs paramètres : {search.best_params_}")
    print(f"F1 macro (validation croisée train) : {search.best_score_:.3f}")
    print(f"F1 macro (test) : {results['f1_macro_test']:.3f}")
    print(f"Rappel Dropout (test) : {recalls[0]:.3f}")
    print(f"Score de Brier multiclasse (test) : {brier_score:.3f}")
    print(
        f"Instances sélectionnées : {len(selected)} "
        f"({int((~selected['is_correct']).sum())} erreurs, "
        f"{int(selected['is_uncertain'].sum())} incertaine)"
    )
    print(f"Instances de mise au point distinctes : {len(debugging)}")
    print(f"Modèle : {model_path}")
    print(f"Résultats : {evaluation_path}")
    print(f"Instances : {selected_path}")
    print(f"Mise au point : {debugging_path}")


if __name__ == "__main__":
    main()