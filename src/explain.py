"""Generate SHAP, Anchors and DiCE explanations for the fixed held-out cases."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LABELS = ["Dropout", "Enrolled", "Graduate"]
# Attributs codés à décoder/présenter avec prudence. Le dictionnaire reste la référence.
CATEGORICAL = ["Marital status", "Application mode", "Course", "Daytime/evening attendance",
"Previous qualification", "Nacionality", "Mother's qualification", "Father's qualification",
"Mother's occupation", "Father's occupation", "Displaced", "Educational special needs", "Debtor",
"Tuition fees up to date", "Gender", "Scholarship holder", "International", "Application order"]
ACTIONABLE = ["Tuition fees up to date", "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)"]

def jsonable(value):
    if isinstance(value, (np.integer,)): return int(value)
    if isinstance(value, (np.floating,)): return float(value)
    if isinstance(value, np.ndarray): return value.tolist()
    return str(value)

def main():
    model = joblib.load(OUT / "model.joblib")
    train = pd.read_csv(ROOT / "data/data.csv", sep=";").drop(columns=["Target"])
    future = [c for c in train.columns if c.startswith("Curricular units 2nd sem")]
    train = train.drop(columns=future)
    selected = pd.read_csv(OUT / "selected_instances.csv")
    meta = {"instance_id", "true_class", "predicted_class", "confidence", "probability_margin", "is_correct", "is_uncertain", *[f"probability_{c}" for c in LABELS]}
    features = [c for c in selected.columns if c not in meta]
    X = selected[features]
    transformed_train = model.named_steps["preprocessing"].transform(train[features])
    transformed_x = model.named_steps["preprocessing"].transform(X)
    names = model.named_steps["preprocessing"].get_feature_names_out()
    explainer = None
    try:
        import shap
        explainer = shap.TreeExplainer(model.named_steps["model"])
        values = explainer.shap_values(transformed_x)
    except Exception as exc:
        raise RuntimeError("SHAP TreeExplainer a échoué; vérifiez shap/sklearn et documentez l'échec.") from exc
    # Normalise les conventions SHAP multiclasse: [n, features, classes] ou liste par classe.
    if isinstance(values, list):
        by_class = np.stack(values, axis=-1)
    else:
        by_class = np.asarray(values)
        if by_class.ndim == 2: by_class = by_class[:, :, None]
        if by_class.shape[1] != len(names) and by_class.shape[2] == len(names): by_class = np.transpose(by_class, (0, 2, 1))
    records = []
    for i, row in X.iterrows():
        pos = X.index.get_loc(i)
        pred = str(selected.iloc[pos]["predicted_class"])
        class_index = LABELS.index(pred)
        vals = by_class[pos, :, class_index]
        top = np.argsort(np.abs(vals))[::-1][:7]
        attribution = [{"feature": str(names[j]), "value": float(transformed_x[pos, j]), "contribution": float(vals[j])} for j in top]
        # Anchors via Alibi; failure is recorded instead of fabricating a rule.
        anchor = {"status": "unavailable"}
        try:
            from alibi.explainers import AnchorTabular
            raw = train[features].to_numpy(dtype=float)
            pred_fn = lambda a: model.predict(pd.DataFrame(a, columns=features))
            cat_idx = [features.index(c) for c in CATEGORICAL if c in features]
            cat_names = {features.index(c): sorted(map(int, train[c].dropna().unique())) for c in CATEGORICAL if c in features}
            ae = AnchorTabular(pred_fn, feature_names=features, categorical_names=cat_names)
            ae.fit(raw, disc_perc=(25, 50, 75))
            result = ae.explain(row.to_numpy(dtype=float), threshold=0.95)
            anchor = {"status": "ok", "rule": list(result.names), "precision": float(result.precision), "coverage": float(result.coverage)}
        except Exception as exc:
            anchor = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        # DiCE random search requires integer class labels; adapter preserves the fitted pipeline.
        counterfactual = {"status": "unavailable"}
        try:
            import dice_ml
            from sklearn.base import BaseEstimator, ClassifierMixin
            class EncodedEstimator(ClassifierMixin, BaseEstimator):
                def __init__(self, fitted, columns): self.fitted, self.columns = fitted, columns
                def predict(self, data):
                    labels = self.fitted.predict(pd.DataFrame(data, columns=self.columns))
                    return np.array([LABELS.index(str(x)) for x in labels])
                def predict_proba(self, data): return self.fitted.predict_proba(pd.DataFrame(data, columns=self.columns))
            train_frame = train[features].copy(); train_frame["Target"] = [LABELS.index(str(y)) for y in pd.read_csv(ROOT / "data/data.csv", sep=";")["Target"]]
            continuous = [c for c in features if c not in CATEGORICAL]
            d = dice_ml.Data(dataframe=train_frame, continuous_features=continuous, outcome_name="Target")
            m = dice_ml.Model(model=EncodedEstimator(model, features), backend="sklearn")
            dice = dice_ml.Dice(d, m, method="random")
            desired = 2 if pred != "Graduate" else 0
            cf = dice.generate_counterfactuals(X.iloc[[pos]], total_CFs=1, desired_class=desired, features_to_vary=[c for c in ACTIONABLE if c in features], random_seed=42)
            cf_df = cf.cf_examples_list[0].final_cfs_df
            counterfactual = {"status": "ok", "target_class": LABELS[desired], "changes": {c: {"before": jsonable(row[c]), "after": jsonable(cf_df.iloc[0][c])} for c in features if c in cf_df and cf_df.iloc[0][c] != row[c]}}
        except Exception as exc:
            counterfactual = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
        record = {"instance_id": str(selected.iloc[pos]["instance_id"]), "true_class": str(selected.iloc[pos]["true_class"]), "predicted_class": pred, "probabilities": {c: float(selected.iloc[pos][f"probability_{c}"]) for c in LABELS}, "shap": {"class_explained": pred, "top7": attribution, "other_contribution_sum": float(np.sum(vals) - sum(v["contribution"] for v in attribution))}, "anchors": anchor, "counterfactual": counterfactual}
        records.append(record)
    OUT.mkdir(exist_ok=True)
    with (OUT / "xai_explanations.jsonl").open("w", encoding="utf-8") as f:
        for r in records: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Écrit {len(records)} cas dans {OUT / 'xai_explanations.jsonl'}")
if __name__ == "__main__": main()
