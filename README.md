# Projet XAI — prévenir le décrochage sans boîte noire

Ce dépôt met en œuvre un pipeline reproductible sur le jeu UCI *Predict Students' Dropout and Academic Success*. Le système est un prototype pédagogique : ses scores ne sont ni un verdict ni une décision automatisée concernant un étudiant.

## État du dépôt

Les étapes déjà présentes comprennent l'exploration initiale, le prétraitement stratifié, la comparaison de modèles et l'entraînement/évaluation du Random Forest final (`src/data_pipeline.py`, `src/train_compare.py`, `src/evaluate.py`). La suite proposée dans ce dépôt ajoute l'explicabilité locale, les appels aux LLMs/juges avec cache et journalisation, et un tableau de bord. Les résultats d'API ne sont pas inclus : ils doivent être produits avec vos propres clés et vérifiés avant le rapport.

> Important : le modèle actuellement retenu utilise les variables disponibles à l'inscription / après le premier semestre et exclut les variables du deuxième semestre afin d'éviter une fuite temporelle. Il ne faut pas le présenter comme une prédiction d'inscription si des résultats du premier semestre sont utilisés. Vérifiez la date de prédiction et la définition du jeu de données dans votre rapport.

## Installation

Python 3.10–3.12 recommandé.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows : .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env       # facultatif, uniquement pour les appels API
```

Le CSV UCI est dans `data/data.csv` (séparateur `;`). Source : UCI Machine Learning Repository, DOI [10.24432/C5MC89](https://doi.org/10.24432/C5MC89), licence CC BY 4.0. Citez Realinho et al. (2022) et UCI dans votre rapport.

## Exécution

Depuis la racine du dépôt :

```bash
python -m src.train_compare
python -m src.evaluate
python -m src.explain
```

Puis configurez `GROQ_API_KEY`, `MISTRAL_API_KEY`, `GEMINI_API_KEY` et `JUDGE_API_KEY` dans `.env` selon les fournisseurs choisis. Les noms exacts de modèles sont configurables dans ce fichier et doivent être consignés avec la date dans `outputs/llm_calls.jsonl`.

```bash
python -m src.llm_pipeline generate  # N1, N2, N3 pour les 15 cas et 3 générateurs
python -m src.llm_pipeline judge    # juge les réponses présentes; indépendant et contexte vierge
streamlit run dashboard/app.py
```

Les appels sont facturés / soumis aux quotas du fournisseur, même lorsque celui-ci offre un niveau gratuit. Le cache est activé; les erreurs ne doivent pas être remplacées par des résultats inventés. Le script reprend les appels manquants au prochain lancement.

## Structure des sorties

- `outputs/model_comparison.csv`, `calibration_curves.png`, `evaluation_results.json` : comparaison et évaluation.
- `outputs/selected_instances.csv` : 15 instances exclusivement tirées du test; `debug_instances.csv` : cas séparés pour mettre au point les prompts.
- `outputs/xai_explanations.jsonl` : une explication technique par méthode et par cas.
- `outputs/llm_responses.jsonl` et `llm_calls.jsonl` : réponses et journal des appels (messages, fournisseur/modèle, température, version, date, hash du prompt).
- `outputs/judge_scores.jsonl` : évaluations structurées par le juge.

Le contenu sous `outputs/` peut contenir des exemples issus de données réelles : ne publiez aucune donnée personnelle ajoutée par votre groupe. Les appels API doivent utiliser exclusivement les données du jeu public et des profils fictifs.

## Interprétation et limites

Le jeu est déséquilibré et Enrolled est une classe difficile. Le modèle présente des faux positifs et faux négatifs; une probabilité n'est pas une probabilité individuelle de réussite au sens causal. Les facteurs SHAP décrivent le comportement du modèle, non les causes du décrochage. Une explication générée par LLM peut être fausse même si elle est fluide. Les sorties doivent être contrôlées par des personnes, et ne doivent jamais servir seules à une décision défavorable.

La méthode d'explication/règle peut échouer ou ne pas trouver de solution; l'absence de sortie n'est pas une explication. Signalez les échecs. Comparez les résultats avec les annotations manuelles demandées par le sujet; les scores d'un LLM-juge ne remplacent pas cette validation.

## Travaux à compléter pour le rendu

1. Produire et contrôler les artefacts en exécutant les scripts dans votre environnement.
2. Remplir le rapport (10 pages maximum) avec les résultats réellement observés, le dictionnaire des variables, la théorie du domaine, les choix et les limites; ne recopiez pas de chiffres non reproduits.
3. Faire les 30 annotations humaines à l'aveugle et calculer les accords inter-annotateurs et humain/juge.
4. Vérifier la disponibilité, les quotas et l'identifiant exact des modèles au jour des expériences; ces offres évoluent.
5. Tester chaque étape et documenter les erreurs rencontrées avant la soutenance.
