# Choix du modèle final

Nous retenons le **Random Forest** pour la suite du projet, en particulier pour
les explications SHAP. Le critère principal est le rappel de la classe
*Dropout*, car manquer un étudiant susceptible de décrocher est un risque que
nous voulons limiter. c'est donc une priorité.

Sur les prédictions hors pli de la validation croisée, le Random Forest obtient
un rappel *Dropout* de **72,2 %**, contre **68,2 %** pour XGBoost, soit environ
**4 points de plus**. Ce choix implique un compromis : son F1 macro est de
**66,9 %**, contre **68,2 %** pour XGBoost (environ **1,4 point de moins**),
et son score de Brier est de **0,410**, contre **0,383** pour XGBoost. XGBoost
présente également un meilleur rappel pour la classe *Enrolled*; le Random
Forest a un meilleur rappel pour *Graduate* dans cette comparaison.

Nous considérons le compromis acceptable parce que la détection de la classe
*Dropout* est notre priorité. Les deux modèles sont des ensembles d’arbres
compatibles avec TreeSHAP. SHAP aidera à expliquer les prédictions du modèle,
mais ces explications décrivent le comportement appris par le modèle et ne
prouvent pas des liens de causalité.

Ces chiffres proviennent de la validation croisée sur les données
d’entraînement; ils ne constituent pas les performances finales sur le jeu de
test. L’équité entre groupes et la stabilité des explications restent à
évaluer séparément.
