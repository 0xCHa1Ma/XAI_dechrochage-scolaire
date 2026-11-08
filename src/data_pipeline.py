from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 1. Chargement des données
data_path = Path(__file__).resolve().parent.parent / "data" / "data.csv"
df = pd.read_csv(data_path, sep=";")
df.columns = df.columns.str.strip()

# 2. Isoler 15 instances "secrètes" pour la toute fin
df_15_instances = df.sample(n=15, random_state=42)
df_main = df.drop(df_15_instances.index)

# 3. Séparer features (X) et target (y)
X = df_main.drop("Target", axis=1) # Target = Dropout, Enrolled, Graduate
y = df_main["Target"]

# 4. Découpage stratifié 70% Train / 30% Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.30, 
    stratify=y,       # Très important pour garder les proportions des 3 classes
    random_state=42
)

# 5. Définir les colonnes selon leur type
categorical_columns = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

binary_columns = [
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

numeric_columns = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

# Application order est ordonnée : on garde son ordre numérique.
ordinal_columns = ["Application order"]

# Le prétraitement est appris sur l'entraînement uniquement pour éviter le leakage. On applique ensuite le même prétraitement sur le test.
preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ("numeric", StandardScaler(), numeric_columns),
        ("binary", "passthrough", binary_columns),
        ("ordinal", "passthrough", ordinal_columns),
    ]
)

X_train_prepared = preprocessor.fit_transform(X_train)
X_test_prepared = preprocessor.transform(X_test)
