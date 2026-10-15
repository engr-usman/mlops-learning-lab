"""Project configuration: paths, schema and validation rules for the Adult dataset."""

from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
RAW_DATA_DIR = ROOT_DIR / "adult"
DATA_DIR = ROOT_DIR / "data"
DATA_PATH = DATA_DIR / "data.csv"

# Target
TARGET = "income"
TARGET_CLASSES = ["<=50K", ">50K"]

# Numeric features with their allowed (min, max) range, inclusive
NUMERIC_FEATURES = {
    "age": (17, 90),
    "fnlwgt": (1, 1_500_000),
    "education-num": (1, 16),
    "capital-gain": (0, 99_999),
    "capital-loss": (0, 4_356),
    "hours-per-week": (1, 99),
}

# Categorical features with their allowed values (from adult/adult.names)
CATEGORICAL_FEATURES = {
    "workclass": [
        "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
        "Local-gov", "State-gov", "Without-pay", "Never-worked",
    ],
    "education": [
        "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
        "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
        "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool",
    ],
    "marital-status": [
        "Married-civ-spouse", "Divorced", "Never-married", "Separated",
        "Widowed", "Married-spouse-absent", "Married-AF-spouse",
    ],
    "occupation": [
        "Tech-support", "Craft-repair", "Other-service", "Sales",
        "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
        "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
        "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces",
    ],
    "relationship": [
        "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried",
    ],
    "race": ["White", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other", "Black"],
    "sex": ["Female", "Male"],
    "native-country": [
        "United-States", "Cambodia", "England", "Puerto-Rico", "Canada",
        "Germany", "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece",
        "South", "China", "Cuba", "Iran", "Honduras", "Philippines", "Italy",
        "Poland", "Jamaica", "Vietnam", "Mexico", "Portugal", "Ireland",
        "France", "Dominican-Republic", "Laos", "Ecuador", "Taiwan", "Haiti",
        "Columbia", "Hungary", "Guatemala", "Nicaragua", "Scotland", "Thailand",
        "Yugoslavia", "El-Salvador", "Trinadad&Tobago", "Peru", "Hong",
        "Holand-Netherlands",
    ],
}

# Expected column order in data.csv
COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", TARGET,
]

# Validation thresholds
MIN_ROWS = 1_000
MAX_DUPLICATE_FRACTION = 0.01  # duplicates above this fraction fail validation

# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42
TEST_DATA_PATH = DATA_DIR / "test.csv"

# Model hyperparameters, keyed by the --model choice in train.py
MODEL_PARAMS = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 200, "max_depth": 15, "min_samples_leaf": 5, "n_jobs": -1},
}
DEFAULT_MODEL = "random_forest"

# MLflow
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT_DIR / 'mlflow.db'}"
MLFLOW_ARTIFACT_DIR = ROOT_DIR / "mlruns"
MLFLOW_EXPERIMENT_NAME = "adult-income"
REGISTERED_MODEL_NAME = "adult-income-classifier"
CHAMPION_ALIAS = "champion"  # alias the API serves

# A new model version is promoted to champion only if it meets these on the test set
PROMOTION_THRESHOLDS = {"accuracy": 0.80, "roc_auc": 0.85}

# Logging
LOG_DIR = ROOT_DIR / "logs"
LOG_LEVEL = "INFO"


def setup_mlflow() -> None:
    """Point MLflow at the project's tracking DB and experiment, wherever the script runs from."""
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME, artifact_location=MLFLOW_ARTIFACT_DIR.as_uri())
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
