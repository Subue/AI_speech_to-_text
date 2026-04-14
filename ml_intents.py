from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "data" / "commands.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "intent_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"


@dataclass
class PredictionResult:
    text: str
    intent: str
    confidence: float


class IntentClassifier:
    def __init__(self, model: Pipeline[str, str]) -> None:
        self.model = model

    @classmethod
    def load(cls) -> "IntentClassifier":
        return cls(joblib.load(MODEL_PATH))

    def predict(self, text: str) -> PredictionResult:
        cleaned_text = text.strip()
        probabilities = self.model.predict_proba([cleaned_text])[0]
        classes = self.model.classes_
        winning_index = int(probabilities.argmax())
        return PredictionResult(
            text=cleaned_text,
            intent=str(classes[winning_index]),
            confidence=float(probabilities[winning_index]),
        )


def load_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    dataset = pd.read_csv(dataset_path)
    required_columns = {"text", "intent"}
    missing_columns = required_columns.difference(dataset.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing columns: {sorted(missing_columns)}")
    dataset["text"] = dataset["text"].astype(str).str.strip()
    dataset["intent"] = dataset["intent"].astype(str).str.strip()
    dataset = dataset[(dataset["text"] != "") & (dataset["intent"] != "")]
    if dataset["intent"].nunique() < 2:
        raise ValueError("Dataset must contain at least two intent classes.")
    return dataset


def build_pipeline() -> Pipeline[str, str]:
    return Pipeline(
        steps=[
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), lowercase=True)),
            ("classifier", LogisticRegression(max_iter=2000)),
        ]
    )


def train_and_save_model(dataset_path: Path = DATASET_PATH) -> dict[str, Any]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_path)

    features = dataset["text"]
    labels = dataset["intent"]
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=42,
        stratify=labels,
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    accuracy = accuracy_score(y_test, predictions)

    metrics = {
        "dataset_path": str(dataset_path),
        "model_path": str(MODEL_PATH),
        "dataset_size": int(len(dataset)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "intent_count": int(labels.nunique()),
        "intents": sorted(labels.unique().tolist()),
        "accuracy": float(accuracy),
        "classification_report": report,
    }

    joblib.dump(pipeline, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def ensure_model_artifacts() -> None:
    if not MODEL_PATH.exists() or not METRICS_PATH.exists():
        train_and_save_model()


def load_metrics() -> dict[str, Any]:
    ensure_model_artifacts()
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def predict_text(text: str) -> dict[str, Any]:
    ensure_model_artifacts()
    classifier = IntentClassifier.load()
    prediction = classifier.predict(text)
    return asdict(prediction)


if __name__ == "__main__":
    metrics = train_and_save_model()
    print(json.dumps(metrics, indent=2))
