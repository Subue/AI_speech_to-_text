# ML Voice Assistant Final Year Project

This project is now a complete end-to-end machine learning assistant with:

- a labeled intent dataset
- a training pipeline using scikit-learn
- saved model artifacts and evaluation metrics
- a FastAPI backend
- a browser frontend for testing predictions and retraining
- a terminal assistant with optional speech input and text-to-speech

## Project Structure

```text
ML/
|-- app.py
|-- assistant.py
|-- actions.py
|-- ml_intents.py
|-- speech_io.py
|-- data/commands.csv
|-- frontend/
|   |-- index.html
|   |-- styles.css
|   |-- app.js
|-- models/
|   |-- intent_model.joblib
|   |-- metrics.json
```

## Setup

```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional for local microphone capture in the terminal assistant on Windows:

```bash
pip install PyAudio
```

## Train The Model

```bash
py -3.11 ml_intents.py
```

This reads `data/commands.csv`, trains the classifier, and writes:

- `models/intent_model.joblib`
- `models/metrics.json`

## Run The Full-Stack Web App

```bash
py -3.11 -m uvicorn app:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## Run The Terminal Assistant

```bash
py -3.11 assistant.py
```

If a microphone is available, it listens for voice input. Otherwise, it automatically falls back to text input mode.

## Dataset Format

The dataset must contain these columns:

- `text`
- `intent`

You can keep adding more samples to `data/commands.csv` and retrain the model from the dashboard or command line.
