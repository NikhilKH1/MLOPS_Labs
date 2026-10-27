# Docker Lab 1: Breast Cancer Classifier in a Container

This lab is based on [Docker Lab 1](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Docker_Labs/Lab1) from the MLOps course repo. The original lab packages a script that trains a Random Forest on the Iris dataset into a Docker image. This version keeps that idea and changes the dataset, the model, and the way the container is built and run.

## What I changed

| | Original lab | This version |
|---|---|---|
| Dataset | Iris (150 samples, 4 features, 3 classes) | Breast Cancer Wisconsin (569 samples, 30 features, benign vs. malignant) |
| Model | A fixed `RandomForestClassifier` | Four candidates (logistic regression, random forest, gradient boosting, SVM) compared with 5-fold stratified cross-validation. The best one by ROC-AUC is kept. Each model is a `Pipeline` with a `StandardScaler`. |
| Evaluation | None, just prints "successful" | Accuracy, precision, recall, F1, ROC-AUC and a confusion matrix on a held-out test set, with malignant as the positive class |
| Output | `iris_model.pkl` saved inside the container, lost when it exits | `model.joblib` and `metrics.json` written to `/app/output`, mounted as a volume so they stay on your machine |
| Configuration | Hard-coded | Environment variables (`MODEL_TYPE`, `TEST_SIZE`, `RANDOM_STATE`, `CV_FOLDS`, `OUTPUT_DIR`) that `docker run -e` can override |
| Inference | None | `predict.py` loads the saved model in a new container and predicts on test samples |
| Base image | `python:3.10` (image builds to 2.06 GB) | `python:3.11-slim` (image builds to 591 MB) |
| Dependencies | Unpinned (`scikit-learn`, `joblib`) | All versions pinned |
| Dockerfile | Copies code before installing packages, runs as root | Installs requirements first so that layer is cached, runs as a non-root user, adds `LABEL`s and a `.dockerignore` |

## Project structure

```
Docker_Lab1/
├── Dockerfile
├── .dockerignore
├── README.md
├── src/
│   ├── main.py           # trains, evaluates and saves the model (default command)
│   ├── predict.py        # loads the saved model and predicts on test samples
│   └── requirements.txt  # pinned dependencies
└── output/               # created when you run with a volume (git-ignored)
    ├── model.joblib
    └── metrics.json
```

## Running the lab

You need [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running. Run all commands from the `Docker_Lab1/` folder.

### 1. Build the image

```bash
docker build -t breast-cancer-trainer:v1 .
```

### 2. Train the model

Mount a local `output/` folder at `/app/output` so the trained model is still there after the container exits:

```bash
docker run --rm -v "$(pwd)/output:/app/output" breast-cancer-trainer:v1
```

Output:

```
Dataset: 569 samples, 30 features (455 train / 114 test)
Cross-validating 4 model(s) with 5 folds:
  logistic_regression  ROC-AUC = 0.9958 (+/- 0.0047)
  random_forest        ROC-AUC = 0.9889 (+/- 0.0070)
  gradient_boosting    ROC-AUC = 0.9908 (+/- 0.0055)
  svm                  ROC-AUC = 0.9949 (+/- 0.0050)
Selected model: logistic_regression
Test metrics (positive class = malignant):
  accuracy   0.9649
  precision  0.9750
  recall     0.9286
  f1         0.9512
  roc_auc    0.9960
Confusion matrix [benign, malignant]: [[71, 1], [3, 39]]
Saved model.joblib and metrics.json to /app/output
The model training was successful
```

`output/metrics.json` records the settings used, the CV score of every candidate, the test metrics, and the container it ran in:

```json
{
  "model": "logistic_regression",
  "config": { "model_type": "auto", "test_size": 0.2, "random_state": 42, "cv_folds": 5, "output_dir": "/app/output" },
  "cv_roc_auc": { "logistic_regression": 0.9958, "random_forest": 0.9889, "gradient_boosting": 0.9908, "svm": 0.9949 },
  "test_metrics": { "accuracy": 0.9649, "precision": 0.975, "recall": 0.9286, "f1": 0.9512, "roc_auc": 0.996 },
  "environment": { "hostname": "896c079d9776", "python": "3.11.18", "scikit_learn": "1.6.1" }
}
```

### 3. Predict in a new container

Override the default command to run `predict.py`. This starts a brand-new container, so it only works if the model was saved to the mounted volume:

```bash
docker run --rm -v "$(pwd)/output:/app/output" breast-cancer-trainer:v1 python predict.py -n 8
```

```
Loaded logistic_regression trained at 2026-10-27T11:22:32+00:00 in container 896c079d9776

  #  actual     predicted  P(malignant)
  0  benign     benign            0.000
  1  malignant  malignant         1.000
  2  benign     benign            0.043
  3  malignant  malignant         0.576
  4  benign     malignant         0.509  <- wrong
  5  benign     benign            0.001
  6  malignant  malignant         0.762
  7  benign     benign            0.001

7/8 correct
```

### 4. Change settings without rebuilding

Every setting is an environment variable with a default set in the Dockerfile. For example, to train only a random forest with a 30% test split:

```bash
docker run --rm -e MODEL_TYPE=random_forest -e TEST_SIZE=0.3 -v "$(pwd)/output-rf:/app/output" breast-cancer-trainer:v1
```

| Variable | Default | Allowed values |
|---|---|---|
| `MODEL_TYPE` | `auto` | `auto`, `logistic_regression`, `random_forest`, `gradient_boosting`, `svm` |
| `TEST_SIZE` | `0.2` | Between 0 and 1 |
| `RANDOM_STATE` | `42` | Any integer |
| `CV_FOLDS` | `5` | An integer of 2 or more |
| `OUTPUT_DIR` | `/app/output` | A path inside the container |

An invalid value stops the container with a clear message, for example `MODEL_TYPE must be one of [...], got 'xgboost'`.

### 5. Save and share the image

These steps are the same as in the original lab:

```bash
docker save breast-cancer-trainer:v1 > breast-cancer-trainer_v1.tar
```

```bash
docker load < breast-cancer-trainer_v1.tar
```

The saved `.tar` is about 122 MB. `.gitignore` and `.dockerignore` both exclude it.

## Notes

- **Linux hosts:** the container runs as user ID 1000. If your user ID is different and the container can't write to `output/`, add `--user "$(id -u):$(id -g)"` to `docker run`.
- **Windows PowerShell:** use `${PWD}/output` instead of `$(pwd)/output`.
- **Inspecting the image:** run `docker run --rm breast-cancer-trainer:v1 whoami` (prints `appuser`) or `docker image inspect breast-cancer-trainer:v1` to see the labels and default environment variables.
