# MLOps Labs

My lab submissions for the MLOps course, based on the labs in [raminmohammadi/MLOps](https://github.com/raminmohammadi/MLOps).

| Lab | Folder | Summary |
|---|---|---|
| Lab Assignment 3: Docker Lab 1 | [`Docker_Lab1/`](Docker_Lab1/) | A breast cancer classifier, trained and served with Docker Compose. Tests run first, then a training container picks the best of four models with cross-validation. A Streamlit dashboard, with a healthcheck, starts only after training succeeds. The trainer and dashboard share the model through a named volume. Run it with `docker compose up --build`. |
