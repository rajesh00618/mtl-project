# Multi-Task Learning with PCGrad

This project demonstrates a custom TensorFlow multi-task learning system that trains a shared backbone on multiple objectives simultaneously. The codebase compares a standard baseline training approach with a gradient surgery method called Projected Conflicting Gradients (PCGrad), which helps mitigate conflicting gradients between tasks and improves multi-task convergence.

The repository includes data handling, model definition, training scripts, model architecture summary generation, and a Streamlit dashboard for visualizing training metrics, gradient conflict behavior, and shared feature representations.

Key goals of the project:

- Showcase how multi-task learning can share representations across tasks while preserving task-specific performance.
- Compare naive loss summation against PCGrad for reducing inter-task gradient conflict.
- Provide tools to inspect model architecture, training progress, and result artifacts.
- Package the app and training workflow so it can run locally or inside Docker.

## Files

- `train_baseline.py`: Train the baseline multi-task model using naive loss summation.
- `train_pcgrad.py`: Train the same model using PCGrad for gradient surgery.
- `generate_model_summary.py`: Print and save the shared backbone and full multi-task model architectures.
- `app.py`: Streamlit dashboard for visualizing training metrics and representations.
- `Dockerfile` / `docker-compose.yml`: Containerize and run the Streamlit app.
- `results/`: Stores metric logs, model weights, analysis, and model architecture summary.

## Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Generate the model architecture summary:

```bash
python generate_model_summary.py
```

3. Train the baseline and PCGrad models:

```bash
python train_baseline.py
python train_pcgrad.py
```

4. Launch the dashboard locally:

```bash
streamlit run app.py
```

5. Or run the app with Docker Compose:

```bash
docker-compose up --build
```

The Streamlit app will be available at `http://localhost:8501`.
