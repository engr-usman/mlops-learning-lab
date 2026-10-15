# MLOps Learning Lab

A hands-on learning repository for exploring **Machine Learning Operations (MLOps)**, with a focus on **MLflow**, **Kubeflow**, and the tools used to take machine learning projects from local experiments to repeatable, deployable workflows.

## Purpose

This repository is a collection of lecture-based exercises, code examples, notebooks, and small projects developed while learning MLOps. The goal is to understand not only how to train a machine learning model, but also how to **track experiments, validate data, package models, automate workflows, and run ML workloads reliably**.

It is a learning workspace rather than a single production application. Examples may evolve as new concepts are introduced, and later lectures may build on or refine earlier implementations.

## What's in the repository?

The repository is organised by lecture, making it easy to follow the progression and revisit individual topics.

| Area | What it covers |
| --- | --- |
| **ML foundations** | Datasets, preprocessing, classification, model training, and evaluation. |
| **MLflow** | Tracking experiments, metrics, parameters, and model artifacts; exploring model lifecycle practices. |
| **Model APIs** | Exposing trained models through Python-based prediction services. |
| **Docker** | Packaging ML applications and their dependencies into reproducible containers. |
| **Kubeflow** | Defining and running ML pipelines and exploring Kubernetes-native ML workloads. |
| **Kubernetes & deployment** | Learning how ML services and workflows run on container infrastructure. |

### Current lecture examples

- **`lecture-02a/`** — Python ML workflow exercises, including data validation, training, evaluation, an API, notebooks, and MLflow-related experimentation.
- **`lecture-03a/`** — An expanded example with data ingestion, configuration, Docker and Docker Compose files, notebooks, and Kubeflow-related material.

Each lecture may include its own `README.md` with setup instructions and implementation details. The root README focuses on the overall learning journey.

## Topics planned for upcoming lectures

As the repository grows, upcoming exercises may explore:

1. **ML fundamentals:** feature engineering, model comparison, and scikit-learn pipelines.
2. **MLflow:** experiment tracking, artifact management, and model registration.
3. **Reproducibility:** configuration management, dependency pinning, and repeatable training runs.
4. **Model serving:** prediction APIs, Docker images, and deployment patterns.
5. **Kubeflow Pipelines:** reusable components, pipeline execution, and workflow orchestration.
6. **Kubeflow ecosystem:** notebooks, training jobs, and hyperparameter tuning with Katib.
7. **Operational practices:** CI/CD for ML, validation, monitoring, and retraining workflows.
8. **Cloud MLOps:** applying these concepts on Kubernetes and managed cloud ML platforms.

These are **learning goals**, not a claim that every topic has already been implemented.

## Learning approach

The examples follow a practical progression:

```text
Explore data → Prepare features → Train → Evaluate
     ↓
Track experiments with MLflow
     ↓
Package and serve the model
     ↓
Run containerised workloads
     ↓
Orchestrate ML workflows with Kubeflow
     ↓
Improve automation, reproducibility, and monitoring
```

The emphasis is on understanding **why** each tool is useful, how it fits into the ML lifecycle, and how it connects to familiar software engineering and DevOps practices.

## Repository notes

- Lecture directories are intended to be explored independently; dependencies and setup steps can vary.
- Local virtual environments, logs, MLflow tracking databases, generated artifacts, and other machine-specific files are excluded from version control where appropriate.
- Some exercises may require datasets or runtime configuration that are not committed to the repository. Refer to lecture-specific documentation for those requirements.
- Code is provided for **education and experimentation** and should be reviewed before production use.

## Who is this for?

Software engineers, DevOps/cloud engineers, and ML beginners who want to learn how machine learning models move from notebooks into repeatable, observable, and deployable systems.

---

**Status:** Ongoing learning repository — new lectures and examples will be added over time.
