# 🪽 Ade's Wealthsimple AI Builder Profile Project: Real-Time Fraud Detection Pipeline

This project serves as a cornerstone for demonstrating enterprise-scale MLOps and fintech relevance, aligning perfectly with the requirements for an AI Builder role.

## 🎯 Goal
To architect and implement a high-throughput, low-latency machine learning pipeline capable of scoring financial transactions for anomalies in near real-time (<50ms end-to-end latency).

## 🧱 Enterprise Focus Areas

| Area | Demonstration |
| :--- | :--- |
| **Scale/Compute** | Utilizing efficient model serving (e.g., compiled models via ONNX Runtime) and asynchronous data handling (Kafka). |
| **Security/Fintech** | Addressing the implicit security and compliance needs of financial data (mentioning concepts like data masking in the design). |
| **Reliability** | Implementing robust logging, monitoring, and health checks (Prometheus/Grafana integration). |

## 🛠️ Technology Stack

| Layer | Primary Tool/Technology | Why? |
| :--- | :--- | :--- |
| **Data Ingestion** | Apache Kafka / Pulsar | Industry standard for scalable, fault-tolerant stream processing. |
| **Inference Engine** | Python (FastAPI) + ONNX Runtime | Demonstrates focus on performance optimization for production serving. |
| **Monitoring** | Prometheus & Grafana | Essential for establishing SLOs/SLAs and tracking model drift in production. |
| **Infrastructure** | Terraform / Docker | Defines the environment reliably and reproducibly (IaC principle). |
| **Data Storage** | PostgreSQL / Redis | PostgreSQL for metadata/state; Redis for fast feature lookups during scoring. |

## 🗺️ Implementation Roadmap (Milestones)

1.  **Phase 1: Data Simulation & Streaming:** Implement a mock producer to pump synthetic transaction data into a Kafka topic. (See: `data_streamer/producer.py`)
2.  **Phase 2: Model Integration:** Containerize a pre-trained, optimized model (placeholder ONNX file) and expose a `/score` endpoint via FastAPI. (See: `model_service/main.py`)
3.  **Phase 3: End-to-End Connectivity:** Connect the Kafka consumer to the FastAPI service, measuring total latency per transaction.
4.  **Phase 4: Observability & IaC:** Implement basic Prometheus metrics on the service and write initial Terraform to define the basic container host environment.

---

### Placeholder Directory Structure
- `data_streamer/`: Scripts for simulating/consuming data streams.
- `model_service/`: Python/FastAPI code for model inference.
- `infrastructure/`: Terraform/Docker files.