# Credit Risk Probability Model for Alternative Data

> **Bati Bank × eCommerce Partner: Buy-Now-Pay-Later Credit Scoring**

## Project Overview

This repository contains an end-to-end credit risk modeling solution built for **Bati Bank** in partnership with an eCommerce platform. The goal is to enable a **buy-now-pay-later (BNPL)** service by scoring customers in real time using alternative (transactional) data.

The project transforms raw eCommerce transaction logs into a predictive risk signal through:
- **RFM-based proxy target engineering** (Recency, Frequency, Monetary)
- **Feature engineering pipelines** with Weight of Evidence (WoE) and Information Value (IV)
- **Machine learning model training** with experiment tracking via MLflow
- **Containerized REST API** for real-time inference
- **CI/CD automation** with GitHub Actions

## Data Source

- **Dataset**: [Xente Fraud Detection Challenge](https://zindi.africa/competitions/xente-fraud-detection-challenge/data) (also available on Kaggle)
- **Records**: ~140,000 transactions from 15 November 2018 to 15 March 2019
- **Context**: Transaction-level data from Xente, an e-commerce and financial service app serving 10,000+ customers in Uganda

## Repository Structure

```
credit-risk-model/
├── .github/workflows/ci.yml      # CI/CD pipeline
├── data/                          # (gitignored) Raw & processed data
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── eda.ipynb                  # Exploratory Data Analysis
├── src/
│   ├── __init__.py
│   ├── data_processing.py         # Feature engineering pipeline
│   ├── train.py                   # Model training & MLflow tracking
│   ├── predict.py                 # Inference utilities
│   └── api/
│       ├── main.py                # FastAPI application
│       └── pydantic_models.py     # Request/response schemas
├── tests/
│   └── test_data_processing.py    # Unit tests
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/credit-risk-model.git
cd credit-risk-model
```

### 2. Create a Virtual Environment
```bash
py -3.11 -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the Dataset
- Download the Xente Challenge dataset from [Zindi](https://zindi.africa/competitions/xente-fraud-detection-challenge/data) or [Kaggle](https://www.kaggle.com/)
- Place the CSV files in `data/raw/`

### 5. Run EDA Notebook
```bash
jupyter notebook notebooks/eda.ipynb
```

### 6. Train the Model
```bash
python src/train.py
```

### 7. Start the API (Development)
```bash
uvicorn src.api.main:app --reload
```

### 8. Run Tests
```bash
pytest tests/
```

### 9. Docker Deployment
```bash
docker-compose up --build
```

---

## Credit Scoring Business Understanding

### 1. Basel II Accord and Model Interpretability

The **Basel II Capital Accord** establishes international regulatory standards for bank capital adequacy, with a strong emphasis on **risk measurement, documentation, and model interpretability**. Under Basel II, banks must maintain rigorous internal ratings-based (IRB) approaches that are transparent, auditable, and defensible to regulators.

This emphasis directly influences our modeling choices in three ways:

- **Interpretability Requirement**: Regulators and internal risk committees must understand *why* a customer receives a particular risk score. Black-box models that cannot explain their decisions create compliance vulnerabilities and audit failures. A model must provide clear, traceable paths from input features to output scores.

- **Documentation Burden**: Basel II mandates comprehensive model documentation covering data sources, feature definitions, methodology, validation results, and monitoring plans. Every modeling assumption must be explicitly recorded and justified, including the proxy variable design.

- **Validation & Monitoring**: Models must be regularly validated against actual outcomes and monitored for drift. This requires stable, well-understood feature engineering pipelines and reproducible training procedures. The use of deterministic random seeds, version-controlled code, and experiment tracking (via MLflow) directly supports this requirement.

For Bati Bank's BNPL product, failing to meet these standards could result in regulatory penalties, increased capital requirements, or suspension of lending authority. Therefore, our model must balance predictive power with regulatory defensibility.

### 2. The Necessity of a Proxy Variable and Its Business Risks

**Why a Proxy is Necessary**

The raw Xente transaction dataset contains **no direct "default" label**—that is, there is no explicit indicator showing which customers failed to repay a loan. This is expected because:

1. The eCommerce partner has not previously offered credit products, so no default history exists.
2. The data captures transactional behavior, not loan performance.
3. We are building a *new* credit product, meaning we must infer creditworthiness from behavioral signals rather than historical defaults.

To train a supervised classification model, we must **engineer a proxy target variable** that approximates credit risk. We use **RFM (Recency, Frequency, Monetary) customer segmentation** via K-Means clustering to identify disengaged, low-value customers and label them as "high-risk" proxies. The underlying assumption is that customers with infrequent transactions, low monetary volumes, and long periods of inactivity exhibit behavioral patterns correlated with higher default probability.

**Business Risks of Proxy-Based Prediction**

Using a proxy variable introduces several material risks that must be explicitly managed:

| Risk | Description | Mitigation |
|------|-------------|------------|
| **Label Noise** | The proxy may misclassify genuinely creditworthy customers as high-risk (false positives) or risky customers as low-risk (false negatives). | Monitor model performance on holdout sets; plan for rapid retraining once actual default data becomes available. |
| **Concept Drift** | The relationship between RFM behavior and default may change over time or differ across populations. | Implement ongoing monitoring; design features to be robust across segments. |
| **Regulatory Scrutiny** | Regulators may challenge the validity of the proxy if it cannot be empirically linked to default probability. | Document the theoretical basis (established literature links low engagement to higher default risk); commit to validation studies. |
| **Fairness & Bias** | The proxy may inadvertently encode demographic or socioeconomic biases present in transactional patterns. | Conduct bias audits; ensure features are behaviorally grounded rather than demographic proxies. |
| **Business Impact** | Overly conservative proxies may reject too many applicants, reducing product uptake and revenue. | Calibrate the proxy threshold using business metrics (expected approval rate, portfolio risk tolerance). |

**Critical Caveat**: The proxy variable is a *modeling assumption*, not ground truth. All downstream predictions are conditional on this assumption holding. Bati Bank's leadership must accept this uncertainty and plan for model updates as real default data accumulates post-launch.

### 3. Trade-offs: Interpretable vs. High-Performance Models

In regulated financial contexts, the choice between a simple, interpretable model and a high-performance complex model involves trade-offs across multiple dimensions:

| Dimension | Interpretable Model (Logistic Regression + WoE) | High-Performance Model (Gradient Boosting) |
|-----------|------------------------------------------------|-------------------------------------------|
| **Regulatory Compliance** | ✅ Highly favorable. Coefficients are directly interpretable; WoE transforms provide monotonic, auditable feature relationships. Regulators can inspect and challenge individual weights. | ⚠️ Challenging. SHAP/LIME explanations are post-hoc approximations; native feature importance lacks directional sign. Requires additional documentation burden. |
| **Predictive Power** | ⚠️ Moderate. Linear decision boundaries may underfit complex, non-linear relationships in behavioral data. | ✅ Superior. Captures interactions, non-linearities, and heterogeneous effects across customer segments. |
| **Feature Engineering** | ✅ WoE binning naturally handles outliers and missing values; produces stable, monotonic features well-suited for scorecard development. | ⚠️ Requires careful handling of outliers, categorical encoding, and missing values; less natural integration with credit scoring conventions. |
| **Implementation & Maintenance** | ✅ Simple deployment; fast inference; easy to update with new data. Scorecards can be implemented in SQL or Excel. | ⚠️ Heavier infrastructure; slower inference; requires MLOps tooling for versioning and monitoring. |
| **Stakeholder Communication** | ✅ Excellent. Risk officers can explain scores in terms of "points per feature." Customers can be given reason codes for adverse decisions (required by regulation in many jurisdictions). | ⚠️ Difficult. Explanations require technical intermediaries; may not satisfy "adverse action" notice requirements without additional tooling. |
| **Stability** | ✅ WoE bins and logistic coefficients tend to be stable across time periods. | ⚠️ Tree-based models can be sensitive to small data changes; require robust validation and ensemble strategies. |

**Recommended Strategy for Bati Bank**

Given Basel II's documentation and interpretability requirements, we propose a **dual-model approach**:

1. **Primary Model**: A **Logistic Regression with WoE-transformed features** serves as the regulatory-compliant baseline. This model meets interpretability requirements, provides reason codes for declined applicants, and can be defended to regulators.

2. **Challenger Model**: A **Gradient Boosting model (XGBoost/LightGBM)** runs in parallel as a shadow model to capture additional predictive signal. Its predictions are used to:
   - Validate the logistic model's rankings (if rankings diverge significantly, investigation is warranted)
   - Inform credit limit and pricing decisions where marginal accuracy gains have direct revenue impact
   - Serve as a fallback if the bank's risk appetite evolves toward performance over interpretability

This approach satisfies the immediate need for regulatory compliance while preserving the option to transition to higher-performance models once sufficient default data validates their superiority and interpretability tooling matures.

---

## Development Workflow

This project follows **Git Flow** with feature branches per task:

1. `task-1`: Business understanding & repository setup
2. `task-2`: Exploratory Data Analysis
3. `task-3`: Feature engineering pipeline
4. `task-4`: Proxy target variable engineering
5. `task-5`: Model training & tracking
6. `task-6`: API deployment & CI/CD

Each task is developed on its own branch and merged to `main` via Pull Request.

---

## License

This project is developed for educational and professional portfolio purposes.

## Contact

For questions regarding this project, please contact the Bati Bank Analytics Team.
