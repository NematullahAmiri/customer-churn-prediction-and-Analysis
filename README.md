# Customer Churn Prediction & Retention Analytics

An end-to-end machine learning project focused on customer churn analysis, predictive modeling, and data-driven customer retention insights in the banking sector.

## Project Overview

Customer churn is an important business challenge for financial institutions because customer attrition can affect recurring revenue, customer relationships, and long-term business performance.

This project applies exploratory data analysis (EDA) and machine learning classification techniques to historical banking customer data to investigate churn patterns and estimate the likelihood of customer attrition.

The project demonstrates how predictive analytics can help organizations identify potential churn risks, understand customer behavior, and support evidence-based retention strategies.

## Business Problem

Financial institutions need to understand which customers are more likely to discontinue their banking relationships.

Without analytical insights, customer retention activities may be reactive rather than data-driven.

This project addresses the following questions:

- What customer characteristics are associated with churn?
- Which customer segments demonstrate higher attrition patterns?
- How effectively can machine learning identify customers who have exited?
- What are the limitations of the predictive model?
- How can predictive insights support customer retention strategies?

## Dataset Description

The analysis uses a banking customer dataset with the following characteristics:

| Attribute | Description |
|---|---|
| Total observations | 10,000 |
| Target variable | `Exited` |
| Churn rate | 20.37% |
| Non-churn rate | 79.63% |
| Problem type | Binary classification |

**Target variable interpretation:**
- `1`: Customer exited.
- `0`: Customer remained.

The observation period and currency units are not established by the available dataset documentation and are therefore not assumed.

**Dataset source:** Add the original dataset source and attribution here.

## Analytical Methodology

The project follows a structured data science workflow:

1. **Business Understanding:** Define the customer churn problem and its business relevance.
2. **Data Understanding:** Examine dataset structure, variables, data quality, and target distribution.
3. **Exploratory Data Analysis:** Investigate customer characteristics and patterns associated with churn.
4. **Data Preparation:** Prepare modeling features and target variables.
5. **Model Development:** Train and evaluate classification models.
6. **Model Evaluation:** Assess predictive performance using classification metrics.
7. **Business Interpretation:** Interpret model performance and discuss potential applications for customer retention.

## Model Evaluation

The reproduced Random Forest evaluation used a stratified 80/20 training and testing split.

### Model Performance

| Evaluation Metric | Result |
|---|---:|
| Accuracy | 86.45% |
| ROC-AUC | 85.26% |
| Precision (Churn Class) | 78.33% |
| Recall (Churn Class) | 46.19% |
| F1-Score (Churn Class) | 58.11% |

*These are reproduced evaluation results and should be verified against the final notebook before being treated as definitive.*

### Performance Interpretation

The model demonstrates an ability to distinguish between customers who exited and those who remained, achieving a ROC-AUC of 85.26%.

However, churn recall is 46.19%, indicating that the model identifies fewer than half of the actual churn cases in the test dataset.

This highlights an important consideration in customer churn prediction: overall accuracy does not necessarily reflect a model's ability to identify the customers who matter most to a retention strategy.

Further investigation of classification thresholds, precision-recall trade-offs, and alternative modeling approaches may improve churn identification.

## Business Insights & Potential Applications

The analytical results can support several customer retention activities:

- **Customer Risk Identification:** Prioritize customers with higher predicted churn probabilities for further assessment.
- **Customer Segmentation:** Investigate characteristics and behavioral patterns associated with customer attrition.
- **Retention Planning:** Help customer relationship teams prioritize potential retention activities.
- **Decision Support:** Use predictive insights alongside customer value, intervention costs, and business knowledge.

These are potential applications of the analytical approach, not proven outcomes of a retention intervention.

## Project Limitations

- Predictive performance does not establish the causes of customer churn.
- The dataset does not establish the observation window or currency units used for its financial variables.
- Model performance may change across customer populations and business environments.
- The results depend on the data preparation, feature selection, and evaluation procedures.
- Customer lifetime value, retention costs, and intervention effectiveness have not been established.
- The model requires further validation before operational deployment.

## Technology Stack

- **Programming:** Python
- **Data Analysis:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Data Visualization:** Matplotlib, Seaborn
- **Development Environment:** Jupyter Notebook

## Repository Structure

Update this structure to match the actual files in your repository.

```text
customer-churn-prediction/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   └── customer_churn_analysis.ipynb
│
├── app/
│   └── app.py
│
├── images/
│   ├── churn_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
└── .gitignore
```

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/NematullahAmiri/Customer-Churn-Analysis.git
```

### 2. Navigate to the project directory

```bash
cd Customer-Churn-Analysis
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Open the analysis notebook

Open the Jupyter Notebook in your preferred environment and execute the notebook cells in sequence.

Ensure that the original dataset is available at the location specified in the notebook.

## Project Links

- **GitHub Repository:** https://github.com/NematullahAmiri/Customer-Churn-Analysis
- **Professional Portfolio:** https://nematullahamiri.github.io/

## Author

**Nematullah Amiri**

Data Science | Business Intelligence | Project Analytics

## Disclaimer

This project is intended for educational and analytical demonstration. Model predictions represent estimated patterns based on historical data and should not be interpreted as verified causes of customer churn or guaranteed outcomes of retention strategies.
