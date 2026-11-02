# Customer Churn Prediction & Retention Analytics
### An End-to-End Data Science, Machine Learning & Business Analytics Project

**Author:** Nematullah Amiri | PMP® | MBA  
**Project Domain:** Banking Analytics | Customer Intelligence | Predictive Modeling  
**Primary Tools:** Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, SciPy

---

## 1. Project Overview

Customer attrition is a significant challenge for banking institutions, affecting recurring revenue, customer lifetime value (CLV), customer relationships, and long-term profitability.

This project applies an end-to-end data science methodology to analyze customer churn, identify its underlying drivers, predict customers at risk of leaving, and develop targeted retention strategies.

Using a dataset of 10,000 banking customers, the analysis combines descriptive, diagnostic, predictive, and prescriptive analytics to transform customer data into actionable business insights.

A Random Forest classification model is developed to estimate customer churn probabilities, while statistical analysis and customer segmentation support the identification of high-risk customer groups and the prioritization of retention interventions.

The project demonstrates how data science and machine learning can support evidence-based customer relationship management and strategic retention planning in the banking sector.

## 2. Business Problem

Customer churn directly affects a bank's ability to maintain its customer base and generate sustainable revenue.

However, applying the same retention strategy to every customer is inefficient and can result in unnecessary marketing expenditure.

The central business problem addressed in this project is:

**How can a bank identify customers who are likely to leave, understand the factors influencing their decisions, and prioritize cost-effective retention interventions?**

### Business Goal

The primary business objective is to develop an analytical framework that supports a targeted reduction in customer churn by 20% over two quarters (six months), while optimizing retention marketing expenditure.

### Project Objectives

- Measure the overall customer churn rate and identify customer segments with elevated attrition.
- Examine demographic, financial, geographic, and behavioral characteristics associated with churn.
- Apply statistical hypothesis testing to investigate differences between churned and retained customers.
- Develop a machine learning model to estimate customer churn probabilities.
- Segment customers according to predicted churn risk.
- Develop value-based retention recommendations and resource allocation strategies.
- Establish a framework for monitoring retention interventions and evaluating business outcomes.

## 3. Dataset Description

The analysis uses the `Churn_Modelling.csv` dataset, containing 10,000 customer records.

The target variable is `Exited`, which indicates whether a customer has left the bank.

| Dataset characteristic | Description |
|---|---|
| Total observations | 10,000 |
| Target variable | Exited |
| Target classes | 0 = Retained, 1 = Churned |
| Overall churn rate | Approximately 20.4% |
| Data format | CSV |
| Missing values | None reported in the notebook |

### Key Variables

| Variable | Description |
|---|---|
| CreditScore | Customer credit score |
| Geography | Customer geographic location |
| Gender | Customer gender |
| Age | Customer age |
| Tenure | Years of customer relationship |
| Balance | Customer account balance |
| NumOfProducts | Number of banking products |
| HasCrCard | Credit card ownership indicator |
| IsActiveMember | Customer activity status |
| EstimatedSalary | Estimated customer salary |
| Exited | Customer churn indicator (target) |

The notebook also creates an `Age_Group` variable to support demographic segmentation.

**Dataset availability:** The notebook expects `Churn_Modelling.csv` to be uploaded into the Google Colab environment. The dataset is not embedded in the notebook. Its original publisher, source URL, and redistribution license must be documented separately before public release.

## 4. Analytical Methodology

The project follows a four-stage analytical framework.

### 4.1 Descriptive Analytics – What Is Happening?

The first stage examines the overall customer base and establishes a baseline understanding of customer attrition.

Key analytical activities include:

- Customer churn distribution.
- Churn rate by geography.
- Age distribution of retained and churned customers.
- Churn rate by customer age groups.
- Account balance comparisons.
- Customer tenure analysis.
- Credit score distribution.
- Gender-based churn comparisons.
- Customer activity and product usage analysis.

Visualizations are developed using Matplotlib and Seaborn to identify patterns and differences across customer segments.

### 4.2 Diagnostic Analytics – Why Are Customers Leaving?

The second stage investigates the factors associated with customer attrition.

The analysis uses statistical hypothesis testing and segmentation techniques.

**Numerical analysis:**
- Independent-samples t-tests.
- Comparison of feature means between churned and retained customers.
- Correlation analysis.

**Categorical analysis:**
- Chi-square tests of independence.
- Churn rate comparisons by geography, gender, age group, and activity status.
- Customer segment analysis.

**Key findings from the notebook:**

| Customer segment / characteristic | Observed churn rate |
|---|---:|
| Overall customer base | 20.4% |
| Germany | 32.4% |
| Age group 51–60 | 56.2% |
| Inactive members | 26.9% |
| Active members | 14.3% |
| Female customers | 25.1% |
| Male customers | 16.5% |
| Customers with four products | 100% |

The four-product segment contains only 60 customers, so this extreme result should be investigated further before drawing broader conclusions.

The notebook also identifies statistically significant differences in customer age, account balance, and credit score between churned and retained customers.

These findings help identify customer segments that may require additional investigation and targeted retention interventions.

### 4.3 Predictive Analytics – Who Is Likely to Churn?

The third stage develops a machine learning classification model to estimate the probability of customer attrition.

**Selected algorithm: Random Forest Classifier**

The model is configured with 100 decision trees and a fixed random state of 42.

#### Model Development Process

1. Separate the predictor variables from the target variable.
2. Remove non-predictive identifiers: `RowNumber`, `CustomerId`, and `Surname`.
3. Apply one-hot encoding to categorical variables.
4. Split the dataset into training and testing subsets.
5. Use an 80:20 stratified train-test split.
6. Train the Random Forest classifier.
7. Generate class predictions and churn probabilities.
8. Evaluate model performance using classification metrics.
9. Examine feature importance and customer risk segmentation.

#### Model Performance

The following results are taken from the executed predictive analytics pipeline in the notebook.

| Evaluation metric | Test-set result |
|---|---:|
| Accuracy | 85.8% |
| ROC-AUC | 84.4% |
| Precision (Churned) | 75.5% |
| Recall (Churned) | 44.7% |
| F1-score (Churned) | 56.2% |

**Performance interpretation**

- Accuracy of 85.8% indicates that the model correctly classifies approximately 85.8% of test observations.
- ROC-AUC of 84.4% indicates useful discrimination between customers who churn and those who remain.
- Precision of 75.5% indicates that approximately three-quarters of customers predicted to churn actually churned in the test set.
- Recall of 44.7% indicates that the model identifies approximately 44.7% of actual churners.
- The F1-score of 56.2% reflects the balance between precision and recall.

The recall result indicates an opportunity for further model optimization, particularly where missing potential churners carries a significant business cost.

#### Customer Risk Segmentation

The notebook applies probability thresholds to classify customers into three risk categories.

| Risk category | Probability threshold | Customers | Share |
|---|---|---:|---:|
| High Risk | Greater than 70% | 1,387 | 13.9% |
| Medium Risk | 30%–70% | 750 | 7.5% |
| Low Risk | Below 30% | 7,863 | 78.6% |
| **Total** | | **10,000** | **100%** |

These are exploratory full-dataset risk segments generated using the fitted model. They are not independent out-of-sample estimates, because the model was also trained on 80% of these observations.

### 4.4 Prescriptive Analytics – What Should the Bank Do?

The final analytical stage translates churn predictions into practical customer retention recommendations.

The notebook introduces a simple estimated customer lifetime value proxy:

`Estimated_CLV = Balance + (EstimatedSalary × 0.1)`

This is an illustrative customer-value proxy, not a validated financial CLV model.

Customer churn probabilities and estimated customer values are then used to prioritize retention interventions.

#### Recommended Retention Actions

| Risk category | Customer value | Recommended intervention |
|---|---|---|
| High risk | High value | Immediate personal retention call |
| High risk | Lower value | Targeted email campaign |
| Medium risk | High value | Personalized retention offer |
| Medium risk | Lower value | SMS reminder |
| Low risk | Any value | Routine monitoring |

The executed customer segmentation produces the following action groups:

| Action category | Customers |
|---|---:|
| Immediate retention calls | 1,061 |
| Email campaigns | 326 |
| Personalized offers | 562 |
| Routine monitoring | 8,051 |
| **Total** | **10,000** |

These recommendations illustrate how a bank could allocate retention resources according to customer risk and estimated value.

The notebook also explores:
- Personalized customer retention offers.
- Retention team resource allocation.
- Campaign budgeting.
- A seven-day retention campaign timeline.
- A proposed A/B testing framework.
- Expected business impact calculations.

The campaign outcomes and ROI estimates are scenario-based assumptions, not measured results from an executed retention campaign.

## 5. Key Business Insights

The analysis highlights several areas requiring managerial attention.

**1. Geographic differences**

Germany has a churn rate of 32.4%, substantially higher than the overall rate of 20.4%. Further investigation into local market conditions, customer experience, pricing, and competitive pressures is recommended.

**2. Age-related customer attrition**

The 51–60 age group exhibits a churn rate of 56.2%. This suggests a need to examine the financial products, service expectations, and retention requirements of older customers.

**3. Customer engagement**

Inactive members have a churn rate of 26.9%, compared with 14.3% among active members. Customer engagement and reactivation initiatives should therefore be considered important retention priorities.

**4. Product portfolio concerns**

Customers holding four products exhibit 100% churn in this dataset, although the segment contains only 60 observations. This warrants a detailed investigation of product suitability, bundling, and customer experience.

**5. Gender-related differences**

Female customers exhibit a higher observed churn rate than male customers. Further investigation should determine whether this difference reflects customer experience, product suitability, or other underlying factors.

## 6. Technology Stack

| Technology | Application |
|---|---|
| Python | Core programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| SciPy | Statistical hypothesis testing |
| Scikit-learn | Machine learning and model evaluation |
| Jupyter Notebook / Google Colab | Interactive analytical environment |
| Tabulate | Analytical reporting and tabular presentation |

## 7. Project Workflow

The overall analytical workflow can be summarized as follows:

```text
Business Understanding
        |
        v
Data Collection & Loading
        |
        v
Data Understanding
        |
        v
Descriptive Analytics
        |
        v
Diagnostic Analytics
        |
        v
Data Preparation
        |
        v
Random Forest Model Development
        |
        v
Model Evaluation
        |
        v
Customer Risk Segmentation
        |
        v
Prescriptive Analytics
        |
        v
Retention Recommendations
        |
        v
Business Decision Support
```

## 8. Repository Structure

The project repository contains the analytical notebook and supporting project files.

```text
customer-churn-prediction-and-Analysis/
│
├── README.md
├── requirements.txt
├── Requirements.txt
├── Bank_Customer_Churn_Analysis.ipynb
├── app.py
├── customer_churn_model.pkl
│
└── .devcontainer/
```

The dataset is not included in the notebook and must be supplied separately.

As the repository evolves, analytical notebooks, application code, trained models, visualizations, and documentation can be organized into dedicated directories.

## 9. Installation and Execution

### Prerequisites

- Python 3.9 or later.
- Jupyter Notebook or Google Colab.
- Required Python libraries.
- The `Churn_Modelling.csv` dataset.

### Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scipy scikit-learn tabulate jupyter
```

### Run using Google Colab

1. Open the Jupyter Notebook in Google Colab.
2. Upload `Churn_Modelling.csv` when prompted.
3. Execute the notebook cells sequentially.
4. Review the descriptive and diagnostic analysis.
5. Examine the Random Forest model evaluation.
6. Review the customer segmentation and retention recommendations.

**Note:** The notebook uses Google Colab's file-upload functionality. If running locally, the dataset-loading section should be adapted to the local file path.

## 10. Limitations and Future Improvements

Although the project provides a structured analytical framework, several improvements are recommended before operational deployment.

**Model performance**
- Improve churn recall through class-weighting, threshold optimization, and additional model comparisons.
- Compare Random Forest with Logistic Regression, Gradient Boosting, and XGBoost.
- Evaluate the model using cross-validation.
- Examine precision-recall curves and threshold-specific business costs.

**Model validation**
- Generate risk scores on unseen customers rather than relying on in-sample predictions.
- Separate model training, evaluation, and operational scoring datasets.
- Validate probability calibration and monitor model performance over time.

**Business analytics**
- Validate the estimated customer lifetime value methodology.
- Test retention interventions through controlled experiments.
- Measure actual retention lift, campaign costs, and realized ROI.
- Investigate the small four-product customer segment before making strategic decisions.

**Deployment**
- Develop a reproducible prediction pipeline.
- Integrate the model into an interactive application or banking analytics environment.
- Establish periodic model monitoring and performance reporting.

## 11. Conclusion

This project demonstrates how descriptive, diagnostic, predictive, and prescriptive analytics can be integrated into a practical customer retention framework.

By combining statistical analysis, machine learning, customer segmentation, and business recommendations, the project moves beyond identifying historical churn patterns toward supporting proactive retention decisions.

The Random Forest model achieves 85.8% accuracy and an ROC-AUC of 84.4% on the test dataset, providing a useful foundation for further development.

More importantly, the analytical framework demonstrates how banking institutions can use customer data to identify risk patterns, prioritize retention activities, and improve the allocation of customer relationship management resources.

The next stage is to improve model recall, validate customer risk probabilities on unseen data, and evaluate the actual effectiveness of proposed retention interventions.

---

## 12. Author

**Nematullah Amiri | PMP® | MBA**

Data Science | Business Intelligence | Project Analytics | Development Research

- **GitHub:** https://github.com/NematullahAmiri
- **Portfolio:** https://nematullahamiri.github.io/
- **LinkedIn:** Add your LinkedIn profile URL here.

*This project is developed for data science, machine learning, banking analytics, and business decision-support portfolio purposes.*
