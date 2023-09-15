# Customer Churn Analysis Project
## Introduction
Since the cost of retaining existing customers is much lower than the cost of acquiring new ones, keeping customers loyal is important to a company's success. Most businesses have
many customers and cannot afford to spend much time on one customer because the cost would be too high and would outweigh the additional revenue. But if companies can predict
in advance which customers are at risk of leaving, they can focus on retaining customers by focusing on these "high-risk" customers.
<br>
This project will help communications companies analyze which customers are stable and which customers are more likely to churn.
## Data Overview
![image](https://github.com/jdenggao/customer-churn-analysis/assets/112433825/92fcac06-5a21-4d68-925b-8227e7b07d7a)
## Analysis with Data Visualization Overview
![image](https://github.com/jdenggao/customer-churn-analysis/assets/112433825/c1e2fa2f-3375-4299-b2e4-337b2ac3b859)
- The left plot shows how many customers are lost, how many are not.
- The right one shows percentage of churn and non-churn customers.
- 26.5% of the customers have churned out. The other 73.4% have stayed with the company.
<br>

![image](https://github.com/jdenggao/customer-churn-analysis/assets/112433825/6694a520-fd03-46ab-9041-ef56cd60ca34)
- Comparison with age.
- Churn customer: 16.2% is seninor citizen, 83.8% is senior citizen.
<br>

![image](https://github.com/jdenggao/customer-churn-analysis/assets/112433825/88a31252-a1d5-4ad2-aa7d-af1550a8bc5b)
- Comparison with multiple lines.
- Churn customer: 42.2% has multiple lines, 48.1% hassingle lines, 9.7% ha no line.
<br>

![image](https://github.com/jdenggao/customer-churn-analysis/assets/112433825/73aefea5-b445-42c7-8254-488705f7adab)
- Comparison with payment methods.
- Churn customer: 22.9% use mailed check, 21.9% use bank transfer(auto), 21.6% use credit card (auto), 33.6% use electronic check.
<br>

![image](https://github.com/jdenggao/customer-churn-analysis/assets/112433825/a7fc65d8-fb16-43df-9dc9-238ca7621766)
- New customers are more likely to leave.
- Customers with higher monthly charges are more likely to leave.

<br>

![image](https://github.com/jdenggao/customer-churn-analysis/assets/112433825/2866dcea-1973-45ac-ae8a-fe6bd3cdfb61)

![image](https://github.com/jdenggao/customer-churn-analysis/assets/112433825/f05f1c77-1db5-4fcc-9e7e-56e949ec6532)
- Correlation of "Churn" with other features.
## Model Building & Evaluation
### Logistic Regression
- Training accuracy: 81.0%
- Test accuracy: 80.4%
- AUC: 0.85
- Cross Validation (k=10): 80%
### Decision Tree
- Training accuracy: 100.0%
- Test accuracy: 72.8%
- AUC: 0.66
- Cross Validation (k=10): 72.8 %
### Gaussian Naive-Bayes
- Training accuracy: 76.0%
- Test accuracy: 75.2%
- AUC: 0.82
- Cross Validation (k=10): 72.5 %
### Model Comparison
![image](https://github.com/jdenggao/customer-churn-analysis/assets/112433825/1aa1d473-948b-4c72-bbb3-0867d971cedc)

![image](https://github.com/jdenggao/customer-churn-analysis/assets/112433825/040402cb-6b9d-42b7-b765-aec6ef161b45)
## Conclusion
- We select some features which are most related our target variable Churn.
  - PaperlessBilling, MonthlyCharges and SeniorCitizen are most positive related features.
    - Paperless customers more likely to leave than non paperless customers.
    - Customers with higher monthly charges are more likely to leave
    - The churn rate of non senior citizen is higher than seninor citizen.
  - Contract, tenure, OnlineSecurity and TechSupport are most negative related features.
    - Monthly contract customers are most likely to leave.
    - Most of the customers who stay with the company are either new or have been with the company for about six years.
    - No Online Security and Tech Support services are more likely to leave.
- As we can see, the best model is the Logistic Regression, which has 80.4% testing score and 85% AUC score.
- Future jobs:
  - Improve data structure, expand data volume, and establish automated data pipelines to
  - Try ensemble model, to get better accuracy.
  - Deploy the project in the cloud to ensure project stability
