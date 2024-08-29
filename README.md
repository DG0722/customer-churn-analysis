# Customer Churn Analysis Project
## Introduction
Since the cost of retaining existing customers is much lower than the cost of acquiring new ones, keeping customers loyal is important to a company's success. Most businesses have
many customers and cannot afford to spend much time on one customer because the cost would be too high and would outweigh the additional revenue. But if companies can predict
in advance which customers are at risk of leaving, they can focus on retaining customers by focusing on these "high-risk" customers.
<br>
This project will help communications companies analyze which customers are stable and which customers are more likely to churn.
## Data Overview
![image](https://github.com/user-attachments/assets/48d140c1-4de4-4a5c-937a-c9b9f3a025f1)
## Analysis with Data Visualization Overview
![image](https://github.com/user-attachments/assets/b4a4943b-e554-40f5-9148-3b78a7ad1ed9)
- The left plot shows how many customers are lost, how many are not.
- The right one shows percentage of churn and non-churn customers.
- 26.5% of the customers have churned out. The other 73.4% have stayed with the company.
<br>

![image](https://github.com/user-attachments/assets/d5aa3ada-dae3-4e9f-ab46-dd1e1a3ddbfd)
- Comparison with age.
- Churn customer: 16.2% is seninor citizen, 83.8% is senior citizen.
<br>

![image](https://github.com/user-attachments/assets/e2aa8015-b929-48e2-b036-3c621c14e813)
- Comparison with seninor citizen.
- There are only 16% of our data set is seninor citizen. The churn rate of non senior citizen is higher than seninor citizen.
<br>

![image](https://github.com/user-attachments/assets/10967055-a4df-4216-923e-d0836129441a)
- Comparison with multiple lines.
- Churn customer: 42.2% has multiple lines, 48.1% hassingle lines, 9.7% ha no line.
<br>

![image](https://github.com/user-attachments/assets/10967055-a4df-4216-923e-d0836129441a)
- Comparison with payment methods.
- Churn customer: 22.9% use mailed check, 21.9% use bank transfer(auto), 21.6% use credit card (auto), 33.6% use electronic check.
<br>

![image](https://github.com/user-attachments/assets/c14d38da-d563-4aad-a8f8-cb8033571aca)

![image](https://github.com/user-attachments/assets/7f193932-5205-4e78-9971-ca6a23105d27)
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
![image](https://github.com/user-attachments/assets/c205f63c-5016-43bc-9b9b-6e0232a34912)
<br>

![image](https://github.com/user-attachments/assets/fddc9a84-5937-4790-9f4b-67e6fabae0cc)

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
