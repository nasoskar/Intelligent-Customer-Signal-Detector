# Intelligent-Customer-Signal-Detector
An AI prototype that analyses customer interaction data — such as feedback, support transcripts, complaints, or usage patterns — and surfaces early warning signals or behavioural trends. 

## Introduction

CustomerIQ is an AI prototype interface that uses ML and GenAI to analyze customer data and create a prioritized view of customers who need attention (high risk customers), along with their respective risk scores, and also provide a retention action plan to be used by the operations team of the company to reduce the churn rate before the customers have already decided to stop using the services of that company.

## Approach

This prototype utilizes artificial intelligence to proactively identify at-risk customers and proposes solutions before they churn. First, a Random Forest classifier is used  to classify the risk level of each customer (low, medium, high) based on structured customer data (tenure, billing, complaints, satisfaction, NPS), as well as provide the risk score of the customer on a scale from 0 to 10. Then, by utilizing Claude API, GenAI is used to perform sentiment analysis (frustrated, positive, neutral) on the feedback text of the client, a simple explanation of the issue, as well as propose a retention action based on the structured signals and the transcript of the customer. The LLM call runs on demand per selected customer rather than in batch, to control latency and cost.
 
A confidence-gap check is also added in the workflow in order to account for the model's ambiguous cases. When the top two predicted risk levels are closely matched in probability, a **Needs Review** label is added, indicating the need for manual review, rather than forcing a tier onto an ambiguous case.

## Tools Used

- **Python, Pandas, Matplotlib, Seaborn Scikit-learn (Random Forest)**: data processing, machine learning (Random Forest), and visualizations
- **Anthropic Claude API (claude-sonnet-4-6)**: transcript analysis
- **Streamlit**: interface of the app

## Assumptions made

- For this project, no real customer data was available or required. An LLM tool (Claude) was used for synthetic data generation. The dataset consists of 1,500 rows and 16 columns (including the ground truth label). 
- In this project, "Risk" is an indicator of churn likelihood. However, in real-life scenarios, actual historical churn data, as well as transactions over a long period of time would be needed so that risk level and churn likelihood could be assessed in a more realistic and accurate way. 
- Satisfaction and NPS scores are assumed available for every customer. However, in production, these are often sparse, hard to be computed, and not available for every customer. 

## Input and Output

The app works by uploading a dataset (usually a csv file) following the structure of the synthetic dataset, without the ground truth column. An indicative example of the input is shown in the below table:


| customer_id | product | contract_type | tenure_months | monthly_charge_gbp | payment_delays_last_6mo | num_complaints_last_6mo | open_tickets | days_since_last_contact | satisfaction_score | nps_score | last_contact_channel | payment_method | latest_support_transcript |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CUST-0001 | Premium Add-on | Annual | 15 | 121.57 | 0 | 0 | 0 | 99 | 4.0 | 7 | Web Form | Credit Card | "I wanted to check on the status of my feature request from a few weeks ago. Happy with everything else though." |
| CUST-0002 | Business Starter | Annual | 14 | 131.19 | 6 | 3 | 4 | 8 | 1.8 | 4 | Phone | Credit Card | "I submitted a cancellation request last week and nobody got back to me. If I don't hear back by Friday I'm going to dispute the charge with my bank." | 

The output of the app consists of a risk level, risk score, sentiment label, confidence level, explanation, and recommended action as shown in the table below.

| Output |
|---|---|
| **Risk tier** | High (score 8.7/10) |
| **Confidence** | Confident — no review flag |
| **Sentiment** | Frustrated |
|**Explanation** | Repeated unresolved complaints with very low satisfaction/NPS; transcript confirms ongoing frustration despite multiple contacts. |
| **Action** | Escalate to senior retention specialist within 24h; offer goodwill credit for unresolved billing issue. |