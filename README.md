# customer-churn-prediction
## Critical Analysis and Interpretation

The following is an in-depth analytical breakdown of the Customer Churn Prediction pipeline, covering Data Preparation, Feature Selection, and Hyperparameter Tuning:

### 1. Dataset Characteristics & The Significance of Class Imbalance
*   **Finding:** The dataset consists of 7,046 records with a highly skewed target distribution (*Churn*), where **73.4% of customers did not churn (No)** and only **26.6% churned (Yes)**[cite: 6].
*   **Critical Interpretation:** This imbalance demonstrates that evaluating the model using *Accuracy* would be highly misleading. If a biased model simply predicted "No Churn" for every customer, it would still achieve a deceptively high accuracy of 73%, completely failing to identify at-risk customers. Therefore, optimizing for metrics like **ROC-AUC** and **Recall** is absolutely essential to ensure the model has strong sensitivity in capturing the signals of churning customers.

### 2. Data Splitting Strategy (Stratified Split)
*   **Finding:** The data was partitioned into three subsets: *Training* (4,932 samples), *Validation* (1,057 samples), and *Testing* (1,057 samples)[cite: 6].
*   **Critical Interpretation:** Implementing a *stratified split* technique was crucial. Verifications across all three subsets confirmed that the ~26.6% churn proportion was identically maintained in each split[cite: 6]. This guarantees that the model is trained and evaluated on distributions that accurately represent the original population. Without stratification, the model risks being trained on unrepresentative data portions, leading to illusory performance metrics when deployed.

### 3. Handling Data Quality Anomalies
*   **Finding:** Initially, the `TotalCharges` feature was parsed as a categorical (`object`/string) data type due to blank rows stemming from brand-new customers (tenure = 0)[cite: 6].
*   **Critical Interpretation:** Overlooking this anomaly would disrupt the modeling process, as algorithms cannot process financial values formatted as text. By converting the empty values to `NaN` and imputing them with the median, `TotalCharges` was successfully recovered as a continuous numerical feature[cite: 6]. This ensures the machine learning pipeline can detect financial behavior patterns without being derailed by data formatting errors.

### 4. Overfitting Prevention via Multicollinearity Analysis
*   **Finding:** Correlation analysis on numerical features revealed that `Tenure` (months with the company) and `TotalCharges` (cumulative billed amount) shared a very high positive correlation (0.83).
*   **Critical Interpretation:** Logically, `TotalCharges` is merely a derivative of `Tenure` multiplied by `MonthlyCharges`. Feeding both overlapping features into the model causes **multicollinearity**, where the model assigns redundant weight to the exact same information. Dropping `TotalCharges` is a strategic move to simplify the model architecture without losing essential variance. This practice directly stabilizes the algorithm and lowers the risk of *overfitting*, forcing the model to anchor its predictions on core foundational indicators: customer loyalty duration (`Tenure`) and monthly financial burden (`MonthlyCharges`).

### 5. The Impact of Hyperparameter Tuning
*   **Finding:** Systematic parameter searches were conducted using `GridSearchCV` / Cross-Validation alongside post-feature selection evaluations.
*   **Critical Interpretation:** Solid data cleaning combined with hyperparameter tuning yielded highly optimized predictive capabilities (e.g., pushing the ROC-AUC score to approximately 0.835 and improving Recall to 0.705). Tuning parameters enables algorithms (such as Random Forest) to find the optimal balance on the *bias-variance tradeoff*. This ensures the model does not merely memorize the training data but actually generalizes underlying customer behaviors to predict future churn accurately and reliably.
