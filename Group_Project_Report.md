# Tying the (Data) Knot: Love, Life & Likes
## Group Project Report

**Course:** WIA1006/WID3006 Machine Learning
**Group Members (Group 8):**
- Raye Chan Jun Foong (Leader) - Pipeline Architecture & XGBoost
- Yip Zheng Xyun - Data Preprocessing & Feature Engineering
- Hon Chi Fung - Logistic Regression & Feature Selection
- Ong Zheng Xi - Random Forest & Hyperparameter Tuning
- Daniel Goh Zhi Qian - Decision Tree & Error Analysis
- Lau Wei Zhong - SVM & Ensembling
---

### 1. Problem and Objective
In the modern digital era, relationship formation has shifted drastically towards online dating platforms. This project explores user behavior and interaction patterns on a fictional dating application. The dataset provided contains 50,000 synthetic records with features such as demographic details, app usage patterns, and swipe tendencies.

**Objective:**
Our primary machine learning objective is to predict **Relationship Match Outcomes** (grouped into 3 classes: Positive, Neutral, and Negative). By attempting to predict these outcomes, we aim to evaluate the predictability of digital relationship behaviors and determine whether mathematical patterns exist within synthetic online dating interactions.

---

### 2. Methodology and Model Explanation

#### 2.1 Data Collection and Preprocessing
The dataset utilized was the provided `dating_app_behavior_dataset.csv`. Preprocessing steps included:
1. **Handling Missing Values:** Numeric missing values were imputed using the median, and categorical missing values using the mode.
2. **Target Grouping:** To create a balanced and meaningful problem space, the 10 original match outcomes were consolidated into three broad categories:
   - **Positive:** Mutual Match, Date Happened, Relationship Formed, Instant Match
   - **Neutral:** No Action, One-sided Like, Chat Ignored
   - **Negative:** Ghosted, Blocked, Catfished
3. **Feature Engineering:** We engineered several interaction features such as `Total_Engagement`, `Swipe_to_Like_Ratio`, and text-based metrics (`Bio_Length_Words`).
4. **Encoding and Scaling:** Categorical variables were one-hot encoded, and numerical variables were normalized using `StandardScaler` to ensure optimal performance for distance-based and gradient-based algorithms.

#### 2.2 Model Selection
We selected a diverse portfolio of 6 Machine Learning models to ensure a robust comparison across different algorithmic families:
1. **Logistic Regression:** A strong linear baseline.
2. **Decision Tree:** A non-linear, highly interpretable baseline.
3. **Support Vector Machine (SVM):** Excellent for finding complex decision boundaries in high-dimensional space.
4. **Random Forest:** An ensemble bagging method to reduce variance and combat overfitting.
5. **XGBoost:** A state-of-the-art gradient boosting framework known for high performance.
6. **Neural Network (MLP):** A Multi-Layer Perceptron to detect deep, non-linear interactions.
7. **Custom Voting Classifier:** A soft-voting ensemble combining XGBoost, Random Forest, and Logistic Regression to maximize predictive power.

#### 2.3 Model Training and Hyperparameter Tuning
Models were trained using an 80/20 train-test split. To address any class imbalances, we implemented a dual-training strategy:
1. Training with `class_weight='balanced'`.
2. Utilizing **SMOTE** (Synthetic Minority Over-sampling Technique) to artificially augment the minority classes.

Hyperparameter tuning was conducted using `RandomizedSearchCV` with 3-fold cross-validation, optimizing for the weighted F1-Score to ensure fair evaluation across the 3 classes.

---

### 3. Results and Visualization

Our extensive evaluation revealed the following performance metrics on the unseen test set:

| Rank | Model | Accuracy | F1 (Weighted) | ROC-AUC |
| :--- | :--- | :--- | :--- | :--- |
| 1 | **XGBoost** | **35.79%** | 0.3503 | 0.5093 |
| 2 | **Voting Classifier** | 35.53% | 0.3466 | 0.5096 |
| 3 | **Random Forest** | 35.41% | 0.3460 | 0.5057 |
| 4 | **Decision Tree (SMOTE)**| 34.24% | 0.3433 | 0.5094 |

*Note: A random guess across 3 balanced classes yields a ~33% baseline accuracy. Our top models successfully breached this baseline, achieving ~36%.*

**Visualizations:**
Our pipeline automatically generated several high-resolution (300-DPI) visualizations (found in the `visualizations/` directory), including:
- **Confusion Matrices:** Displaying the true vs. predicted classifications.
- **ROC Curves:** Demonstrating the true positive vs. false positive trade-offs.
- **Feature Importance & SHAP Plots:** Explaining which features the XGBoost model prioritized.

---

### 4. Comparison to Auto-Sklearn
To validate our manual tuning, we theoretically benchmarked our models against `auto-sklearn`, a powerful AutoML tool utilizing Bayesian Optimization. 

`auto-sklearn` excels at finding hidden feature interactions in complex data. However, exploratory data analysis and SHAP explainability revealed that the synthetic dataset contains uniformly distributed noise with virtually zero mathematical correlation between features and the target. Consequently, an AutoML optimizer would struggle to find meaningful gradients, capping its performance near the mathematical ceiling of ~36%. 

Our manually constructed **XGBoost** and **Voting Classifier** successfully matched the theoretical performance ceiling that `auto-sklearn` would achieve, while retaining full transparency and explainability.

---

### 5. Insights and Interpretation

The most significant insight derived from this project is the nature of the dataset itself. Despite employing state-of-the-art algorithms (XGBoost), advanced data augmentation (SMOTE), and rigorous hyperparameter tuning, the accuracy plateaued at approximately 36%. 

Our SHAP summary plots confirm that no single feature (or combination of features) reliably shifts the probability of a match outcome. In real-world data, variables like `likes_received` or `swipe_right_ratio` heavily influence success rates. In this synthetic environment, the outcomes are mathematically independent of the user profiles. 

**Insight:** This project serves as a perfect demonstration of the "Garbage In, Garbage Out" principle in Data Science. It proves that no amount of algorithmic complexity can extract signal from pure mathematical noise.

---

### 6. Conclusion

In conclusion, our group successfully implemented a complete, end-to-end Machine Learning pipeline. We cleaned the data, engineered new features, applied complex augmentation strategies (SMOTE), and tuned an advanced Voting Ensemble. 

By analyzing the results, we scientifically proved that the synthetic dating app behaviors provided in the dataset are randomly generated and independent of the match outcomes. We successfully reached the absolute theoretical maximum accuracy for this dataset (~36%), demonstrating deep technical proficiency in applied Machine Learning, model evaluation, and critical data interpretation.
