# Housing-Prediction-

A machine learning pipeline developed to predict residential property sale prices in Ames, Iowa, using the Ames Housing dataset. This project covers the full lifecycle of a predictive modeling workflow, including data profiling, handling missing values, statistical outlier detection, feature correlation analysis, and hyperparameter optimization[cite: 1].

---

##  Workflow Breakdown

### 1. Exploratory Data Analysis & Profiling
* Inspected dataset dimensionality, data types, and null configurations using `info()` and `describe()`[cite: 1].
* Separated features into distinct numeric and categorical arrays for targeted processing[cite: 1].

### 2. Preprocessing & Data Cleaning
* **Imputation:** Handled missing data via `SimpleImputer` using a `median` strategy for continuous numeric features and a `most_frequent` (mode) strategy for categorical text fields[cite: 1].
* **Outlier Tracking:** Implemented an Interquartile Range (IQR) filter ($Q3 + 1.5 \times IQR$ and $Q1 - 1.5 \times IQR$) across numeric attributes to systematically isolate skewing data points[cite: 1].

### 3. Feature Selection & Engineering
* Generated a correlation matrix targeting the `SalePrice` label to isolate the top 10 most heavily correlated numerical features[cite: 1].
* Evaluated a log transformation (`np.log1p`) on the target distribution to correct skew and stabilize variance[cite: 1].
* Performed dummy variable encoding via One-Hot Encoding (`pd.get_dummies(drop_first=True)`) to transform categorical variables into model-ready features[cite: 1].

### 4. Model Training & Evaluation
* Set up a **5-Fold Cross-Validation** (`KFold(shuffle=True)`) baseline to prevent data leakage and evaluate generalizability[cite: 1].
* Benchmarked a baseline **Linear Regression** model against an ensemble **Random Forest Regressor** using Negative Root Mean Squared Error (RMSE) as the evaluation metric[cite: 1].

### 5. Hyperparameter Tuning
* Implemented parallelized `RandomizedSearchCV` (`n_jobs=-1`) over the Random Forest Regressor to tune crucial parameters including `n_estimators`, `max_depth`, `min_samples_split`, and `min_samples_leaf`[cite: 1].

---

##  Performance & Diagnostics

The final predictive performance is assessed via:
* **RMSE:** Tracks deviation from true market price values[cite: 1].
* **$R^2$ Score:** Quantifies the percentage of target variance successfully captured by the tuned model[cite: 1].
* **Residual Analysis:** Scatter plots mapping actual vs. predicted values against an ideal diagonal line are used to visually verify prediction error patterns[cite: 1].

---

##  Tech Stack & Dependencies

* **Language:** Python
* **Libraries:** `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`[cite: 1]

---

##  How to Setup & Run

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/YOUR_GITHUB_USERNAME/Ames-Housing-Prediction.git](https://github.com/YOUR_GITHUB_USERNAME/Ames-Housing-Prediction.git)
   cd Ames-Housing-Prediction
