# **Predicting Insurance**

## **Team Members**
- **Martinut Francesco** (324CA)  
  - Contributions: Exploratory Data Analysis (EDA), data preprocessing, RMSLE function, integration of H2O AutoML, and finalization of this README.
- **Mitrofan Theodor** (324CB)  
  - Contributions: Exploratory Data Analysis (EDA), implementation of LGBM, XGB, HGB, hyperparameter tuning with Optuna, a Python-based Neural Network, and managing version control (Git Hero).
- **Raris Vlad** (321CC)  
  - Contributions: Development of two additional models, evaluation of the best-performing model, and test predictions.

---

## **Project Description**
This project aims to predict insurance costs based on user data using advanced machine learning models. The objectives of this challenge are to predict insurance premiums based on various factors, building a robust pipeline to analyze, preprocess, and model data efficiently. By leveraging multiple algorithms and AutoML tools, we evaluate the most suitable approach for generating accurate predictions.

---

## **GitHub Link**
Find the complete project [here](https://github.com/francesco481/predict-insurance).

---

## **Necessary Skills and Knowledge**
To understand and replicate this project, you should have the following skills:
- Proficiency in Python and Jupyter Notebooks.
- Familiarity with machine learning frameworks like H2O, LightGBM (LGBM), XGBoost (XGB), and Histogram-based Gradient Boosting (HGB).
- Experience with Exploratory Data Analysis (EDA), data preprocessing techniques, and neural networks.
- Knowledge of hyperparameter tuning and model evaluation.
- Basics of working with Kaggle datasets.

---

## **Installation and Use Instructions**
1. **Install Required Libraries**:
   - Ensure you have Python 3.8+ and install the necessary libraries.
   - Dependencies include:
     - `pandas`, `numpy`, `matplotlib`, `seaborn` (for EDA)
     - `scikit-learn`, `lightgbm`, `xgboost`, `h2o`, `tensorflow`, and `keras` (for modeling)
   - You may also need Jupyter Notebook for running the project interactively.

2. **Download Dataset**:
   - Obtain the dataset from Kaggle. [Dataset Link](https://www.kaggle.com/competitions/playground-series-s4e12/data)

3. **Run the Notebook**:
   - Open the provided Jupyter Notebook.
   - Execute all cells sequentially (`Run All`).

4. **Verify Other Models** (if uncommented):
   - If other models are uncommented in the notebook, you can verify their performance by:
     - Running those cells to train and evaluate the models.
     - Comparing their results to the previously tuned models.
     - Analyzing their predictive power to identify any inefficiencies and decide whether they should be included in the final model ensemble.

5. **Model Training and Predictions**:
   - The notebook will preprocess the data, train multiple models, and evaluate them.
   - Final predictions for the test set will be generated and saved as a CSV file.

---

## **Key Contributions**

### **Francesco**:
- **EDA**:
  - Conducted detailed exploratory data analysis to understand the dataset's structure and relationships.
  - Visualized data distributions, correlations, and outliers.
- **Preprocessing**:
  - Filling missing values across key columns like `Age`, `Annual Income`, `Number of Dependents`, `Health Score`, and `Credit Score`.
  - Handling outliers in `Previous Claims` using quantile-based filtering.
  - Applied MinMax scaling to normalize `Age`, and robust scaling for `Annual Income` and `Health Score` to improve model convergence.
  - Handled categorical variables (`Marital Status`, `Occupation`, `Customer Feedback`) by filling missing values and encoding them appropriately.
- **H2O AutoML**:
  - Integrated H2O AutoML to automate model selection and compare results against other models.

### **Theo** (Git Hero):
- **EDA**:
  - Supported Francesco in refining EDA and ensuring actionable insights for preprocessing.
- **Model Implementation**:
  - Developed models using LGBM, XGB, and HGB algorithms.
  - Tuned hyperparameters for optimal performance using Optuna.
  - Conducted cross-validation to validate model robustness.
- **Neural Network**:
  - Built a Python-based Neural Network using TensorFlow and Keras for regression tasks.
  - Tuned the architecture with layers, activation functions, and optimizers.
  - Compared the performance of the neural network with other models.
- **Git Management**:
  - Managed version control and collaboration workflows effectively, ensuring smooth integration of contributions from all team members.

### **Vlad**:
- **Additional Models**:
  - Introduced and trained two alternative models to further diversify the predictive approach.
- **Model Evaluation**:
  - Compared model performances using RMSE and R² metrics.
  - Identified the best-performing model.
- **Test Predictions**:
  - Used the final model to make predictions on the test set.
  - Saved the predictions as a CSV file for submission.

---

## **Model Evaluation**
We compared the following models:
- **H2O AutoML**: Automated model building and tuning.
- **LGBM**: Lightweight Gradient Boosting Machine.
- **XGB**: Extreme Gradient Boosting.
- **HGB**: Histogram-based Gradient Boosting.
- **Neural Network**: Custom implementation using TensorFlow and Keras.

Evaluation metrics include RMSLE, R². The best-performing model was selected based on these metrics and was used for generating final predictions.

---

## **Difficulties We Faced During the Project**:

1. **Data Quality Issues**
	- **Handling Outliers**: Identified outliers in the `Previous Claims` column using quantile-based filtering. This step helped in maintaining the data quality and ensured that the models were not unduly influenced by extreme values.
	- **Missing Values**: Managed missing values across key columns such as `Age`, `Annual Income`, `Number of Dependents`, `Health Score`, and `Credit Score` using appropriate imputation methods.
	- **Date Columns**: Removed date-related columns (`Policy Start Date`, `Month`, `Day`) that were not necessary for predictive modeling. Created cyclical encoding for `Month` and `Day` using cosine and sine transformations (`Month_cos`, `Day_cos`). This approach helps the model understand temporal relationships, such as the sequence of months (e.g., after December, January comes).

2. **Model Selection and Tuning**:
   - **Choosing the Right Algorithms**: Selecting the appropriate machine learning algorithms among H2O AutoML, LGBM, XGB, HGB, and neural networks was difficult. We had to compare their performance on the training and validation sets to decide on the best models.
   - **Hyperparameter Tuning**: Fine-tuning hyperparameters using methods like Optuna was complex and time-consuming, requiring extensive experimentation to achieve optimal results.

3. **Integration of Different Approaches**:
   - **Version Control**: Managing version control and collaboration among team members was challenging, especially when working with large datasets and multiple models. Git Hero (Theo) played a crucial role in maintaining a smooth workflow.

4. **Evaluation and Interpretation**:
   - **Final Prediction Interpretation**: Interpreting the predictions from the final models and understanding their implications for real-world insurance pricing was a key challenge, requiring insights from both EDA and model results.

