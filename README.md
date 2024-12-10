# detecting_fraud
# Martinut Francesco 324CA  
# Mitrofan Theodor 324CB  
# Raris Vlad 321CC  

---

## Introduction  
The project aims to create a Large Language Model (LLM) that can predict insurance and
make decisions in real-time. Unlike traditional algorithms, this method leverages the
capabilities of LLMs to identify suspicious patterns.

We will use:
- **Python** for implementing the LLM and for data processing
- **Dataset** imported from Kaggle, providing a large set of transactions
- The model will likely be trained using the **BERT** library, and we will aim to
optimize parameters for enhanced performance

---

## Detailed Component Design  

### 1. Data Import and Preprocessing  
- **Data Import**: We will manually download the dataset, and use the **Pandas** library
to read and manage data tables in our program.
- **Data Processing**: For data manipulation and cleaning, we will use **Pandas** and
employ libraries like **Matplotlib** and **Plotly** to visualize and analyze the data,
identifying necessary features, statistics, etc.

### 2. Model Training  
- We will test various LLMs and parameters to identify the optimal configuration.
- To evaluate model performance, the dataset will be split into training and testing
subsets. For each model, we will calculate the error rate, with a lower error indicating
a better-performing model.

### 3. Showing Results  
- We will display the error matrix and other relevant statistics to illustrate the model’s
performance.

### 4. Conclusion  
The project implementation involves several stages: data import and preprocessing, model
training and evaluation, and, ultimately, saving the model for production-level use. The
model training component is the most complex task.

---

## Estimated Work Hours and Required Resources  
This project is estimated to require **80-100 hours** of work, covering documentation,
model training, optimization, and performance testing. If additional infrastructure for
ongoing model performance evaluation is desired, the time estimate could increase.

---

## Necessary Skills and Knowledge  
To successfully complete this project, the following skills are essential:

- **Data Processing with Pandas**, including preprocessing for any text data,
if present
- **Machine Learning and Deep Learning**, with a focus on fine-tuning LLMs 
and understanding evaluation metrics for imbalanced classification
- **Python Programming** and familiarity with libraries like **Hugging Face Transformers**
for LLMs and **Scikit-learn** for model evaluation
- **Handling and Processing Large Datasets**, including data cleaning and feature engineering
