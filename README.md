# 🏢 Data-Driven Analysis of High-Growth Enterprise Characteristics

## 📘 Overview
This project presents a **data-driven framework** for identifying and predicting **high-growth enterprises**.  
By integrating multi-source enterprise data—including **capital activity**, **organizational maturity**, **industry positioning**, and **legal risk**—it builds a **multi-dimensional indicator system** and **machine learning models** to uncover the key drivers of enterprise growth.

The project applies **feature engineering**, **clustering**, and **supervised learning algorithms** to evaluate enterprise growth potential and construct a robust predictive system with both high interpretability and accuracy.

---

## 🧠 Research Methodology

### 1. Multi-Dimensional Indicator Construction
Four key evaluation dimensions are developed:
- **Industry Positioning**
- **Capital Activity**
- **Organizational Maturity**
- **Legal Risk**

A weighted composite score (“Growth Potential Index”) is computed, with the top 30% of enterprises labeled as *high-growth potential*.

### 2. Feature Engineering & Data Processing
Nine core quantitative features are extracted, including:
- Financing Frequency (times/year)  
- Paid-in Capital Completion Rate (%)  
- Investor Quality Score  
- Number of Shareholders  
- Log Registered Capital  
- Number of Cases (last year)  
- Defendant Ratio (%)  
- Number of High-Tech Tags  
- Number of Government Support Tags  

Data cleaning, normalization, and outlier handling ensure robust feature quality.

### 3. Clustering Analysis
Unsupervised **K-Means clustering** identifies enterprise subgroups such as:
- High-Risk Capital-Driven Firms  
- Low-Activity Transitional Firms  
- Capital-Driven Growth Firms  
- Policy & Tech-Driven Stable Firms  

### 4. Feature Evaluation & Selection
Using **correlation analysis**, **t-tests**, and **Information Value (IV)** with **Weight of Evidence (WOE)** encoding, the most influential features are ranked.  
Top predictors include:
- Cluster type  
- Recent case frequency  
- Defendant ratio  
- Government & tech support tags

### 5. Predictive Modeling
Multiple supervised algorithms were tested:
- Logistic Regression  
- Support Vector Machine (SVM)  
- XGBoost  
- Random Forest ✅ *(Best performance)*  

**Random Forest achieved:**
- Accuracy: **95%**  
- F1 Score: **0.91**  
- AUC: **0.9915**

---

## 📊 Key Insights

1. **Active Market Engagement Drives Growth**  
   High-growth firms actively participate in market and legal activities, showing strong operational initiative and strategic competition.

2. **Cumulative Effect of Quality Resources**  
   Government support and technological innovation have strong, positive, and compounding effects on growth.

3. **Capital Quality > Capital Quantity**  
   The *Investor Quality Score* contributes more to growth prediction than financing frequency or shareholder count.

4. **Stable and Interpretable Predictive System**  
   Combines WOE/IV-based interpretability with ensemble model accuracy for practical enterprise growth forecasting.

---

## 🛠️ Tech Stack
- **Language:** Python  
- **Libraries:** pandas · scikit-learn · XGBoost · matplotlib · seaborn  
- **Core Techniques:** Feature engineering, clustering, WOE/IV encoding, model optimization, cross-validation

---

## 📈 Results Summary
| Model | F1 Score | AUC | Precision | Recall |
|:------|:---------:|:----:|:---------:|:-------:|
| Logistic Regression | 0.864 | 0.959 | 0.854 | 0.875 |
| SVM | 0.857 | 0.973 | 0.892 | 0.825 |
| XGBoost | 0.900 | 0.991 | 0.900 | 0.900 |
| **Random Forest** | **0.909** | **0.992** | **0.946** | **0.875** |

---

