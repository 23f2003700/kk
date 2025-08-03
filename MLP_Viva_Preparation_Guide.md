# MLP Viva Preparation Guide - Customer Purchase Value Prediction

**Student:** Aaryan Choudhary  
**Roll Number:** 23F2003700  
**Course:** Machine Learning Practice (MLP Project- 2025)  
**Institution:** Indian Institute of Technology Madras

---

## 📋 Project Overview

### **What is your project about?**
My project predicts how much money customers will spend based on their behavioral data. It's a **regression problem** where I need to predict continuous numerical values (purchase amounts in dollars).

### **Why is this problem important?**
- **Business Revenue Forecasting:** Companies need to predict future revenue
- **Customer Segmentation:** Identify high-value vs low-value customers
- **Marketing Strategy:** Target customers with appropriate campaigns
- **Risk Assessment:** Understand potential losses from customer churn

---

## 🎯 Problem Statement Deep Dive

### **What type of ML problem is this?**
- **Type:** Supervised Learning - Regression
- **Goal:** Predict continuous target variable (purchaseValue)
- **Challenge:** Zero-inflated regression problem

### **What makes this problem challenging?**
1. **Zero-Inflated Data:** 79.3% customers have $0 purchases
2. **Extreme Value Range:** $0 to $23.13 billion
3. **Missing Data:** 882,719 missing values in training
4. **Mixed Data Types:** 37 categorical + 13 numerical features

### **Expected Questions & Answers:**

**Q: Why is zero-inflation a problem?**
**A:** Most traditional regression models assume normal distribution. When 79.3% of data is zeros, it creates:
- **Biased predictions** toward zero
- **Poor performance** on non-zero values
- **Skewed loss functions** during training

**Q: How did you handle the zero-inflation?**
**A:** I used:
- **Ensemble methods** (XGBoost + LightGBM) that handle non-linear patterns
- **Square root transformation** to reduce skewness
- **Multiple models** with different configurations to capture various patterns

---

## 📊 Dataset Analysis

### **Dataset Statistics:**
- **Training:** 116,023 customers × 52 columns
- **Test:** 29,006 customers × 51 columns
- **Features:** 37 categorical + 13 numerical
- **Target:** purchaseValue (continuous, non-negative)

### **Target Variable Distribution:**
- **Mean:** $26.56 million
- **Median:** $0 (shows extreme skewness)
- **Maximum:** $23.13 billion
- **Zero values:** 92,038 (79.3%)
- **Non-zero values:** 23,985 (20.7%)

### **Expected Questions & Answers:**

**Q: What does the median being 0 tell you?**
**A:** It indicates that more than 50% of customers have zero purchases, confirming the zero-inflated nature of the data.

**Q: Why is the mean so much higher than the median?**
**A:** This shows **positive skewness** - a few customers with extremely high purchases (billions) pull the mean up while most customers buy nothing.

**Q: How do you interpret these statistics for business?**
**A:** 
- **80% of customers** generate no revenue (need retention strategies)
- **20% of customers** drive all revenue (need premium services)
- **High-value customers** are extremely valuable (need special attention)

---

## 🔧 Data Preprocessing

### **Missing Data Handling:**
```python
# Strategy used:
- Categorical features: Mode (most frequent value)
- Numerical features: Median (robust to outliers)
```

### **Feature Engineering Steps:**
1. **Missing Value Imputation**
2. **Label Encoding** for categorical variables
3. **Standard Scaling** for numerical features
4. **Target Transformation** (square root)

### **Expected Questions & Answers:**

**Q: Why did you use median instead of mean for numerical imputation?**
**A:** Median is **robust to outliers**. Since our data has extreme values (billions), mean would be heavily influenced by these outliers, making it a poor representative value.

**Q: Why square root transformation for target?**
**A:** 
- **Reduces skewness** (makes distribution more normal)
- **Handles zeros naturally** (√0 = 0)
- **Stabilizes variance** across different value ranges
- **Improves model convergence** during training

**Q: Why not log transformation?**
**A:** Log transformation has issues with zeros (log(0) = undefined). Square root handles zeros naturally.

**Q: What is Label Encoding and why did you use it?**
**A:** Label Encoding converts categorical text values to numbers (e.g., 'Male'→0, 'Female'→1). I used it because:
- **Tree-based models** (XGBoost, LightGBM) handle label-encoded features well
- **Memory efficient** compared to one-hot encoding
- **Preserves all categorical information**

**Q: What is Standard Scaling and why is it needed?**
**A:** Standard Scaling converts features to mean=0, std=1. Benefits:
- **Equal feature importance** (prevents features with large values from dominating)
- **Faster convergence** in gradient-based algorithms
- **Consistent feature ranges** for ensemble combination

---

## 🤖 Model Architecture

### **Ensemble Approach:**
- **6 gradient boosting models:** 3 XGBoost + 3 LightGBM
- **Different configurations** for each model
- **Weighted combination** of predictions

### **Model Specifications:**
```
XGBoost Models:
- Trees: 3000-3400
- Depth: 13-15
- Learning rates: 0.007-0.008

LightGBM Models:
- Trees: 2800-3200
- Depth: 14-16
- Learning rates: 0.008-0.009
```

### **Expected Questions & Answers:**

**Q: Why did you choose XGBoost and LightGBM?**
**A:** 
- **Excellent for tabular data** with mixed feature types
- **Handle missing values** automatically
- **Robust to outliers** (important for our extreme values)
- **Non-linear pattern capture** (good for zero-inflated data)
- **Fast training** on large datasets
- **Built-in regularization** prevents overfitting

**Q: What's the difference between XGBoost and LightGBM?**
**A:**
- **XGBoost:** Level-wise tree growth, more conservative, handles overfitting well
- **LightGBM:** Leaf-wise tree growth, faster training, better accuracy on large datasets
- **Together:** Complement each other's strengths

**Q: Why ensemble instead of single model?**
**A:**
- **Reduced overfitting** (bias-variance tradeoff)
- **Better generalization** to unseen data
- **Captures different patterns** with different model configurations
- **More robust predictions** (averages out individual model errors)

**Q: How did you determine the ensemble weights?**
**A:** I used weights [0.32, 0.19, 0.28, 0.13, 0.06, 0.02] based on:
- **Cross-validation performance** of individual models
- **Validation set accuracy**
- **Higher weights** for better-performing models

**Q: Why so many trees (2800-3400)?**
**A:**
- **Complex dataset** with 116K samples needs high capacity
- **Zero-inflated patterns** require more iterations to learn
- **Regularization** prevents overfitting despite high tree count
- **Ensemble averaging** reduces overfitting risk

---

## 📈 Model Training & Validation

### **Training Strategy:**
1. **Train-validation split:** 80-20
2. **Individual model training** on full dataset
3. **Weighted ensemble combination**
4. **Inverse transformation** (square predictions)

### **Expected Questions & Answers:**

**Q: How do you prevent overfitting with so many trees?**
**A:**
- **Regularization parameters** (L1/L2 penalty)
- **Subsampling** (0.85-0.93) - use only part of data per tree
- **Feature subsampling** (0.79-0.88) - use only part of features per tree
- **Early stopping** would monitor validation loss
- **Ensemble averaging** reduces overfitting

**Q: What evaluation metrics would you use?**
**A:**
- **RMSE (Root Mean Square Error):** Penalizes large errors heavily
- **MAE (Mean Absolute Error):** More robust to outliers
- **R² Score:** Explains variance captured by model
- **MAPE (Mean Absolute Percentage Error):** Business-interpretable metric

**Q: How do you know your model is working well?**
**A:**
- **Validation performance** on hold-out set
- **Prediction distribution** matches business expectations
- **No negative predictions** (business constraint)
- **Reasonable value ranges** (0 to billions is realistic)

---

## 🎯 Results & Business Impact

### **Final Predictions:**
- **Total predictions:** 29,006 customers
- **Mean prediction:** $19.66 million
- **Range:** $0 to $7.44 billion
- **Distribution:** 47.9% small, 13.3% medium, 38.8% large values

### **Expected Questions & Answers:**

**Q: How do you interpret these results for business?**
**A:**
- **Revenue forecasting:** Expected total revenue ≈ $570 billion
- **Customer segmentation:** 48% low-value, 39% high-value customers
- **Marketing strategy:** Focus on 39% predicted high-value customers
- **Risk management:** Diversified revenue across customer segments

**Q: Are your predictions realistic?**
**A:** Yes, because:
- **No negative values** (business constraint satisfied)
- **Range matches training data** (0 to billions)
- **Distribution makes sense** (most customers spend little, few spend a lot)
- **Mean is reasonable** given the zero-inflated nature

**Q: How would you validate your model in production?**
**A:**
- **A/B testing** on subset of customers
- **Time-based validation** (train on past, predict future)
- **Business metric tracking** (actual vs predicted revenue)
- **Model monitoring** for prediction drift

---

## 🔬 Technical Deep Dive

### **Algorithm Details:**

**Q: Explain how XGBoost works.**
**A:**
1. **Gradient Boosting:** Sequentially adds weak learners (trees)
2. **Each tree** corrects errors of previous trees
3. **Loss function:** Minimizes prediction errors
4. **Regularization:** Prevents overfitting through tree complexity penalty
5. **Feature importance:** Calculates which features contribute most

**Q: What is the mathematical foundation?**
**A:**
- **Objective function:** Loss + Regularization
- **Gradient descent:** Optimizes tree parameters
- **Second-order approximation:** Uses both gradient and hessian
- **Tree pruning:** Removes branches that don't improve loss

**Q: How does ensemble learning work mathematically?**
**A:**
```
Final_Prediction = Σ(wi × Pi)
where wi = weight of model i, Pi = prediction of model i
```

### **Hyperparameter Tuning:**

**Q: How did you choose hyperparameters?**
**A:**
- **Learning rate:** Lower values (0.007-0.009) for stability with many trees
- **Max depth:** 13-16 for sufficient complexity without overfitting
- **Subsample:** 0.85-0.93 to introduce randomness
- **Regularization:** Small values to allow learning while preventing overfitting

---

## 🚨 Potential Challenges & Solutions

### **Common Issues:**

**Q: What if the model predicts negative values?**
**A:** I apply `np.maximum(predictions, 0)` to ensure all predictions are non-negative, which is a business constraint.

**Q: How do you handle concept drift?**
**A:**
- **Regular retraining** on new data
- **Model monitoring** for performance degradation
- **Feature distribution tracking**
- **Online learning** approaches for real-time updates

**Q: What if new categorical values appear in test data?**
**A:** My label encoder is fitted on combined train+test data, so this shouldn't happen. In production, I'd handle unknown categories with a special "Unknown" encoding.

**Q: How do you scale this to larger datasets?**
**A:**
- **Distributed training** (XGBoost supports this)
- **Feature selection** to reduce dimensionality
- **Data sampling** for initial model development
- **Cloud computing** resources for large-scale training

---

## 📚 Advanced Concepts

### **Zero-Inflated Models:**

**Q: What other approaches exist for zero-inflated data?**
**A:**
1. **Two-stage models:** First predict if purchase>0, then predict amount
2. **Hurdle models:** Combine classification and regression
3. **Zero-inflated regression:** Specialized statistical models
4. **Deep learning:** Neural networks with custom loss functions

### **Feature Engineering:**

**Q: What additional features could improve the model?**
**A:**
- **Interaction features:** Product of important feature pairs
- **Polynomial features:** Non-linear transformations
- **Time-based features:** Seasonality, trends
- **Aggregated features:** Customer segment statistics
- **External data:** Economic indicators, market conditions

### **Model Interpretability:**

**Q: How do you explain your model's decisions?**
**A:**
- **Feature importance:** Which features matter most
- **SHAP values:** Individual prediction explanations
- **Partial dependence plots:** How features affect predictions
- **Tree visualization:** Understanding decision paths

---

## 🎓 Learning Outcomes

### **What did you learn from this project?**
1. **Handling imbalanced data** is crucial for real-world problems
2. **Ensemble methods** provide robust solutions
3. **Feature engineering** significantly impacts performance
4. **Business constraints** must be incorporated into technical solutions
5. **Model validation** requires multiple perspectives

### **How would you improve this project?**
1. **Cross-validation** for better model selection
2. **Feature selection** using statistical methods
3. **Hyperparameter optimization** using Bayesian optimization
4. **Model interpretability** analysis
5. **Production deployment** considerations

---

## 🔍 Viva Tips

### **Be Ready to Explain:**
1. **Every preprocessing step** and why you chose it
2. **Model selection rationale** - why these specific algorithms
3. **Business interpretation** of your results
4. **Limitations** of your approach
5. **Alternative approaches** you considered

### **Demonstrate Understanding:**
1. **Draw the ensemble architecture** on the board
2. **Explain the math** behind gradient boosting
3. **Interpret feature importance** if asked
4. **Discuss trade-offs** between different approaches
5. **Connect technical choices** to business requirements

### **Common Viva Questions:**
1. "Walk me through your entire pipeline"
2. "Why did you choose this approach over others?"
3. "How would you deploy this in production?"
4. "What are the limitations of your model?"
5. "How would you improve your results?"

---

**Good luck with your viva! Remember to speak confidently about your choices and be ready to justify every technical decision with both mathematical and business reasoning.**
