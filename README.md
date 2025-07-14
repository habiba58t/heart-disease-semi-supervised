# Project Overview
This project was completed as part of my Advanced Machine Learning course and was done in collaboration with two colleagues.

This project focuses on solving a semi-supervised learning problem using the UCI
Heart Disease dataset. The dataset contains both labeled and unlabeled data, where the target column
("num“) has values ranging from 0 to 4, representing the severity of heart disease (0 = no disease, 4 =
highest severity). 
The  goal is to use clustering techniques to get the missing labels and apply classification
models for accurate disease prediction.

## Dataset
- Dataset is downloaded form the kaggle competition page 
- Dataset Features: The dataset includes the following key features:
• ID(Unique ID for each patient)
• age (Age of the patient in years)
• origin (place of study)
• sex (Male/Female)
• cp chest pain type ([typical angina, atypical angina, non-anginal, asymptomatic])
• trestbps resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
• chol(serum cholesterol in mg/dl)
• fbs (if fasting blood sugar > 120 mg/dl)
• restecg (resting electrocardiographic results) - Values: [normal, stt abnormality, lv hypertrophy]
• thalach: maximum heart rate achieved
• exang: exercise-induced angina (True/ False)
• oldpeak:ST depression induced by exercise relative to rest
• slope: the slope of the peak exercise ST segment
• ca: number of major vessels (0-3) colored by fluoroscopy
• thal: [normal; fixed defect; reversible defect]
• num: the predicted attribute


## Tools & Libraries Used

- Python 3.x
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn

## Methodology / Steps

1. **Data Cleaning & Preprocessing**  
   - Handled missing values using different methods based on column, some columns where imputed using mean , other columns like ca column were imputed using a Gradient Boosting Regressor trained on related features (age, thalch, oldpeak). The model predicted missing ca values in both training and test sets, ensuring no missing data remained. This method provides more accurate imputation than simple averages .
   - Encoded categorical features using [One-Hot Encoding / Label Encoding].  
   - Removed outliers based on [e.g., IQR method or Z-score].  
   - Normalized/standardized features to ensure consistent scale for model training.

2. **Data Visualization**  
   - Created Boxplots to identify outliers.  
   - Created a correlation heatmap to observe relationships between features.  
   - Generated pairplots to visualize patterns between features and the target variable.

3. **Semi-Supervised Learning — Filling Missing Labels**  
   Missing num (Target Label) values were imputed using multiple clustering methods (K-Means, Spectral, Fuzzy C-Means) on scaled and reduced features. Clusters were mapped to known labels, and missing labels were assigned by majority vote across methods. If clustering failed, missing values were filled with the most frequent label. This approach improved label accuracy and dataset completeness

4. **Classification Model Training**  
   - Added some new features form existing features to the dataset to inhance model an classification perfomance 
   - Trained supervised models including AdaBoost and gradient descent-based models on the completed dataset.  
   - Further improved performance by training Random Forest, XGBoost, Support Vector Machine (SVM), and Logistic Regression models.  
   - Evaluated model performance using accuracy, F1-score, and cross-validation.

## Results & Conclusion

Several classification models were trained and evaluated on the dataset:

- **Support Vector Machine (SVM):** Achieved an accuracy of 27.5% with moderate precision on the majority class but struggled with minority classes.
- **AdaBoost:** Improved performance with 50.7% accuracy and better balanced precision and recall across classes.
- **Stochastic Gradient Descent (SGD):** Lower accuracy of 18.8%, indicating limited effectiveness for this problem.
- **XGBoost:** Achieved the best accuracy of 61.6%, making it the top-performing model for this task.

The classification reports indicate challenges with class imbalance, as shown by warnings about undefined precision for some classes due to no predicted samples. Future work could focus on handling class imbalance using techniques such as resampling or class-weighting to improve minority class predictions.

Overall, the semi-supervised learning approach combined with XGBoost provides a promising method for heart disease severity prediction on this dataset.
