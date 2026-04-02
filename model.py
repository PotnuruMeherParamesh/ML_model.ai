import streamlit as st
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# models
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier

# metrics
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score, precision_score, 
                             recall_score, f1_score, classification_report, confusion_matrix)


# For AI insights
from analysis import generate_summary, suggest_improvements


st.set_page_config('ML & AL Insights 🤖')
st.title('Automated ML + AI Insights APP 📋')
st.subheader(':green[To learn the given data and to fit the ML models amd to get AL insights using Gemini]֎🇦🇮')
file = st.file_uploader('Upload the CSV file here 💾', type=['csv'])

if file:
    df = pd.read_csv(file)
    st.write('### Data Preview: 👀')
    st.dataframe(df)
    
    target = st.selectbox(':blue[Select the target variable 🎯]', df.columns)
    
    if target:
        x = df.drop(columns=[target]).copy()
        y = df[target]
        
        # =============
        # preprocessing
        # =============
        
        numeric_cols = x.select_dtypes(include=['int64', 'float64']).columns.to_list()
        categorical_cols = x.select_dtypes(include=['object','string']).columns.to_list()
        
        x[numeric_cols] = x[numeric_cols].fillna(x[numeric_cols].median())
        x[categorical_cols] = x[categorical_cols].fillna('Unknown')
        
        # ==============================
        # Encoding categorical variables
        # ==============================
        
        x = pd.get_dummies(x, columns=categorical_cols, drop_first=True, dtype=int)
        
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            
        # =======================
        # Detect the problem type
        # =======================
        
        if df[target].dtype == 'object' or len(df[target].unique()) < 15:
            problem_type = 'classification'
        else:
            problem_type = 'regression'
            
        st.write(f'### 🔎 Detected Problem Type: :orange[{problem_type.upper()}] 🧠')
        
        # ================
        # Train-test split
        # ================
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        for i in x_train.columns:
            x_train[i] = scaler.fit_transform(x_train[[i]])
            x_test[i] = scaler.transform(x_test[[i]])
            
        # ===============
        # Models Training
        # ===============
        
        results = []
        if problem_type == 'regression':
            models = {'Linear Regression': LinearRegression(),
                      'Random Forest Regressor': RandomForestRegressor(),
                      'Gradient Boosting Regressor': GradientBoostingRegressor()}
            
            for name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
                results.append({'Model Name': name,
                                'R2 Score': round(r2_score(y_test, y_pred),3),
                                'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred)),3)})
                                
        else:
            models = {'Logistic Regression': LogisticRegression(),
                      'Random Forest Classifier': RandomForestClassifier(),
                      'Gradient Boosting Classifier': GradientBoostingClassifier()}
           
            for name, model in models.items():
                model.fit(x_train, y_train)
                y_pred = model.predict(x_test)
                
                results.append({'Model Name': name,
                                'Accuracy': round(accuracy_score(y_test, y_pred),3),
                                'Precision': round(precision_score(y_test, y_pred,average='weighted'),3),
                                'Recall': round(recall_score(y_test, y_pred,average='weighted'),3),
                                'F1 Score': round(f1_score(y_test, y_pred, average='weighted'),3)})
        results_df = pd.DataFrame(results)
        st.write('### :red[Model Results]: 📊')
        st.dataframe(results_df)
        
        if problem_type == 'regression':
            st.bar_chart(results_df.set_index('Model Name')['R2 Score'])
            st.bar_chart(results_df.set_index('Model Name')['RMSE'])
            
        else:
            st.bar_chart(results_df.set_index('Model Name')['Accuracy'])
            st.bar_chart(results_df.set_index('Model Name')['F1 Score'])
            
        # =============
        # AI Insights
        # =============
        
        if st.button(':blue[Generate summary]'):
            summary = generate_summary(results_df)
            st.write(summary)
            
        if st.button(':blue[Suggest improvements]'):
            improvements = suggest_improvements(results_df)
            st.write(improvements)
            
        # ================
        # Download results
        # ================
        
        csv = results_df.to_csv(index=False).encode('utf-8')
        st.download_button('Download Results Here as CSV📋',csv,'Model_results.csv')
        
        
        