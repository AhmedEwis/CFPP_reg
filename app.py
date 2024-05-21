###
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error

# Load the model
model = joblib.load('xgb_model_cfpp1.pkl')

# Load the dataset
df = pd.read_csv('mixing.csv')

# Function to get model metrics
def get_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mse, mae, r2, rmse, mape

# Streamlit app
st.title("CFPP Prediction App")

# Sidebar for user input
st.sidebar.header("User Input Parameters")

def user_input_features():
    Tank1_percentage = st.sidebar.slider('Tank1_percentage', 10, 90, 50)
    Tank2_percentage = st.sidebar.slider('Tank2_percentage', 10, 90, 50)
    CFPP_Tank_1 = -20
    CFPP_Tank_2 = st.sidebar.slider('CFPP_Tank_2', -53, -8, -30)
    data = {
        'Tank1_percentage': Tank1_percentage,
        'Tank2_percentage': Tank2_percentage,
        'CFPP_Tank_1': CFPP_Tank_1,
        'CFPP_Tank_2': CFPP_Tank_2
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Main panel
st.write("""
# CFPP Prediction
""")

# Display user input
st.write('### User Input Parameters', input_df)

# Prediction
prediction = model.predict(input_df)
st.write('### Predicted Final_CFPP', prediction[0])

# Tabs for additional analysis
st.write('## Additional Analysis')

tab1, tab2, tab3 = st.tabs(["Graphs", "Analytics", "Metrics"])

# Graphs tab
with tab1:
    st.write("### Feature Distributions")
    fig, ax = plt.subplots(2, 2, figsize=(15, 10))
    
    sns.histplot(df['Tank1_percentage'], kde=True, bins=10, ax=ax[0, 0])
    ax[0, 0].set_title('Tank1_percentage Distribution')
    
    sns.histplot(df['Tank2_percentage'], kde=True, bins=10, ax=ax[0, 1])
    ax[0, 1].set_title('Tank2_percentage Distribution')
    
    sns.histplot(df['CFPP_Tank_2'], kde=True, bins=10, ax=ax[1, 0])
    ax[1, 0].set_title('CFPP_Tank_2 Distribution')
    
    sns.histplot(df['Final_CFPP'], kde=True, bins=10, ax=ax[1, 1])
    ax[1, 1].set_title('Final_CFPP Distribution')
    
    st.pyplot(fig)

# Analytics tab
with tab2:
    st.write("### Relationships")
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    
    sns.scatterplot(x='Tank1_percentage', y='Final_CFPP', data=df, ax=ax[0])
    ax[0].set_title('Tank1_percentage vs Final_CFPP')
    
    sns.scatterplot(x='Tank2_percentage', y='Final_CFPP', data=df, ax=ax[1])
    ax[1].set_title('Tank2_percentage vs Final_CFPP')
    
    st.pyplot(fig)

# Metrics tab
with tab3:
    st.write("### Model Metrics")
    X = df[['Tank1_percentage', 'Tank2_percentage', 'CFPP_Tank_1', 'CFPP_Tank_2']]
    y = df['Final_CFPP']
    y_pred = model.predict(X)
    mse, mae, r2, rmse, mape = get_metrics(y, y_pred)
    
    st.write(f"**Mean Squared Error (MSE)**: {mse}")
    st.write(f"**Mean Absolute Error (MAE)**: {mae}")
    st.write(f"**R-squared (R2)**: {r2}")
    st.write(f"**Root Mean Squared Error (RMSE)**: {rmse}")
    st.write(f"**Mean Absolute Percentage Error (MAPE)**: {mape * 100:.2f}%")
