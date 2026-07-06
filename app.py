import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# Set page config
st.set_page_config(
    page_title="Telecom Customer Churn Predictor",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    /* Gradient Background and general style */
    .reportview-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
    }
    
    /* Title and headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Glassmorphism style cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .metric-title {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    /* Alert style cards */
    .churn-alert {
        background: rgba(239, 68, 68, 0.1);
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 12px;
        padding: 20px;
        color: #fca5a5;
    }
    
    .stay-alert {
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 12px;
        padding: 20px;
        color: #6ee7b7;
    }
</style>
""", unsafe_allow_html=True)

# Load pipeline models and meta config
@st.cache_resource
def load_resources():
    pipeline = joblib.load("best_model.pkl")
    states = joblib.load("states_list.pkl")
    with open("model_features.json", "r") as f:
        features = json.load(f)
    return pipeline, states, features

try:
    pipeline, states_list, model_features = load_resources()
except Exception as e:
    st.error(f"Error loading models or config: {e}. Please ensure 'train_pipeline.py' was run first.")
    st.stop()

# Sidebar Information
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/phone-office.png", width=120)
    st.title("Customer Churn Portal")
    st.markdown("""
    Predict which telecom customers are at high risk of churning to allow proactive retention strategies.
    
    **Business Objectives:**
    - Reduce Churn Rate
    - Increase Retention
    - Maximize Customer Lifetime Value (CLV)
    """)
    
    st.markdown("---")
    st.markdown("### Model Details")
    st.markdown("**Algorithm:** Tuned Decision Tree")
    st.markdown("**Metric:** Accuracy & Recall Focused")

# Main Content Header
st.title("📞 Customer Churn Predictor")
st.markdown("Deploying machine learning models to maximize customer retention.")

tab1, tab2 = st.tabs(["👤 Single Customer Prediction", "📂 Batch Upload Prediction"])

# Tab 1: Single Prediction
with tab1:
    st.markdown("### Enter Customer Profile")
    
    with st.form("single_customer_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            state = st.selectbox("State / Location", options=states_list)
            account_length = st.number_input("Account Length (Days)", min_value=1, max_value=365, value=100)
            intl_plan = st.selectbox("International Plan", ["No", "Yes"])
            vmail_plan = st.selectbox("Voicemail Plan", ["No", "Yes"])
            cust_service_calls = st.number_input("Customer Service Calls", min_value=0, max_value=20, value=1)
            
        with col2:
            day_mins = st.number_input("Total Day Minutes", min_value=0.0, max_value=500.0, value=180.0)
            day_calls = st.number_input("Total Day Calls", min_value=0, max_value=200, value=100)
            day_charge = st.number_input("Total Day Charge (₹)", min_value=0.0, max_value=100.0, value=30.0)
            eve_mins = st.number_input("Total Evening Minutes", min_value=0.0, max_value=500.0, value=200.0)
            eve_calls = st.number_input("Total Evening Calls", min_value=0, max_value=200, value=100)
            eve_charge = st.number_input("Total Evening Charge (₹)", min_value=0.0, max_value=100.0, value=17.0)

        with col3:
            night_mins = st.number_input("Total Night Minutes", min_value=0.0, max_value=500.0, value=200.0)
            night_calls = st.number_input("Total Night Calls", min_value=0, max_value=200, value=100)
            night_charge = st.number_input("Total Night Charge (₹)", min_value=0.0, max_value=100.0, value=9.0)
            intl_mins = st.number_input("Total Intl Minutes", min_value=0.0, max_value=100.0, value=10.0)
            intl_calls = st.number_input("Total Intl Calls", min_value=0, max_value=50, value=4)
            intl_charge = st.number_input("Total Intl Charge (₹)", min_value=0.0, max_value=50.0, value=2.7)
            
        submit = st.form_submit_button("Predict Churn Risk")

    if submit:
        # Calculate engineered features
        total_charges = day_charge + eve_charge + night_charge + intl_charge
        total_usage = day_mins + eve_mins + night_mins + intl_mins
        service_stress = cust_service_calls / (account_length + 1)
        
        # Categorize Revenue Segment based on training boundaries
        if total_charges <= 55.05:
            rev_seg = 'Low'
        elif total_charges <= 63.91:
            rev_seg = 'Medium'
        else:
            rev_seg = 'High'
            
        # Reconstruct the exact feature dict/DataFrame as expected by model_features
        input_data = {feat: 0 for feat in model_features}
        
        # Direct assignments
        input_data["Account length"] = account_length
        input_data["International plan"] = 1 if intl_plan == "Yes" else 0
        input_data["Voice mail plan"] = 1 if vmail_plan == "Yes" else 0
        input_data["Total day minutes"] = day_mins
        input_data["Total day calls"] = day_calls
        input_data["Total eve minutes"] = eve_mins
        input_data["Total eve calls"] = eve_calls
        input_data["Total night minutes"] = night_mins
        input_data["Total night calls"] = night_calls
        input_data["Total intl minutes"] = intl_mins
        input_data["Total intl calls"] = intl_calls
        input_data["Customer service calls"] = cust_service_calls
        input_data["Total Charges"] = total_charges
        input_data["Total_Usage"] = total_usage
        input_data["Service_Stress"] = service_stress
        
        # Set dummy features
        state_col = f"State_{state}"
        if state_col in input_data:
            input_data[state_col] = 1
            
        rev_col = f"Revenue_Segment_{rev_seg}"
        if rev_col in input_data:
            input_data[rev_col] = 1
            
        # Convert to DataFrame with exact column order
        input_df = pd.DataFrame([input_data])[model_features]
        
        # Perform Inference
        prediction = pipeline.predict(input_df)[0]
        probabilities = pipeline.predict_proba(input_df)[0]
        churn_probability = probabilities[1]
        
        st.markdown("### Prediction Result")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            if prediction == 1:
                st.markdown(f"""
                <div class="churn-alert">
                    <h4>⚠️ High Risk Customer</h4>
                    <p>This customer is predicted to <b>CHURN</b>.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="stay-alert">
                    <h4>✅ Loyal Customer</h4>
                    <p>This customer is predicted to <b>STAY</b>.</p>
                </div>
                """, unsafe_allow_html=True)
                
        with col_res2:
            st.metric(label="Churn Probability", value=f"{churn_probability * 100:.1f}%")
            st.progress(float(churn_probability))

# Tab 2: Batch Upload
with tab2:
    st.markdown("### Batch Prediction via CSV Upload")
    st.markdown("Upload a CSV file containing customer profiles. The system will process it, run predictions, and allow downloading the output.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            st.dataframe(input_df.head())
            
            if st.button("Generate Predictions"):
                # Copy df to work on
                df_proc = input_df.copy()
                
                # Perform necessary transformations
                df_proc['Total Charges'] = df_proc['Total day charge'] + df_proc['Total eve charge'] + df_proc['Total night charge'] + df_proc['Total intl charge']
                df_proc['Total_Usage'] = df_proc['Total day minutes'] + df_proc['Total eve minutes'] + df_proc['Total night minutes'] + df_proc['Total intl minutes']
                df_proc['Service_Stress'] = df_proc['Customer service calls'] / (df_proc['Account length'] + 1)
                
                # Revenue Segment bins
                def get_revenue_segment(tc):
                    if tc <= 55.05:
                        return 'Low'
                    elif tc <= 63.91:
                        return 'Medium'
                    else:
                        return 'High'
                df_proc['Revenue_Segment'] = df_proc['Total Charges'].apply(get_revenue_segment)
                
                # Map binary plans
                df_proc['International plan'] = df_proc['International plan'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
                df_proc['Voice mail plan'] = df_proc['Voice mail plan'].apply(lambda x: 1 if str(x).strip().lower() in ['yes', '1', 'true'] else 0)
                
                # Get dummies matching training
                df_proc = pd.get_dummies(df_proc, columns=['State', 'Revenue_Segment'], dtype=int)
                
                # Add missing dummy columns with 0 values
                for col in model_features:
                    if col not in df_proc.columns:
                        df_proc[col] = 0
                        
                # Subset to exact columns in order
                X_batch = df_proc[model_features]
                
                # Predict
                preds = pipeline.predict(X_batch)
                probs = pipeline.predict_proba(X_batch)[:, 1]
                
                # Append outputs
                output_df = input_df.copy()
                output_df['Predicted_Churn'] = preds
                output_df['Churn_Probability'] = probs
                
                st.markdown("### Prediction Results preview")
                st.dataframe(output_df.head(10))
                
                # Download link
                csv_data = output_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Predictions CSV",
                    data=csv_data,
                    file_name="customer_churn_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing CSV file: {e}")
