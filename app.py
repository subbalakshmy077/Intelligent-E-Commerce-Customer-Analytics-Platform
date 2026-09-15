import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

# Set Streamlit Page Config
st.set_page_config(
    page_title="Intelligent E-Commerce Customer Analytics",
    page_icon="🛍️",
    layout="wide"
)

# ---------------------------------------------------------
# Load Saved Data & Artifacts
# ---------------------------------------------------------
@st.cache_data
def load_data():
    master_df = pd.read_csv('olist_cleaned_master.csv')
    customer_df = pd.read_csv('customer_segments.csv')
    master_df['order_purchase_timestamp'] = pd.to_datetime(master_df['order_purchase_timestamp'])
    return master_df, customer_df

@st.cache_resource
def load_models():
    clf_model = joblib.load('rf_repeat_purchase_model.pkl')
    clf_preprocessor = joblib.load('clf_preprocessor.pkl')
    reg_model = joblib.load('rf_clv_model.pkl')
    reg_preprocessor = joblib.load('reg_preprocessor.pkl')
    recommender_data = joblib.load('recommender_system.pkl')
    return clf_model, clf_preprocessor, reg_model, reg_preprocessor, recommender_data

# Load Datasets and Models
df_master, df_customers = load_data()
clf_model, clf_preprocessor, reg_model, reg_preprocessor, category_popularity = load_models()

# ---------------------------------------------------------
# Dashboard Header
# ---------------------------------------------------------
st.title("🛍️ Intelligent E-Commerce Customer Analytics Platform")
st.markdown("A platform for customer segmentation, ML predictions, personalized recommendations, and model interpretability.")

# Create Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Dashboard",
    "👥 Customer Segments",
    "🤖 ML Predictions",
    "💡 Product Recommendations",
    "🔍 Explainable AI (XAI)"
])

# ---------------------------------------------------------
# Tab 1: Executive Dashboard
# ---------------------------------------------------------
with tab1:
    st.header("Executive Overview & KPIs")
    
    # Calculate Key Metrics
    total_revenue = df_master['price'].sum()
    total_orders = df_master['order_id'].nunique()
    total_customers = df_customers['customer_unique_id'].nunique()
    avg_order_value = df_master.groupby('order_id')['price'].sum().mean()
    
    # Render KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"${total_revenue:,.2f}")
    col2.metric("Total Orders", f"{total_orders:,}")
    col3.metric("Unique Customers", f"{total_customers:,}")
    col4.metric("Average Order Value (AOV)", f"${avg_order_value:,.2f}")
    
    st.markdown("---")
    
    # Revenue Trend Chart
    st.subheader("Monthly Revenue Trend")
    # UPDATE THIS LINE (Line 76 in app.py):
    monthly_rev = df_master.set_index('order_purchase_timestamp').resample('ME')['price'].sum().reset_index()
    monthly_rev['Year-Month'] = monthly_rev['order_purchase_timestamp'].dt.strftime('%Y-%m')
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(monthly_rev['Year-Month'], monthly_rev['price'], marker='o', color='#2b5c8f', linewidth=2)
    ax.set_xticklabels(monthly_rev['Year-Month'], rotation=45)
    ax.set_ylabel("Revenue ($)")
    ax.set_title("Sales Growth Over Time")
    ax.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(fig)

# ---------------------------------------------------------
# Tab 2: Customer Segments
# ---------------------------------------------------------
with tab2:
    st.header("Customer Behavior & RFM Segmentation")
    
    col_a, col_b = st.columns([1, 2])
    
    with col_a:
        st.subheader("Segment Breakdown")
        segment_counts = df_customers['Customer_Segment'].value_counts()
        st.dataframe(segment_counts.rename("Customer Count"))
        
    with col_b:
        st.subheader("Average RFM Metrics per Segment")
        rfm_summary = df_customers.groupby('Customer_Segment')[['Recency', 'Frequency', 'Monetary', 'Avg_Order_Value']].mean()
        st.dataframe(rfm_summary.style.format("${:.2f}", subset=['Monetary', 'Avg_Order_Value']))
    
    st.markdown("---")
    st.subheader("Search Customer Profile")
    selected_customer = st.selectbox("Select Customer ID:", df_customers['customer_unique_id'].head(100))
    cust_profile = df_customers[df_customers['customer_unique_id'] == selected_customer]
    st.write(cust_profile[['customer_unique_id', 'Customer_Segment', 'Recency', 'Frequency', 'Monetary', 'Avg_Review_Score']])

# ---------------------------------------------------------
# Tab 3: Machine Learning Predictions
# ---------------------------------------------------------
with tab3:
    st.header("Real-Time ML Predictions")
    st.markdown("Predict Repeat Purchase Probability & Expected Customer Lifetime Value (CLV).")
    
    # Input Form for Custom Prediction
    with st.form("prediction_form"):
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            recency_in = st.number_input("Recency (Days since last order)", min_value=1, max_value=700, value=30)
            avg_order_in = st.number_input("Average Order Value ($)", min_value=5.0, max_value=5000.0, value=120.0)
        with col_p2:
            freq_in = st.number_input("Frequency (Total Orders)", min_value=1, max_value=50, value=1)
            review_in = st.slider("Average Review Score", 1.0, 5.0, 4.5)
        with col_p3:
            payment_in = st.selectbox("Preferred Payment Method", ['credit_card', 'boleto', 'voucher', 'debit_card'])
        
        submit_btn = st.form_submit_button("Run Predictions")
        
    if submit_btn:
        # Construct Input DataFrames
        clf_input = pd.DataFrame([{
            'Recency': recency_in,
            'Avg_Order_Value': avg_order_in,
            'Avg_Review_Score': review_in,
            'Preferred_Payment': payment_in
        }])
        
        reg_input = pd.DataFrame([{
            'Recency': recency_in,
            'Frequency': freq_in,
            'Avg_Order_Value': avg_order_in,
            'Avg_Review_Score': review_in,
            'Preferred_Payment': payment_in
        }])
        
        # Transform & Predict
        clf_prep = clf_preprocessor.transform(clf_input)
        repeat_prob = clf_model.predict_proba(clf_prep)[0][1]
        
        reg_prep = reg_preprocessor.transform(reg_input)
        predicted_clv = reg_model.predict(reg_prep)[0]
        
        # Display Results
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Predicted Repeat Purchase Probability", f"{repeat_prob * 100:.1f}%")
        res_col2.metric("Predicted Customer Lifetime Value (CLV)", f"${predicted_clv:,.2f}")

# ---------------------------------------------------------
# Tab 4: Product Recommendations
# ---------------------------------------------------------
with tab4:
    st.header("Personalized Product Recommendations")
    
    target_cust = st.selectbox("Select Customer to Recommend For:", df_customers['customer_unique_id'].head(50), key="rec_select")
    cust_data = df_customers[df_customers['customer_unique_id'] == target_cust]
    pref_cat = cust_data['Preferred_Category'].values[0] if not cust_data.empty else None
    
    st.write(f"**Preferred Category:** `{pref_cat}`")
    
    # Recommendation Logic
    recs = category_popularity[category_popularity['product_category_name_english'] == pref_cat].head(5)
    if recs.empty:
        recs = category_popularity.head(5)
        
    st.subheader("Top Recommended Products")
    st.table(recs[['product_id', 'avg_price', 'avg_rating', 'total_sales']].rename(
        columns={'avg_price': 'Average Price ($)', 'avg_rating': 'Rating', 'total_sales': 'Total Sales Volume'}
    ))

# ---------------------------------------------------------
# Tab 5: Explainable AI (XAI)
# ---------------------------------------------------------
with tab5:
    st.header("Model Interpretability & Feature Importance")
    st.markdown("Understand the global drivers behind model predictions using SHAP & Feature Importance.")
    
    col_x1, col_x2 = st.columns(2)
    
    with col_x1:
        st.subheader("Global Feature Importance")
        if os.path.exists('global_feature_importance.png'):
            st.image('global_feature_importance.png', use_container_width=True)
        else:
            st.info("Global feature importance plot not found. Run Phase 9 script to generate it.")
            
    with col_x2:
        st.subheader("SHAP Summary Plot")
        if os.path.exists('shap_summary_plot.png'):
            st.image('shap_summary_plot.png', use_container_width=True)
        else:
            st.info("SHAP summary plot not found. Run Phase 9 script to generate it.")