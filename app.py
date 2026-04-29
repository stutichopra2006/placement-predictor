import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# --- CONFIG & STYLING ---
st.set_page_config(page_title="Placement Predictor AI", layout="wide")

# --- DATA GENERATION (Simulating your Dataset) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    n_samples = 500
    data = {
        'CGPA': np.random.uniform(6.0, 10.0, n_samples),
        'Aptitude': np.random.randint(50, 100, n_samples),
        'Coding': np.random.randint(50, 100, n_samples),
        'Comm_Skills': np.random.randint(1, 10, n_samples),
        'Projects': np.random.randint(0, 5, n_samples),
        'Internships': np.random.randint(0, 3, n_samples),
        'Attendance': np.random.randint(60, 100, n_samples)
    }
    df = pd.DataFrame(data)
    # Logic for Placement (Simplified ML target)
    df['Placed'] = ((df['CGPA'] * 10 + df['Coding'] + df['Aptitude']) / 3 > 75).astype(int)
    return df

df = load_data()

# --- ML MODEL TRAINING ---
X = df.drop('Placed', axis=1)
y = df['Placed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- SIDEBAR / NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home & Predictor", "Analytics & Metrics"])

if page == "Home & Predictor":
    st.title("🎓 Placement Probability Predictor")
    st.markdown("### Predict your chances with Machine Learning")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Enter Your Stats")
        cgpa = st.slider("CGPA", 0.0, 10.0, 8.0)
        apti = st.number_input("Aptitude Score (0-100)", 0, 100, 75)
        coding = st.number_input("Coding Score (0-100)", 0, 100, 70)
        comm = st.select_slider("Communication Rating", options=list(range(1, 11)), value=7)
        
    with col2:
        proj = st.number_input("Number of Projects", 0, 10, 2)
        intern = st.number_input("Internships", 0, 5, 1)
        attend = st.slider("Attendance %", 0, 100, 85)
        
    if st.button("Predict Probability"):
        user_data = np.array([[cgpa, apti, coding, comm, proj, intern, attend]])
        prediction = model.predict(user_data)[0]
        probability = model.predict_proba(user_data)[0][1]
        
        st.divider()
        if prediction == 1:
            st.success(f"### High Chance! Probability: {probability:.2%}")
            st.balloons()
        else:
            st.warning(f"### Low to Medium Chance. Probability: {probability:.2%}")
            st.info("Tip: Focusing on Coding scores or Internships could boost your profile.")

elif page == "Analytics & Metrics":
    st.title("📊 Model Performance & Insights")
    
    m1, m2 = st.columns(2)
    m1.metric("Model Accuracy", f"{acc:.2%}")
    m2.metric("Dataset Size", f"{len(df)} Students")

    st.subheader("Feature Importance")
    # Showing what matters most for placement
    importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    fig_imp = px.bar(importances, x='Importance', y='Feature', orientation='h', color='Importance')
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("CGPA vs. Placement Correlation")
    fig_scatter = px.scatter(df, x="CGPA", y="Coding", color="Placed", template="plotly_dark")
    st.plotly_chart(fig_scatter, use_container_width=True)
