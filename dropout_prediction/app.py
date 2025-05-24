import streamlit as st
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the trained model
model = pickle.load(open('dropout_model.pkl', 'rb'))

# Title and description
st.title("🎓 **Dropout Risk Predictor**")
st.write("This application helps predict whether a student is at risk of dropping out based on various factors. Fill in the student details and click *'Predict Dropout Risk'* to get the prediction.")

# Sidebar
st.sidebar.header("Student Details")
st.sidebar.write("Please provide the following details for the student.")

# Inputs - Organizing in columns for better UI
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    age = st.slider("Age", 15, 25, 18)
    gpa = st.number_input("GPA", min_value=0.0, max_value=4.0, step=0.1)
    attendance = st.slider("Attendance (%)", 0, 100, 75)
    behavior_score = st.slider("Behavior Score (1-10)", 1, 10, 5)
    engagement = st.slider("Engagement Level (1-10)", 1, 10, 5)
    health = st.slider("Health Status (1-5)", 1, 5, 3)

with col2:
    parent_education = st.selectbox("Parental Education Level", ['Primary', 'Secondary', 'Higher'])
    internet_access = st.selectbox("Internet Access", ['Yes', 'No'])
    study_time = st.slider("Daily Study Time (hours)", 0, 5, 2)
    failures = st.slider("Past Failures", 0, 4, 0)
    school_support = st.selectbox("School Support", ['Yes', 'No'])
    family_support = st.selectbox("Family Support", ['Yes', 'No'])
    activities = st.selectbox("Extracurricular Activities", ['Yes', 'No'])
    absences = st.slider("Absences", 0, 30, 5)
    paid_classes = st.selectbox("Attends Paid Classes", ['Yes', 'No'])

# Convert categorical inputs to numeric (match your training preprocessing!)
gender = 1 if gender == 'Male' else 0
parent_education = {'Primary': 0, 'Secondary': 1, 'Higher': 2}[parent_education]
internet_access = 1 if internet_access == 'Yes' else 0
school_support = 1 if school_support == 'Yes' else 0
family_support = 1 if family_support == 'Yes' else 0
activities = 1 if activities == 'Yes' else 0
paid_classes = 1 if paid_classes == 'Yes' else 0

# Combine inputs into a single numpy array
features = np.array([[gender, age, gpa, attendance, behavior_score, engagement,
                      parent_education, internet_access, study_time, failures,
                      school_support, family_support, activities, health,
                      absences, paid_classes]])

# Prediction Button
if st.button("Predict Dropout Risk"):
    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("⚠️ **At Risk of Dropping Out!** ⚠️")
    else:
        st.success("✅ **Not At Risk of Dropping Out!** ✅")

# Show Model Insights (Feature Importance and Confusion Matrix)
if st.checkbox("Show Model Insights"):
    # Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_names = ['Gender', 'Age', 'GPA', 'Attendance', 'Behavior', 'Engagement', 
                     'Parent Education', 'Internet Access', 'Study Time', 'Failures',
                     'School Support', 'Family Support', 'Activities', 'Health', 'Absences', 'Paid Classes']
    
    st.subheader("📊 Feature Importance")
    st.write("This bar chart shows the relative importance of each feature in predicting dropout risk.")
    
    plt.figure(figsize=(10,6))
    sns.barplot(x=importances[indices], y=np.array(feature_names)[indices])
    plt.title('Feature Importance from Random Forest')
    st.pyplot(plt)

    # Confusion Matrix
    st.subheader("🔴 Confusion Matrix")
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix([1], prediction)  # Dummy test case with true label '1'

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not At-Risk', 'At-Risk'], yticklabels=['Not At-Risk', 'At-Risk'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Footer
st.markdown("""
    <div style="text-align: center; color: gray;">
    <p><small>Powered by Streamlit & Random Forest. Model trained on dropout risk prediction data.</small></p>
    </div>
    """, unsafe_allow_html=True)
