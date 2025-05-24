# 🎓 Dropout Risk Predictor

This is a machine learning-powered web app built with **Streamlit** that predicts whether a student is at risk of dropping out, based on various academic, demographic, and behavioral features.

## 🚀 Demo

To run the app locally:

```bash
git clone https://github.com/your-username/dropout-risk-predictor.git
cd dropout-risk-predictor
pip install -r requirements.txt
streamlit run streamlit_app.py

## 📊 Dataset
We used the xAPI-Edu-Data dataset to train the model. It includes student performance metrics, demographic data, and class outcomes.

## 🧠 Model
Algorithm: Random Forest Classifier

Target: Predict if a student is "At-Risk" (Class = L or M)

Features: Gender, Age, GPA, Attendance, Behavior, Engagement, etc.

## 📈 Evaluation Metrics
Accuracy

Precision

Recall

F1 Score

Confusion Matrix

Feature Importance
