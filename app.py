
import streamlit as st
from src.model import StudentPlacementModel

# Initialize the model
model = StudentPlacementModel("data/realistic_student_placement_data.csv")
model.train()

st.set_page_config(page_title="Student Placement Predictor", layout="centered")
st.title("🎓 Student Placement Predictor")
st.markdown("Enter student details to check if they're likely to be placed.")

# Input fields
cgpa = st.number_input("Enter CGPA (e.g., 7.5)", min_value=0.0, max_value=10.0, step=0.1)
iq = st.number_input("Enter IQ (e.g., 110)", min_value=0, max_value=200, step=1)

threshold_option = st.radio("Prediction Mode", ["Default Threshold (0.5)", "Best Threshold (Auto-Optimized)"])
use_best = threshold_option == "Best Threshold (Auto-Optimized)"

if st.button("🔮 Predict Placement"):
    result = model.predict(cgpa, iq, use_best_threshold=use_best)
    st.write("### Prediction Result")
    st.write(f"**CGPA:** {result['cgpa']}")
    st.write(f"**IQ:** {result['iq']}")
    st.write(f"**Predicted Probability of Placement:** `{result['predicted_probability']:.2f}`")
    st.write(f"**Threshold Used:** `{result['threshold_used']:.2f}`")
    
    if result['placement_prediction']:
        st.success("🎉 The student is likely to be placed!")
    else:
        st.error("📌 The student is less likely to be placed.")
