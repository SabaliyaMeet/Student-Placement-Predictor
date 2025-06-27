# 🎓 Student Placement Predictor

An end-to-end machine learning project that predicts student placement outcomes based on CGPA and IQ. Includes data preprocessing, model evaluation, threshold analysis, and an interactive web interface using Streamlit.

---

## 📌 Problem Statement

Campus placement is a critical phase for students and institutions alike. This project aims to predict whether a student is likely to get placed based on two academic indicators:
- **CGPA** (Cumulative Grade Point Average)
- **IQ** (Intelligence Quotient)

By using logistic regression, we build a simple yet effective binary classification model with a real-time prediction interface.

---

## 🚀 Features

- 📊 **EDA & Visualization** — Univariate and bivariate plots using Seaborn
- 🤖 **Logistic Regression** — Binary classification using scikit-learn
- 📉 **Model Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix
- ⚖️ **Threshold Tuning** — Compare default vs optimal classification thresholds
- 🧪 **Cross-Validation** — Model validation using stratified folds
- 🌐 **Web Interface** — Deployed Streamlit app for real-time predictions

---

## 🧠 Tech Stack

| Category | Tools |
|---------|-------|
| Language | Python 3 |
| ML | scikit-learn, joblib |
| Data Analysis | pandas, numpy |
| Visualization | matplotlib, seaborn |
| App Interface | Streamlit |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

- student-placement-predictor/
- ├── data/ # Dataset CSV
- ├── notebooks/ # EDA and model development
- ├── src/ # Modular Python scripts
- │ └── model.py
- ├── app.py # Streamlit app
- ├── model.pkl # Trained model
- ├── requirements.txt # Python dependencies
- ├── README.md # Project documentation
- └── .gitignore

---


---

## 📊 Dataset Overview

The dataset contains the following columns:

| Column     | Type    | Description                         |
|------------|---------|-------------------------------------|
| `cgpa`     | float   | Student’s CGPA                      |
| `iq`       | int     | Student’s IQ score                  |
| `placement`| binary  | Target variable: 1 = Placed, 0 = Not Placed |

> 📌 Note: This is a synthetic but realistic dataset generated for demonstration purposes.

---

## 📦 Setup Instructions

```bash
# Clone the repository
git clone https://github.com/SabaliyaMeet/student-placement-predictor.git
cd student-placement-predictor

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py

```
### 🔗 Live Demo
🎯 Try it out here: [https://your-streamlit-link.streamlit.app](https://student-placement-predictor.streamlit.app/)
(Enter CGPA and IQ → get instant prediction)

---

### ✅ Results & Metrics
- Metric	Value (Example)
- Accuracy	91.2%
- Precision	89.5%
- Recall	93.1%
- F1 Score	91.2%
- ROC-AUC	0.94

The model performs well on imbalanced classes with proper threshold tuning.

---

### 📈 Future Improvements
- Add more features: internships, projects, communication score

- Use advanced models (e.g., XGBoost, Random Forest)

- Integrate database storage for user input

- Improve UI design with Streamlit components

---

### 🧑‍💻 Author
- Meet Sabaliya
- ML & Python Developer | MCA Graduate
- 📫 LinkedIn: https://www.linkedin.com/in/meet-sabaliya-61ab34207/ 
- ✉️ Mail: sabaliyameet7@gmail.com


