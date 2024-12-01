import streamlit as st
import pickle
import numpy as np


def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    st.title("Software Developer Salary Prediction")

    st.write("""### We need some information to predict the salary""")

    countries = (
        "United States",
        "India",
        "United Kingdom",
        "Germany",
        "Canada",
        "Brazil",
        "France",
        "Spain",
        "Australia",
        "Netherlands",
        "Poland",
        "Italy",
        "Russian Federation",
        "Sweden",
    )

    education = (
        "Less than a Bachelors",
        "Bachelor’s degree",
        "Master’s degree",
        "Post grad",
    )


    undergradmajor = {
        "Computer science, computer engineering, or software engineering",
        "Another engineering discipline (such as civil, electrical, mechanical, etc.)",
        "Information systems, information technology, or system administration",
        "A natural science (such as biology, chemistry, physics, etc.)",
        "Mathematics or statistics ",
        "Web development or web design",
        "A business discipline (such as accounting, finance, marketing, etc.)",
        "A humanities discipline (such as literature, history, philosophy, etc.)",
        "A social science (such as anthropology, psychology, political science, etc.)",
        "Fine arts or performing arts (such as graphic design, music, studio art, etc.)",
        "Other"
    }

    platformworkedwith = {
        "Windows",
        "Linux;Windows",
        "Linux",
        "Microsoft Azure;Windows"
    }

    techworkedwith = {
        "Node.js",
        ".NET;.NET Core",
        ".NET",
        ".NET;.NET Core;Node.js",
        "Pandas ",
        "Node.js;React Native",
        "Ansible",
        ".NET;Node.js "
    }

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)
    
    undergradmajor = st.selectbox("Undergraduate Major", undergradmajor)
    platformworkedwith = st.selectbox("Platfrom worked with", platformworkedwith)
    techworkedwith = st.selectbox("Techstack worked with", techworkedwith)

    expericence = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, expericence ]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])
        X = X.astype(float)

        salary = regressor.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f}")
