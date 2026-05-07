import streamlit as st
import numpy as np
import pickle
import shap
from keras.models import load_model
from sklearn.metrics.pairwise import euclidean_distances

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Nano-QSAR Predictor",
    page_icon="🧪",
    layout="wide"
)

# -------------------------
# LOAD MODEL
# -------------------------
model = load_model("nano_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
X_train = np.load("X_train.npy")

# -------------------------
# SIDEBAR
# -------------------------
st.sidebar.title("🧪 Nano-QSAR Tool")
st.sidebar.info("""
Predict nanoparticle toxicity using:
- ANN Model
- SHAP Explainability
- Applicability Domain
""")

# Sample button
if st.sidebar.button("Load Sample Data"):
    st.session_state.sample = True

# -------------------------
# TITLE
# -------------------------
st.title("🔬 Explainable Nano-QSAR Toxicity Predictor")
st.markdown("Predict cytotoxicity of nanoparticles with interpretability and reliability check.")

# -------------------------
# INPUT SECTION
# -------------------------
st.subheader("🧾 Input Features")

col1, col2 = st.columns(2)

with col1:
    coresize = st.slider("Core Size (nm)", 0.0, 200.0, 50.0)
    hydrosize = st.slider("Hydrodynamic Size (nm)", 0.0, 300.0, 100.0)
    surfcharge = st.slider("Surface Charge (mV)", -100.0, 100.0, 0.0)
    surfarea = st.slider("Surface Area", 0.0, 500.0, 50.0)

with col2:
    Ec = st.slider("Ec", 0.0, 10.0, 1.0)
    Expotime = st.slider("Exposure Time (hrs)", 0.0, 72.0, 24.0)
    dosage = st.slider("Dosage", 0.0, 100.0, 10.0)
    e = st.slider("e", 0.0, 10.0, 1.0)
    NOxygen = st.slider("Number of Oxygen Atoms", 0.0, 50.0, 5.0)

# -------------------------
# APPLICABILITY DOMAIN
# -------------------------
def check_ad(train_data, new_input):
    dist = euclidean_distances(train_data, new_input)
    threshold = np.mean(dist)
    return "Inside Applicability Domain" if np.min(dist) < threshold else "Outside Applicability Domain"

# -------------------------
# PREDICT BUTTON
# -------------------------
if st.button("🔍 Predict Toxicity"):

    input_data = np.array([[
        coresize, hydrosize, surfcharge,
        surfarea, Ec, Expotime, dosage, e, NOxygen
    ]])

    # input_scaled = scaler.transform(input_data)

    pred = model.predict(input_data)[0][0]

    # -------------------------
    # RESULT DISPLAY
    # -------------------------
    st.subheader("📊 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Probability", f"{pred:.4f}")

    with col2:
        if pred > 0.5:
            st.error("⚠️ Toxic")
        else:
            st.success("✅ Non-Toxic")

    # -------------------------
    # APPLICABILITY DOMAIN
    # -------------------------
    st.subheader("🧠 Applicability Domain")

    ad_result = check_ad(X_train, input_data)

    if "Inside" in ad_result:
        st.success(ad_result)
    else:
        st.warning(ad_result)

    # -------------------------
    # SHAP EXPLAINABILITY
    # -------------------------
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    st.subheader("📈 Explainability (SHAP)")

    try:
        # Ensure correct shape
        input_scaled = np.array(input_data).reshape(1, -1)

        # Create explainer
        explainer = shap.KernelExplainer(model.predict, X_train[:50])

        shap_values = explainer.shap_values(input_scaled)

        # Handle list/array cases
        if isinstance(shap_values, list):
            shap_vals = shap_values[0]
        else:
            shap_vals = shap_values

        # Flatten properly
        shap_vals = shap_vals.flatten()

        feature_names = [
            "Core Size", "Hydro Size", "Surface Charge",
            "Surface Area", "Ec", "Exposure Time",
            "Dosage", "e", "NOxygen"
        ]

        # Create dataframe
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Impact": shap_vals
        })

        shap_df = shap_df.sort_values(by="Impact", key=abs, ascending=False)

        # Plot
        fig, ax = plt.subplots()
        ax.barh(shap_df["Feature"], shap_df["Impact"])
        ax.set_title("SHAP Feature Importance")
        st.pyplot(fig)

        # Show table
        st.subheader("🔍 Feature Contribution Ranking")
        st.dataframe(shap_df)

    except Exception as e:
        st.error(f"SHAP error: {e}")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.markdown("Developed for Nano-QSAR Research with Explainable AI")
