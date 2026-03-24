import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

# -------------------------------
# Load trained Random Forest model
# -------------------------------
model = joblib.load("yield_model.pkl")

st.title("Wheat Yield Prediction")
st.markdown("""
This tool predicts wheat yield (kg/ha) based on input agricultural indicators.
""")

# -------------------------------
# Load dataset for permutation importance
# -------------------------------
# Make sure you have the same cleaned dataset used for training saved as CSV
data = pd.read_csv("final_merged_crop_dataset.csv") 

# Features and target
features = ["Cereal_Yield_kg_ha", "Agri_Land_Percent", "Fertilizer_Use", "Arable_Land_Percent"]
X = data[features]
y = data["Wheat_Yield_kg_ha"]

# -------------------------------
# Input sliders for prediction
# -------------------------------
cereal_yield = st.slider("Cereal Yield (kg/ha)", 300.0, 40000.0, 4000.0)
agri_land_percent = st.slider("Agricultural Land (%)", 0.0, 100.0, 40.0)
fertilizer_use = st.slider("Fertilizer Use (kg/ha)", 0.0, 200000.0, 1000.0)
arable_land_percent = st.slider("Arable Land (%)", 0.0, 70.0, 15.0)

# -------------------------------
# Predict button
# -------------------------------
if st.button("Predict Wheat Yield"):
    input_features = np.array([[cereal_yield, agri_land_percent, fertilizer_use, arable_land_percent]])
    prediction = model.predict(input_features)
    st.success(f"Predicted Wheat Yield: {prediction[0]:.2f} kg/ha")

    # -------------------------------
    # Permutation Importance
    # -------------------------------
    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
    feat_imp_df = pd.DataFrame({
        "Feature": features,
        "Importance": perm_importance.importances_mean
    }).sort_values(by="Importance", ascending=False)

    st.write("Permutation Feature Importance:")

    # Plot
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x="Importance", y="Feature", data=feat_imp_df, ax=ax, palette="viridis")
    plt.title("Permutation Feature Importance")
    st.pyplot(fig)
