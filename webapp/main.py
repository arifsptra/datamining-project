import streamlit as st
from web_functions import load_data
from Tabs import home
from Tabs import predict
from Tabs import visualisasi

Tabs = {
    "Home": home,
    "Prediction": predict,
    "Visualization": visualisasi
}

# Membuat sidebar
st.sidebar.title("Navigasi")

# Membuat radio option
page = st.sidebar.radio("Halaman", list(Tabs.keys()))

# Load dataset
dh, x, y = load_data()

# Kondisi memanggil fungsi app
if page in ["Prediction", "Visualization"]:
    Tabs[page].app(dh, x, y)
else:
    Tabs[page].app()
