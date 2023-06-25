import warnings
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

from web_functions import train_model, calculate_error_rate, load_data, proses_data


warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)


def app(dh, x, y):
    st.title("Visualisasi Prediksi Penyakit Jantung")

    # Load data
    x_train, x_test, y_train, y_test = proses_data(x, y)

    # Train model
    model, score = train_model(x_train, y_train)

    # Predict on test data
    y_pred = model.predict(x_test)

    # Confusion Matrix
    if st.checkbox("Plot Confusion Matrix"):
        cm = confusion_matrix(y_test, y_pred, labels=np.unique(y))
        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        st.pyplot()
        st.subheader("Deskripsi:")
        st.markdown(" - :Jumlah True Positives (TP): 119 Ini berarti model telah dengan benar mengklasifikasikan 119 data sebagai positif berdasarkan label yang sebenarnya.")
        st.markdown(" - :Jumlah False Positives (FP): 4 Model telah salah mengklasifikasikan 4 data sebagai positif, padahal sebenarnya data tersebut adalah negatif.")
        st.markdown(" - :Jumlah True Negatives (TN): 116 Model telah dengan benar mengklasifikasikan 116 data sebagai negatif berdasarkan label yang sebenarnya.")
        st.markdown(" - :Jumlah False Negatives (FN): 18 Model telah salah mengklasifikasikan 29 data sebagai negatif, padahal sebenarnya data tersebut adalah positif.")


    # Error Rate vs K
    if st.checkbox("Plot Error Rate vs K"):
        max_k = 40
        error_rate = calculate_error_rate(x_train, y_train, x_test, y_test, max_k)

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max_k + 1), error_rate, color='blue', linestyle='--', marker='o', markerfacecolor='red',
                 markersize=10)
        plt.title('Error Rate vs K')
        plt.xlabel('K')
        plt.ylabel('Error Rate')
        st.pyplot()
        st.subheader("Deskripsi:")
        st.markdown("Visualisasi tersebut adalah sebuah grafik yang menampilkan garis yang menggambarkan perubahan tingkat kesalahan (error rate) seiring dengan perubahan nilai n_neighbors (K). Grafik ini berguna untuk memvisualisasikan bagaimana performa KNN dipengaruhi oleh jumlah tetangga terdekat yang digunakan. Dengan melihat grafik ini, Kita dapat memilih nilai n_neighbors yang menghasilkan tingkat kesalahan yang paling rendah untuk model KNN yang kita gunakan.")