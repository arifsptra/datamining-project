import streamlit as st
import numpy as np
from web_functions import predict, load_data, proses_data, train_model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def app(dh, x, y):
    st.title("Halaman Prediksi")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('input Age :')
        sex = st.text_input('input Sex :')
        cp = st.text_input('input CP :')
        trestbps = st.text_input('input TrestBps :')
    with col2:
        chol = st.text_input('input Chol :')
        fbs = st.text_input('input Fbs :')
        restecg = st.text_input('input RestecG :')
        thalach = st.text_input('input Thalac :')
    with col3:
        exang = st.text_input('input Exang :')
        oldpeak = st.text_input('input OldPeak :')
        slope = st.text_input('input Slope :')
        ca = st.text_input('input CA :')
        thal = st.text_input('input Thal :')

    # Convert input values to numpy array
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

    dh, x, y = load_data()
    x_train, x_test, y_train, y_test = proses_data(x, y)

    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
    classifier.fit(x_train_scaled, y_train)

    y_pred = classifier.predict(x_test_scaled)

    ac = accuracy_score(y_test, y_pred)

    if col1.button("Prediksi"):
        if any(feature == '' for feature in features):
            st.warning("Mohon lengkapi semua input.")
        else:
            # Konversi nilai atribut dari string menjadi float
            features_float = np.array(features, dtype=float)

            if any(np.isnan(features_float)):
                st.warning("Terdapat nilai yang tidak valid. Mohon periksa kembali input.")
            else:
                # Skala atribut input menggunakan StandardScaler
                features_scaled = sc.transform(features_float.reshape(1, -1))

                prediction = classifier.predict(features_scaled)

                st.info("Prediksi Sukses....")

                if prediction == 1:
                    st.warning("Orang tersebut rentan terkena penyakit jantung")
                else:
                    st.success("Orang tersebut relatif aman dari penyakit jantung")

                st.write("Model yang digunakan memiliki tingkat akurasi ", ac * 100, "%")
