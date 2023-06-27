import streamlit as st
import numpy as np
from web_functions import predict, load_data, proses_data, train_model
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def app(dh, x, y):
    st.title("Prediksi Penyakit Jantung")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input('Usia :', 17, 150)
        sex = st.selectbox('Jenis Kelamin :', [0, 1])
        cp = st.selectbox('Tipe Nyeri Dada :', [0, 1, 2, 3])
        trestbps = st.number_input('Tekanan Darah Istirahat :', 80, 200)
    with col2:
        chol = st.number_input('Kolesterol Serum :', 110, 350)
        fbs = st.number_input('Gula Darah Puasa :', 110, 350)
        restecg = st.selectbox('Hasil Elektrokardiogram Istirahat :', [0, 1, 2])
        thalach = st.number_input('Denyut Jantung Maksimum Tercapai :', 50, 200)
    with col3:
        exang = st.selectbox('Angina yang Dipicu Olahraga :', [0, 1])
        oldpeak = st.number_input('Depresi ST yang Diinduksi oleh Olahraga Terhadap Istirahat :', 0., 10.)
        slope = st.selectbox('Kemiringan Segmen ST Puncak Saat Olahraga :', [0, 1, 2])
        ca = st.selectbox('Jumlah Pembuluh Darah Utama :', [0, 1, 2, 3])
        thal = st.selectbox('Jenis Kelainan pada Thalassemia :', [0, 1, 2])
        
        # nk = st.number_input('Masukkan nilai k :')
        nk=4

    # Convert input values to numpy array
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])

    dh, x, y = load_data()
    x_train, x_test, y_train, y_test = proses_data(x, y)

    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)

    classifier = KNeighborsClassifier(n_neighbors=nk, metric='euclidean')
    classifier.fit(x_train_scaled, y_train)

    y_pred = classifier.predict(x_test_scaled)

    ac = accuracy_score(y_test, y_pred)

    if col1.button("Prediksi"):
        if any(feature == '' for feature in features):
            st.warning("Mohon lengkapi semua inputan.")
        else:
            # Konversi nilai atribut dari string menjadi float
            features_float = np.array(features, dtype=float)

            if any(np.isnan(features_float)):
                st.warning("Terdapat nilai yang tidak valid. Mohon periksa kembali inputan anda.")
            else:
                # Skala atribut input menggunakan StandardScaler
                features_scaled = sc.transform(features_float.reshape(1, -1))

                prediction = classifier.predict(features_scaled)

                # st.info("Prediksi Sukses....")

                if prediction == 1:
                    st.warning("Anda rentan terkena penyakit jantung")
                    st.write("Silahkan periksakan kondisi anda saat ini lebih lanjut ke dokter terdekat!")
                else:
                    st.success("Anda relatif aman dari penyakit jantung")
                    st.write("Tetap jaga kondisi kesehatan anda! Jangan lupa olahraga!")

                # st.write("Model yang digunakan memiliki tingkat akurasi ", ac * 100, "%")
