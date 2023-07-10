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
        sex = st.selectbox('Jenis Kelamin :', ['Laki-Laki', 'Perempuan'])
        if(sex == 'Perempuan'):
            sex=0
        else:
            sex=1
        cp = st.selectbox('Tipe Nyeri Dada :', ['Tidak ada nyeri dada', 'Nyeri dada tipe non-anginal', 'Nyeri dada tipe angina tidak stabil', 'Nyeri dada tipe angina stabil'])
        if(cp == 'Tidak ada nyeri dada'):
            cp=0
        elif(cp == 'Nyeri dada tipe non-anginal'):
            cp=1
        elif(cp == 'Nyeri dada tipe angina tidak stabil'):
            cp=2
        elif(cp == 'Nyeri dada tipe angina stabil'):
            cp=3
        trestbps = st.number_input('Tekanan Darah Istirahat :', 0, 500)
    with col2:
        chol = st.number_input('Kolesterol Serum :', 0, 500)
        fbs = st.number_input('Gula Darah Puasa :', 0, 500)
        restecg = st.selectbox('Hasil Elektrokardiogram Istirahat :', ['Hasil normal', 'Memperlihatkan adanya kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST yang ≥ 0.05 mV)', 'Memperlihatkan adanya hipertrofi ventrikel kiri yang pasti menurut kriteria Estes'])
        if(restecg == 'Hasil normal'):
            restecg=0
        elif(restecg == 'Memperlihatkan adanya kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST yang ≥ 0.05 mV)'):
            restecg=1
        elif(restecg == 'Memperlihatkan adanya hipertrofi ventrikel kiri yang pasti menurut kriteria Estes'):
            restecg=2
        thalach = st.number_input('Denyut Jantung Maksimum Tercapai :', 0, 500)
    with col3:
        exang = st.selectbox('Angina yang Dipicu Olahraga :', ['ya', 'tidak'])
        if(exang == 'ya'):
            exang=1
        else: 
            exang=0
        oldpeak = st.number_input('Depresi ST yang Diinduksi oleh Olahraga Terhadap Istirahat :', 0., 10.)
        slope = st.selectbox('Kemiringan Segmen ST Puncak Saat Olahraga :', ['Kemiringan tidak dapat ditentukan', 'Kemiringan naik', 'Kemiringan turun'])
        if(slope == 'Kemiringan tidak dapat ditentukan'):
            slope=0
        elif(slope == 'Kemiringan naik'):
            slope=1
        elif(slope == 'Kemiringan turun'):
            slope=2
        ca = st.number_input('Jumlah Pembuluh Darah Utama :', 0, 3)
        thal = st.selectbox('Jenis Kelainan pada Thalassemia :', ['Normal', 'Cacat tetap', 'Cacat yang dapat dipulihkan'])
        if(thal == 'Normal'):
            thal=0
        elif(thal == 'Cacat tetap'):
            thal=1
        elif(thal == 'Cacat yang dapat dipulihkan'):
            thal=2
        
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
