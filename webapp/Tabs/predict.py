import pickle
import streamlit as st
import numpy as np
import pandas as pd
from web_functions import predict, load_data, proses_data, train_model
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def app(dh, x, y):
    st.title("Prediksi Penyakit Jantung")

    option = st.radio("Pilih Opsi: ", ('Prediksi Penyakit Jantung', 'Input Form', 'Upload File'))

    if(option == 'Prediksi Penyakit Jantung'):
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

    elif(option == 'Input Form'):
        st.text("Silahkan input data dibawah ini: ");
        
        col1, col2, col3, col4 = st.columns(4)
        num_rows = st.number_input("Jumlah Baris", min_value=1, value=1, step=1)
        data = []

        for i in range(num_rows):
            st.header(f"Input data ke-{i+1} ")
            row={}

            row['usia'] = st.number_input("Usia", min_value=17, step=1, key=f"age_{i}")
            row['jenis_kelamin'] = st.selectbox('Jenis Kelamin',['Laki-Laki', 'Perempuan'], key=f"jenis_kelamin_{i}")
            row['tipe_nyeri_dada'] = st.selectbox('Tipe Nyeri Dada', ['Tidak ada nyeri dada', 'Nyeri dada tipe non-anginal', 'Nyeri dada tipe angina tidak stabil', 'Nyeri dada tipe angina stabil'], key=f"tipe_nyeri_dada_{i}") 
            row['tekanan_darah_istirahat'] = st.number_input('Tekanan Darah Istirahat', min_value=0, step=1, key=f"tekanan_darah_istirahat_{i}")
            row['kolesterol_serum'] = st.number_input('Kolesterol Serum', min_value=0, step=1, key=f"kolesterol_serum_{i}")
            row['gula_darah_puasa'] = st.number_input('Gula Darah Puasa', min_value=0, step=1, key=f"gula_darah_puasa_{i}")
            row['hasil_elektrokardiogram_istirahat'] = st.selectbox('Hasil Elektrokardiogram Istirahat', ['Hasil normal', 'Memperlihatkan adanya kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST yang ≥ 0.05 mV)', 'Memperlihatkan adanya hipertrofi ventrikel kiri yang pasti menurut kriteria Estes'], key=f"hasil_elektrokardiogram_istirahat_{i}")
            row['denyut_jantung_maksimum_tercapai'] = st.number_input('Denyut Jantung Maksimum Tercapai', min_value=0, step=1, key=f"denyut_jantung_maksimum_tercapai_{i}")
            row['agina'] = st.selectbox('Angina yang Dipicu Olahraga', ['Tidak', 'Ya'], key=f"agina_{i}")
            row['depresi_st'] = st.number_input('Depresi ST yang Diinduksi oleh Olahraga Terhadap Istirahat', min_value=0.0, step=0.1, key=f"depresi_st_{i}", format='%.1f')
            row['kemiringan_st'] = st.selectbox('Kemiringan Segmen ST Puncak Saat Olahraga', ['Kemiringan tidak dapat ditentukan', 'Kemiringan naik', 'Kemiringan turun'], key=f"kemiringan_st_{i}")
            row['jumlah_pembuluh_darah'] = st.number_input('Jumlah Pembuluh Darah Utama', min_value=0, step=1, key=f"jumlah_pembuluh_darah_{i}")
            row['jenis_kelainan'] = st.selectbox('Jenis Kelainan pada Thalassemia', ['Normal', 'Cacat tetap', 'Cacat yang dapat dipulihkan'], key=f"jenis_kelainan_{i}")
            row['heart_disease'] = st.selectbox('Heart Disease', ['Ya', 'Tidak'], key=f"heart_disease_{i}")

            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            data.append(row)

        dataFrame = pd.DataFrame(data)
        st.title("Data yang anda input: ")
        df = pd.DataFrame(dataFrame)
        st.dataframe(df)

        if st.button("predict"):    
            dataFrame['jenis_kelamin'] = dataFrame['jenis_kelamin'].replace({'Laki-Laki':0, 'Perempuan':1})
            dataFrame['tipe_nyeri_dada'] = dataFrame['tipe_nyeri_dada'].replace({'Tidak ada nyeri dada':0, 'Nyeri dada tipe non-anginal':1, 'Nyeri dada tipe angina tidak stabil':2, 'Nyeri dada tipe angina stabil':3})
            dataFrame['hasil_elektrokardiogram_istirahat'] = dataFrame['hasil_elektrokardiogram_istirahat'].replace({'Hasil normal':0, 'Memperlihatkan adanya kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST yang ≥ 0.05 mV)':1, 'Memperlihatkan adanya hipertrofi ventrikel kiri yang pasti menurut kriteria Estes':2})
            dataFrame['agina'] = dataFrame['agina'].replace({'Tidak':0, 'Ya':1})
            dataFrame['kemiringan_st'] = dataFrame['kemiringan_st'].replace({'Kemiringan tidak dapat ditentukan':0, 'Kemiringan naik':1, 'Kemiringan turun':2})
            dataFrame['jenis_kelainan'] = dataFrame['jenis_kelainan'].replace({'Normal':0, 'Cacat tetap':1, 'Cacat yang dapat dipulihkan':2})
            dataFrame['heart_disease'] = dataFrame['heart_disease'].replace({'Ya':0, 'Tidak':1})

            x_data = dataFrame[['usia', 'jenis_kelainan', 'tipe_nyeri_dada', 'tekanan_darah_istirahat', 'kolesterol_serum', 'gula_darah_puasa', 'hasil_elektrokardiogram_istirahat', 'denyut_jantung_maksimum_tercapai', 'agina', 'depresi_st', 'kemiringan_st', 'jumlah_pembuluh_darah', 'jenis_kelainan']]
            y_target = dataFrame['heart_disease']

            scaler = MinMaxScaler()
            x_data_scaled = scaler.fit_transform(x_data)
            x_data_scaled = pd.DataFrame(x_data_scaled, columns=['usia', 'jenis_kelainan', 'tipe_nyeri_dada', 'tekanan_darah_istirahat', 'kolesterol_serum', 'gula_darah_puasa', 'hasil_elektrokardiogram_istirahat', 'denyut_jantung_maksimum_tercapai', 'agina', 'depresi_st', 'kemiringan_st', 'jumlah_pembuluh_darah', 'jenis_kelainan'])

            knnModel = pickle.load(open('knn_model.sav', 'rb'))
            pred_data = knnModel.predict(x_data_scaled)
            knnAccuracy = accuracy_score(pred_data, y_target)
            st.text("Accuracy from your data: " + str(knnAccuracy * 100) + "%")



    elif(option == 'Upload File'):
        st.text("Hallo Dunia, ini upload"); 