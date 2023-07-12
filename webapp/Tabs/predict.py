from io import StringIO
import pickle
from more_itertools import tabulate
import streamlit as st
import numpy as np
import pandas as pd
from web_functions import predict, load_data, proses_data, train_model, load_data_sample
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def app(dh, x, y):
    st.title("Prediksi Penyakit Jantung")

    option = st.radio("Pilih Opsi: ", ('Prediksi Penyakit Jantung', 'Input Form', 'Upload File'))

    if(option == 'Prediksi Penyakit Jantung'):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input('Age :', 17, 150)
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

            row['Age'] = st.number_input("Usia", min_value=17, step=1, key=f"Age_{i}")
            row['Sex'] = st.selectbox('Jenis Kelamin',['Laki-Laki', 'Perempuan'], key=f"Sex_{i}")
            row['ChestPainType'] = st.selectbox('Tipe Nyeri Dada', ['Tidak ada nyeri dada', 'Nyeri dada tipe non-anginal', 'Nyeri dada tipe angina tidak stabil', 'Nyeri dada tipe angina stabil'], key=f"ChestPainType_{i}") 
            row['RestingBP'] = st.number_input('Tekanan Darah Istirahat', min_value=0, step=1, key=f"RestingBP_{i}")
            row['Cholesterol'] = st.number_input('Kolesterol Serum', min_value=0, step=1, key=f"Cholesterol_{i}")
            row['FastingBP'] = st.number_input('Gula Darah Puasa', min_value=0, step=1, key=f"FastingBP_{i}")
            row['RestingECG'] = st.selectbox('Hasil Elektrokardiogram Istirahat', ['Hasil normal', 'Memperlihatkan adanya kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST yang ≥ 0.05 mV)', 'Memperlihatkan adanya hipertrofi ventrikel kiri yang pasti menurut kriteria Estes'], key=f"RestingECG_{i}")
            row['MaxHR'] = st.number_input('Denyut Jantung Maksimum Tercapai', min_value=0, step=1, key=f"MaxHR_{i}")
            row['ExerciseAgina'] = st.selectbox('Angina yang Dipicu Olahraga', ['Tidak', 'Ya'], key=f"ExerciseAgina_{i}")
            row['Oldpeak'] = st.number_input('Depresi ST yang Diinduksi oleh Olahraga Terhadap Istirahat', min_value=0.0, step=0.1, key=f"Oldpeak_{i}", format='%.1f')
            row['ST_Slop'] = st.selectbox('Kemiringan Segmen ST Puncak Saat Olahraga', ['Kemiringan tidak dapat ditentukan', 'Kemiringan naik', 'Kemiringan turun'], key=f"ST_Slop_{i}")
            row['CA'] = st.number_input('Jumlah Pembuluh Darah Utama', min_value=0, step=1, key=f"CA_{i}")
            row['Thal'] = st.selectbox('Jenis Kelainan pada Thalassemia', ['Normal', 'Cacat tetap', 'Cacat yang dapat dipulihkan'], key=f"Thal_{i}")
            row['HeartDisease'] = st.selectbox('Heart Disease', ['Ya', 'Tidak'], key=f"HeartDisease_{i}")

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
            dataFrame['Sex'] = dataFrame['Sex'].replace({'Laki-Laki':0, 'Perempuan':1})
            dataFrame['ChestPainType'] = dataFrame['ChestPainType'].replace({'Tidak ada nyeri dada':0, 'Nyeri dada tipe non-anginal':1, 'Nyeri dada tipe angina tidak stabil':2, 'Nyeri dada tipe angina stabil':3})
            dataFrame['RestingECG'] = dataFrame['RestingECG'].replace({'Hasil normal':0, 'Memperlihatkan adanya kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST yang ≥ 0.05 mV)':1, 'Memperlihatkan adanya hipertrofi ventrikel kiri yang pasti menurut kriteria Estes':2})
            dataFrame['ExerciseAgina'] = dataFrame['ExerciseAgina'].replace({'Tidak':0, 'Ya':1})
            dataFrame['ST_Slop'] = dataFrame['ST_Slop'].replace({'Kemiringan tidak dapat ditentukan':0, 'Kemiringan naik':1, 'Kemiringan turun':2})
            dataFrame['Thal'] = dataFrame['Thal'].replace({'Normal':0, 'Cacat tetap':1, 'Cacat yang dapat dipulihkan':2})
            dataFrame['HeartDisease'] = dataFrame['HeartDisease'].replace({'Ya':0, 'Tidak':1})

            x_data = dataFrame[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBP', 'RestingECG', 'MaxHR', 'ExerciseAgina', 'Oldpeak', 'ST_Slop', 'CA', 'Thal']]
            y_target = dataFrame['HeartDisease']

            scaler = MinMaxScaler()
            x_data_scaled = scaler.fit_transform(x_data)
            x_data_scaled = pd.DataFrame(x_data_scaled, columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBP', 'RestingECG', 'MaxHR', 'ExerciseAgina', 'Oldpeak', 'ST_Slop', 'CA', 'Thal'])

            knnModel = pickle.load(open('knn_model.sav', 'rb'))
            pred_data = knnModel.predict(x_data_scaled)
            knnAccuracy = accuracy_score(pred_data, y_target)
            st.text("Accuracy from your data: " + str(knnAccuracy * 100) + "%")



    elif(option == 'Upload File'):
        st.text("Silahkan upload data, dengan struktur file seperti data dibawah ini")
        if st.button("Another Sample"):
            button_clicked = True
            if button_clicked:
                data = load_data_sample()
                updated_df = data.sample(5)
                st.dataframe(updated_df)
        else:
            data = load_data_sample()
            st.dataframe(data.sample(5))

        uploaded_file = st.file_uploader("Choose a CSV file", key="file1", type="csv")

        upload_file2 = st.file_uploader("Choose a CSV file", key="file2", type="csv")

        if uploaded_file is not None:
            dataFrame = pd.read_csv(uploaded_file)
            dataFrameTesting = pd.read_csv(upload_file2)
            # dataFrameFinal = pd.read_csv(uploaded_file)
            # dataFrameFinal = dataFrame
            st.title("Your Data Training:")
            df = pd.DataFrame(dataFrame)
            df.index = range(1, len(df) + 1)
            st.dataframe(df)
            st.text("")

            dataFrame['Sex'] = dataFrame['Sex'].replace({'Laki-Laki':0, 'Perempuan':1})
            dataFrame['ChestPainType'] = dataFrame['ChestPainType'].replace({'Tidak ada nyeri dada':0, 'Nyeri dada tipe non-anginal':1, 'Nyeri dada tipe angina tidak stabil':2, 'Nyeri dada tipe angina stabil':3})
            dataFrame['RestingECG'] = dataFrame['RestingECG'].replace({'Hasil normal':0, 'Memperlihatkan adanya kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST yang ≥ 0.05 mV)':1, 'Memperlihatkan adanya hipertrofi ventrikel kiri yang pasti menurut kriteria Estes':2})
            dataFrame['ExerciseAgina'] = dataFrame['ExerciseAgina'].replace({'Tidak':0, 'Ya':1})
            dataFrame['ST_Slop'] = dataFrame['ST_Slop'].replace({'Kemiringan tidak dapat ditentukan':0, 'Kemiringan naik':1, 'Kemiringan turun':2})
            dataFrame['Thal'] = dataFrame['Thal'].replace({'Normal':0, 'Cacat tetap':1, 'Cacat yang dapat dipulihkan':2})
            dataFrame['HeartDisease'] = dataFrame['HeartDisease'].replace({'Ya':0, 'Tidak':1})

            x_train = dataFrame[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBP', 'RestingECG', 'MaxHR', 'ExerciseAgina', 'Oldpeak', 'ST_Slop', 'CA', 'Thal']]
            y_train = dataFrame['HeartDisease']

            # scaler = MinMaxScaler()
            # x_data_scaled = scaler.fit_transform(x_data)
            # x_data_scaled = pd.DataFrame(x_data_scaled, columns=['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBP', 'RestingECG', 'MaxHR', 'ExerciseAgina', 'Oldpeak', 'ST_Slop', 'CA', 'Thal'])

            # knnModel = pickle.load(open('knn_model.sav', 'rb'))
            # pred_data = knnModel.predict(x_data_scaled)
            # knnAccuracy = accuracy_score(pred_data, y_target)
            # st.text("Accuracy from your data: " + str(knnAccuracy * 100) + "%")

            # x_train, x_test, y_train, y_test = proses_data(x_data, y_target)

            st.title("Your Data Testing:")
            df2 = pd.DataFrame(dataFrameTesting)
            df2.index = range(1, len(df2) + 1)
            st.dataframe(df2)
            st.text("")

            dataFrame['Sex'] = dataFrame['Sex'].replace({'Laki-Laki':0, 'Perempuan':1})
            dataFrame['ChestPainType'] = dataFrame['ChestPainType'].replace({'Tidak ada nyeri dada':0, 'Nyeri dada tipe non-anginal':1, 'Nyeri dada tipe angina tidak stabil':2, 'Nyeri dada tipe angina stabil':3})
            dataFrame['RestingECG'] = dataFrame['RestingECG'].replace({'Hasil normal':0, 'Memperlihatkan adanya kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST yang ≥ 0.05 mV)':1, 'Memperlihatkan adanya hipertrofi ventrikel kiri yang pasti menurut kriteria Estes':2})
            dataFrame['ExerciseAgina'] = dataFrame['ExerciseAgina'].replace({'Tidak':0, 'Ya':1})
            dataFrame['ST_Slop'] = dataFrame['ST_Slop'].replace({'Kemiringan tidak dapat ditentukan':0, 'Kemiringan naik':1, 'Kemiringan turun':2})
            dataFrame['Thal'] = dataFrame['Thal'].replace({'Normal':0, 'Cacat tetap':1, 'Cacat yang dapat dipulihkan':2})
            dataFrame['HeartDisease'] = dataFrame['HeartDisease'].replace({'Ya':0, 'Tidak':1})

            x_test = dataFrame[['Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBP', 'RestingECG', 'MaxHR', 'ExerciseAgina', 'Oldpeak', 'ST_Slop', 'CA', 'Thal']]
            y_test = dataFrame['HeartDisease']

            sc = StandardScaler()

            x_train_scaled = sc.fit_transform(x_train)
            x_test_scaled = sc.transform(x_test)

            classifier = KNeighborsClassifier(n_neighbors=4, metric='euclidean')
            classifier.fit(x_train_scaled, y_train)

            y_pred = classifier.predict(x_test_scaled)

            ac = accuracy_score(y_test, y_pred)

            st.text("Accuracy from your data: " + str(ac * 100) + "%")

            # st.header("Confusion Matrix from your data prediction")
            # cm = confusion_matrix(pred_data,y_target)
            # plt.figure(figsize=(8,6))
            # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            # plt.xlabel("Predicted Labels")
            # plt.ylabel("True Labels")
            # st.pyplot(plt.gcf())

            # st.title("Your Data:")

            # # Create sample data
            # empty = "    "
            # data = {
            #     "No ": empty,
            #     "Label Sebenarnya": y_target,
            #     "Hasil Prediksi": pred_data,
            # }

            # # Convert data to DataFrame
            # df = pd.DataFrame(data)
            # df.index = range(1, len(df) + 1)

            # # Display the DataFrame using tabulate
            # # table = tabulate(df, headers="keys", tablefmt="grid")
            # st.text(df)

            # st.header("Tabel Prediksi: ")
            # # dataFrameHasil = pd.read_csv(uploaded_file)
            # dataHasil = {
            #     "Hasil Prediksi": pred_data,
            # }

            # dataFrameFinal['Hasil Prediksi'] = dataHasil["Hasil Prediksi"]
            # dataFrameFinal.index = range(1, len(dataFrameFinal) + 1)
            # dataFrameFinal['Sex'] = dataFrameFinal['Sex'].replace({0:'Laki-Laki', 1:'Perempuan'})
            # dataFrameFinal['ChestPainType'] = dataFrameFinal['ChestPainType'].replace({0:'Tidak ada nyeri dada', 1:'Nyeri dada tipe non-anginal', 2:'Nyeri dada tipe angina tidak stabil', 3:'Nyeri dada tipe angina stabil'})
            # dataFrameFinal['RestingECG'] = dataFrameFinal['RestingECG'].replace({0:'Hasil normal', 1:'Memperlihatkan adanya kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST yang ≥ 0.05 mV)', 2:'Memperlihatkan adanya hipertrofi ventrikel kiri yang pasti menurut kriteria Estes'})
            # dataFrameFinal['ExerciseAgina'] = dataFrameFinal['ExerciseAgina'].replace({0:'Tidak',1: 'Ya'})
            # dataFrameFinal['ST_Slop'] = dataFrameFinal['ST_Slop'].replace({0:'Kemiringan tidak dapat ditentukan', 1:'Kemiringan naik', 2:'Kemiringan turun'})
            # dataFrameFinal['Thal'] = dataFrameFinal['Thal'].replace({0:'Normal', 1:'Cacat tetap', 2:'Cacat yang dapat dipulihkan'})
            # dataFrameFinal['HeartDisease'] = dataFrameFinal['HeartDisease'].replace({0:'Ya',1: 'Tidak'})
            # dataFrameFinal['Hasil Prediksi'] = dataFrameFinal['Hasil Prediksi'].replace({0:'Ya', 1:'Tidak'})

            # st.dataframe(dataFrameFinal)

