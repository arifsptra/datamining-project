import streamlit as st
from web_functions import load_data
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


def app():
    dh, x, y = load_data()
    st.title("Infomarsi Dataset Heart Disease")
    st.markdown("Dataset yang saya dapat dari website Kaggle ini merupakan dataset yang berasal dari tahun 1988 dan terdiri dari empat basis data: Cleveland, Hungary, Switzerland, dan Long Beach V. Dataset ini terdiri dari 76 atribut, termasuk atribut yang diprediksi, tetapi semua eksperimen yang dipublikasikan mengacu pada penggunaan subset 14 atribut tersebut. Kolom target mengacu pada keberadaan penyakit jantung pada pasien. Nilainya merupakan bilangan bulat, di mana 0 = tidak ada penyakit dan 1 = ada penyakit.")

    st.subheader("Informasi Atribut Dataset")
    st.markdown("1. **Age (Usia):** Usia pasien dalam tahun.")
    st.markdown("2. **Sex (Jenis Kelamin):** Jenis kelamin pasien. Nilai 0 menandakan perempuan, dan nilai 1 menandakan laki-laki.")
    st.markdown("3. **Chest Pain Type (Tipe Nyeri Dada):** Tipe nyeri dada yang dialami oleh pasien. Terdapat 4 nilai yang mungkin:")
    st.markdown("   - 0: Tidak ada nyeri dada.")
    st.markdown("   - 1: Nyeri dada tipe non-anginal.")
    st.markdown("   - 2: Nyeri dada tipe angina tidak stabil.")
    st.markdown("   - 3: Nyeri dada tipe angina stabil.")
    st.markdown("4. **Resting Blood Pressure (Tekanan Darah Istirahat):** Tekanan darah pasien dalam keadaan istirahat (dalam mmHg).")
    st.markdown("5. **Serum Cholestoral (Kolesterol Serum):** Kadar kolesterol serum pasien dalam mg/dl.")
    st.markdown("6. **Fasting Blood Sugar (Gula Darah Puasa):** Kadar gula darah pasien setelah puasa selama 12 jam. Jika nilai > 120 mg/dl, maka nilai adalah 1; jika ≤ 120 mg/dl, maka nilai adalah 0.")
    st.markdown("7. **Resting Electrocardiographic Results (Hasil Elektrokardiogram Istirahat):** Hasil elektrokardiogram (EKG) pada saat istirahat. Terdapat 3 nilai yang mungkin:")
    st.markdown("   - 0: Hasil normal.")
    st.markdown("   - 1: Memperlihatkan adanya kelainan gelombang ST-T (inversi gelombang T dan/atau elevasi atau depresi ST yang ≥ 0.05 mV).")
    st.markdown("   - 2: Memperlihatkan adanya hipertrofi ventrikel kiri yang pasti menurut kriteria Estes.")
    st.markdown("8. **Maximum Heart Rate Achieved (Denyut Jantung Maksimum Tercapai):** Denyut jantung maksimum yang dicapai oleh pasien.")
    st.markdown("9. **Exercise Induced Angina (Angina yang Dipicu Olahraga):** Menunjukkan apakah pasien mengalami angina yang dipicu oleh aktivitas fisik. Jika nilai adalah 1, maka pasien mengalami angina yang dipicu oleh olahraga; jika nilai adalah 0, maka tidak mengalami angina yang dipicu oleh olahraga.")
    st.markdown("10. **Oldpeak: ST Depression Induced by Exercise Relative to Rest (Depresi ST yang Diinduksi oleh Olahraga Terhadap Istirahat):** Perbedaan depresi segmen ST yang dihasilkan oleh aktivitas fisik dibandingkan dengan keadaan istirahat.")
    st.markdown("11. **Slope of the Peak Exercise ST Segment (Kemiringan Segmen ST Puncak Saat Olahraga):** Kemiringan segmen ST puncak saat melakukan aktivitas fisik. Terdapat 3 nilai yang mungkin:")
    st.markdown("    - 0: Kemiringan tidak dapat ditentukan.")
    st.markdown("    - 1: Kemiringan naik.")
    st.markdown("    - 2: Kemiringan turun.")
    st.markdown("12. **Number of Major Vessels (Jumlah Pembuluh Darah Utama):** Jumlah pembuluh darah utama yang diwarnai dengan flourosopi selama prosedur angiografi. Nilai ini berkisar antara 0 hingga 3, yang mengindikasikan jumlah pembuluh darah utama yang mengalami penyempitan atau kerusakan.")
    st.markdown("13. **Thal: Jenis Kelainan pada Thalassemia (Thalassemia):** Menunjukkan jenis kelainan pada thalassemia. Terdapat 3 nilai yang mungkin:")
    st.markdown("   - 0: Normal.")
    st.markdown("   - 1: Cacat tetap (fixed defect).")
    st.markdown("   - 2: Cacat yang dapat dipulihkan (reversible defect).")
    st.markdown("14. **Target(label kelas):** Kolom target mengacu pada keberadaan penyakit jantung pada pasien dimana Nilainya merupakan bilangan bulat, di mana 0 = tidak ada penyakit dan 1 = ada penyakit.")

    st.subheader("Tampilan Dataset")
    st.dataframe(dh)
    st.markdown("Berikut merupakan tampilan dataset HeartDisease dengan atribut yang telah dijelaskan sebelumnya.")
    st.markdown("**Size Dataset:**")
    st.write(dh.size)
    st.markdown("**Shape Dataset:**")
    st.write(dh.shape)
    st.markdown("**Deskripsi Dataset:**")
    st.write(dh.describe())

    st.subheader("Visualisasi Dataset")
    # Visualisasi pertama
    colors = px.colors.cyclical.Twilight
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Countplot', 'Percentages'),
                        specs=[[{"type": "xy"}, {'type': 'domain'}]])
    fig.add_trace(go.Bar(y=dh['target'].value_counts().values.tolist(),
                         x=dh['target'].value_counts().index,
                         text=dh['target'].value_counts().values.tolist(),
                         textfont=dict(size=15),
                         textposition='outside',
                         showlegend=False,
                         marker=dict(color=colors, line_color='black', line_width=3)),
                  row=1, col=1)
    fig.add_trace(go.Pie(labels=dh['target'].value_counts().keys(),
                         values=dh['target'].value_counts().values, textfont=dict(size=16),
                         hole=0.4, marker=dict(colors=colors), textinfo='label+percent', hoverinfo='label'),
                  row=1, col=2)
    fig.update_yaxes(range=[0, 550])
    fig.update_layout(paper_bgcolor='#FFFDE7', plot_bgcolor='#FFFDE7',
                      title=dict(text="HeartDisease Distribution", x=0.5, y=0.95), title_font_size=30)
    st.plotly_chart(fig)

    st.markdown("Visualisasi tersebut terdiri dari dua bagian, yaitu countplot (grafik batang) dan pie chart (grafik lingkaran), yang menggambarkan distribusi data pada kolom 'target' dalam dataset penyakit jantung.")
    st.markdown("**Countplot:**")
    st.markdown("Grafik batang menunjukkan jumlah kasus penyakit jantung (nilai target) yang terdapat dalam dataset. Sumbu x pada grafik menunjukkan nilai target, yaitu 0 (tidak ada penyakit) dan 1 (terdapat penyakit). Setiap batang pada grafik mewakili jumlah kasus dengan nilai target yang sesuai. Nilai teks yang terletak di luar batang menunjukkan jumlah kasus secara numerik.")
    st.markdown("**Pie Chart:**")
    st.markdown("Grafik lingkaran menunjukkan persentase pembagian kasus penyakit jantung berdasarkan nilai target. Setiap bagian pada lingkaran mewakili nilai target 0 atau 1, dengan label yang sesuai. Ukuran bagian pada lingkaran mencerminkan persentase kasus dalam dataset. Nilai teks pada bagian lingkaran menunjukkan label dan persentase kasus.")
    st.markdown("Kedua visualisasi ini memberikan gambaran tentang sebaran kasus penyakit jantung dalam dataset, dengan countplot menunjukkan jumlah kasus secara langsung dan pie chart memberikan persentase relatif dari setiap nilai target.")

    # Visualisasi kedua
    fig = px.scatter(dh, 
                     x='age', 
                     y='chol', 
                     color='target', 
                     facet_col='fbs',
                     facet_row='sex',
                     color_discrete_map={1: "#FF5722", 0: "#7CB342"},
                     width=950, 
                     height=800,
                     title="Heart Disease Data")

    fig.update_layout(
        plot_bgcolor="#dcedc1",
        paper_bgcolor="#FFFDE7",
    )
    st.plotly_chart(fig)
    
    st.markdown("Visualisasi dataset yang telah dibuat menggunakan scatter plot dengan bantuan library Plotly Express. Scatter plot ini memperlihatkan penyebaran data dari dataset penyakit jantung yang telah dimuat.")
    st.markdown("- Pada sumbu x (horizontal), kita menggunakan atribut 'age' yang merepresentasikan usia pasien.")
    st.markdown("- Pada sumbu y (vertikal), kita menggunakan atribut 'chol' yang merepresentasikan kadar kolesterol serum pasien dalam mg/dl.")
    st.markdown("- Warna titik-titik pada scatter plot ini memperlihatkan atribut target 'target' yang menunjukkan keberadaan penyakit jantung. Nilai 1 (merah) menandakan pasien dengan penyakit jantung, sedangkan nilai 0 (hijau) menandakan pasien tanpa penyakit jantung.")
    st.markdown("- Subplot kolom (facet_col) dibentuk berdasarkan atribut 'fbs' yang menunjukkan kadar gula darah puasa pasien. Terdapat dua kolom pada subplot, yaitu kolom '0' dan '1' yang merepresentasikan dua kategori kadar gula darah puasa.")
    st.markdown("- Subplot baris (facet_row) dibentuk berdasarkan atribut 'sex' yang menunjukkan jenis kelamin pasien. Terdapat dua baris pada subplot, yaitu baris '0' dan '1' yang merepresentasikan dua kategori jenis kelamin.")
    st.markdown("Visualisasi ini membantu kita melihat pola penyebaran data berdasarkan usia, kadar kolesterol, keberadaan penyakit jantung, kadar gula darah puasa, dan jenis kelamin. Dengan membagi plot menjadi subplots berdasarkan kadar gula darah puasa dan jenis kelamin, kita dapat dengan mudah membandingkan karakteristik pasien dalam konteks yang berbeda.")

    st.subheader("Metode Pemrosesan Data")
    st.markdown("Dalam aplikasi praktis, dataset ini dapat digunakan untuk melatih model Machine Learning, seperti K-Nearest Neighbors (KNN), untuk melakukan prediksi penyakit jantung berdasarkan atribut-atribut yang diberikan.")
    st.markdown("Dalam aplikasi ini berisi tentang infomasi-informasi dataset, prediksi databaru, dan visualisasi dataset berdasarkan confusion matrix dan error rate nilai K, dengan menggunakan algoritma K-Nearest Neighbors (KNN).")
    st.markdown("Metode yang saya gunakan untuk pemrosesan dataset ini menggunakan metode K-Nearest Neighbors (KNN) dengan mengatur nilai K/Neighbors nya menjadi '4' dan dengan menggunakan metric euclidean")