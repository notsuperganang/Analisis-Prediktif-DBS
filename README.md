# Laporan Proyek Machine Learning - Ganang Setyo Hadi

## Domain Proyek

Kanker payudara merupakan salah satu jenis kanker yang paling umum dan menjadi penyebab utama kematian terkait kanker pada wanita di seluruh dunia. Deteksi dini dan diagnosis yang akurat sangat krusial untuk meningkatkan prognosis dan tingkat kelangsungan hidup pasien. Biopsi Aspirasi Jarum Halus (Fine Needle Aspiration - FNA) adalah salah satu prosedur diagnostik umum, di mana sampel sel diambil dari massa payudara dan dianalisis secara mikroskopis. Fitur-fitur dari inti sel yang diamati, seperti radius, tekstur, perimeter, dan area, dapat memberikan indikasi apakah sel tersebut bersifat jinak (benign) atau ganas (malignant).

Proyek ini berfokus pada pengembangan model machine learning yang mampu melakukan klasifikasi diagnosis kanker payudara berdasarkan fitur-fitur yang diekstrak dari citra digital hasil FNA. Dengan memanfaatkan teknik machine learning, diharapkan dapat dibangun sebuah sistem pendukung keputusan yang objektif dan akurat untuk membantu para profesional medis dalam proses diagnosis.

**Mengapa dan Bagaimana Masalah Ini Harus Diselesaikan:**
Diagnosis kanker payudara secara manual melalui pemeriksaan mikroskopis bisa bersifat subjektif dan memakan waktu. Kesalahan diagnosis, baik false positive (menyatakan ganas padahal jinak) maupun false negative (menyatakan jinak padahal ganas), memiliki konsekuensi serius. False positive dapat menyebabkan kecemasan pasien dan tindakan medis yang tidak perlu, sedangkan false negative dapat menunda pengobatan yang vital.

Machine learning menawarkan potensi untuk menganalisis data fitur sel secara komprehensif dan mengidentifikasi pola kompleks yang mungkin sulit dideteksi oleh manusia. Dengan melatih model pada dataset historis yang besar berisi kasus-kasus yang telah terkonfirmasi, kita dapat mengembangkan alat prediksi yang:
- Meningkatkan akurasi diagnosis.
- Mengurangi variabilitas antar pengamat.
- Mempercepat proses diagnosis.
- Memberikan opini kedua yang objektif kepada ahli patologi.

**Referensi Terkait:**
1.  World Health Organization (WHO). (2023). *Breast cancer*. [Online]. Available: [https://www.who.int/news-room/fact-sheets/detail/breast-cancer](https://www.who.int/news-room/fact-sheets/detail/breast-cancer)
2.  Street, W. N., Wolberg, W. H., & Mangasarian, O. L. (1993). Nuclear feature extraction for breast tumor diagnosis. *IS&T/SPIE's Symposium on Electronic Imaging: Science & Technology*, *1905*, 861-870. San Jose, CA. https://doi.org/10.1117/12.148698
3.  Wolberg, W., Mangasarian, O., Street, N., & Street, W. (1993). Breast Cancer Wisconsin (Diagnostic) [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B.

## Business Understanding

Proyek ini bertujuan untuk memanfaatkan data hasil biopsi untuk mengembangkan model prediktif yang dapat membantu dalam diagnosis kanker payudara.

### Problem Statements
- Bagaimana cara membangun model machine learning yang dapat secara akurat mengklasifikasikan tumor payudara sebagai jinak (benign) atau ganas (malignant) berdasarkan fitur-fitur seluler yang diukur?
- Algoritma klasifikasi machine learning manakah (di antara Logistic Regression, Decision Tree, dan Random Forest) yang memberikan performa terbaik untuk dataset Kanker Payudara Wisconsin dalam hal akurasi, presisi, dan recall?
- Fitur-fitur seluler apa saja yang memiliki pengaruh paling signifikan dalam membedakan antara tumor jinak dan ganas?

### Goals
- Mengembangkan model machine learning dengan akurasi tinggi untuk klasifikasi diagnosis kanker payudara (benign/malignant).
- Membandingkan kinerja tiga algoritma klasifikasi (Logistic Regression, Decision Tree, Random Forest) dan memilih model yang paling optimal berdasarkan metrik evaluasi yang relevan (akurasi, presisi, recall, F1-score).
- Mengidentifikasi fitur-fitur seluler yang paling berpengaruh dalam prediksi diagnosis kanker payudara untuk memberikan insight medis.

### Solution statements
- **Solusi 1: Pengembangan dan Perbandingan Model Klasifikasi:**
  Mengimplementasikan dan melatih tiga model machine learning yang berbeda: Logistic Regression, Decision Tree, dan Random Forest, menggunakan dataset Breast Cancer Wisconsin. Kinerja masing-masing model akan dievaluasi secara komprehensif menggunakan metrik akurasi, presisi, recall, dan F1-score pada data uji yang terpisah. Selain itu, kurva ROC dan nilai AUC akan digunakan untuk menilai kemampuan diskriminatif masing-masing model. Model dengan performa terbaik secara keseluruhan akan diidentifikasi sebagai solusi utama.

- **Solusi 2: Analisis Fitur Penting untuk Interpretasi Model:**
  Setelah model terbaik dipilih, dilakukan analisis untuk mengidentifikasi fitur-fitur yang paling berpengaruh terhadap prediksi. Untuk model berbasis tree seperti Random Forest, ini dapat dilakukan menggunakan atribut `feature_importances_`. Untuk Logistic Regression, koefisien model dapat dianalisis. Selain itu, analisis korelasi fitur terhadap variabel target yang telah dilakukan pada tahap EDA juga akan mendukung identifikasi ini. Tujuannya adalah untuk memberikan pemahaman yang lebih baik mengenai faktor-faktor biologis yang mendorong klasifikasi, yang dapat berguna secara klinis.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah **Breast Cancer Wisconsin (Diagnostic) Dataset**. Dataset ini bersumber dari UCI Machine Learning Repository dan juga tersedia di Kaggle. Untuk proyek ini, data dimuat dari file `data.csv` yang merupakan representasi dari dataset tersebut.
Tautan sumber dataset Kaggle: [https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)

Fitur-fitur dalam dataset ini dihitung dari citra digital hasil aspirasi jarum halus (FNA) dari massa payudara. Fitur-fitur tersebut mendeskripsikan karakteristik inti sel yang ada dalam citra. Beberapa contoh citra dapat ditemukan di [http://www.cs.wisc.edu/~street/images/](http://www.cs.wisc.edu/~street/images/).

Bidang pemisah (separating plane) yang dijelaskan (dalam konteks asli dataset) diperoleh menggunakan *Multisurface Method-Tree* (MSM-T), sebuah metode klasifikasi yang menggunakan pemrograman linier untuk membangun pohon keputusan. Fitur-fitur yang relevan dipilih menggunakan pencarian menyeluruh dalam ruang 1-4 fitur dan 1-3 bidang pemisah.

Program linier aktual yang digunakan untuk mendapatkan bidang pemisah dalam ruang 3 dimensi adalah yang dijelaskan dalam: [K. P. Bennett and O. L. Mangasarian: "Robust Linear Programming Discrimination of Two Linearly Inseparable Sets", Optimization Methods and Software 1, 1992, 23-34. http://dx.doi.org/10.1080/10556789208805504].

Dataset ini terdiri dari 569 sampel (baris) dan 32 kolom. Kolom-kolom tersebut mencakup ID pasien, diagnosis (target variable), dan 30 fitur numerik yang merupakan hasil pengukuran karakteristik inti sel.

### Variabel-variabel pada Breast Cancer Wisconsin (Diagnostic) dataset adalah sebagai berikut:
Dataset ini memiliki variabel-variabel berikut:
- **id**: Nomor identifikasi unik untuk setiap pasien.
- **diagnosis**: Variabel target yang menunjukkan diagnosis kanker payudara, dengan nilai 'M' untuk Malignant (ganas) dan 'B' untuk Benign (jinak).
- **Fitur (_mean, _se, _worst)**: Terdapat 30 fitur numerik lainnya. Fitur-fitur ini adalah hasil komputasi dari citra digital sel inti yang diambil melalui Fine Needle Aspirate (FNA). Untuk setiap karakteristik inti sel, dihitung tiga nilai:
    - `_mean`: Rata-rata dari karakteristik tersebut (misalnya, `radius_mean`, `texture_mean`).
    - `_se`: Standar error dari karakteristik tersebut (misalnya, `radius_se`, `texture_se`).
    - `_worst` atau `_largest`: Nilai terburuk atau terbesar dari karakteristik tersebut (misalnya, `radius_worst`, `texture_worst`).

    Fitur-fitur ini mencakup pengukuran seperti:
    - `radius`: Jari-jari inti sel.
    - `texture`: Tekstur inti sel (standar deviasi dari nilai skala abu-abu).
    - `perimeter`: Keliling inti sel.
    - `area`: Luas inti sel.
    - `smoothness`: Kehalusan kontur inti sel (variasi lokal dalam panjang jari-jari).
    - `compactness`: Kekompakan inti sel (perimeter^2 / area - 1.0).
    - `concavity`: Tingkat kecekungan kontur inti sel.
    - `concave points`: Jumlah titik cekung pada kontur inti sel.
    - `symmetry`: Simetri inti sel.
    - `fractal_dimension`: Dimensi fraktal kontur inti sel ("coastline approximation" - 1).

```python
# Contoh kode untuk menampilkan kolom
import pandas as pd
df = pd.read_csv('dataset/data.csv')
print(df.columns.tolist())
```
Outputnya adalah: `['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst']`


### Exploratory Data Analysis (EDA)
Tahapan EDA dilakukan untuk memahami data lebih dalam.
- **Pengecekan Nilai Hilang dan Duplikat:** Hasil pemeriksaan menunjukkan bahwa tidak ada nilai hilang maupun data duplikat dalam dataset ini, menandakan kualitas data yang baik.
- **Statistik Deskriptif:**
    - Dataset memiliki 569 observasi.
    - Fitur `id` tidak akan digunakan untuk pemodelan.
    - Fitur-fitur `_mean`, `_se`, dan `_worst` menunjukkan variasi nilai yang cukup besar, dengan fitur `_worst` umumnya memiliki nilai rata-rata dan maksimum yang lebih tinggi.
    - Beberapa fitur seperti `concavity_mean` dan `concave points_mean` memiliki nilai minimum 0.00.
    - Adanya perbedaan signifikan antara Q3 dan nilai maksimum pada beberapa fitur (misalnya `area_mean`) mengindikasikan potensi adanya outliers atau distribusi data yang miring.
- **Analisis Variabel Target (`diagnosis`):**
    - Kelas 'B' (Benign): 357 sampel (62.7%).
    - Kelas 'M' (Malignant): 212 sampel (37.3%).
    - Terdapat ketidakseimbangan kelas yang moderat.
    - Variabel target ini kemudian diubah menjadi numerik ('M': 1, 'B': 0) untuk keperluan visualisasi dan pemodelan.

- **Visualisasi Distribusi Target:**
  ![Visualisasi Distribusi Target](https://i.imgur.com/ABESGSl.png)
  Count plot dan pie chart digunakan untuk memvisualisasikan distribusi kelas target, yang mengkonfirmasi proporsi 62.7% Benign dan 37.3% Malignant.

```python
# # Contoh kode untuk visualisasi distribusi target (setelah mapping ke numerik)
if df[target_col].dtype == 'object':
  target_mapping = {'M': 1, 'B': 0}
  df[target_col] = df[target_col].map(target_mapping)
  sns.countplot(data=df, x=target_col)
  plt.title('Distribusi Kelas Target')
  plt.show()
```

- **Analisis Korelasi Fitur:**
![Analisis Korelasi Fitur](https://i.imgur.com/i7LMPgc.png)
    - Heatmap korelasi menunjukkan adanya multikolinearitas yang tinggi antar fitur-fitur yang berkaitan dengan ukuran dan bentuk sel (misalnya, `radius_mean`, `perimeter_mean`, `area_mean`).
    - 10 fitur dengan korelasi absolut tertinggi terhadap variabel target (`diagnosis`) diidentifikasi, dengan `concave points_worst`, `perimeter_worst`, dan `concave points_mean` menempati posisi teratas.
    - Visualisasi distribusi dua fitur teratas (`concave points_worst` dan `perimeter_worst`) berdasarkan kelas target menunjukkan pemisahan yang cukup jelas, di mana nilai yang lebih tinggi cenderung berasosiasi dengan diagnosis Malignant.

## Data Preparation
Proses persiapan data dilakukan untuk memastikan data siap dan optimal untuk tahap pemodelan.

1.  **Konversi Target Variabel ke Numerik**: Variabel target `diagnosis` yang awalnya berupa string ('M' dan 'B') diubah menjadi format numerik (Malignant=1, Benign=0). Ini dilakukan pada tahap EDA sebelum visualisasi distribusi target dan digunakan secara konsisten untuk pemodelan. Alasan kenapa tahap ini dilakukan disana adalah untuk memudahkan proses EDA dan membuat hasil EDA menjadi lebih baik.
    ```python
    # Contoh kode (asumsi target_col adalah 'diagnosis')
    if df['diagnosis'].dtype == 'object':
        target_mapping = {'M': 1, 'B': 0}
        df['diagnosis'] = df['diagnosis'].map(target_mapping)
    ```
    *Alasan*: Banyak algoritma machine learning, terutama yang berbasis matematis seperti Logistic Regression, memerlukan input numerik. Konversi ini juga memudahkan perhitungan metrik evaluasi.

2.  **Pemisahan Fitur dan Target**: Dataset dipisahkan menjadi matriks fitur (X) yang berisi semua fitur numerik relevan (30 fitur setelah `id` dan `diagnosis` dikeluarkan dari fitur prediktif), dan vektor target (y) yang berisi variabel `diagnosis`.
    ```python
    numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
    # Pastikan target_col (misal, 'diagnosis' yang sudah numerik) dan 'id' (jika ada di numeric_features) dihapus
    if 'diagnosis' in numeric_features: numeric_features.remove('diagnosis')
    if 'id' in numeric_features: numeric_features.remove('id')
    X = df[numeric_features]
    y = df['diagnosis'] # Pastikan ini sudah numerik
    ```
    *Alasan*: Ini adalah langkah standar dalam supervised learning, di mana model belajar dari fitur (X) untuk memprediksi target (y).

3.  **Pembagian Data (Train-Test Split)**: Data (X dan y) dibagi menjadi set pelatihan (80%) dan set pengujian (20%). Pembagian ini dilakukan secara stratified berdasarkan variabel target `y` untuk memastikan proporsi kelas (Benign dan Malignant) tetap terjaga di kedua set. `random_state=42` digunakan untuk memastikan reproduktibilitas hasil.
    ```python
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    ```
    *Alasan*: Pembagian data memungkinkan evaluasi model yang objektif pada data yang belum pernah dilihat sebelumnya (test set), mencegah overfitting. Stratifikasi penting untuk dataset dengan ketidakseimbangan kelas agar kedua kelas terwakili secara proporsional di set training dan testing.

4.  **Standarisasi Fitur (Feature Scaling)**: Fitur-fitur numerik pada set pelatihan (`X_train`) dan set pengujian (`X_test`) distandarisasi menggunakan `StandardScaler` dari scikit-learn. Scaler di-fit hanya pada `X_train` dan kemudian digunakan untuk mentransformasi `X_train` dan `X_test`.
    ```python
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    ```
    *Alasan*: Standarisasi mengubah fitur sehingga memiliki rata-rata 0 dan standar deviasi 1. Ini penting untuk algoritma yang sensitif terhadap skala fitur, seperti Logistic Regression dan algoritma berbasis jarak. Standarisasi dapat membantu mempercepat konvergensi selama pelatihan dan memastikan semua fitur memberikan kontribusi yang setara. Data `X_train_scaled` digunakan untuk melatih Logistic Regression, sementara data `X_train` (tidak diskalakan) digunakan untuk Decision Tree dan Random Forest karena model berbasis tree umumnya tidak sensitif terhadap skala fitur.

Hasil dari tahapan ini adalah data yang telah bersih, terstruktur, dan siap untuk digunakan dalam pelatihan model machine learning. Proporsi kelas pada training dan testing set juga telah diverifikasi tetap terjaga.

## Modeling
Tiga algoritma klasifikasi machine learning diimplementasikan dan dilatih untuk memprediksi diagnosis kanker payudara.

1.  **Logistic Regression**
    -   **Tahapan**: Model Logistic Regression diinisialisasi dengan `random_state=42` untuk reproduktibilitas dan `max_iter=1000` untuk memastikan konvergensi. Model ini dilatih menggunakan data fitur yang telah distandarisasi (`X_train_scaled`) dan variabel target `y_train`.
    -   **Parameter**: `random_state=42`, `max_iter=1000`.
    -   **Kelebihan**: Cepat dilatih, mudah diinterpretasikan (koefisien dapat menunjukkan pentingnya fitur), memberikan output probabilitas, dan merupakan baseline yang baik.
    -   **Kekurangan**: Mengasumsikan hubungan linier antara fitur dan log-odds dari target, bisa kurang performan pada masalah yang sangat non-linier.

2.  **Decision Tree Classifier**
    -   **Tahapan**: Model Decision Tree diinisialisasi dengan `random_state=42` dan `max_depth=10` untuk mengontrol kompleksitas pohon dan mencegah overfitting yang berlebihan. Model ini dilatih menggunakan data fitur asli (`X_train`) dan variabel target `y_train`.
    -   **Parameter**: `random_state=42`, `max_depth=10`.
    -   **Kelebihan**: Sangat mudah diinterpretasikan (struktur pohonnya intuitif), mampu menangani hubungan non-linier antar fitur, dan tidak memerlukan penskalaan fitur.
    -   **Kekurangan**: Cenderung overfitting terutama jika kedalaman pohon tidak dibatasi, bisa tidak stabil (perubahan kecil pada data dapat menghasilkan pohon yang berbeda).

3.  **Random Forest Classifier**
    -   **Tahapan**: Model Random Forest diinisialisasi dengan `random_state=42` dan `n_estimators=100` (jumlah pohon dalam forest). Model ini merupakan ensemble dari Decision Trees dan dilatih menggunakan data fitur asli (`X_train`) dan variabel target `y_train`.
    -   **Parameter**: `random_state=42`, `n_estimators=100`.
    -   **Kelebihan**: Umumnya memberikan akurasi yang tinggi, lebih robust terhadap overfitting dibandingkan Decision Tree tunggal, mampu menangani data dengan fitur yang banyak, dan memberikan ukuran pentingnya fitur.
    -   **Kekurangan**: Lebih kompleks dan kurang interpretatif dibandingkan Decision Tree tunggal (merupakan "black box"), dan memerlukan lebih banyak waktu komputasi untuk training.

**Hasil Akurasi Training:**
- Logistic Regression: 0.9868
- Decision Tree: 1.0000
- Random Forest: 1.0000

Akurasi training yang sangat tinggi, terutama 100% untuk Decision Tree dan Random Forest, bisa menjadi indikasi awal adanya *overfitting*. Kinerja generalisasi akan dievaluasi pada data test.

**Analisis Feature Importance Model**

Untuk lebih memahami fitur mana yang dianggap paling penting oleh model Decision Tree dan Random Forest dalam membuat prediksi, kita melakukan analisis *feature importance*. Fitur dengan nilai *importance* yang lebih tinggi memiliki kontribusi yang lebih besar dalam proses pengambilan keputusan model.

Berikut adalah hasil analisis *feature importance*:

* **Top 10 Feature Importance - Decision Tree:**
    1.  `perimeter_worst`: 0.7232
    2.  `concave points_worst`: 0.0677
    3.  `smoothness_worst`: 0.0408
    4.  `texture_mean`: 0.0292
    5.  `area_mean`: 0.0225
    6.  `texture_worst`: 0.0187
    7.  `area_worst`: 0.0156
    8.  `concavity_worst`: 0.0143
    9.  `fractal_dimension_worst`: 0.0134
    10. `texture_se`: 0.0125

* **Top 10 Feature Importance - Random Forest:**
    1.  `area_worst`: 0.1310
    2.  `perimeter_worst`: 0.1303
    3.  `concave points_worst`: 0.1032
    4.  `radius_worst`: 0.0978
    5.  `concave points_mean`: 0.0874
    6.  `area_mean`: 0.0584
    7.  `perimeter_mean`: 0.0484
    8.  `concavity_worst`: 0.0478
    9.  `area_se`: 0.0419
    10. `concavity_mean`: 0.0366

Visualisasi dari 15 fitur teratas untuk kedua model dapat dilihat pada gambar berikut:

![Feature Importance Decision Tree dan Random Forest](https://i.imgur.com/VqRinTZ.png)

**Interpretasi Feature Importance:**
-   Pada model **Decision Tree**, fitur `perimeter_worst` mendominasi dengan sangat signifikan (importance ~0.72). Ini menunjukkan bahwa model Decision Tree tunggal ini sangat bergantung pada satu fitur ini. Fitur-fitur lain memiliki kontribusi yang jauh lebih kecil.
-   Pada model **Random Forest**, *feature importance* tersebar lebih merata di antara beberapa fitur teratas. Fitur-fitur seperti `area_worst`, `perimeter_worst`, `concave points_worst`, `radius_worst`, dan `concave points_mean` semuanya menunjukkan kontribusi yang signifikan. Hal ini menunjukkan bahwa Random Forest mempertimbangkan lebih banyak fitur dalam membuat keputusan, yang umumnya mengarah pada model yang lebih robust dan general.
-   Fitur-fitur yang konsisten muncul sebagai penting di kedua model (meskipun dengan peringkat yang berbeda) adalah fitur-fitur `_worst` (misalnya `perimeter_worst`, `concave points_worst`, `area_worst`) dan beberapa fitur `_mean` (misalnya `concave points_mean`, `area_mean`). Ini sejalan dengan analisis korelasi sebelumnya yang juga menyoroti pentingnya fitur-fitur ini.

Analisis *feature importance* ini memperkuat pemahaman kita tentang data dan bagaimana model mengambil keputusan, serta memberikan insight fitur mana yang paling prediktif untuk diagnosis kanker payudara.

**Pemilihan Model Terbaik (setelah evaluasi pada data test):**
Berdasarkan evaluasi pada data test, **Logistic Regression** menunjukkan performa keseluruhan yang paling seimbang dan tinggi, dengan akurasi tertinggi (0.9649) dan F1-score tertinggi (0.9512). Meskipun Random Forest memiliki presisi sempurna untuk kelas Malignant (1.0000), recall-nya sedikit lebih rendah (0.8810) dibandingkan Logistic Regression (0.9286). Mengingat pentingnya mendeteksi kasus Malignant (meminimalkan False Negatives, yang berhubungan dengan recall tinggi), dan keseimbangan metrik secara keseluruhan, Logistic Regression dipilih sebagai model terbaik untuk kasus ini. AUC score Logistic Regression (0.996) juga sedikit lebih unggul dari Random Forest (0.992).

## Evaluation
Evaluasi model dilakukan menggunakan data test untuk mengukur seberapa baik model dapat menggeneralisasi pada data baru. Metrik yang digunakan adalah Akurasi, Presisi, Recall, dan F1-Score. Kurva ROC dan AUC juga digunakan untuk analisis lebih lanjut.

**Penjelasan Metrik yang Digunakan:**
-   **Akurasi (Accuracy)**: Proporsi dari total prediksi yang benar (baik prediksi Benign maupun Malignant yang benar) dibagi dengan jumlah total sampel.
    -   *Formula*: `(TP + TN) / (TP + TN + FP + FN)`
    -   *Interpretasi*: Mengukur seberapa sering model membuat prediksi yang benar secara keseluruhan. Cocok untuk dataset yang seimbang.
-   **Presisi (Precision)**: Dari semua sampel yang diprediksi sebagai Malignant, berapa proporsi yang benar-benar Malignant.
    -   *Formula*: `TP / (TP + FP)`
    -   *Interpretasi*: Mengukur kemampuan model untuk tidak salah melabeli sampel Benign sebagai Malignant. Presisi tinggi penting ketika biaya False Positive tinggi.
-   **Recall (Sensitivity atau True Positive Rate)**: Dari semua sampel yang sebenarnya Malignant, berapa proporsi yang berhasil diprediksi sebagai Malignant oleh model.
    -   *Formula*: `TP / (TP + FN)`
    -   *Interpretasi*: Mengukur kemampuan model untuk menemukan semua kasus Malignant. Recall tinggi sangat krusial dalam diagnosis medis untuk meminimalkan False Negative.
-   **F1-Score**: Rata-rata harmonik dari Presisi dan Recall. Memberikan keseimbangan antara kedua metrik tersebut.
    -   *Formula*: `2 * (Precision * Recall) / (Precision + Recall)`
    -   *Interpretasi*: Berguna ketika ada ketidakseimbangan kelas atau ketika penting untuk menyeimbangkan Presisi dan Recall.
-   **ROC Curve (Receiver Operating Characteristic Curve)**: Plot yang menggambarkan kinerja model klasifikasi biner pada berbagai ambang batas klasifikasi. Kurva ini memplot True Positive Rate (Recall) terhadap False Positive Rate.
    -   *Interpretasi*: Model yang baik memiliki kurva yang mendekati sudut kiri atas.
-   **AUC (Area Under the ROC Curve)**: Mengukur keseluruhan kemampuan model untuk membedakan antara kelas positif dan negatif. Nilai AUC berkisar dari 0 hingga 1.
    -   *Interpretasi*: AUC = 1.0 berarti classifier sempurna; AUC = 0.5 berarti classifier tidak lebih baik dari tebakan acak.

Dimana:
-   TP (True Positive): Sampel Malignant yang diprediksi benar sebagai Malignant.
-   TN (True Negative): Sampel Benign yang diprediksi benar sebagai Benign.
-   FP (False Positive): Sampel Benign yang salah diprediksi sebagai Malignant (Type I Error).
-   FN (False Negative): Sampel Malignant yang salah diprediksi sebagai Benign (Type II Error).

**Hasil Proyek Berdasarkan Metrik Evaluasi:**

Tabel Perbandingan Model pada Data Test:
| Model                | Akurasi  | Presisi (M) | Recall (M) | F1-Score (M) | AUC    |
| :------------------- | :------- | :---------- | :--------- | :----------- | :----- |
| Logistic Regression  | 0.9649   | 0.9750      | 0.9286     | 0.9512       | 0.996  |
| Decision Tree        | 0.9211   | 0.9024      | 0.8810     | 0.8916       | 0.913  |
| Random Forest        | 0.9561   | 1.0000      | 0.8810     | 0.9367       | 0.992  |

*Catatan: Presisi, Recall, dan F1-Score di atas dilaporkan untuk kelas positif (Malignant).*

**Analisis Confusion Matrix untuk Model Terbaik (Logistic Regression):**
![Analisis Confusion Matrix](https://i.imgur.com/exCmJck.png)
- True Negatives (TN / Benign diprediksi Benign): 70
- False Positives (FP / Benign diprediksi Malignant): 1
- False Negatives (FN / Malignant diprediksi Benign): 3
- True Positives (TP / Malignant diprediksi Malignant): 40

Interpretasi untuk Logistic Regression:
- Model ini dengan benar memprediksi 70 kasus sebagai Benign.
- Model ini salah memprediksi 1 kasus Benign sebagai Malignant (Type I Error).
- Model ini salah memprediksi 3 kasus Malignant sebagai Benign (Type II Error).
- Model ini dengan benar memprediksi 40 kasus sebagai Malignant.

**Analisis Kurva ROC dan AUC:**
![Analisis Kurva ROC dan AUC](https://i.imgur.com/xvWNlqL.png)
- Logistic Regression memiliki AUC tertinggi (0.996), menunjukkan kemampuan diskriminasi terbaik.
- Random Forest juga menunjukkan performa sangat baik dengan AUC 0.992.
- Decision Tree memiliki AUC terendah (0.913) dibandingkan dua model lainnya.

**Kesimpulan Evaluasi:**
Semua model menunjukkan kinerja yang baik, namun Logistic Regression menonjol dengan keseimbangan terbaik antara berbagai metrik evaluasi dan kemampuan generalisasi yang tinggi pada data test. Dengan akurasi 96.49%, presisi 97.5% untuk kelas Malignant, recall 92.86% untuk kelas Malignant, dan F1-score 95.12% untuk kelas Malignant, serta AUC 0.996, model Logistic Regression dianggap sebagai solusi yang paling efektif dan seimbang untuk masalah prediksi kanker payudara ini. Jumlah False Negatives yang rendah (3 kasus) juga sangat penting dalam konteks medis.