# Agglomerative-Clustering

## Deskripsi Project

Project ini bertujuan untuk menganalisis dampak perkembangan Artificial Intelligence (AI) terhadap berbagai jenis pekerjaan di tahun 2030. Dataset berisi informasi seperti gaji rata-rata, tingkat pengalaman, indeks keterpaparan AI, tingkat pertumbuhan teknologi, dan probabilitas otomatisasi.

Dengan menggunakan Agglomerative Clustering (Hierarchical Clustering), project ini mengelompokkan pekerjaan ke dalam beberapa cluster berdasarkan kemiripan karakteristiknya.
Hasilnya dapat digunakan untuk:

* Mengidentifikasi pekerjaan yang berisiko tinggi terkena otomatisasi.

* Mengetahui kelompok pekerjaan dengan gaji tinggi/rendah.

* Melihat keterkaitan antara skill dan eksposur AI.

* Membantu perusahaan dan analis kebijakan dalam merencanakan reskilling/upskilling tenaga kerja.

## Fungsi Utama Program

### Program 1 - Import Library

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import fcluster
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
```
Fungsi: Mengambil seluruh library yang dibutuhkan (pandas, matplotlib, sklearn, scipy).
Tujuan: Menyiapkan seluruh alat analisis, preprocessing, clustering, dan visualisasi.

### program 2 - Load Dataset

```python
df = pd.read_csv('AI_Impact_on_Jobs_2030.csv')
df.head()
```
<img width="1880" height="214" alt="image" src="https://github.com/user-attachments/assets/06ec3a82-9045-443e-9cb8-4e1952a08207" />

Fungsi: Membaca file CSV ke dalam DataFrame.
Tujuan: Memuat dataset utama yang akan dianalisis.

### program 3 - Info mengenai dataset

```python
df.info()
```
<img width="464" height="445" alt="image" src="https://github.com/user-attachments/assets/f731edf4-41f6-471f-a1bb-12a5bbf9cb3a" />

Fungsi: Menampilkan tipe data, jumlah baris & kolom.
Tujuan: Mengetahui struktur dataset untuk menentukan langkah preprocessing.

### program 4 - Hasil Statistik Dekriptif Kolom Numerik DataFrame

```python
df.describe()
```
Fungsi: Menghasilkan statistik deskriptif.
Tujuan: Membantu memahami distribusi numerik (mean, std, min, max).

### program 5 - Cek missing value

```python
df.isna().sum()
```
<img width="244" height="640" alt="image" src="https://github.com/user-attachments/assets/dd26f548-da28-44e3-bd4f-29af796639cc" />

Fungsi: Menghitung missing value.
Tujuan: Menentukan apakah perlu dilakukan imputasi/cleaning.

### program 6 - one- hot encoding

```python
categorical_cols_to_encode = ['Job_Title', 'Education_Level', 'Risk_Category']
df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)

print("Bentuk DataFrame setelah one-hot encoding:", df_encoded.shape)
print("5 baris pertama DataFrame setelah one-hot encoding:")
print(df_encoded.head())
```
<img width="650" height="788" alt="image" src="https://github.com/user-attachments/assets/31ebc420-6b8a-4942-bc47-b82bfe8a9ebd" />

Tujuan: Mengubah kolom kategori menjadi kolom numerik, agar bisa diproses oleh algoritma machine learning seperti Agglomerative Clustering yang hanya menerima data numerik.

a. mendefinisikan kolom kategori

```python
categorical_cols_to_encode = ['Job_Title', 'Education_Level', 'Risk_Category']
```
Menyimpan daftar kolom yang bertipe kategori (string) yang perlu dikonversi menjadi bentuk numerik.

b. encoding

```python
df_encoded = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
```
Fungsi:

* Mengubah setiap nilai kategori menjadi kolom biner (0/1).
* Contoh:
Kolom Education_Level dengan nilai {Highschool, Diploma, Bachelor} → menjadi:
* Education_Level_Diploma
* Education_Level_Bachelor
(karena drop_first=True, satu kategori sengaja dihapus untuk menghindari dummy variable trap)
Tujuan:
* Agar data kategori bisa diproses oleh algoritma machine learning.
* Menghindari multikolinearitas karena drop_first=True.

c. data setelah encoding

```python
print("Bentuk DataFrame setelah one-hot encoding:", df_encoded.shape)
```
Fungsi:
* Menampilkan ukuran (jumlah kolom dan baris) dataset yang sudah di-encoding.
Tujuan:
* Mengecek apakah jumlah kolom bertambah sesuai dengan jumlah kategori yang dipecah menjadi kolom baru.

d. menampilkan hasil encoding

```python
print(df_encoded.head())
```
Fungsi:
* Menampilkan data setelah transformasi untuk memastikan encoding berjalan benat

### program 7 - standarisasi(scaling)

```python
numerical_cols_to_scale = [
    'Average_Salary',
    'Years_Experience',
    'AI_Exposure_Index',
    'Tech_Growth_Factor',
    'Automation_Probability_2030',
    'Skill_1', 'Skill_2', 'Skill_3', 'Skill_4', 'Skill_5',
    'Skill_6', 'Skill_7', 'Skill_8', 'Skill_9', 'Skill_10'
]

# Step 1: Standardize numerical columns
scaler = StandardScaler()
df_standardized_numerical = pd.DataFrame(
    scaler.fit_transform(df_encoded[numerical_cols_to_scale]),
    columns=numerical_cols_to_scale,
    index=df_encoded.index
)
print("Bentuk data setelah standardisasi:", df_standardized_numerical.shape)
print("5 baris pertama data setelah standardisasi:")
df_standardized_numerical.head()
```
<img width="1578" height="239" alt="image" src="https://github.com/user-attachments/assets/33e52c17-29a2-4d3d-9261-1190a8766deb" />

Tujuan utama langkah ini adalah:
Menstandarkan (standardize) seluruh fitur numerik agar berada pada skala yang sama
Ini penting karena dalam clustering, perhitungan jarak (distance) sangat sensitif terhadap perbedaan skala antar fitur.
Misal:
* „Average Salary“ = 100,000
* „Skill_1“ = 3
Tanpa scaling, fitur gaji akan mendominasi hasil klaster.

a. menentukan kolom numerik
```python
numerical_cols_to_scale = [...]
```
Fungsi:
Menyimpan daftar kolom numerik yang akan distandardisasi.
Tujuan:
Memastikan hanya fitur numerik yang diproses oleh scaler.

b. membuat objek standar scaler
```python
scaler = StandardScaler()
```
Fungsi:
Membuat scaler yang menstandardisasi data menjadi:

<img width="171" height="72" alt="image" src="https://github.com/user-attachments/assets/22c5c723-985b-4412-abb3-6cdac56e1abb" />

Tujuan:
Mengubah semua fitur menjadi skala yang sama (mean = 0, std = 1)

c. melakuakn fit dan transform
```python
df_standardized_numerical = pd.DataFrame(
    scaler.fit_transform(df_encoded[numerical_cols_to_scale]),
    columns=numerical_cols_to_scale,
    index=df_encoded.index
)
```
Fungsi:
* fit_transform() menghitung mean & std setiap kolom
* Mengubah semua nilai numerik berdasarkan nilai tersebut
* Membuat DataFrame baru berisi nilai yang sudah distandardisasi
Tujuan:
Menyusun dataset numerik yang ideal untuk clustering

d. menampilkan hasil
```python
print(df_standardized_numerical.shape)
df_standardized_numerical.head()
```
Fungsi:
* Mengecek bentuk dataframe setelah scaling
* Melihat perubahan nilai setelah distribusinya disetarakan

### program 8 - Normalisasi

```python
min_max_scaler = MinMaxScaler()
df_normalized_numerical = pd.DataFrame(
    min_max_scaler.fit_transform(df_standardized_numerical),
    columns=numerical_cols_to_scale,
    index=df_encoded.index
)

print("Bentuk data setelah normalisasi:", df_normalized_numerical.shape)
print("5 baris pertama data setelah normalisasi:")
df_normalized_numerical.head()
```
Tujuan 
Program ini mengubah fitur numerik menjadi rentang 0–1 menggunakan MinMaxScaler untuk membuat data siap dipakai model atau clustering.

a. min max scaler

```python
min_max_scaler = MinMaxScaler()
```
Scaler ini akan mengubah nilai:

<img width="251" height="84" alt="image" src="https://github.com/user-attachments/assets/51d04069-aeb9-42b5-add2-a188e5355e39" />

Semua nilai menjadi dalam range 0 sampai 1.

b. melakukan normalisasi
```python
df_normalized_numerical = pd.DataFrame(
    min_max_scaler.fit_transform(df_standardized_numerical),
    columns=numerical_cols_to_scale,
    index=df_encoded.index
)
```
fungsi:
* mengambil data numerik yang sudah distandarisasi (Z-score)
* menormalkannya lagi dengan MinMaxScaler
* membungkus hasilnya menjadi DataFrame baru

c. menampilkan bentuk data
```python
print("Bentuk data setelah normalisasi:", df_normalized_numerical.shape)
```
fungsi: Untuk memastikan jumlah baris & kolom benar.

d. menapilkan 5 baris pertama
```python
df_normalized_numerical.head()
```
fungsi: Untuk melihat data yang sudah dinormalisasi (semua dalam range 0–1).

### program 9 - menggabungkan fitur numerik + katerogikal

```python
df_processed = pd.concat(
    [
        df_normalized_numerical,
        df_encoded.drop(columns=numerical_cols_to_scale)
    ],
    axis=1
)

print("Bentuk data akhir:", df_processed.shape)
df_processed.head()
```
fungsi:
* df_normalized_numerical → berisi kolom numerik yang sudah dinormalisasi
* df_encoded.drop(columns=numerical_cols_to_scale) → berisi semua kolom kategorikal one-hot, dan fitur lain kecuali kolom numerik (karena kolom numerik sudah diganti dengan versi normalisasi)
* Gabung secara horizontal (axis=1).
* Data akhir = numerik baru + kategorikal one-hot
Tujuan:
Program ini menggabungkan fitur numerik yang telah dinormalisasi dan fitur kategorikal yang telah terenkripsi menjadi dataset final yang bisa langsung digunakan untuk training model atau clustering.


a. menampilkan bentuk shape
```python
print("Bentuk data akhir:", df_processed.shape)
```

### program 10 - dendogram 
```python
Z = linkage(df_processed, method='ward')
print("Matriks linkage berhasil dihitung. Bentuk:", Z.shape)

plt.figure(figsize=(25, 10))
plt.title('Dendrogram untuk Pengelompokan Agglomerative')
plt.xlabel('Indeks Sampel')
plt.ylabel('Jarak')
dendrogram(
    Z,
    leaf_rotation=90.,
    leaf_font_size=8.,
    truncate_mode='lastp',
    p=50
)
plt.show()
```
<img width="1919" height="858" alt="image" src="https://github.com/user-attachments/assets/4a6ed80d-af0e-457a-bee5-769ea6f75d08" />

Dendogram:
* Tinggi garis vertikal menunjukkan jarak (distance) pada saat dua cluster digabung. Garis yang tinggi berarti dua cluster yang digabung sangat jauh (misalnya sangat berbeda).
* Daun (leaves) mewakili data individu (sampel).
* Jika kamu memotong dendrogram di garis horizontal tertentu, potongan itu menunjukkan pembagian cluster — misalnya, cut di jarak 𝑑
berarti “gabungkan semua cluster yang lebih dekat dari 𝑑 menjadi satu cluster besar”.

Tujuan:
* Mengelompokkan data (df_processed) menjadi cluster secara hierarkis, tanpa menentukan jumlah cluster di awal.
* Menggunakan metode Ward pada linkage untuk membentuk cluster dengan meminimalkan varians dalam cluster.
* Memvisualisasikan struktur hierarki cluster melalui dendrogram, agar bisa melihat bagaimana sampel digabung step-by-step dan menentukan di mana “cut” cluster terbaik.

### program 11 - pemilihan Jumlah Cluster Optimal Berdasarkan Lompatan Jarak

```python
merge_distances = Z[:, 2]
distance_jumps = [merge_distances[i+1] - merge_distances[i] for i in range(len(merge_distances)-1)]

if distance_jumps:
    max_jump_index = distance_jumps.index(max(distance_jumps))
    optimal_cut_distance = merge_distances[max_jump_index]
    clusters_at_optimal_cut = fcluster(Z, optimal_cut_distance, criterion='distance')
    suggested_num_clusters = len(set(clusters_at_optimal_cut))

    print(f"Jumlah klaster optimal yang disarankan berdasarkan lompatan jarak terbesar: {suggested_num_clusters}")
    print(f"(Ini sesuai dengan pemotongan dendrogram pada jarak sekitar {optimal_cut_distance:.2f})")
else:
    print("Tidak dapat menentukan klaster optimal secara terprogram menggunakan lompatan jarak.")
    suggested_num_clusters = None

n_clusters = 4
print(f"\nMelanjutkan dengan jumlah klaster yang diasumsikan: {n_clusters}")
```
<img width="606" height="77" alt="image" src="https://github.com/user-attachments/assets/e135e456-fc51-48c6-a228-7017c38c27ef" />

Fungsi Program
* Kode ini secara otomatis menentukan jumlah klaster optimal dari hasil hierarchical clustering dengan cara:
* Mengambil merge distances dari matriks linkage Z.
* Menghitung selisih (lompatan) antar jarak penggabungan.
* Menemukan lompatan jarak terbesar → titik terbaik untuk memotong dendrogram.
8 Menentukan jumlah klaster pada titik potong tersebut menggunakan fcluster.

Tujuan Program
* Mengidentifikasi jumlah cluster optimal tanpa harus melihat dendrogram secara manual.
* Meningkatkan objektivitas pemilihan jumlah cluster.
* Mendukung proses clustering dengan pendekatan data-driven.
* Menentukan titik pemotongan dendrogram yang menghasilkan struktur klaster paling stabil.

a. mengambil jarak dari penggabungan linkage
```python
merge_distances = Z[:, 2]
```
Tujuan:
* Mengambil kolom ke-3 (index 2) dari linkage matrix.
* Berisi jarak saat dua cluster digabung.
* Semakin besar nilainya, semakin “jauh” dua cluster tersebut.

b. menghitung lompatan jarak antar penggabungan 
```python
distance_jumps = [merge_distances[i+1] - merge_distances[i] for i in range(len(merge_distances)-1)]
```
Tujuan:
* Menghitung selisih jarak antar penggabungan berturut-turut.
* Lompatan terbesar menandakan perubahan struktur klaster yang signifikan.
* Digunakan untuk menentukan titik pemotongan dendrogram terbaik.
  
c. Jika ada lompatan jarak → tentukan klaster optimal
```python
if distance_jumps:
    max_jump_index = distance_jumps.index(max(distance_jumps))
    optimal_cut_distance = merge_distances[max_jump_index]
```
Tujuan:
* Mengecek apakah ada lompatan jarak.
* Mencari indeks lompatan terbesar.
* Menentukan jarak pemotongan dendrogram berdasarkan indeks tersebut.

d. Membentuk klaster berdasarkan pemotongan optimal
```python
clusters_at_optimal_cut = fcluster(Z, optimal_cut_distance, criterion='distance')
suggested_num_clusters = len(set(clusters_at_optimal_cut))
```
Tujuan:
* clusters_at_optimal_cut = fcluster(Z, optimal_cut_distance, criterion='distance')
* suggested_num_clusters = len(set(clusters_at_optimal_cut))

e. Menampilkan rekomendasi hasil cluster
```python
print(f"Jumlah klaster optimal yang disarankan berdasarkan lompatan jarak terbesar: {suggested_num_clusters}")
print(f"(Ini sesuai dengan pemotongan dendrogram pada jarak sekitar {optimal_cut_distance:.2f})")
```
Tujuan:
Menampilkan jumlah klaster optimal secara terprogram.

f. Jika tidak ada lompatan jarak
```python
else:
    print("Tidak dapat menentukan klaster optimal secara terprogram menggunakan lompatan jarak.")
    suggested_num_clusters = None
```
Tujuan:
Jika tidak ada perubahan signifikan pada jarak penggabungan, program tidak dapat menentukan klaster optimal.

g. Melanjutkan dengan jumlah cluster manual.
```python
n_clusters = 4
print(f"\nMelanjutkan dengan jumlah klaster yang diasumsikan: {n_clusters}")
```
Tujuan:
* Pengguna tetap bisa menjalankan clustering menggunakan jumlah klaster default (misalkan 4).
* Berguna jika metode otomatis gagal.

### program 12 - Melabeli data berdasarkan hasil klasterisasi Agglomerative Clustering.
```python
agglomerative_model = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
df_processed['Cluster'] = agglomerative_model.fit_predict(df_processed)
print(f"Label klaster ditambahkan ke df_processed. Jumlah klaster: {n_clusters}")
print("5 baris pertama dengan kolom 'Cluster' baru:")
df_processed.head()
```
<img width="668" height="802" alt="image" src="https://github.com/user-attachments/assets/0bb892ec-dd16-4ff9-98c6-3bcca13cf63f" />

Fungsi:
* Melakukan proses agglomerative hierarchical clustering pada data yang sudah diproses.
* Menggunakan parameter jumlah klaster (n_clusters) yang sudah ditentukan sebelumnya.
* Menambahkan kolom baru bernama "Cluster" ke DataFrame (df_processed) berisi label klaster tiap baris.
* Menampilkan hasil 5 baris pertama untuk memastikan proses clustering berhasil.

Tujuan:
* Mengelompokkan data ke dalam n klaster berdasarkan kedekatan jarak (similarity).
* Memberikan informasi label klaster untuk setiap baris data agar dapat digunakan untuk analisis selanjutnya.

a. Membuat model agglomerative clustering
```python
agglomerative_model = AgglomerativeClustering(
    n_clusters=n_clusters, 
    metric='euclidean', 
    linkage='ward'
)
```
Tujuan:
* Membuat model Agglomerative Clustering.
* n_clusters: menentukan jumlah kelompok yang ingin dibentuk.
* metric='euclidean': menghitung jarak antar data menggunakan Euclidean distance.
* linkage='ward': metode penggabungan untuk meminimalkan variansi dalam klaster.

b, Melakukan Klaterisasi
```python
df_processed['Cluster'] = agglomerative_model.fit_predict(df_processed)
```
Fungsi:
* Menjalankan proses klasterisasi pada seluruh data.
* Menghasilkan label klaster untuk setiap baris.
* Menyimpan hasilnya ke kolom baru 'Cluster'.

c. Menampilkan Hasil Klasterisasi
```python
print(f"Label klaster ditambahkan ke df_processed. Jumlah klaster: {n_clusters}")
print("5 baris pertama dengan kolom 'Cluster' baru:")
df_processed.head()
```
Fungsi: 
* Memberi informasi bahwa proses klasterisasi berhasil.
* Menampilkan contoh hasil klasterisasi pada 5 baris pertama.

### Program 13 - Visualisasi Cluster dengan PCA 2D
```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

X = df_processed.drop(columns=['Cluster'])
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
df_pca_2d = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
df_pca_2d['Cluster'] = df_processed['Cluster']
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PC1',
    y='PC2',
    hue='Cluster',
    palette='viridis',
    data=df_pca_2d,
    legend='full',
    alpha=0.7
)
plt.title('Visualisasi Klaster Aglomeratif (PCA 2D)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()
```
<img width="868" height="703" alt="image" src="https://github.com/user-attachments/assets/88902cdb-5436-41e9-8995-e84e53a100e6" />

Fungsi:
* Mengubah data berdimensi tinggi menjadi 2 dimensi menggunakan PCA.
* Memvisualisasikan hasil clustering dengan diagram scatter 2D.
* Melihat apakah cluster yang terbentuk oleh Agglomerative Clustering terpisah dengan baik di ruang PCA.
* Memberikan gambaran intuitif tentang struktur data setelah clustering.

Manfaat:
* Mengetahui apakah cluster terpisah dengan jelas atau saling overlap.
* Mengevaluasi kualitas clustering secara visual.
* Menjadi bukti pendukung laporan analisis atau penelitian.
* Memahami bentuk distribusi data di ruang PCA.

Program 14 - 








