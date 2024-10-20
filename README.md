# LAPORAN PROYEK MACHINE LEARNING - Ahmad Fauzi
## Domain Proyek
Di era persaingan pasar ponsel yang semakin ketat, inovasi teknologi dan fitur-fitur baru terus bermunculan, menjadikan harga produk sebagai salah satu faktor penentu keberhasilan perusahaan dalam menarik konsumen. Bob, seorang pengusaha yang baru memulai perusahaan ponselnya, menghadapi tantangan dalam menetapkan harga yang kompetitif. Perusahaan besar seperti Apple dan Samsung mendominasi pasar dengan produk-produk yang memiliki rentang harga yang jelas sesuai dengan spesifikasi teknis dan inovasi yang ditawarkan [Gartner, 2021](https://www.gartner.com/en/newsroom/press-releases/2022-03-01-4q21-smartphone-market-share). Untuk menyaingi perusahaan-perusahaan ini, Perusahaan membutuhkan cara yang efisien untuk menentukan harga ponsel berdasarkan fitur-fitur teknis seperti RAM, memori internal, dan kapasitas baterai. Dalam konteks ini, pendekatan manual tidak memadai, dan metode berbasis data seperti machine learning dapat menawarkan solusi yang lebih efisien dan akurat [Choudhary et al., 2021](https://www.gartner.com/en/newsroom/press-releases/2022-03-01-4q21-smartphone-market-share).

Machine learning telah terbukti menjadi alat yang sangat berguna dalam memproses data besar dan menemukan pola yang tersembunyi, terutama dalam masalah klasifikasi harga. Melalui algoritma seperti Random Forest dan Support Vector Machine (SVM), model machine learning dapat mengolah fitur-fitur penting dari ponsel dan menghasilkan prediksi yang akurat tentang kisaran harga produk [Mulla & Desai, 2020](https://link.springer.com/article/10.1007/s40622-020-00260-8). Penggunaan metode ini memungkinkan perusahaan untuk memahami segmentasi pasar dengan lebih baik dan menetapkan harga yang kompetitif sesuai dengan kebutuhan konsumen. Ini penting karena fitur teknis yang ditawarkan oleh ponsel memiliki korelasi langsung dengan kisaran harga yang diharapkan, sebagaimana ditunjukkan dalam studi-studi prediksi harga berbasis fitur produk [Zhang et al., 2019](https://link.springer.com/article/10.1007/s10115-022-01679-4).

Dalam perusahaan, memerlukan strategi penetapan harga yang tepat untuk memaksimalkan keuntungan dan meningkatkan daya saing produknya di pasar yang sangat kompetitif. Klasifikasi harga dengan menggunakan machine learning memungkinkan perusahaan untuk menetapkan harga berdasarkan data empiris dan tren pasar. Selain itu, hal ini dapat membantu dalam menargetkan segmen pelanggan yang sesuai dan mengembangkan strategi pemasaran yang lebih efektif [Mulla & Desai, 2020](https://link.springer.com/article/10.1007/s40622-020-00260-8). Oleh karena itu, proyek ini tidak hanya relevan bagi perusahaan, tetapi juga merupakan pendekatan yang sesuai dengan kebutuhan industri yang bergerak cepat di mana inovasi teknologi terus berkembang.

## Business Understanding
### Problem Statements
Perusahaan yang baru memulai bisnis ponsel, menghadapi tantangan besar dalam menetapkan harga yang kompetitif untuk produknya. Dalam pasar ponsel yang kompetitif, penetapan harga yang salah dapat mengakibatkan hilangnya peluang pasar atau potensi pendapatan yang terlewatkan. Perusahaan tidak memiliki pengetahuan teknis yang cukup dalam machine learning untuk memprediksi harga produk berdasarkan fitur-fitur teknis seperti RAM, memori internal, kapasitas baterai, dan sebagainya. Tantangan utama yang dihadapi adalah bagaimana mengklasifikasikan ponsel ke dalam kisaran harga yang sesuai berdasarkan fitur-fitur tersebut, sehingga dapat menargetkan segmen pasar yang tepat.
### Goals
Tujuan utama dari proyek ini adalah membantu perusahaan mengembangkan sistem yang dapat mengklasifikasikan ponsel yang diproduksi ke dalam berbagai kisaran harga berdasarkan spesifikasi teknis. Dengan adanya model machine learning ini, Perusahaan dapat membuat keputusan penetapan harga yang lebih cerdas dan kompetitif, serta memposisikan produknya dengan lebih baik di pasar. Dengan prediksi kisaran harga yang akurat, perusahaan dapat bersaing dengan brand besar, dan meraih pangsa pasar dengan lebih efektif.
### Solution statements
Untuk menyelesaikan masalah ini, akan digunakan pendekatan machine learning berbasis klasifikasi dengan dua algoritma utama: Random Forest dan Support Vector Machine (SVM). Algoritma ini akan digunakan untuk memprediksi kisaran harga berdasarkan data fitur ponsel seperti RAM, ukuran layar, kapasitas baterai, dan memori internal.
Agar hasil prediksi lebih optimal, model yang dihasilkan akan disempurnakan dengan menggunakan GridSearch untuk Hyperparameter Tuning. Optimasi hyperparameter ini akan membantu dalam menemukan kombinasi parameter yang paling optimal untuk meningkatkan performa model.
  
  * Random Forest adalah algoritma ensemble learning yang menggabungkan prediksi dari banyak decision tree untuk meningkatkan akurasi dan stabilitas model. Dalam konteks proyek ini, Random Forest akan membantu dalam menentukan kisaran harga berdasarkan fitur-fitur seperti RAM, ukuran layar, dan kapasitas baterai. Setiap decision tree dalam Random Forest dibangun berdasarkan subset acak dari data dan subset acak dari fitur, yang membuat model ini tahan terhadap overfitting, terutama ketika data memiliki banyak fitur yang saling berkorelasi. Keunggulan Random Forest adalah kemampuan generalisasi yang baik dengan menggabungkan hasil dari banyak decision tree, Random Forest mampu menghasilkan prediksi yang lebih akurat dan mengurangi risiko overfitting. Kemudian dalam pemilihan fitur model ini dapat menangani sejumlah besar fitur dan secara otomatis memberikan peringkat fitur mana yang paling berkontribusi terhadap prediksi, sehingga dapat mengidentifikasi fitur ponsel yang paling penting dalam menentukan kisaran harga.
  
  * Support Vector Machine (SVM) adalah algoritma yang bekerja dengan menemukan hyperplane terbaik yang memisahkan kelas-kelas data. Dalam proyek ini, SVM akan digunakan untuk memetakan data fitur ponsel ke dalam ruang berdimensi tinggi, kemudian menemukan garis atau kurva (hyperplane) yang memisahkan ponsel berdasarkan kisaran harga. SVM sangat efektif ketika ada perbedaan yang jelas antara kategori harga, dan algoritma ini mampu bekerja dengan baik bahkan ketika data tidak linear, melalui penggunaan kernel trick. Keunggulan SVM yaitu memiliki keakuratan pada data yang tidak seimbang SVM mampu memberikan hasil klasifikasi yang baik, bahkan dalam kasus di mana data tidak seimbang atau memiliki sedikit kesalahan klasifikasi. Serta keunggulan lainnya yaitu dalam penggunaan kernel trick dimana SVM dapat menangani data yang tidak linear dan membuat model lebih fleksibel untuk berbagai macam distribusi data fitur ponsel.
  
  * Hyperparameter tuning dapat digunakan untuk memastikan performa terbaik dari model yang diterapkan, kita akan menggunakan Hyperparameter Tuning dengan GridSearch. Baik Random Forest maupun SVM memiliki hyperparameter yang dapat mempengaruhi performa model secara signifikan. Misalnya, pada Random Forest, jumlah tree (n_estimators) atau kedalaman maksimum tree (max_depth) perlu dioptimalkan, sedangkan pada SVM, parameter seperti C (regularization) dan kernel (linear, polynomial, atau RBF) harus disesuaikan.
  
  * GridSearch adalah metode yang memungkinkan kita untuk menguji kombinasi berbagai nilai hyperparameter dan memilih yang terbaik berdasarkan kinerja model. Dalam proyek ini, GridSearch akan menguji berbagai kombinasi parameter dan mengevaluasi model berdasarkan metrik seperti akurasi, precision, recall, dan F1-score. Dengan melakukan tuning yang tepat, model dapat dioptimalkan untuk memberikan hasil klasifikasi yang lebih baik dan akurat dalam memprediksi kisaran harga ponsel. Keunggulan Hyperparameter Tuning dengan GridSearch adalah dapat meningkatkan performa model, dengan menemukan kombinasi hyperparameter terbaik, model akan bekerja lebih optimal dan memberikan hasil klasifikasi yang lebih akurat. Serta keunngulan lainnya dapat mencegah overfitting, dengan pengaturan hyperparameter yang tepat, kita dapat menghindari overfitting dan memastikan bahwa model dapat bekerja dengan baik pada data baru.
Setelah dilakukan optimasi, model yang terbaik akan dievaluasi menggunakan metrik seperti akurasi, precision, recall, dan F1-score, untuk memastikan bahwa prediksi kisaran harga yang dihasilkan dapat diimplementasikan secara efektif dalam pengambilan keputusan penetapan harga perusahaan Bob.
## Data Understanding
Dalam tahapan Data Understanding, kita akan berfokus pada pemahaman mendalam terhadap dataset yang digunakan untuk proyek klasifikasi kisaran harga ponsel. Data diambil dari [Kaggle](https://www.kaggle.com/) [Mobile Price Classification Dataset](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification), yang terdiri dari dua folder: data train dengan 2000 entri dan data test dengan 1000 entri. Dataset ini memuat berbagai fitur ponsel seperti kapasitas baterai, RAM, resolusi layar, dan banyak fitur lain yang akan digunakan untuk memprediksi variabel target, yaitu price_range, yang mewakili empat kategori harga. Tahap pertama adalah memuat data train dan test ke dalam lingkungan pemrograman untuk dianalisis lebih lanjut. Data train berisi 2000 observasi dengan 21 variabel, sedangkan data test memiliki 1000 observasi dengan 20 variabel (tanpa variabel target price_range). Berikut detail variabelnya:

| No | Variabel	| Deskripsi |
| -- | -------- | --------- |
| 1	| battery_power	| Total energi yang dapat disimpan baterai dalam satu waktu diukur dalam mAh |
| 2 |	blue | Memiliki bluetooth atau tidak (1: ya, 0: tidak) |
| 3	| clock_speed |	Kecepatan mikroprosesor mengeksekusi instruksi (GHz) |
| 4	| dual_sim	| Memiliki dukungan dual sim atau tidak (1: ya, 0: tidak) |
| 5 |	fc	| Mega piksel Kamera Depan |
| 6 |	four_g	| Memiliki 4G atau tidak (1: ya, 0: tidak) |
| 7	| int_memory	| Memori Internal dalam Gigabyte |
| 8	| m_dep |	Kedalaman Seluler dalam cm |
| 9	| mobile_wt	| Berat ponsel dalam gram |
| 10	| n_cores	| Jumlah inti prosesor |
| 11	| pc	| Mega piksel Kamera Utama |
| 12	| px_height	| Tinggi Resolusi Piksel |
| 13 |	px_width |	Lebar Resolusi Piksel |
| 14	| ram |	Memori Akses Acak dalam Mega Byte |
| 15	| sc_h |	Tinggi Layar ponsel dalam cm |
| 16 |	sc_w	| Lebar Layar ponsel dalam cm |
| 17	| talk_time	| Waktu terlama yang dapat digunakan untuk satu kali pengisian daya baterai |
| 18	| three_g	| Memiliki 3G atau tidak (1: ya, 0: tidak) |
| 19	| touch_screen	| Memiliki layar sentuh atau tidak (1: ya, 0: tidak) |
| 20	| wifi |	Memiliki wifi atau tidak (1: ya, 0: tidak) |
| 21	| price_range	| Variabel target dengan nilai 0 (biaya rendah), 1 (biaya sedang), 2 (biaya tinggi), 3 (biaya sangat tinggi) |

Variabel pada Data Test:

> Hampir sama dengan data train, namun tanpa variabel price_range dan tambahan kolom Id sebagai pengenal unik.

Berikut informasi mengenai jumlah data ,tipe data dan informasi data hilang (missing value) yang terdapat pada dataset ini:

![info](https://github.com/user-attachments/assets/015637ae-d8b0-4965-8e47-3afb2fbfd649)

![Missing](https://github.com/user-attachments/assets/0d40b733-dec5-48d2-a71f-b00d14c9c9ad)

Dalam memudahkan proses analisis diperlukan beberapa visualisasi data, seperti:

  * sns.boxplot, untuk mendeteksi adanya data yang berada di luar batas atas dan batas bawah data (outliers).

![outlier](https://github.com/user-attachments/assets/4977c809-5305-44b8-9c92-6d4dd2c84923)

  * count.plot, untuk menganalisa fitur battery_power.

![battery_power](https://github.com/user-attachments/assets/8c710888-4c9b-4fef-a4ad-9019034e99f4)

  * sns.catplot, untuk mempertimbangkan price_range dengan fitur fc.

![price_fc](https://github.com/user-attachments/assets/129cac2a-1bce-492c-b09f-bc87d1deb008)

  * sns.pairplot, untuk menunjukkan semua grafik fitur numerik.

![grafik](https://github.com/user-attachments/assets/a207416a-339d-45f4-8033-5e940337624e)

> NOTE: Kita belum dapat menarik kesimpulan, dikarenakan sebaran data yang masih acak(random)

  * sns.heatmap, untuk menunjukkan matrik korelasi fitur numerik.

![korelasi](https://github.com/user-attachments/assets/b8caf6b9-1337-4fed-b35b-856e0eeebb59)

Berdasarkan hasil korelasi diatas dapat diketahui:

  * Variabel yang memiliki korelasi paling kuat adalah battery_power dan ram hal ini sangat dapat dipahami karena kebutuhan akan battery dan sangat berpengaruh terhadap kinerja HP.
  * Pada kolerasi diata terdapat variabel yang tidak memilki korelasi dengan price_range yaitu variabel n_cores, m_dep, dan clock_speed, sehingga saya akan melakukan drop atau penghapusan pada tiga variabel tersebut.

Dengan visualisasi data yang telah dilakukan, diharapkan dapat memudahkan kita didalam proses analisa data.

## Data Preparation
Setelah proses data understanding dengan melakukan pengecekan missing value dan outlier pada data, dimana data tidak memiliki missing value dan outlier. Selanjutnya melakukan data preparation dengan dimulai dari:
* Berdasarkan hasil korelasi diatas dapat diketahui bahwa terdapat variabel yang tidak memiliki korelasi dengan variabel label yaitu:
  * variabel n_cores, m_dep, dan clock_speed, sehingga saya akan melakukan drop atau penghapusan pada tiga variabel tersebut.
```
#Menghapus fitur yang memiliki korelasi kecil yaitu fitur "n_cores, m_dep, dan clock_speed"
df_train = df_train.drop(['n_cores', 'm_dep', 'clock_speed'], axis=1)
df_train.head()
```
Setelah data sudah siap selanjutnya adalang dengan membagi data berikut langkahnya:
  * Train Test Split : Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model.
  * Data Transform
    * Scaling: 
      * Scaling Data Train (Standarisasi) : Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses standarisasi dapat membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.
      * Scaling Data Test : kita perlu melakukan proses scaling fitur numerik pada data test/uji. Hal ini harus dilakukan agar skala antara data train dan data test sama dan kita bisa melakukan evaluasi.

## Modeling
Terdapat beberapa algoritma yang dapat diterapkan pada kasus klasifikasi. Mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik adalah cara yang dapat kita lakukan sebagai solusi utama. Ketiga algoritma yang digunakan, antara lain:
  1.	Random Forest
  2.	Support Vektor Machine
  3.	Hyperparameter GridSearch

* Model prediksi dengan algoritma Random Forest:
```
#Inisialisasi model
model_rf = RandomForestClassifier(n_estimators=100, random_state=42)
#Latih model dengan data latih
model_rf.fit(X_train_scaled, y_train)

#Prediksi pada data uji
y_pred_rf = model_rf.predict(X_test_scaled)
#Evaluasi model
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
```
> Berikut merupakan penjelasan terhadap setiap parameter yang digunakan:

> * n_estimators = menunjukkan jumlah model Decision Tree yang digunakan pada Random Forest
> * random_state = mengontrol random number generator yang digunakan. Parameter ini berupa bilangan integer dan nilainya bebas. Parameter ini bertujuan untuk memastikan bahwa hasil pembagian dataset konsisten dan memberikan data yang sama setiap kali model dijalankan. Jika tidak ditentukan, maka tiap kali melakukan split, kita akan mendapatkan data train dan tes berbeda. Hal ini berpengaruh terhadap akurasi model ML yang menjadi berbeda tiap kali di-run.

* Confusion Matrix model random forest.
  Confusion matrix atau matriks kebingungan adalah alat yang digunakan untuk menggambarkan kinerja model klasifikasi pada data uji yang sudah diketahui hasil sebenarnya. Confusion matrix merupakan cara kita mencatat poin benar dan poin salah tersebut. Di dalam matriks ini, kita tulis semua kemungkinan jawaban yang benar dan jawaban yang salah. Dari catatan ini, kita bisa mengetahui seberapa baik model kita dalam menebak kelereng dan apa yang perlu diperbaiki agar model kita makin pintar menebak. Dalam confusion matrix, ada 4 bagian yang penting untuk kita ketahui.
  
![Confusion Matrix Rf](https://github.com/user-attachments/assets/68c06849-f6a7-4d51-ade6-2e1429c8c0a3)

* Hyperparameter Model Random Forest dengan GridSearch

Parameter yang digunakan untuk optimasi model random forest menggunakan GridSearch yaitu:

  * 'n_estimators': [50, 100, 200]
  * 'max_depth': [None, 10, 20, 30]
  * 'min_samples_split': [2, 5, 10]

Dari parameter diatas akan dicari nilai parameter terbaik menggunakan GridSearch untuk model klasifikasi random forest.

* Model optimasi algoritma Random Forest dengan Hyperparameter GridSearch:
```
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
  }

grid_search = GridSearchCV(estimator=model_rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train_scaled, y_train)
#Best parameters dan evaluasi ulang model
print(grid_search.best_params_)
y_pred_best_rf = grid_search.best_estimator_.predict(X_test_scaled)
print(classification_report(y_test, y_pred_best_rf))
```
* Confusion Matrix model random forest dengan hyperparameter GridSearch:
```
#Melihat Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()
```

* Model prediksi dengan algoritma Support Vektor Machine:
```
  # Inisialisasi model SVM
model_svm = SVC(kernel='rbf', random_state=42)

# Latih model
model_svm.fit(X_train, y_train)

# Prediksi data uji
from sklearn.metrics import accuracy_score, classification_report

y_pred_svm = model_svm.predict(X_test)

# Evaluasi performa model
print(f"Akurasi: {accuracy_score(y_test, y_pred_svm)}")
print(classification_report(y_test, y_pred_svm))
```
> Berikut merupakan penjelasan terhadap setiap parameter yang digunakan:

> * kernel = Algoritma SVM menggunakan serangkaian fungsi matematika yang didefinisikan sebagai kernel. Fungsi kernel adalah mengambil data sebagai input dan mengubahnya ke dalam bentuk yang dibutuhkan. Algoritma SVM yang berbeda menggunakan berbagai jenis fungsi kernel. Fungsi-fungsi ini dapat memiliki tipe yang berbeda. Misalnya linear, nonlinier, polinomial, fungsi basis radial (RBF), dan sigmoid.
> * random_state = mengontrol random number generator yang digunakan. Parameter ini berupa bilangan integer dan nilainya bebas. Parameter ini bertujuan untuk memastikan bahwa hasil pembagian dataset konsisten dan memberikan data yang sama setiap kali model dijalankan. Jika tidak ditentukan, maka tiap kali melakukan split, kita akan mendapatkan data train dan tes berbeda. Hal ini berpengaruh terhadap akurasi model ML yang menjadi berbeda tiap kali di-run.

* Confusion Matrix model Support Vektor Machine (SVM):
```
#Confusion Matrix Model SVM
cm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
```

* Model optimasi algoritma Support Vektor Machine (SVM) dengan Hyperparameter GridSearch:

Parameter yang digunakan untuk optimasi model SVM menggunakan GridSearch yaitu:

  * 'C': [0.1, 1, 10, 100]
  * 'gamma': [1, 0.1, 0.01, 0.001]
  * 'kernel': ['rbf', 'poly', 'sigmoid']

dari parameter diatas akan dicari nilai parameter terbaik menggunakan GridSearch untuk model klasifikasi SVM.

```
# Tentukan parameter yang ingin di-tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

# Lakukan Grid Search dengan 5-fold cross-validation
grid = GridSearchCV(model_svm, param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train, y_train)

# Cetak parameter terbaik yang ditemukan oleh GridSearchCV
print("Best Hyperparameters:", grid.best_params_)

# Prediksi dengan model terbaik
y_pred_SVM = grid.predict(X_test)

# Evaluasi hasil prediksi
print(f"Akurasi: {accuracy_score(y_test, y_pred_SVM)}")
print(classification_report(y_test, y_pred_SVM))
```
* Confusion Matrix model Support Vektor Machine (SVM) dengan Hyperparameter GridSearch:
```
#Confusion Matrix Hyperparameter SVM
cm = confusion_matrix(y_test, y_pred_SVM)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
```
## Evaluasi
Untuk melihat hasil pelatihan dari masing-masing model klasifikasi dengan menggunakan akurasi pada nilai yang dihasilkan pada setiap model, nilai akurasi menggunakan library dari [sklearn](https://scikit-learn.org/dev/modules/generated/sklearn.metrics.accuracy_score.html). Selainmelihat nilai akurasi pada proyek ini melakukan visualisasi hasil pelatihan dengan confusion matrix. Berikut merupakan hasil akurasi pada setiap model:

* Evaluasi Hasil Model Random Forest:

![Akurasi RF](https://github.com/user-attachments/assets/e8c08739-c5b5-49d0-8980-d23ce916a524)

> Dari hasil evaluasi diatas model random forest memiliki hasil akurasi sebesar 89%. hasil ini akan kita tingkatkan dengan optimasi menggunakan Hyperparameter GridSearch.

* Berikut merupakan hasil dari confusion matrix pada model Random Forest:

![Hasil CM Rf](https://github.com/user-attachments/assets/2186566e-5190-429b-95be-d7280b4e0c9c)

> Hasil confusion matrix diatas diketahui bahwa:

>  * Pada label 0 terdapat 101 label yang diprediksi label 0 dan terdapat 4 yang diprediksi menjadi label 1 serta tidak ada prediksi yang berlabel 2 ataupun 3.
>  * Pada label 1 terdapat 79 label yang diprediksi label 1 dan tedapat 6 yang diprediksi menjadi label 0 dan 2, serta tidak ada prediksi yang berlabel 3.
>  * Pada label 2 terdapat 75 label yang diprediksi label 2, dan terdapat 10 yang diprediksi menjadi label 1, serta terdapat 7 yang diprediksi menjadi label 3, untuk prediksi label 0 tidak ada.
>  * Pada label 3 terdapat 99 label yang diprediksi label 3, dan terdapat 13 yang diprediksi menjadi label 2, serta tidak ada prediksi yang berlabel 0 ataupun 1.

> Dari confusion matrix tersebut dapat diketahui dataset memiliki akurasi yang cukup baik dengan penghitungan model menggunakan Random Forest.

* Evaluasi Hasil Model Random Forest dengan Hyperparameter GridSearch:

![Akurasi RF_Tuning](https://github.com/user-attachments/assets/99b484a1-8774-427f-b16c-687a270a2661)

> Hasil parameter terbaik dari Hyperparameter GridSearch yaitu:

  > * 'max_depth': 20
  > * 'min_samples_split': 5
  > * 'n_estimators': 200

> Dari hasil optimasi menggunakan Hyperparameter GridSearch dapat diketahui peningkatan dari hasil akurasi sebesar 3% yaitu dari 89% menjadi 91%. Peningkatan ini tejadi dari parameter terbaik yang dihasilkan.

* Berikut merupakan hasil dari confusion matrix pada model Random Forest dengan Hyperparameter GridSearch:

![CM RF_Tuning](https://github.com/user-attachments/assets/d4dcd251-35e8-4cc3-96e8-e543a979031a)

> Hasil confusion matrix diatas diketahui bahwa:

> * Pada label 0 terdapat 101 label yang diprediksi label 0 dan terdapat 4 yang diprediksi menjadi label 1 serta tidak ada prediksi yang berlabel 2 ataupun 3.
> * Pada label 1 terdapat 83 label yang diprediksi label 1 dan tedapat 4 yang diprediksi menjadi label 0 dan 2, serta tidak ada prediksi yang berlabel 3.
> * Pada label 2 terdapat 77 label yang diprediksi label 2, dan terdapat 8 yang diprediksi menjadi label 1, serta terdapat 7 yang diprediksi menjadi label 3, untuk prediksi label 0 tidak ada.
> * Pada label 3 terdapat 101 label yang diprediksi label 3, dan terdapat 11 yang diprediksi menjadi label 2, serta tidak ada prediksi yang berlabel 0 ataupun 1.

> Dari confusion matrix tersebut dapat diketahui perbaikan nilai prediksi dari dataset dengan penghitungan model optimasi menggunakan Hyperparameter GridSearch.

* Evaluasi Hasil Model Support Vektor Machine:

![Akurasi SVM](https://github.com/user-attachments/assets/8a91ca7c-8a41-4920-88c3-587682bc9761)

> Dari hasil evaluasi diatas model SVM memiliki hasil akurasi sebesar 96%. hasil yang cukup tinggi dan melebihi nilai akurasi dari model random forest. Selanjutnya hasil ini akan kita tingkatkan dengan optimasi yang sama sebelumnya yaitu menggunakan Hyperparameter GridSearch.

* Berikut merupakan hasil dari confusion matrix pada model Support Vektor Machine:

![CM SVM](https://github.com/user-attachments/assets/71614ca3-c1ba-4dd5-b9e0-2fa922088575)

> Hasil confusion matrix diatas diketahui bahwa:

> * Pada label 0 terdapat 103 label yang diprediksi label 0 dan terdapat 2 yang diprediksi menjadi label 1 serta tidak ada prediksi yang berlabel 2 ataupun 3.
> * Pada label 1 terdapat 90 label yang diprediksi label 1 dan tedapat 1 yang diprediksi menjadi label 0, serta tidak ada prediksi yang berlabel 2 dan 3.
> * Pada label 2 terdapat 87 label yang diprediksi label 2, dan terdapat 3 yang diprediksi menjadi label 1, serta terdapat 2 yang diprediksi menjadi label 3, untuk prediksi label 0 tidak ada.
> * Pada label 3 terdapat 106 label yang diprediksi label 3, dan terdapat 6 yang diprediksi menjadi label 2, serta tidak ada prediksi yang berlabel 0 ataupun 1.

> Dari confusion matrix tersebut dapat diketahui dataset memiliki akurasi prediksi yang cukup baik dengan penghitungan model menggunakan SVM.>

* Evaluasi Hasil Model SVM dengan Hyperparameter GridSearch:

![Akurasi SVM_Tuning](https://github.com/user-attachments/assets/644ede6c-7d3d-4446-b31f-feef2b47c8a3)

> Hasil parameter terbaik dari Hyperparameter GridSearch yaitu:

  > * 'C': 0.1
  > * 'gamma': 1
  > * 'kernel': 'poly'

> Dari hasil optimasi menggunakan Hyperparameter GridSearch dapat diketahui peningkatan dari hasil akurasi sebesar 1% yaitu dari 96% menjadi 97%. Peningkatan ini tejadi dari parameter terbaik yang dihasilkan.

* Berikut merupakan hasil dari confusion matrix pada model SVM dengan Hyperparameter GridSearch:

![CM SVM_Tuning](https://github.com/user-attachments/assets/f3b17eae-2299-484b-b9b5-891cd6aca847)

> Hasil confusion matrix diatas diketahui bahwa:

> * Pada label 0 terdapat 103 label yang diprediksi label 0 dan terdapat 2 yang diprediksi menjadi label 1 serta tidak ada prediksi yang berlabel 2 ataupun 3.
> * Pada label 1 terdapat 90 label yang diprediksi label 1 dan tedapat 1 yang diprediksi menjadi label 2, serta tidak ada prediksi yang berlabel 0 dan 3.
> * Pada label 2 terdapat 88 label yang diprediksi label 2, dan terdapat 3 yang diprediksi menjadi label 1, serta terdapat 1 yang diprediksi menjadi label 3, untuk prediksi label 0 tidak ada.
> * Pada label 3 terdapat 108 label yang diprediksi label 3, dan terdapat 4 yang diprediksi menjadi label 2, serta tidak ada prediksi yang berlabel 0 ataupun 1.

Dari confusion matrix tersebut dapat diketahui perbaikan nilai prediksi dari dataset dengan penghitungan model optimasi menggunakan Hyperparameter GridSearch.

* Membuat Plot metrik Akurasi dengan bar chart:

![Perbandingan Model](https://github.com/user-attachments/assets/8ccaebae-4de2-4ae9-bda0-bd12ca124b6e)

> Berdasarkan perbandingan diatas diketahui model SVM memiliki nilai akurasi tertinggi yaitu 97%, untuk model optimasi dengan hyperparameter GridSearch peningkatan nilai akurasi terbesar adalah model algoritma Random Forest dengan peningkatan 3%.

* Pengujian model prediksi dengan model SVM menggunakan data test yang sudah disediakan dengan data tanpa ada variabel price_range:
```
#Buat model SVM
model_prediksi = SVC(kernel='poly', C=0.1, gamma=1)

#Latih model pada data latih
model_prediksi.fit(X_train, y_train)

#Lakukan prediksi pada data uji
y_prediksi = model_prediksi.predict(df_test)

#Buat DataFrame dari hasil prediksi
predictions_df = pd.DataFrame(y_prediksi, columns=['price_range'])

#Simpan prediksi ke file CSV
predictions_df.to_csv('/content/drive/MyDrive/Machine Learning/Latihan Dicoding ML/svm_predictions.csv', index=False)
print("Prediksi telah selesai dan disimpan dalam 'svm_predictions.csv'")
```
Hasil prediksi kemudian tersimpan dalam drive dengan file csv, berikut tabel hasil prediksi data test menggunakan model SVM:

![Prediksi dataset](https://github.com/user-attachments/assets/29d68c34-0fd4-4121-be3b-8275304e7f98)

## Kesimpulan

Berdasarkan hasil pelatihan model dengan dua algoritma machine learning yaitu Random Forest dan Support Vektor Machine, masing-masing model mampu memprediksi diatas 85% dan mengalami peningkatan nilai akurasi saat dilakukan optimasi menggunakan Hyperparameter GridSearch, untuk akurasi tertinggi dihasilkan oleh model SVM dengan akurasi sebesar 96% sebelum optimasi dan 97% sesudah dilakukan optimasi, sementara untuk algoritma Random Forest memiliki nilai akurasi sebesar 89% dengan peningkatan yang cukup tinggi dibandingkan dengan SVM setelah dilakukan optimasi yaitu sebesar 3% dengan akurasi menjadi 91%. Dari hasil tersebut model algoritma Support Vektor Machine (SVM) lebih tepat dan baik untuk digunakan sebagai model klasifikasi pada dataset Mobile Price Kalsifikasi.

## Penutup

Demikian hasil dari laporan proyek machine learning, predicitive analytics dengan dataset Mobile Price Klasifikasi. Bilamana didalam penyampaian serta penjelasan yang kurang berkenaan, saya memohon maaf. Atas waktu dan perhatiannya, saya ucapkan Terima kasih telah membaca laporan ini. Semoga dapat memberi manfaat bagi kita semuanya.

