# LAPORAN PROYEK MACHINE LEARNING - Ahmad Fauzi
## Domain Proyek
Di era persaingan pasar ponsel yang semakin ketat, inovasi teknologi dan fitur-fitur baru terus bermunculan, menjadikan harga produk sebagai salah satu faktor penentu keberhasilan perusahaan dalam menarik konsumen. Bob, seorang pengusaha yang baru memulai perusahaan ponselnya, menghadapi tantangan dalam menetapkan harga yang kompetitif. Perusahaan besar seperti Apple dan Samsung mendominasi pasar dengan produk-produk yang memiliki rentang harga yang jelas sesuai dengan spesifikasi teknis dan inovasi yang ditawarkan [Gartner, 2021](https://www.gartner.com/en/newsroom/press-releases/2022-03-01-4q21-smartphone-market-share). Untuk menyaingi perusahaan-perusahaan ini, Bob membutuhkan cara yang efisien untuk menentukan harga ponsel berdasarkan fitur-fitur teknis seperti RAM, memori internal, dan kapasitas baterai. Dalam konteks ini, pendekatan manual tidak memadai, dan metode berbasis data seperti machine learning dapat menawarkan solusi yang lebih efisien dan akurat [Choudhary et al., 2021](https://www.gartner.com/en/newsroom/press-releases/2022-03-01-4q21-smartphone-market-share).

Machine learning telah terbukti menjadi alat yang sangat berguna dalam memproses data besar dan menemukan pola yang tersembunyi, terutama dalam masalah klasifikasi harga. Melalui algoritma seperti Random Forest dan Support Vector Machine (SVM), model machine learning dapat mengolah fitur-fitur penting dari ponsel dan menghasilkan prediksi yang akurat tentang kisaran harga produk [Mulla & Desai, 2020](https://link.springer.com/article/10.1007/s40622-020-00260-8). Penggunaan metode ini memungkinkan perusahaan seperti milik Bob untuk memahami segmentasi pasar dengan lebih baik dan menetapkan harga yang kompetitif sesuai dengan kebutuhan konsumen. Ini penting karena fitur teknis yang ditawarkan oleh ponsel memiliki korelasi langsung dengan kisaran harga yang diharapkan, sebagaimana ditunjukkan dalam studi-studi prediksi harga berbasis fitur produk [Zhang et al., 2019](https://link.springer.com/article/10.1007/s10115-022-01679-4).

Sebagai perusahaan baru, Bob memerlukan strategi penetapan harga yang tepat untuk memaksimalkan keuntungan dan meningkatkan daya saing produknya di pasar yang sangat kompetitif. Klasifikasi harga dengan menggunakan machine learning memungkinkan Bob untuk menetapkan harga berdasarkan data empiris dan tren pasar. Selain itu, hal ini dapat membantu dalam menargetkan segmen pelanggan yang sesuai dan mengembangkan strategi pemasaran yang lebih efektif [Mulla & Desai, 2020](https://link.springer.com/article/10.1007/s40622-020-00260-8). Oleh karena itu, proyek ini tidak hanya relevan bagi Bob, tetapi juga merupakan pendekatan yang sesuai dengan kebutuhan industri yang bergerak cepat di mana inovasi teknologi terus berkembang.

## Business Understanding
### Problem Statements
Bob, pengusaha yang baru memulai perusahaan ponselnya, menghadapi tantangan besar dalam menetapkan harga yang kompetitif untuk produknya. Dalam pasar ponsel yang kompetitif, penetapan harga yang salah dapat mengakibatkan hilangnya peluang pasar atau potensi pendapatan yang terlewatkan. Bob tidak memiliki pengetahuan teknis yang cukup dalam machine learning untuk memprediksi harga produk berdasarkan fitur-fitur teknis seperti RAM, memori internal, kapasitas baterai, dan sebagainya. Tantangan utama yang dihadapi adalah bagaimana mengklasifikasikan ponsel ke dalam kisaran harga yang sesuai berdasarkan fitur-fitur tersebut, sehingga dapat menargetkan segmen pasar yang tepat.
### Goals
Tujuan utama dari proyek ini adalah membantu Bob mengembangkan sistem yang dapat mengklasifikasikan ponsel yang diproduksi ke dalam berbagai kisaran harga berdasarkan spesifikasi teknis. Dengan adanya model machine learning ini, Bob dapat membuat keputusan penetapan harga yang lebih cerdas dan kompetitif, serta memposisikan produknya dengan lebih baik di pasar. Dengan prediksi kisaran harga yang akurat, perusahaan Bob dapat bersaing dengan brand besar seperti Apple dan Samsung, dan meraih pangsa pasar dengan lebih efektif.
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
  * Train Test Split : Membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus kita lakukan sebelum membuat model.
  * Data Transform
    * Scaling: 
      * Scaling Data Train (Standarisasi) : Algoritma machine learning memiliki performa lebih baik dan konvergen lebih cepat ketika dimodelkan pada data dengan skala relatif sama atau mendekati distribusi normal. Proses         standarisasi dapat membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma.
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
> n_estimators = menunjukkan jumlah model Decision Tree yang digunakan pada Random Forest
> random_state = mengontrol random number generator yang digunakan. Parameter ini berupa bilangan integer dan nilainya bebas. Parameter ini bertujuan untuk memastikan bahwa hasil pembagian dataset konsisten dan memberikan data yang sama setiap kali model dijalankan. Jika tidak ditentukan, maka tiap kali melakukan split, kita akan mendapatkan data train dan tes berbeda. Hal ini berpengaruh terhadap akurasi model ML yang menjadi berbeda tiap kali di-run.


