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
```
  * Hampir sama dengan data train, namun tanpa variabel price_range dan tambahan kolom Id sebagai pengenal unik.
```
