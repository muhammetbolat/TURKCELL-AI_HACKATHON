1.) TEST DATALARININ YÜKLENMESİ

	Bize gönderilen eğitim verilerinde olduğu gibi test edilmek istenen resimler aynı şekilde dosyalara koyulmalıdır. Dosya isimlerinin birebir aynı olması önemlidir. Çünkü etiketleme bu isimlendirmelere göre yapılmıştır.
	Örneğin:

			TEST/
			├── Armut
			│   ├── armut_2200.jpg
			│   ├── armut_2201.jpg
			│   └── armut_2507.jpg
			├── Cilek
			│   ├── cilek_1372.jpg
			│   ├── cilek_1598.jpg
			│   └── cilek_1599.jpg
			├── Elma_Kirmizi
			│   ├── elma_kirmizi_1661.jpg
			│   ├── elma_kirmizi_1950.jpg
			│   └── elma_kirmizi_1951.jpg
			├── Elma_Yesil
			│   ├── elma_yesil_1507.jpg
			│   ├── elma_yesil_1508.jpg
			│   └── elma_yesil_1721.jpg
			├── Mandalina
			│   ├── mandalina_1474.jpg
			│   ├── mandalina_1475.jpg
			│   └── mandalina_1656.jpg
			├── Muz
			│   ├── muz_1639.jpg
			│   ├── muz_1931.jpg
			│   └── muz_1932.jpg
			└── Portakal
			    ├── portakal_1517.jpg
			    ├── portakal_1662.jpg
			    └── portakal_1663.jpg

	TEST dosyasının mutlak ya da göreceli yol ifadesini inference.py dosyasının içerisinde TEST_PATH isimli değişkene atayabilirsiniz.
		TEST_PATH = '/Users/tcmbolat/Desktop/AI-HACKATHON/HAM_DATA/TEST/'

2.) MODELİN YÜKLENMESİ

	Zip dosyasının içerisinde muhammetbolat_model.h5 adında bir dosya bulunmaktadır. Bu dosya içerisinde mevcut model ve ağırlık katsayıları bulunmaktadır. Model dosyasının göreceli yol ifadesi ANN_MODEL_PATH = 'muhammetbolat_model.h5' değişkenine atanmıştır. Ancak spesifik bir
	dosya olarak da değiştirilebilinir.



3.) GORSELLERİN SIZE'LARI

	Model resimleri 200X200'e resize edilmektedir. Bu yüzden herhangi bir değişiklik yapmanıza gerek kalmamaktadır.


4.) PREDICTION SONUÇLARI

	PREDICTION yapıldıktan sonra size çıktı olarak detaylı bir rapor sunulacaktır. Örnek rapor aşağıdaki gibidir.

	### Result of the predictions using 1607 test data ###

	Classification Report:

	              precision    recall  f1-score   support

	       Armut       0.98      0.79      0.87       295
	       Cilek       0.99      0.98      0.99       196
	Elma_Kirmizi       0.96      0.99      0.98       288
	  Elma_Yesil       0.99      0.99      0.99       207
	   Mandalina       0.92      0.87      0.89       183
	         Muz       0.92      0.98      0.95       293
	    Portakal       0.77      0.99      0.86       145

	 avg / total       0.94      0.94      0.94      1607


	Confusion Matrix:


	              Armut  Cilek  Elma_Kirmizi  Elma_Yesil  Mandalina  Muz  Portakal
	Armut           232      0             8           1         10   22        22
	Cilek             0    193             3           0          0    0         0
	Elma_Kirmizi      0      2           286           0          0    0         0
	Elma_Yesil        0      0             0         204          3    0         0
	Mandalina         1      0             0           0        159    3        20
	Muz               3      0             0           1          0  287         2
	Portakal          0      0             0           0          1    0       144

	Accuracy: 0.93653


