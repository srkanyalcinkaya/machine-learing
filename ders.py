#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 20:03:13 2024

@author: serkanyalcinkaya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv("eksikveriler.csv")

# eksik verilerin ortalama ile doldurulması
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

yas = veriler.iloc[:,1:4].values

imputer = imputer.fit(yas[:,1:4])

yas[:,1:4] = imputer.transform(yas[:,1:4])


# verileri kategorileştirme
"""
encoder: Kategorik -> Numeric
"""
from sklearn import preprocessing

ulke = veriler.iloc[:,0:1].values

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])

#print(ulke)

ohe = preprocessing.OneHotEncoder()

ulke = ohe.fit_transform(ulke).toarray()

#print(ulke)


## verileri birleştirme
"""
numpy dizilerini dataframe donusumu
"""
sonuc = pd.DataFrame(data=ulke, index=range(22), columns=["fr","tr","us"])

sonuc2 = pd.DataFrame(data=yas, index=range(22), columns=["boy","kilo","yas"])

##Cinsiyeti cvs'den alıyoruz ve iloc ile kolonu alıyoruz
cinsiyet = veriler.iloc[:,-1]

sonuc3 = pd.DataFrame(data=cinsiyet, index=range(22), columns=["cinsiyet"])

## hepsini birleştirme işlemi yapıyoruz
s = pd.concat([sonuc,sonuc2], axis=1)

s2 = pd.concat([s,sonuc3], axis=1)

#print(s2)

"""
Verilerin egitimi ve test için bölünmesi
train_size -> yüzde kaçı egitim seti olarak kullanması belirlenmesi
"""
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s,sonuc3, train_size=0.33, random_state=0)


##öznitelik ölçeklemesi

from sklearn.preprocessing import StandardScaler

sc  =StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

