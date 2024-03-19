import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler = pd.read_csv("eksikveriler.csv")

boy = veriler[["boy"]]
boy_kilo = veriler[["boy","kilo"]]

#print(boy_kilo)

# eksik verilerin ortalama ile doldurulması

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

"""
iloc: verilerin içinden kolonları alınması
"""
yas = veriler.iloc[:,1:4].values

imputer = imputer.fit(yas[:,1:4])

yas[:,1:4] = imputer.transform(yas[:,1:4])

##print(yas)