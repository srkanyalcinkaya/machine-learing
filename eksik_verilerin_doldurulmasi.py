# -*- coding: utf-8 -*-
"""
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