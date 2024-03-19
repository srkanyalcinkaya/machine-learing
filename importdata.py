import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler = pd.read_csv("data.csv")

boy = veriler[["boy"]]
boy_kilo = veriler[["boy","kilo"]]

print(boy_kilo)