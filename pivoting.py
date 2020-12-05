import pandas as pd
import numpy as np
import re

df = pd.read_csv('Preprocessing_columns.csv', sep =';', encoding='Latin-1',names=["user-id","rating","book_title"])
print(df)
df.to_csv('prepivoting.csv') ## We add columns in the previous csv

