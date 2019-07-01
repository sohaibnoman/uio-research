import numpy as np
import pandas as ps
from fastai import *                # import common functionallity
from fastai.tabular import *        # import tabular functionallity

path = untar_data(URLs.ADULT_SAMPLE)
df = pd.read_csv(path/'adult.csv')
train_df, valid_df = df[:-2000].copy(), df[-2000:].copy()

print(valid_df.head())
