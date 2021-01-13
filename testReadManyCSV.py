import os, sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
#--------------------------------------------------------------------------

import glob2 as glob
import pandas as pd

files = glob.glob(os.getcwd() + "\\*Story*.csv")
initDataframe = [pd.read_csv(f, header=None, sep=";") for f in files]

#print(dataframe[0][0][1]) # actual row

df = pd.DataFrame({'initDF':initDataframe})
print(type(df))
print(df.shape)
print(df.size)

for k,v in df.iterrows():
    print(v[0][0][1])
