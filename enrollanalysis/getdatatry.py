import csv
import pandas as pd
import sys
x=[]
y=[]
file="Admission_predict_Vermiss.csv"
df=pd.read_csv(file, sep=",")
print (df.isnull().sum())
df=df.dropna()
#data=open(file,"r")
#reader=csv.reader(data)
df=df.values
xuse=df[:,:-1]
print ("After delete missing data, there are still %s data avaliable\n\n"%len(xuse))
yuse=df[:,-1]

