import matplotlib.pyplot as plt
import pandas as pd
import getdata
import numpy as np

xuse=np.array(getdata.x).astype(np.float64)
yuse=np.array(getdata.y).astype(np.float64)

def Outliercheck(df):
	IQR=df.quantile(.75)-df.quantile(.25)
	print("IQR1 is %s"%df.quantile(.25))
	print("IQR3 is %s"%df.quantile(.75))
	for i in range (len(yuse)):
		if df[i]>(df.quantile(.75)+(1.5*IQR)) or df[i]<(df.quantile(.25)-(1.5*IQR)):
			print ("Outlier point: %s, value is %s"%((i+1), yuse[i]))

dfy=pd.Series(yuse)
dfg=pd.Series(xuse[:,1])
dft=pd.Series(xuse[:,2])
dfc=pd.Series(xuse[:,6])
print("outlier check for Admission Percentage")
Outliercheck(dfy)
print("outlier check for GRE score")
Outliercheck(dfg)
print("outlier check for TOEFL score")
Outliercheck(dft)
print("outlier check for GPA")
Outliercheck(dfc)
# figure,axes=plt.subplots() #得到画板、轴
# axes.boxplot(yuse,patch_artist=True) #描点上色
# plt.show()
