import getdatatry
import numpy as np
from sklearn import linear_model
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score
import pandas as pd
import time
import matplotlib.pyplot as plt
xuse=(getdatatry.xuse).astype(np.float64)
yuse=(getdatatry.yuse).astype(np.float64)
testx=np.zeros((int(0.3*len(getdatatry.yuse)),7))
testy=np.arange(int(0.3*len(getdatatry.yuse)), dtype=np.float64)
trainx=np.zeros(((len(getdatatry.yuse)-int(0.3*len(getdatatry.yuse))),7))
trainy=np.arange(len(getdatatry.yuse)-int(0.3*len(getdatatry.yuse)),dtype=np.float64)

randli=np.random.choice(len(getdatatry.yuse),int(0.3*len(getdatatry.yuse)),replace=False)
randli=np.sort(randli)

test=0
train=0
for i in range (len(yuse)):
	if len(randli)>1 and i==(randli[0]):
		testx[test]=xuse[i][1:]
		testy[test]=yuse[i]
		randli=randli[1:]
		test=test+1
	elif len(randli)==1 and i==(randli[0]):
		testx[test]=xuse[i][1:]
		testy[test]=yuse[i]	
	else:
		trainx[train]=xuse[i][1:]
		trainy[train]=yuse[i]
		train=train+1

start=time.clock()
mlr=linear_model.LinearRegression()
mlr.fit(trainx.astype(np.float64),trainy.astype(np.float64))
elapseML=(time.clock()-start)
y_predict=mlr.predict(testx.astype(np.float64))	

start1=time.clock()
clf=DecisionTreeRegressor()
clf=clf.fit(trainx.astype(np.float64),trainy.astype(np.float64))
elapseTR=(time.clock()-start1)
y_predict_tree=clf.predict(testx.astype(np.float64))	

mse_test=np.sum((y_predict-testy)**2)/len(testy)
print ("MSE test for linear_model: %s"%mse_test)
mse_test_tree=np.sum((y_predict_tree-testy)**2)/len(testy)
print ("MSE test for DecisionTreeRegressor: %s"%mse_test_tree)

mae_test=median_absolute_error(testy, y_predict)
print ("MAE test for linear_model: %s"%mae_test)
mae_test_tree=median_absolute_error(testy, y_predict_tree)
print ("MAE test for DecisionTreeRegressor: %s"%mae_test_tree)

r2_test=r2_score(testy, y_predict)
print ("R-squared test for linear_model: %s"%r2_test)
r2_test_tree=r2_score(testy, y_predict_tree)
print ("R-squared test for DecisionTreeRegressor: %s"%r2_test_tree)

LMTP=TRTP=LMTN=TRTN=LMFP=TRFP=LMFN=TRFN=0
LMerror=TRerror=0

for i in range (len(testy)):
	LMerror=LMerror+abs(y_predict[i]-(testy[i]))
	if (testy[i]>=0.75) and (y_predict[i]>=0.75):
		LMTP=LMTP+1
	elif (testy[i]>=0.75) and (y_predict[i]<0.75):
		LMFN=LMFN+1	
	elif (testy[i]<0.75) and (y_predict[i]>=0.75):
		LMFP=LMFP+1	
	elif (testy[i]<0.75) and (y_predict[i]<0.75):
		LMTN=LMTN+1
for i in range (len(testy)):
	TRerror=TRerror+abs(y_predict_tree[i]-(testy[i]))				
	if (testy[i]>=0.75) and (y_predict_tree[i]>=0.75):
		TRTP=TRTP+1		
	elif (testy[i]>=0.75) and (y_predict_tree[i]<0.75):
		TRFN=TRFN+1	
	elif (testy[i]<0.75) and (y_predict_tree[i]>=0.75):
		TRFP=TRFP+1	
	elif (testy[i]<0.75) and (y_predict_tree[i]<0.75):
		TRTN=TRTN+1	
print ("\n\nIf we say 75% is the line to decide if a student should apply this university.")
LMpre=LMTP/(LMTP+LMFP)
print ("Precision for linear model: %s"%LMpre)
TRpre=TRTP/(TRTP+TRFP)
print ("Precision for Decision Tree Regressor: %s"%TRpre)
LMrecall=LMTP/(LMTP+LMFN)
print ("Recall for linear model: %s"%LMrecall)
TRrecall=TRTP/(TRTP+TRFN)
print ("Recall for Decision Tree Regressor: %s"%TRrecall)
LMacc=(LMTP+LMTN)/(LMTP+LMFP+LMTN+LMFN)
print ("Accuracy for linear model: %s"%LMacc)
TRacc=(TRTP+TRTN)/(TRTP+TRFP+TRTN+TRFN)
print ("Accuracy for Decision Tree Regressor: %s"%TRacc)
F1LM=2*LMpre*LMrecall/(LMpre+LMrecall)
F1TR=2*TRpre*TRrecall/(TRpre+TRrecall)
print ("F1 for Linear Model: %s"%F1LM)
print ("F1 for Decision Tree Regressor: %s"%F1TR)
print ("\n\nAverage error for linear model: %s"%(LMerror/len(testy)))
print ("Average error for Decision Tree Regressor: %s"%(TRerror/len(testy)))	
print ("\n\nUsing 10 fold validation to check error")
divten=int(len(yuse)/10)

TELM=0
TETR=0
Allerrlm=[]
Allerrtr=[]
for i in range(10):
	print("fold %s"%(i+1))
	if i==9:
		ttestx=xuse[(i)*divten:,1:]
		ttesty=yuse[(i)*divten:]
	else:
		ttestx=xuse[(i)*divten:((i+1)*divten),1:]
		ttesty=yuse[(i)*divten:((i+1)*divten)]

	if i==0:
		ttrainx=xuse[((i+1)*divten):,1:]
		ttrainy=yuse[((i+1)*divten):]
	elif i==9:
		ttrainx=xuse[:(i)*divten,1:]
		ttrainy=yuse[:(i)*divten]
	else:
		trainx1=xuse[:(i)*divten,1:]
		trainx2=xuse[((i+1)*divten):,1:]
		trainy1=yuse[:(i)*divten]
		trainy2=yuse[((i+1)*divten):]
		ttrainx=np.r_[trainx1,trainx2]
		ttrainy=np.r_[trainy1,trainy2]
	mlr=linear_model.LinearRegression()
	clf=DecisionTreeRegressor()
	mlr.fit(ttrainx.astype(np.float64),ttrainy.astype(np.float64))
	clf.fit(ttrainx.astype(np.float64),ttrainy.astype(np.float64))
	ty_predict=mlr.predict(ttestx.astype(np.float64))	
	ty_predict_tree=clf.predict(ttestx.astype(np.float64))
	grouperrLM=0
	grouperrTR=0
	for i in range (len(ttesty)):
		sigerrorLM=abs(ty_predict[i]-ttesty[i])
		grouperrLM=grouperrLM+sigerrorLM
		sigerrorTR=abs(ty_predict_tree[i]-ttesty[i])
		grouperrTR=grouperrTR+sigerrorTR
	grouperrLM=grouperrLM/len(ttesty)
	grouperrTR=grouperrTR/len(ttesty)
	print ("group averange error Linear Model:%s"%grouperrLM)
	print ("group averange error Decision Tree Regressor:%s"%grouperrTR)
	TELM=TELM+grouperrLM
	TETR=TETR+grouperrTR
	Allerrtr.append(grouperrTR)
	Allerrlm.append(grouperrLM)
	print ("-------------------||||||||||||||||||||------------------")
TELM=TELM/10
TETR=TETR/10

print ("total averange error linear model:%s "%TELM)
print ("total averange Decision Tree Regressor:%s "%TETR)
trarr_var=np.var(Allerrtr)
t=(TETR-TELM)/(trarr_var/(10**0.5))
print("t value is %s"%t)
if (t>2.262):
	print("There is no significant different in confidence level 95%, 90%, and 80%")
elif(2.262>t>1.833):
	print("There is no significant different in confidence level 90%, and 80% but in 95%")
elif(1.833>t>1.383):
	print("There is no significant different in confidence level 80% but in 95%, 90% ")
else:
		print("There is significant different in confidence level 95%, 90%, and 80%")

print ("------------------------------------------------------------------------------")
print("Linear model train time: %s"%elapseML)
print("Decision Tree Regressor train time: %s"%elapseTR)