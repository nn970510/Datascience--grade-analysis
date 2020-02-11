import csv
x=[]
y=[]
file="Admission_predict_Ver1.1.csv"
data=open(file,"r")
reader=csv.reader(data)
for item in reader:
	if reader.line_num==1:
		continue
	x.append(item[:-1])
	y.append(item[-1])