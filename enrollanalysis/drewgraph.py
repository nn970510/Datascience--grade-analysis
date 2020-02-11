import getdata
import numpy as np
import matplotlib.pyplot as plt
GRE=[]
TOEFL=[]
GPA=[]
for i in range (len(getdata.x)):
	GRE.append(getdata.x[i][1])
	TOEFL.append(getdata.x[i][2])
	GPA.append(getdata.x[i][6])

GRE=np.array(GRE).astype(np.float64)
GPA=np.array(GPA).astype(np.float64)
TOEFL=np.array(TOEFL).astype(np.float64)
yuse=np.array(getdata.y).astype(np.float64)

plt.figure(figsize=(7,7))
ax=plt.subplot(221)
T=np.arctan2(yuse,GRE)
plt.scatter(GRE, yuse, c=T)
plt.xlim(287, 342)
plt.ylim(0.25, 1)
ax.set_title("GRE")

ax=plt.subplot(222)
T=np.arctan2(yuse,TOEFL)
plt.scatter(TOEFL, yuse, c=T)
plt.xlim(90, 122)
plt.ylim(0.25, 1)
ax.set_title("TOEFL")

ax=plt.subplot(223)
T=np.arctan2(yuse,GPA)
plt.scatter(GPA, yuse, c=T)
plt.xlim(6.5, 10)
plt.ylim(0.25, 1)
ax.set_title("GPA")

plt.show()
