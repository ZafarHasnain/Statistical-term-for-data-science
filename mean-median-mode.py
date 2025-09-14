#mean,median,mode,
#std,iqr,variance


import numpy as np
import statistics as st
import matplotlib.pyplot as plt
x=[3,4,2,4,2,5,56,3,2,2,6,8]
y=np.mean(x)
print(y)
y1=np.median(x)
print(y1)
y2=st.mode(x)
print(y2)


#standard deviation
y3=np.std(x)
print(y3)

#variance
y4=np.var(x)
print("variance:",y4)

#interquartile range
q1 = np.percentile(x, 25)   # 25th percentile
q3 = np.percentile(x, 75)   # 75th percentile
iqr = q3 - q1
print("Q1 (25th percentile):", q1)
print("Q3 (75th percentile):", q3)
print("Interquartile Range (IQR):", iqr)
