import os
import numpy as np

# path = "C:/Users/ASUS/Documents/MIT/Thesis/python code"
# files = os.listdir(path)
# for f in files:
# 	if f[0:2] == "L2":
# 		print(f[0:8])

blah = [[1, 2, 3], [4,5,6], [7,8,9],[10,11,12]]
blah = np.array(blah)
print(blah)
print(np.mean(blah))
print(np.mean(blah, axis=0))
print(np.mean(blah,axis=1))