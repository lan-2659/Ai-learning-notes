import numpy as np
arr1=np.array([1,2,3,4,5])
print(id(arr1))
arr2=np.array([10,20,30,40,50])
# arr1=arr2
arr1[::]=arr2
print(arr1)
print(id(arr1))
