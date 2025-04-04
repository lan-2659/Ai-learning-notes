import numpy as np

arr = np.arange(64).reshape(4, 4, 4)
print(arr[::, ::-1, ::])
print(arr[::, ::, -999:100:-1])
print(arr[::, -999:100:-1, ::])
print(arr[-999:100:-1, ::, ::])