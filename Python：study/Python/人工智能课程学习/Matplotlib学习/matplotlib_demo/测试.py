import pandas as pd
import numpy as np

df = pd.DataFrame(np.arange(16).reshape(4, 4), index=['a', 'b', 'c', 'd'], columns=['A', 'B', 'C', 'D'])
print(df)
print()

re = df.reindex(index=['a', 'b', 'c', 'd', 'e'], method='ffill')
print(re)
print()

re = df.reindex(index=['a', 'e', 'f'], method='ffill')
print(re)
print()

