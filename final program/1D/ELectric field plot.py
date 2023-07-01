import csv
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 1, 101)
df = pd.read_csv('electric_field.csv')
y = []
z = []
for index, row in df.iterrows():
    a = row
    b = index
    y.append(b)
    z.append(a)
plt.contourf(x, y, z, levels=100)
plt.xlabel('Length in arb. units')
plt.ylabel("Iteration (time)")
plt.title("Electric Field Plot")
plt.colorbar()
plt.show()
