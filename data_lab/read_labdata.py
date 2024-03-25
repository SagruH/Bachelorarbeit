import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

csvfile= ['M2CS137.csv', 'M1Co60.csv', 'M4Na22.csv', 'M3Ra226.csv']
label = ['Cs137', 'Co60', 'Na22','Ra226']


for i in range(4):
    #read data
    data = pd.read_csv(csvfile[i], delimiter=';', skiprows=2, decimal =',')
    signal = data.iloc[:, 1].to_numpy()
    channel = data.iloc[:, 2].to_numpy()
    #plot
    plt.plot(channel, signal, marker='o', linestyle='', color='b', markersize=1, label=label[i])
    plt.xlabel('bin')
    plt.yscale('log')
    plt.ylabel('Spectrum')
    plt.title('Spectrum Lab')
    plt.legend()
    plt.grid(True)
    plt.show()