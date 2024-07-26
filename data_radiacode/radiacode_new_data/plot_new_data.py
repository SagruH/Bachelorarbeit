# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 05:15:21 2024

@author: sagru_8xqc8px
"""

import yaml
import numpy as np
import matplotlib.pyplot as plt
import PhyPraKit as ppk

def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            return yaml_data

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

co60_data = read_yaml_file('r_co60_20min.yaml')
cs137_data = read_yaml_file('r_CS137_20min.yaml')
na22_data = read_yaml_file('r_na22_20min.yaml')
ra226_data = read_yaml_file('r_ra226_20min.yaml')


spectrum = np.array([cs137_data['spectrum'], co60_data['spectrum'], na22_data['spectrum'], ra226_data['spectrum']])

channel = np.arange(len(spectrum[0]))
name =['Cs137', 'Co60', 'Na22','Ra226']

for i in range(0,4):
    plt.figure(figsize=(10, 6))
    plt.plot(channel[0:600], spectrum[i][0:600], label=name[i], 
             marker='o', linestyle='', color='b', markersize=1)
    
    
    # Add labels and title
    plt.xlabel('Channel')
    plt.ylabel('N')
    plt.title('Spectrum Plot')
    
    
    # Add a grid for better readability
    plt.grid(True)
    
    # Show legend
    plt.legend()
    
    # Display the plot
    plt.show()