import yaml
import numpy as np
import matplotlib.pyplot as plt

#from scipy.signal import find_peaks, savgol_filter
from scipy.interpolate import UnivariateSpline

#reads yaml file from given path and returns data
def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            return yaml_data

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def moving_average(data, window_size):
    """Apply a moving average filter to smooth out the data."""
    window = np.ones(window_size) / window_size
    return np.convolve(data, window, mode='same')

def peak_finder(data, threshold):
    """Find peaks in data above a certain threshold."""
    peaks = []
    for i in range(1, len(data) - 1):
        if data[i] > threshold and data[i] > data[i - 1] and data[i] > data[i + 1]:
            peaks.append(i)
    return peaks

def polyfit_data(x, y, degree):
    """Smooth the data by fitting a polynomial."""
    coeffs = np.polyfit(x, y, degree)
    smoothed_y = np.polyval(coeffs, x)
    return smoothed_y

#read data 
cs137_data = read_yaml_file('m1Cs137_240226-1235.yaml')
co60_data = read_yaml_file('m2C060_240226-1246.yaml')
na22_data = read_yaml_file('m3Na22_240226-1256.yaml')
ra226_data = read_yaml_file('m4Ra226_240226-1303.yaml')

#extracte data
#rates = np.array([cs137_data['rates'], co60_data['rates'], na22_data['rates'], ra226_data['rates']])
spectrum = np.array([cs137_data['spectrum'], co60_data['spectrum'], na22_data['spectrum'], ra226_data['spectrum']])


#find splines for all plots to smooth out data
#find peaks for all isotops
x_values = np.arange(1, len(spectrum[0]) + 1)
s_factor = [ 1e5 , 2.5e4 , 2e4 , 6e3 ] # individual smoothing factore for splines 


splines = []
peaks = []
smoothed_spectrum = []

for i in range(4):
    spline = UnivariateSpline(x_values, spectrum[i], s=s_factor[i])  # Adjust smoothing factor 's' as needed
    smoothed_spectrum.append( spline(x_values))                        # y for given x
    peaks.append( peak_finder(smoothed_spectrum[i], 10) )                 # find peaks
    splines.append(spline)


# Plotting
label = ['Cs137', 'Co60', 'Na22','Ra226']
for i in range(4):
    plt.plot(x_values, smoothed_spectrum[i], marker='o', linestyle='-', color='b', markersize=1)
    plt.plot(x_values, spectrum[i], marker='o', linestyle='', color='r', markersize=1, label=label[i])
    plt.xlabel('bin')
   # plt.yscale('log')
    plt.ylabel('Spectrum')
    plt.title('Spectrum Plot')
    plt.legend()
    plt.grid(True)
    plt.show()
   