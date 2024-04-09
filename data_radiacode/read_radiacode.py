import yaml
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks, savgol_filter
#from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from scipy.optimize import curve_fit

#reads yaml file from given path and returns data
def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            return yaml_data

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def gauss(x, A, mu, sigma):
    return A*( (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma** 2)) )

def lin_interp(x, y, i, half):
    return x[i] + (x[i+1] - x[i]) * ((half - y[i]) / (y[i+1] - y[i]))

def half_max_x(x, y):
    half = max(y)/2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]

#values
cs137_energy = [662]        #kev #channel num 265
co60_energy  = [1173, 1333] #kev #            452,509
na22_energy  = [511, 1275]  #kev #            209, 489


#read data and extract values
cs137_data = read_yaml_file('m1Cs137_240226-1235.yaml')
co60_data = read_yaml_file('m2C060_240226-1246.yaml')
na22_data = read_yaml_file('m3Na22_240226-1256.yaml')
ra226_data = read_yaml_file('m4Ra226_240226-1303.yaml')

spectrum = np.array([cs137_data['spectrum'], co60_data['spectrum'], na22_data['spectrum'], ra226_data['spectrum']])
#rates = np.array([cs137_data['rates'], co60_data['rates'], na22_data['rates'], ra226_data['rates']])

#-----------------------------------------------------------------
#smooth out data
#find peaks for all isotops
#calibrate x axis by finding cor. energy

peaks = []
smoothed_spectrum = []

x_channel = np.arange(0, len(spectrum[0]))

for i in range( len(spectrum) ):
    smoothed_spectrum.append( savgol_filter(spectrum[i],20,3))
    temp, temp2 = find_peaks(smoothed_spectrum[i], height=10, prominence=5)
    peaks.append(temp)                 # find peaks

#find energy for x_values
channel_energy_linreg = linregress([209, 265, 452, 489, 509] , [511,662,1173,1275,1333])
x_energy =  channel_energy_linreg[0] * x_channel + channel_energy_linreg[1]

#-----------------------------------------------------------------
#find energy of peaks
#find energy fit quality
#fit gauss curves over peaks

#Na22 1 ; CS137 ; Co60 1 ; Na22 2 ; Co60 2
peak_channel = [209, 265, 452, 489, 509] #channel of peak found by find_peaks
peak_energy = [x_energy[peak_channel]]  #energy of peaks above 
peak_pos = [[167,240] , [230,340] , [414,480] , [438,540] , [481,544]]   #channels over which gaus fit is done
gaus_popt = []
gaus_pcov = []
fwhm = []
#lin reg of list of messured energies to real ones, y= real
#print( linregress(x_energy[[209, 265, 452, 489, 509]], [511,662,1173,1275,1333]))


#fit gaussian
for i,j in zip([2, 0, 1, 2, 1],range(len(peak_pos))):
    popt, pcov = curve_fit(gauss, x_energy[peak_pos[j][0] : peak_pos[j][1]],
                           spectrum[i][peak_pos[j][0] : peak_pos[j][1]],
                           p0 = [1,x_energy[peak_channel[j]],1])
    gaus_pcov.append(pcov)
    gaus_popt.append(popt)
    #print(popt)


#find FWHM (full width half maximum)
for j in range(len(gaus_popt)):
    if j == 2:
        continue;
    x = x_energy[peak_pos[j][0] : peak_pos[j][1]]
    y = gauss(x,gaus_popt[j][0],gaus_popt[j][1],gaus_popt[j][2])
    hmx = half_max_x(x, y)
    fwhm.append(hmx[1] - hmx[0])

print(fwhm)

# Plotting
label = ['Cs137', 'Co60', 'Na22','Ra226']
for i,j in zip([2, 0, 1, 2, 1],range(len(peak_pos))):
    i=i
    j=j
    plt.plot(x_energy, smoothed_spectrum[i], marker='o', linestyle='-', color='b', markersize=1)
    plt.plot(x_energy, spectrum[i], marker='o', linestyle='', color='r', markersize=1, label=label[i])
    # plt.plot(x_channel, smoothed_spectrum[i], marker='o', linestyle='-', color='b', markersize=1)
    # plt.plot(x_channel, spectrum[i], marker='o', linestyle='', color='r', markersize=1, label=label[i])
    plt.plot(x_energy[peak_pos[j][0] : peak_pos[j][1]], 
             gauss(x_energy[peak_pos[j][0] : peak_pos[j][1]],gaus_popt[j][0],gaus_popt[j][1],gaus_popt[j][2]),
             marker='o', linestyle='-', color='g', markersize=1,)
    plt.xlabel('Energy in keV')
    #plt.yscale('log')
    plt.ylabel('N')
    plt.title('Spectrum Plot')
    plt.legend()
    plt.grid(True)
    plt.show()
   