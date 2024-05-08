import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PhyPraKit as ppk


from scipy.signal import find_peaks, savgol_filter
from scipy.stats import linregress
from scipy.optimize import curve_fit

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

csvfile= ['M2CS137.csv', 'M1Co60.csv', 'M4Na22.csv', 'M3Ra226.csv']
label = ['Cs137', 'Co60', 'Na22','Ra226']

cs137_energy = [662]        #kev #channel num 399
c060_energy  = [1173, 1333] #kev #            691, 782
na22_energy  = [511, 1275]  #kev #            310, 754

peaks = []
spectrum = []
channel = []
smoothed_spectrum = []


for i in range(4):
    #read data
    data = pd.read_csv(csvfile[i], delimiter=';', skiprows=2, decimal =',')
    spectrum.append( data.iloc[:, 1].to_numpy() )
    channel.append( data.iloc[:, 2].to_numpy() )
    
    #smooth data and find peaks
    smoothed_spectrum.append( savgol_filter(spectrum[i],30,3))
    temp, temp2 = find_peaks(smoothed_spectrum[i], height=20, prominence=50)
    peaks.append(temp)


#assign energys to channel
x_channel = np.arange(0, len(spectrum[0]))
channel_energy_linreg = linregress([310, 399, 691, 754, 782] , [511,662,1173,1275,1333])
x_energy =  channel_energy_linreg[0] * x_channel + channel_energy_linreg[1]

#-----------------------------------------------------------------
#find energy of peaks
#find energy fit quality
#fit gauss curves over peaks

#Na22 1 ; CS137 ; Co60 1 ; Na22 2 ; Co60 2
peak_channel = [310, 399, 691, 754, 782] #channel of peak found by find_peaks
peak_energy = [x_energy[peak_channel]]  #energy of peaks above 
peak_pos = [[250,360] , [330,460] , [640,735] , [680,810] , [736,840]]   #channels over which gaus fit is done
gaus_popt = []
gaus_pcov = []
fwhm = []

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


#plot
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
    plt.title('Spectrum Lab')
    plt.legend()
    plt.grid(True)
    plt.show()