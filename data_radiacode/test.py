# import numpy as np
# import matplotlib.pyplot as plt
# import PhyPraKit as ppk

# def generate_gaussian_data(mean, std_dev, num_points, seed=None):
#     """
#     Generate Gaussian-shaped random data.
#     """
#     if seed is not None:
#         np.random.seed(seed)
    
#     data = np.random.normal(loc=mean, scale=std_dev, size=num_points)
#     return data

# def gauss(x, A=1.0, mu=1.0, sigma=1.0):
#     return A * (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# # Example usage
# mean = 1  # Mean of the Gaussian distribution
# std_dev = 1  # Standard deviation of the Gaussian distribution
# num_points = 10000  # Number of data points to generate
# seed = 42  # Predefined seed for reproducibility

# data = generate_gaussian_data(mean, std_dev, num_points, seed)

# # Plotting the data
# counts, bin_edges = np.histogram(data, bins=30, density=True)
# bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

# # Fitting the data
# fit = ppk.mnFit(fit_type="hist")
# fit.set_hOptions(run_minos=False, use_GaussApprox=True, fit_density=True)
# fit.init_hData(counts, bin_centers, DeltaMu=1)
# fit.init_hFit(gauss, p0=[1, mean, std_dev])
# fit.do_fit()
# fit.plotModel()

# # Plotting the Gaussian function for comparison
# x = np.linspace(min(bin_centers), max(bin_centers), 100)
# p = np.exp(-0.5 * ((x - mean) / std_dev) ** 2) / (std_dev * np.sqrt(2 * np.pi))
# plt.plot(x, p, 'k', linewidth=2)
# title = "Histogram of Gaussian data"
# plt.title(title)
# plt.show()


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------



# #! /usr/bin/env python3
# """test_histFit.py
#    histogram fit  with iminiut

# .. moduleauthor:: Guenter Quast <g.quast@kit.edu>

# """

# import numpy as np, matplotlib.pyplot as plt

# ##from PhyPraKit.phyFit import hFit
# from PhyPraKit import hFit

# if __name__ == "__main__":  # --------------------------------------
#     #
#     # Example of a histogram fit
#     #

#     # define the model function to fit
#     def model(x, mu=6.0, sigma=0.5, s=0.3):
#         """pdf of a Gaussian signal on top of flat background"""
#         normal = np.exp(-((x - mu) ** 2) / (2 * sigma** 2)) / np.sqrt(
#             2.0 * np.pi * sigma**2
#         )
#         flat = 1.0 / (max - min)
#         return s * normal + (1 - s) * flat

#     #
#     # generate Histogram Data
#     #
#     # parameters of data sample, signal and background parameters
#     N = 100  # number of entries
#     min = 0.0  # range of data, mimimum
#     max = 10.0  # maximum
#     s = 0.25  # signal fraction
#     pos = 6.66  # signal position
#     width = 0.33  # signal width

#     # fix random generator seed
#     np.random.seed(314159)  # initialize random generator

#     def generate_data(N, min, max, p, w, s):
#         """generate a random dataset:
#         Gaussian signal at position p with width w and signal fraction s
#         on top of a flat background between min and max
#         """
#         # signal sample
#         data_s = np.random.normal(loc=pos, scale=width, size=int(s * N))
#         # background sample
#         data_b = np.random.uniform(low=min, high=max, size=int((1 - s) * N))
#         return np.concatenate((data_s, data_b))

#     # generate a data sample ...
#     SplusB_data = generate_data(N, min, max, pos, width, s)
#     # ... and create the histogram
#     bc, be = np.histogram(SplusB_data, bins=40)

#     #
#     # ---  perform fit
#     #
#     rdict = hFit(
#         model,
#         bc,
#         be,  # bin entries and bin edges
#         p0=None,  # initial guess for parameter values
#         #  constraints=['name', val ,err ],   # constraints within errors
#         limits=("s", 0.0, None),  # limits
#         use_GaussApprox=False,  # Gaussian approximation
#         fit_density=True,  # fit density
        
#         plot=True,  # plot data and model
#         plot_band=True,  # plot model confidence-band
#         plot_cor=False,  # plot profiles likelihood and contours
#         quiet=False,  # suppress informative printout
#         axis_labels=["x", "entries / bin   |  f(x, *par)"],
#         data_legend="pseudo-data",
#         model_legend="model",
#     )

#     # Print results to illustrate how to use output
#     print("\n*==* Results of Histgoram Fit:")
#     #
#     pvals, perrs, cor, gof, pnams = rdict.values()
#     print(" goodness-of-fit: {:.3g}".format(gof))
#     print(" parameter names:       ", pnams)
#     print(" parameter values:      ", pvals)
#     print(" neg. parameter errors: ", perrs[:, 0])
#     print(" pos. parameter errors: ", perrs[:, 1])
#     print(" correlations : \n", cor)

# # - alternatively print results dictionaray directly
# #  for key in rdict:
# #    print("{}\n".format(key), rdict[key])


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------


import yaml
import numpy as np
import matplotlib.pyplot as plt
import PhyPraKit as ppk

from scipy.signal import find_peaks, savgol_filter
#from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from scipy.optimize import curve_fit

from peak_class import *

def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            return yaml_data

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
cs137_data = read_yaml_file('m1Cs137_240226-1235.yaml')
co60_data = read_yaml_file('m2C060_240226-1246.yaml')
na22_data = read_yaml_file('m3Na22_240226-1256.yaml')
ra226_data = read_yaml_file('m4Ra226_240226-1303.yaml')

spectrum = np.array([cs137_data['spectrum'], co60_data['spectrum'], na22_data['spectrum'], ra226_data['spectrum']])

cs = Peak("cs137","main",662, spectrum[0],265,220,400)

cs.do_fit()
#cs.plot_peak()


# def gauss(x, s=10, mu=8.0, sigma=0.5):
#     """
#     pdf of a Gaussian signal 
#     on top of flat background to fit peak with
    
#     """
    
#     bmin = 0
#     bmax = 1
#     normal = np.exp(-((x - mu) ** 2) / (2 * sigma** 2)) / np.sqrt(
#         2.0 * np.pi * sigma**2)
#     flat = 1.0 / (bmax - bmin)
#     return s * normal + (1 - s) * flat
    
    
# plt.figure(figsize=(10, 6))
# plt.plot(np.arange(0,20,0.01 ), gauss((np.arange(0,20,0.01 )),10,8),
#           marker='o', linestyle='', color='b', markersize=1)
# plt.plot(np.arange(0,20,0.01 ), gauss((np.arange(0,20,0.01 )),5,8),
#           marker='o', linestyle='', color='r', markersize=1)
# plt.plot(np.arange(0,20,0.01 ), gauss((np.arange(0,20,0.01 )),15,8),
#           marker='o', linestyle='', color='g', markersize=1)

 
# # Add labels and title
# plt.xlabel('Index')
# plt.ylabel('Intensity (arbitrary units)')
# plt.title('Spectrum Plot')

# plt.grid(True)
# plt.show()   