import yaml
import numpy as np
import matplotlib.pyplot as plt
from PhyPraKit import hFit

from scipy.signal import find_peaks, savgol_filter
#from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
from scipy.optimize import curve_fit




#-----------------------------------------------------------------



#reads yaml file from given path and returns data
def read_yaml_file(file_path):
    try:
        with open(file_path, 'r') as file:
            yaml_data = yaml.safe_load(file)
            return yaml_data

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    
    
if __name__ == "__main__":  # --------------------------------------
    #
    # Example of a histogram fit
    #

    # define the model function to fit
    def model(x, mu=6.0, sigma=0.5, s=0.3):
        """pdf of a Gaussian signal on top of flat background"""
        normal = np.exp(-0.5 * ((x - mu) / sigma) ** 2) / np.sqrt(
            2.0 * np.pi * sigma**2
        )
        flat = 1.0 / (1 - 0)
        return s * normal + (1 - s) * flat

    #
    # generate Histogram Data
    #
    # parameters of data sample, signal and background parameters
    N = 100  # number of entries
    min = 0.0  # range of data, mimimum
    max = 10.0  # maximum
    s = 0.25  # signal fraction
    pos = 6.66  # signal position
    width = 0.33  # signal width

    # fix random generator seed
    np.random.seed(314159)  # initialize random generator

    def generate_data(N, min, max, p, w, s):
        """generate a random dataset:
        Gaussian signal at position p with width w and signal fraction s
        on top of a flat background between min and max
        """
        # signal sample
        data_s = np.random.normal(loc=pos, scale=width, size=int(s * N))
        # background sample
        data_b = np.random.uniform(low=min, high=max, size=int((1 - s) * N))
        return np.concatenate((data_s, data_b))

    # generate a data sample ...
    SplusB_data = generate_data(N, min, max, pos, width, s)
    
    
    
    cs137_data = read_yaml_file('m1Cs137_240226-1235.yaml')
    co60_data = read_yaml_file('m2C060_240226-1246.yaml')
    na22_data = read_yaml_file('m3Na22_240226-1256.yaml')
    ra226_data = read_yaml_file('m4Ra226_240226-1303.yaml')

    spectrum = np.array([cs137_data['spectrum'], co60_data['spectrum'], na22_data['spectrum'], ra226_data['spectrum']])
    
    spectrum = spectrum[0][230:340]
    # ... and create the histogram
    
    
    #
    # ---  perform fit
    #
    rdict = hFit(
        model,
        spectrum,
        np.arange(230,341),  # bin entries and bin edges
        p0=None,  # initial guess for parameter values
        #  constraints=['name', val ,err ],   # constraints within errors
        limits= (("s", 0.0, None),("mu",1.,1000.)),  # limits
        use_GaussApprox=False,  # Gaussian approximation
        fit_density=True,  # fit density
        plot=True,  # plot data and model
        plot_band=False,  # plot model confidence-band
        plot_cor=False,  # plot profiles likelihood and contours
        quiet=False,  # suppress informative printout
        axis_labels=["x", "entries / bin   |  f(x, *par)"],
        data_legend="pseudo-data",
        model_legend="model",
    )

    # Print results to illustrate how to use output
    print("\n*==* Results of Histgoram Fit:")
    #
    pvals, perrs, cor, gof, pnams = rdict.values()
    print(" goodness-of-fit: {:.3g}".format(gof))
    print(" parameter names:       ", pnams)
    print(" parameter values:      ", pvals)
    print(" neg. parameter errors: ", perrs[:, 0])
    print(" pos. parameter errors: ", perrs[:, 1])
    print(" correlations : \n", cor)

# - alternatively print results dictionaray directly
#  for key in rdict:
#    print("{}\n".format(key), rdict[key])