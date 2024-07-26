
import numpy as np
import matplotlib.pyplot as plt
import PhyPraKit as ppk


class Peak:
    def __init__(self, isotope, description, book_value, spectrum, position, p_left, p_right):
        """
        Initialize a Peak instance.

        Parameters:
        - isotope : name of the Isotope for the peak.
        - description : Additional Information for the peak.
        - spectrum : histogram measured by detector, length = number of channels
        - position : The channel of the peak.
        - book_value : Known Value for the energy of the peak
        - p_left : The channel number for the left peak boarder.
        - p_right : The channel number for the right peak boarder.

        
        """
        self.isotope = isotope
        self.desc = description
        self.book_value = book_value
        
        self.position = position      
        self.left = p_left
        self.right = p_right

        self.spectrum = spectrum
        self.channel = np.array(range(len(spectrum)))
        
        self.peak_channel = np.arange(p_left,p_right+1)
        self.peak_spectrum = spectrum[p_left:p_right]
        
        self.fit = None
        
    def gauss(self, x, s=10, mu=6.0, sigma=0.5):
        """
        pdf of a Gaussian signal 
        on top of sloped background to fit peak with
        
        """
        
        amin = 0
        amax = 1
        normal = np.exp(-((x - mu) ** 2) / (2 * sigma** 2)) / np.sqrt(
            2.0 * np.pi * sigma**2)
        
        
        slope = (amax - amin) / (amax - amin)  # Slope coefficient for background
        background = amin + slope * (x - amin)  # Linear slope
        
        return s * normal + (1 - s) * background
    
    def do_fit(self):
        """
        Does fit based on the hfit function from ppk
     
        """
        print(type(self.peak_channel))
        self.fit = ppk.hFit(
            self.gauss,
            self.peak_spectrum,
            self.peak_channel,  # bin entries and bin edges
            p0=[10, 200, 10],  # initial guess for parameter values
            #  constraints=['name', val ,err ],   # constraints within errors
            limits = (["s",0.0,None],["mu", 100.,400.]),  # limits
            use_GaussApprox=False,  # Gaussian approximation
            fit_density=True,  # fit density
            
            plot=True,  # plot data and model
            plot_band=False,  # plot model confidence-band
            plot_cor=False,  # plot profiles likelihood and contours
            quiet=False,  # suppress informative printout
            axis_labels=["123"],
            data_legend="123",
            model_legend="gauss",
        )

    def plot_input(self,log=False):
        # Create the plot
       plt.figure(figsize=(10, 6))
       plt.plot(self.channel, self.spectrum, label=self.isotope, marker='o', linestyle='', color='b', markersize=1)

       # Add labels and title
       plt.xlabel('Index')
       plt.ylabel('Intensity (arbitrary units)')
       plt.title('Spectrum Plot')
       
       
       if(log): plt.yscale('log')
       
       # Add a grid for better readability
       plt.grid(True)
       
       # Show legend
       plt.legend()

       # Display the plot
       plt.show()

    def plot_peak(self,log=False):
        # Create the plot
       plt.figure(figsize=(10, 6))
       plt.plot(self.channel, self.spectrum, label=self.isotope, 
                marker='o', linestyle='', color='b', markersize=1)
       plt.plot(self.peak_channel, self.peak_spectrum, label=self.isotope, 
                marker='o', linestyle='', color='r', markersize=1)
       
       # Add labels and title
       plt.xlabel('Index')
       plt.ylabel('Intensity (arbitrary units)')
       plt.title('Spectrum Plot')
       
       
       if(log): plt.yscale('log')
       
       # Add a grid for better readability
       plt.grid(True)
       
       # Show legend
       plt.legend()

       # Display the plot
       plt.show()



















