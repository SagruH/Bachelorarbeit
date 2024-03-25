#! /usr/bin/env python3

import xml.etree.ElementTree as ET
import sys, numpy as np, matplotlib.pyplot as plt

# some constants
rho_CsJ = 4.51  # density of CsJ in g/cm^3
m_sensor = rho_CsJ * 1e-3  # Volume is 1 cm^3, mass in kg
keV2J = 1.602e-16

# Helper funcitons for conversion of channel numbers to energies
global a0, a1, a2  # calibration constants
# approx. calibration, overwritten by first  retrieved spectrum
a0 = 0.17
a1 = 2.42
a2 = 0.0004


def Chan2En(C):
    # convert Channel number to Energy
    #  E = a0 + a1*C + a2 C^2
    return a0 + a1 * C + a2 * C**2


def En2Chan(E):
    # convert Energies to Channel Numbers
    # inverse E = a0 + a1*C + a2 C^2
    c = a0 - E
    return (np.sqrt(a1**2 - 4 * a2 * c) - a1) / (2 * a2)


# end helpers ---------------------------------------

fname = sys.argv[1]
if len(sys.argv) >= 3:
    debug = sys.argv[2]
else:
    debug = False

# open xml file ...
tree = ET.parse(fname)
root = tree.getroot()
# ... and work through tree structure
ResultsDataList = None
ResultData = None
EnergySpectrum = None
EnergyCalibration = None
MeasurementTime = None
SpectrumName = None
Spectrum = None

for c in root:
    if debug:
        print('found tags', c.tag, c.text)
    if c.tag == 'ResultDataList':
        ResultsDataList = c
if ResultsDataList is None:
    sys.exit("could not find tag 'ResultsDataList'")

for c in ResultsDataList:
    if debug:
        print('found tags', c.tag, c.text)
    if c.tag == 'ResultData':
        ResultData = c
if ResultData is None:
    sys.exit("could not find tag 'ResultsData'")

for c in ResultData:
    if debug:
        print('found tags', c.tag, c.text)
    if c.tag == 'EnergySpectrum':
        EnergySpectrum = c
if EnergySpectrum is None:
    sys.exit("could not find tag 'EnergySpectrum'")

for c in EnergySpectrum:
    if debug:
        print('found tags', c.tag, c.text)
    if c.tag == 'EnergyCalibration':
        EnergyCalibration = c
    if c.tag == 'MeasurementTime':
        MeasurementTime = c
    if c.tag == 'Spectrum':
        Spectrum = c
    if c.tag == 'SpectrumName':
        SpectrumName = c
if Spectrum is None:
    sys.exit("could not find tag 'Spectrum'")

calibCoeffs = None
for c in EnergyCalibration:
    if c.tag == 'Coefficients':
        calibCoeffs = []
        for v in c:
            calibCoeffs.append(np.float32(v.text))

spectrum = []
for v in Spectrum:
    spectrum.append(np.int32(v.text))

print("*==* Spectrum found:     ", SpectrumName.text)
if MeasurementTime is not None:
    T = np.int32(MeasurementTime.text)
    print("      Measruement Time: ", T)
if calibCoeffs is not None:
    print("      calibration coefficients: ", calibCoeffs)

Energies = None
Channels = np.asarray(range(len(spectrum))) + 0.5
if calibCoeffs is not None:
    a0 = calibCoeffs[0]
    a1 = calibCoeffs[1]
    a2 = calibCoeffs[2]
    Energies = a0 + a1 * Channels + a2 * Channels**2

# some statistical analysis
print(f'time: {T:.4g} s')
rate = np.sum(np.asarray(spectrum)) / T
print(f'rate: {rate:.4g} Hz')
deposited_energy = np.sum(np.asarray(spectrum) * Energies)
# dose in nGy/h = nJ/(kg*h)
dose = deposited_energy * keV2J * 3600 / T / m_sensor * 1e6
print(f'dose: {dose:.4g} µGy/h')

# define a Figure
fig = plt.figure("Gamma Spectrum", figsize=(10.0, 7.5))
fig.suptitle('RadiaCode 102: Spectrum ' + SpectrumName.text, size='x-large', color='b')
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.97, top=0.89, wspace=None, hspace=0.25)  #
# define subplots
axE = fig.add_subplot(1, 1, 1)
if Energies is None:
    axE.plot(Channels, spectrum, label=SpectrumName.text)
    axE.set_xlabel('Channel Number', size='large')
else:
    axE.plot(Energies, spectrum, label=SpectrumName.text)
    axE.set_xlabel('Energy (keV)', size='large')
axE.set_ylabel('Frequency', size='large')
axE.set_yscale('log')
axE.grid()
axE.legend(loc='best', numpoints=1, prop={'size': 10})
txt = axE.text(
    0.8,
    0.85,
    f'time: {T:.4g} s\n' + f'rate: {rate:.3g} Hz\n' + f'dose: {dose:.3g} µGy/h',
    transform=axE.transAxes,
    color='darkblue',
    # backgroundcolor='white',
    alpha=0.7,
)
# a second x-axis for channels
axC = axE.secondary_xaxis('top', functions=(En2Chan, Chan2En))
axC.set_xlabel('Channel #')
plt.show()
