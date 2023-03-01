# Templated Polymerization of Chemically Fueled Oligomers

(c) Ludwig Burger, 2023

This software was used in the manuscript "Template-based information transfer in chemically fueled dynamic combinatorial
libraries" by Christine M. E. Kriebisch, Ludwig Burger, Oleksii Zozulia, Michele Stasi, Alexander
Floroni, Dieter Braun, Ulrich Gerland, and Job Boekhoven.

## Description

Upon activation by the fuel EDC, monomers (isophthalic acid thymine conjugates) undergo polymerization to form longer oligomers (anhydrides). The activation consumes fuel and limits the amount of oligomers produced. After some time, the anhydrides break down via hydrolysis. Besides the polymerization, fuel and monomers are consumed via side reactions that form unwanted side product. This software computes the time-evolution (and associated observables of interest) by numerical integration of the chemical rate equations.

## Execution
The executables are contained in the following directories:

- ```/without_template__length-independent_rate_constants```: contains functions needed to analyze the time evolution of the concentrations for the system without template and length-independent reaction rate constants.
   * ```./compute_timeevolution.py```: computes the time-evolution of concentrations.
- ```/without_template__length-dependent_rate_constants```: contains functions needed to analyze the time evolution of the concentrations for the system without template and length-dependent activation rate constants and length-dependent rate constants of anhydride hydrolysis.
  * ```./compute_goodness_of_fit.py```: computes the goodness of fit $\chi$ for multiple sets of length-dependent reaction rate constants. The sets of reaction rate constants span a contour of parameter sets that fit the experimental data optimally. This contour was determined by performing a collection of multiple least square curve fits.
  * ```./compute_timeevolution.py```: computes the time-evolution of concentrations for a single set of length-dependent reaction rate constants.
- ```/with_template__length-dependent_rate_constants```: functions needed to analyze the time evolution of the concentrations for the system with template and length-dependent activation rate constants and length-dependent rate constants of anhydride hydrolysis. By specifying the variable ```name``` the user can choose any of the parameter sets in ```rate_constants_variable_klig.xlsx```.
  * ```./compute_timeevolution.py```: computes the time-evolution of concentrations for a single set of length-dependent reaction rate constants in solution and length-independent reaction rate constants on the template. By specifying the variable ```name``` the user can choose any of the parameter sets in ```parameters.xlsx```. Note that the integration of the time-evolution will take around 2 hours (dependent on your hardware specifications) and use about 6 GB of RAM.
  * ```./compute_fluxes.py```: computes the reaction rates for (almost) all chemical reactions as a function of time
  * ```./compute_EDC_consumption.py```: computes the total concentration of consumed EDC separated by the oligomers that consumed the EDC.
  * ```./compute_percentages.py```: computes the percentages of oligomers (separated by length and terminal group) in solution and on template.
  * ```./compute_mean_oligomer_length.py```: computes the average oligomer length in solution and on template.
 
Remarks for ```/with_template__length-dependent_rate_constants```:
- ```./compute_fluxes.py```, ```./compute_percentages.py``` and ```./compute_mean_oligomer_length.py``` can only be run after running ```./compute_timeevolution.py```
- ```./compute_EDC_consumption.py``` can only be run after running ```./compute_timeevolution.py``` and ```./compute_fluxes.py```

## Outputs
The executables produce plots in PDF-format showing the observable of interest. In the special case of ```/with_template__length-dependent_rate_constants/compute_timeevolution.py``` and ```/with_template__length-dependent_rate_constants/compute_fluxes.py``` the output is also written to a pickle-file in order to reuse the outputs for subsequent calculations. All outputs are written to the directories that contain the respective executables.

## Dependencies
 - [Matplotlib](https://matplotlib.org/)
 - [Numba](https://numba.pydata.org/)
 - [NumPy](https://numpy.org/)
 - [pandas](https://pandas.pydata.org/)
 - [SciPy](https://scipy.org/)

