# Missing-wave-data-imputation-

## Content
This repository contains the codes for imputation of missing values in the order of minutes for wave data obtained from different [CDIP buoys](https://cdip.ucsd.edu/). The buoys are located in different water depths at different locations. The method initially employs an ocean wave model to break down wave surface elevations into time series data of several slow varying amplitudes. 
The slow varying amplitudes are then utilized for missing data imputation either through neural network predictions or gap filling using singular spectrum analysis. The chosen neural networks include recurrent neural networks as well as a combination of a convolutional and a recurrent neural network enhanced through attention mechanism. 
The efficacy of the different gap imputation approaches are compared here alongwith the influence of the wave modelling process.![An overview of the gap filling process is represented here](https://github.com/SamarpanChakraborty97/Missing-wave-data-imputation-/blob/main/image_overview.jpg). 
