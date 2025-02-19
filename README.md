# Missing-wave-data-imputation-

## Content
This repository contains the codes for imputation of missing values in the order of minutes for wave data obtained from different [CDIP buoys](https://cdip.ucsd.edu/). The buoys are located in different water depths at different locations. The method initially employs an ocean wave model to break down wave surface elevations into time series data of several slow varying amplitudes. 
The slow varying amplitudes are then utilized for missing data imputation either through neural network predictions or gap filling using singular spectrum analysis. The chosen neural networks include recurrent neural networks (LSTM) as well as a combination of a convolutional and a recurrent neural network enhanced through attention mechanism (CNN+LSTM with attention). 
The efficacy of the different gap imputation approaches are compared here alongwith the influence of the wave modelling process.![An overview of the gap filling process is represented here](https://github.com/SamarpanChakraborty97/Missing-wave-data-imputation-/blob/main/image_overview.jpg).
Alongwith this, the effect of the attention mechanism on a forecasting task has also been investigated through a number of noise experimental siumulations.

## License
This software is made public for research use only. It may be modified and redistributed under the terms of the MIT License.

## Citation
Please cite [1] and [2] if you use the codes here in your work.

## References
1. [Breunung, T., & Balachandran, B. (2024). Prediction of freak waves from buoy measurements. Scientific Reports, 14(1), 16048](https://doi.org/10.1038/s41598-024-66315-3)
2.  [Chakraborty, C., Ide, K., & Balachandran, B. (2025). Missing values imputation in ocean buoy time series data. Ocean Engineering, 318, 120145](https://www.sciencedirect.com/science/article/pii/S0029801824034838)

