The scripts in this repository were designed to train and test a spatio-temporal Gaussian process-based model for forecasting traffic congestions using Waze data.

REQUIREMENTS
------------

In order to use this code, there are two requirements:

	1- Octave (version 4.0.0)
	
	2- GPML (version 3.5)

Using different versions may lead to inconsistencies!

GPML must also be in the Octave path, so that its functions can be accessed anywhere.
For that:

	1- Enter Octave
	
	2- Generate the path to GPML using "s=genpath(<full-path-to-GPML>)"
	
	3- Add the path to Octave path using "addpath(s)"
	
	4- Save path to ~/.octaverc so that we don't have to do it again using "savepath()"

USAGE
-----

There are two scripts associated with this project, one for generating a configuration file and the other for fitting the model:

	1- prepare.m: generates configuration file config.txt
	
	2- runGP.m: hyperparameter optimization and inference

The proposed model uses spatio-temporal data. On other words, it assumes we have a 3D grid of binary observations (existence or not of traffic jams), where 2 dimensions are associated with a MxM spatial grid and the remaining dimension is associated with time.

To generate the configuration file, one must provide a dataset as input. To avoid passing 3D grids, which may difficult to format correctly, this training set must consist of a Nx(M^2) matrix, where N is the number of observations per cell (remember: each observation corresponds to a moment in time). Therefore, each collum corresponds to a spatial grid cell and the values on each collum are observations associated with the corresponding cell. Collums are also expected to be ordered such that row-wise. On the other hand, the first M collumns are associated with the first row of the MxM spatial grid, the next M collumns are associated with the second row, an so on.

The configuration script (prepare.m) outputs one file: config.txt. It contains both the data to be used for hyperparameter learning and inference as information regarding the GP prior distribution (prior mean, covariance and likelihood function and inference method).

The execution script (runGP.m) takes as input the configuration file and the number of the cell grid for which hyperparameter learning and inference is required. It outputs two files: forecasts_N.txt and hypers_N.txt, where N is the number of the cell grid given as input. The first file contains a Tx2 matrix with predictive mean and variance, where T is the number of time intervals required for testing. Predictions are in the interval [-1,+1], where predictions closer to -1 indicate greater probability of being associated with label -1 and predictions closer to +1 indicate the opposite scenario. These predictions can be turned into probabilities by turning them into the interval [0,1] (by summing 1 and then diving them by 2). The second output file consists of a vector with learned hyperparameters.

DEMO
----

A sample file has been added to guide formatting new datafiles and testing of the code. The file sample.txt contains data of a 50x50 spatial grid with hourly observations for 28 days (on a total of 672 observations per grid cell).

For testing, run:

	1- octave prepare.m sample.txt
	2- octave runGP.m config.txt N

where N is the number of a cell grid (1 <= N <= 2500).
