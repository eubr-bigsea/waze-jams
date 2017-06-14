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

There are two scripts associated with this project, one for training and the other for testing:

	1- trainGP.m: hyperparameter optimization and estimation of predictions for the training set
	
	2- testGP.m: prediction of future traffic jams

The proposed model uses spatio-temporal data. On other words, it assumes we have a 3D grid of binary observations (existence or not of traffic jams), where 2 dimensions are associated with a MxM spatial grid and the remaining dimension is associated with time.

To train the model (hyperparameter learning), one must provide a training set as input. To avoid passing 3D grids, which may difficult to format correctly, this training set must consist of a Nx(M^2) matrix, where N is the number of observations per cell (remember: each observation corresponds to a moment in time). Therefore, each collum corresponds to a spatial grid cell and the values on each collum are observations associated with the corresponding cell. Collums are also expected to be ordered such that row-wise. On the other hand, the first M collumns are associated with the first row of the MxM spatial grid, the next M collumns are associated with the second row, an so on.

The training script (runGPInference.m) outputs three files: yhat_fitting.txt, yvar_fitting.txt and hyp_fitting.txt. The first file consists of a Nx(M^2) matrix with estimates issued by the model for the training set. Predictions are in the interval [-1,+1], where predictions closer to -1 indicate greater probability of being associated with label -1 and predictions closer to +1 indicate the opposite scenario. These predictions can be turned into probabilities by turning them into the interval [0,1] (by summing 1 and then diving them by 2). The second output file also consists of a Nx(M^2) matrix of variances associated with predictions. Finally, the third file consists of a 7xM matrix with learned hyperparameters via likelihood maximization.

To test the model, one must provide the training set, the test set (both in the same format as before) and the hyperparameters obtained after likelihood maximization. It outputs two files: yhat_forecasting.txt and yvar_forecasting.txt. The first file consists of a Tx(M^2) matrix, where T is the number of testing instances in the test set, containing the forecasts issued by the model. The second file consists of a Tx(M^2) matrix with predictive variances.

DEMO
----

Two sample files have been added to guide formatting new train and test files and testing of the code. The file sample_train.txt contains data of a 50x50 spatial grid with hourly observations for 28 days (on a total of 672 observations per grid cell). The file sample_test.txt contains data of the same spatial grid for the following 7 days (on a total of 168 observations per grid cell).

Training and test using this files takes hours (possibly more than a day), so feel free to produce smaller datasets (perharps a week for training and one day for testing) to speed up both procedures.
