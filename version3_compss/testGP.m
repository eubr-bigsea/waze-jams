%%% SCRIPT FOR FORECASTING TRAFFIC JAMS %%%

% Input: paths to training data, test data and hyperparameters
trainfile = argv(){1};
testfile = argv(){2}
hypfile = argv(){3};

% Defining prior structure
covft = {@covSum, {@covSEiso, @covPeriodic}};														% temporal component
covfa = {@covSum, {@covConst, @covLINiso}};															% adjacency component
covfunc = {@covSum, {{@covMask, {1, covft}}, {@covMask, {2, covfa}}}};	% sum over all components
meanfunc = @meanZero;																										% specification of mean function
hyp.mean = [];
likfunc = @likLogistic;																									% specification of likelihood function
hyp.lik = [];
infmet = @infLaplace;																										% Laplace approximation for likelihood function

% Reading training, test and hyperparameter data and forming 3D spatio-temporal grids
ytab_train = load(trainfile);
[N M] = size(ytab_train);
ytab_test = load(testfile);
[T M2] = size(ytab_test)
Hyps = load(hypfile);
M3 = size(hypfile, 2)
if (M != M2 || M != M3), error('Wrong format of input data!'); end
yg_train = reshape(ytab_train', sqrt(M), sqrt(M), N);
yg_test = reshape(ytab_test', sqrt(M), sqrt(M), T);
yg = zeros(sqrt(M), sqrt(M), N+T);
yg(:,:,1:N) = yg_train;
yg(:,:,N+1:end) = yg_test;

% Forming adjacency-related covariate: proportion of jams on neighboring cells
adj = zeros(sqrt(M), sqrt(M), N+T);
meanAdj = zeros(sqrt(M), sqrt(M), 24);
for (ii = 1 : sqrt(M))
	for (jj = 1 : sqrt(M))
		c = 0;
		if (ii > 1 && jj > 1), adj(ii,jj,:) = adj(ii,jj,:) + yg(ii-1,jj-1,:); c = c+1; end
		if (ii > 1), adj(ii,jj,:) = adj(ii,jj,:) + yg(ii-1,jj,:); c = c+1; end
		if (ii > 1 && jj < sqrt(M)), adj(ii,jj,:) = adj(ii,jj,:) + yg(ii-1,jj+1,:); c = c+1; end
		if (jj > 1), adj(ii,jj,:) = adj(ii,jj,:) + yg(ii,jj-1,:); c = c+1; end
		if (jj < sqrt(M)), adj(ii,jj,:) = adj(ii,jj,:) + yg(ii,jj+1,:); c = c+1; end
		if (ii < sqrt(M) && jj > 1), adj(ii,jj,:) = adj(ii,jj,:) + yg(ii+1,jj-1,:); c = c+1; end
		if (ii < sqrt(M)), adj(ii,jj,:) = adj(ii,jj,:) + yg(ii+1,jj,:); c = c+1; end
		if (ii < sqrt(M) && jj < sqrt(M)), adj(ii,jj,:) = adj(ii,jj,:) + yg(ii+1,jj+1,:); c = c+1; end
		adj(ii,jj,:) = adj(ii,jj,:) / c;

		for (kk = 1 : 24)	% estimate for future proportion of neighboring cells with traffic jams
			meanAdj(ii,jj,kk) = mean(adj(ii,jj,kk:24:N)(:));
		end
		adj(ii,jj,N+1:end) = repmat(meanAdj(ii,jj,:), 1, 1, T/24);
	end
end

% Fixing labels to be +/- 1
yg = yg * 2 - 1;

% Running inference for all grid cells
Yhat = repmat(-1, T, M);
Yvar = repmat(-1, T, M);
tic;
for (ii = 1 : sqrt(M))
	for (jj = 1 : sqrt(M))
		disp(['Predicting for grid cell (' num2str(ii) ',' num2str(jj) ')...'])
		hyp.cov = Hyps(:, (ii-1)*sqrt(M)+jj);
		for (tt = 1 : T)
			Xtrain = [ (1:N+tt-1)' adj(ii,jj,1:N+tt-1)(:) ];
			Xtest = [ N+tt adj(ii,jj,N+tt) ];
			ytrain = yg(ii,jj,1:N+tt-1)(:);

			if (sum(ytrain == 1) > 0)	% forecasting is performed only for cells with at least one traffic congestion
				[yhat yvar] = gp(hyp, infmet, meanfunc, covfunc, likfunc, Xtrain, ytrain, Xtest);
				Yhat((ii-1)*sqrt(M)+jj,tt) = yhat;
				Yvar((ii-1)*sqrt(M)+jj,tt) = yvar;
			end

			if (mod(tt,24) == 0)		% updating estimate of adjacency-related covariate
				for (kk = 1 : 24), meanAdj(ii,jj,kk) = mean(adj(ii,jj,kk:24:N+tt)(:)); end
				adj(ii,jj,N+tt+1:end) = repmat(meanAdj(ii,jj,:), 1, 1, (T-tt)/24);
			end
		end
	end
end
toc;

% Storing results
save('-ascii', 'yhat_forecasting.txt', 'Yhat');
save('-ascii', 'yvar_forecasting.txt', 'Yvar');
