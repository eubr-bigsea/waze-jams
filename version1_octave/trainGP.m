%%% SCRIPT FOR HYPERPARAMETER LEARNING AND ESTIMATING PREDICTIONS OVER THE TRAINING SET %%%

% Input: path to file with training data
datafile = argv(){1};

% Defining prior structure
covft = {@covSum, {@covSEiso, @covPeriodic}};				% temporal components
covfa = {@covSum, {@covConst, @covLINiso}};				% adjacency component
covfunc = {@covSum, {{@covMask, {1, covft}}, {@covMask, {2, covfa}}}};	% sum over all components
hyp.cov = log([1;1;1;24;1;1;1]);					% tnitial guess for hypers
meanfunc = @meanZero;							% specification of mean function
hyp.mean = [];
likfunc = @likLogistic;							% specification of likelihood function
hyp.lik = [];
infmet = @infLaplace;							% Laplace approximation for likelihood function

% Reading training data and constructing 3-D spatio-temporal grid
ytab = load(datafile);
[N M] = size(ytab);
yg = reshape(ytab', sqrt(M), sqrt(M), N);

% Forming adjacency-related covariate: proportion of jams on neighboring cells
adj = zeros(sqrt(M), sqrt(M), N);
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
	end
end

% Fixing labels to be +/- 1
yg = yg * 2 - 1;

Yhat = [];
Yvar = [];
Hyp = [];
tic;
% Running inference per grid cells
for (ii = 1 : sqrt(M))
	for (jj = 1 : sqrt(M))
		disp(['Predicting for grid cell (' num2str(ii) ',' num2str(jj) ')...'])
		if (sum(yg(ii,jj,:)) > -1*N)	% inference is perfomed only if there is at least one traffic jam at current cell
			Xcur = [ (1:N)' adj(ii,jj,:)(:) ];
			opt = minimize(hyp, @gp, -100, infmet, meanfunc, covfunc, likfunc, Xcur, yg(ii,jj,:)(:));
			[yhat yvar] = gp(opt, infmet, meanfunc, covfunc, likfunc, Xcur, yg(ii,jj,:)(:), Xcur);
			Yhat = [Yhat yhat];
			Yvar = [Yvar yvar];
			Hyp = [Hyp opt.cov];
		else				% otherwise, we simply assume that there will be no traffic jams
			Yhat = [Yhat repmat(-1, N, 1)];
			Yvar = [Yvar repmat(1, N, 1)];
			Hyp = [Hyp hyp.cov];
		end
	end
end
toc;

% Storing results
save('-ascii', 'yhat_fitting.txt', 'Yhat');
save('-ascii', 'yvar_fitting.txt', 'Yvar');
save('-ascii', 'hyp_fitting.txt', 'Hyp');
