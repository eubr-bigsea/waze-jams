
function  config = prepare (ytab) %[data, covf, meanfunc, likf, infm, hyp, Ntrain, Ntest, M]


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

		%printf ("File:%s",datafile)

		% Reading training data and constructing 3-D spatio-temporal grid
		%ytab = load(datafile);
		[N M] = size(ytab);
		Ntrain = -1;
		if (Ntrain == -1), Ntrain = N; end
		if (Ntrain > N), error('Dataset has too few instances!'); end
		yg = reshape(ytab', sqrt(M), sqrt(M), N);

		size(yg); % Lucas

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

		% Saving pre-computations
		config.data = [ adj(:) yg(:) ];
		% config.covf = covfunc;
		% config.meanf = meanfunc;
		% config.likf = likfunc;
		%config.infm = infmet;
		config.hyp = hyp;
		config.Ntrain = Ntrain;
		config.Ntest = N - Ntrain + 1;
		config.M = M;



		%save(name_output, 'config');

		%code = 42
endfunction
