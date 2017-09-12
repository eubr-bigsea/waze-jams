
function [code] = runGP (config,cellnum,out1,out2)

		printf ("HEREEE")

		% Input: path to file with configuration, cell number
		%configfile = argv(){1};
		%cellnum = str2num(argv(){2});
		%printf("File: %s",configfile)
		% Loading configuration file and preparing data
		covft 		= {@covSum, {@covSEiso, @covPeriodic}};				% temporal components
		covfa 		= {@covSum, {@covConst, @covLINiso}};				% adjacency component
		covfunc	 	= {@covSum, {{@covMask, {1, covft}}, {@covMask, {2, covfa}}}};	% sum over all components
		meanfunc 	= @meanZero;							% specification of mean function
		likfunc 	= @likLogistic;							% specification of likelihood function
		infmet 		= @infLaplace;							% Laplace approximation for likelihood function

		config.covf 	= covfunc;
		config.meanf 	= meanfunc;
		config.likf 	= likfunc;
		config.infm 	= infmet;

		%load(configfile);
		if (config.M < cellnum), error('Wrong cell number!'); end
		adj = config.data(:,1);
		adj_tab = reshape(adj, config.M, config.Ntrain + config.Ntest - 1);
		adj = adj_tab(cellnum, :)';						% filtering relevant data
		y = config.data(:,2);
		y_tab = reshape(y, config.M, config.Ntrain + config.Ntest - 1);
		y = y_tab(cellnum, :)';							% filtering relevant data

		% Hyperparameter optimization
		Xtrain = [ (1:config.Ntrain)' adj(1:config.Ntrain) ];
		ytrain = y(1:config.Ntrain);
		opt = minimize(config.hyp, @gp, -100, config.infm, config.meanf, config.covf, config.likf, Xtrain, ytrain);
		hyp = opt.cov;


		% Estimating future probability of traffic jams
		yhat = zeros(config.Ntest, 1);
		yvar = zeros(config.Ntest, 1);
		for (tt = 1 : config.Ntest)
			disp(['Predicting instance ' num2str(tt) ' out of ' num2str(config.Ntest) '...']);
			Xtest = [ config.Ntrain+tt mean(adj(mod(tt-1,24)+1:24:config.Ntrain+tt-1)) ];					% forming test instance
			[yhat(tt) yvar(tt)] = gp(opt, config.infm, config.meanf, config.covf, config.likf, Xtrain, ytrain, Xtest);	% forecasting test instance
			if (tt < config.Ntest)
				Xtrain = [ Xtrain; [ config.Ntrain+tt adj(config.Ntrain+tt) ] ];					% updating training set
				ytrain = y(1:config.Ntrain+tt);										% updating training set
			end
		end

		% Storing results
		Forecasts = [ yhat yvar ];
		save('-ascii', out1 , 'Forecasts'); %['forecasts_' num2str(cellnum) '.txt']
		save('-ascii', out2, 'hyp'); %['hypers_' num2str(cellnum) '.txt']

		code = 42
endfunction
