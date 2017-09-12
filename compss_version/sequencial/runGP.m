% Input: path to file with configuration, cell number
configfile = argv(){1};
cellnum = str2num(argv(){2});

% Loading configuration file and preparing data
load(configfile);
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
save('-ascii', ['forecasts_' num2str(cellnum) '.txt'], 'Forecasts');
save('-ascii', ['hypers_' num2str(cellnum) '.txt'], 'hyp');
