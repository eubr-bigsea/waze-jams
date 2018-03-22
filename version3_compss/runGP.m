function result = runGP (adj, yg, M, cellnum, Ntrain, Ntest, hypers)

  addpath(genpath('/opt/gpml'))

	%  Defining prior structure
  covft = {@covSum, {@covSEiso, @covPeriodic}};  % temporal components
  covfa = {@covSum, {@covConst, @covLINiso}};  % adjacency component
  covf = {@covSum, {{@covMask, {1, covft}}, {@covMask, {2, covfa}}}};  % sum over all components

  meanf = @meanZero;  % specification of mean function
  likf = @likLogistic;  % specification of likelihood function
  infm = @infLaplace;  % Laplace approximation for likelihood function

  % Load parameters
  Ntrain = double(Ntrain);
  Ntest = double(Ntest);
  cellnum = double(cellnum)
  opt.cov	= double(hypers);

  hyp.mean = [];
  hyp.lik = [];
  hyp.cov = opt.cov;  % previous hypers

  disp(['Filtering relevant data...']);
  adj_tab = reshape(adj, M, Ntrain + Ntest - 1);
  adj = adj_tab(cellnum, :)';
  y_tab = reshape(yg, M,  Ntrain + Ntest - 1);
  yg = y_tab(cellnum, :)';

  disp(['Hyperparameter optimization...']);
  xtrain = [ (1:Ntrain)' adj(1:Ntrain) ];
  ytrain = yg(1:Ntrain);

  disp(['Estimating future probability of traffic jams...']);
  yhat = zeros(Ntest, 1);
  yvar = zeros(Ntest, 1);
  for (tt = 1 : Ntest)
    disp(['Predicting instance ' num2str(tt) ' out of ' num2str(Ntest) '...']);
    Xtest = [ Ntrain+tt mean(adj(mod(tt-1,24)+1:24:Ntrain+tt-1)) ];  % forming test instance
    [yhat(tt) yvar(tt)] = gp(hyp, infm, meanf, covf, likf, xtrain, ytrain, Xtest);  % forecasting test instance
    if (tt < Ntest)
      xtrain = [ xtrain; [ Ntrain+tt adj(Ntrain+tt) ] ];  % updating training set
      ytrain = yg(1:Ntrain+tt);  % updating training set
    end
  end

  disp(['Storing results']);
  Forecasts = [ yhat yvar ];
  % Forecasts = (Forecasts + 1)/2; % turning them into the interval [0,1]
  result.Forecasts = Forecasts;
  result.hyp = hyp;

endfunction
