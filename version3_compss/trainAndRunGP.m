function result = trainAndRunGP (adj, yg, M, Ntrain, Ntest, cellnums)

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
  cellnums = double(cellnums);

  hyp.mean = [];
  hyp.lik = [];
  hyp.cov = log([1;1;1;24;1;1;1]);  % initial guess for hypers

  adj_tab = reshape(adj, M, Ntrain + Ntest - 1);
  y_tab = reshape(yg, M,  Ntrain + Ntest - 1);

  result.Forecasts = [];
  result.hyp = [];

  for (c = 1:columns(cellnums))
    cellnum =  cellnums(1,c);
    disp(['Predicting grid #' num2str(cellnum) ]);

    adj = adj_tab(cellnum, :)';
    yg = y_tab(cellnum, :)';

    % disp(['Hyperparameter optimization...']);
    xtrain = [ (1:Ntrain)' adj(1:Ntrain) ];
    ytrain = yg(1:Ntrain);
    % The minimize function minimizes the negative log marginal likelihood,
    % which is returned by the gp function, together with the partial
    % derivatives wrt the hyperparameters. The inference method is specified
    % to be infm exact inference. The minimize function is allowed a
    % computational budget of 100 function evaluations
    opt = minimize(hyp, @gp, -100, infm, meanf, covf, likf, xtrain, ytrain);
    Hyp = opt.cov;

    % disp(['Estimating future probability of traffic jams...']);
    %Estimating future probability of traffic jams
    yhat = zeros(Ntest, 1);
    yvar = zeros(Ntest, 1);
    for (tt = 1 : Ntest)
      Xtest = [ Ntrain+tt mean(adj(mod(tt-1,24)+1:24:Ntrain+tt-1)) ];  % forming test instance
      [yhat(tt) yvar(tt)] = gp(opt, infm, meanf, covf, likf, xtrain, ytrain, Xtest);  % forecasting test instance
      if (tt < Ntest)
        xtrain = [ xtrain; [ Ntrain+tt adj(Ntrain+tt) ] ];  % updating training set
        ytrain = yg(1:Ntrain+tt);  % updating training set
  		end
    end
    % Storing results
    % Forecasts = [yhat yvar];
    result.Forecasts = [result.Forecasts; yhat yvar];
    result.hyp = [result.hyp; Hyp'];

  end

endfunction
