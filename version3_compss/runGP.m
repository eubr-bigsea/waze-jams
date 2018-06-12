function result = runGP (adj, yg, M, Ntrain, Ntest, hypers, cellnums)

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
  hypers = double(hypers);

  adj_tab = reshape(adj, M, Ntrain + Ntest - 1);
  y_tab = reshape(yg, M,  Ntrain + Ntest - 1);

  result.Forecasts = [];
  result.hyp = [];

  hyp.mean = [];
  hyp.lik = [];

  for (c = 1 :columns(cellnums))

    cellnum = cellnums(1,c);
    disp(['Predicting grid #' num2str(cellnum) ]);

    hyp.cov = hypers(c, 2:8);  % previous hypers

    % disp(['Filtering relevant data...']);
    adj = adj_tab(cellnum, :)';
    yg = y_tab(cellnum, :)';

    % disp(['Hyperparameter optimization...']);
    xtrain = [ (1:Ntrain)' adj(1:Ntrain) ];
    ytrain = yg(1:Ntrain);

    % disp(['Estimating future probability of traffic jams...']);
    yhat = zeros(Ntest, 1);
    yvar = zeros(Ntest, 1);
    for (tt = 1 : Ntest)
      Xtest = [ Ntrain+tt mean(adj(mod(tt-1,24)+1:24:Ntrain+tt-1)) ];  % forming test instance
      [yhat(tt) yvar(tt)] = gp(hyp, infm, meanf, covf, likf, xtrain, ytrain, Xtest);  % forecasting test instance
      if (tt < Ntest)
        xtrain = [ xtrain; [ Ntrain+tt adj(Ntrain+tt) ] ];  % updating training set
        ytrain = yg(1:Ntrain+tt);  % updating training set
      end
    end

  % disp(['Storing values'])
  % Forecasts = (Forecasts + 1)/2; % turning them into the interval [0,1]
  result.Forecasts = [result.Forecasts; yhat yvar];

  end

endfunction
