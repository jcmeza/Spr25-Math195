function [x, it, time, crit] = FBPD(x, f, g, h, opt)

% default inputs
if nargin < 2 || isempty(f),   f.fun = @(x) 0; f.prox = @(x,gamma) x;                                                   end
if nargin < 3 || isempty(g),   g.fun = @(x) 0; g.grad = @(x)       0; g.beta   = 0;                                     end
if nargin < 4 || isempty(h),   h.fun = @(y) 0; h.prox = @(y,gamma) y; h.dir_op = @(x) x; h.adj_op = @(y) y; h.beta = 1; end
if nargin < 5 || isempty(opt), opt.tol = 1e-4; opt.iter = 500;                                                          end

% select the step-sizes
tau = 2 / (g.beta+2);
sigma = (1/tau - g.beta/2) / h.beta;

% initialize the dual solution
y = h.dir_op(x);

% execute the algorithm
time = zeros(1, opt.iter);
crit = zeros(1, opt.iter);
hdl = waitbar(0, 'Running FBPD...');
for it = 1:opt.iter
    
    tic;
    
    % primal forward-backward step
    x_old = x;
    x = x - tau * ( g.grad(x) + h.adj_op(y) );
    x = f.prox(x, tau);
    
    % dual forward-backward step
    y = y + sigma * h.dir_op(2*x - x_old);
    y = y - sigma * h.prox(y/sigma, 1/sigma);   

    % time and criterion
    time(it) = toc;
    crit(it) = f.fun(x) + g.fun(x) + h.fun(h.dir_op(x));
           
    % stopping rule
    if norm( x(:) - x_old(:) ) < opt.tol * norm( x_old(:) ) && it > 10
        break;
    end
    
    waitbar(it/opt.iter, hdl);
end

close(hdl);
crit = crit(1:it);
time = cumsum(time(1:it));