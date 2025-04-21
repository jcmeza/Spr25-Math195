%% original image
x_bar = double( imread('firemen.jpg') );

% blur operator
psf = fspecial('average', 3);

% noisy image
rng('default');
z = imfilter(x_bar, psf) + 20 * randn( size(x_bar) );

% visualization
figure; imshow(x_bar/255,[]); title('Original image');
figure; imshow(    z/255,[]); title('Noisy image');

%% constraint
f.prox = @(x,tau) project_box(x, 0, 255);

% data fidelity
A_dir = @(x) imfilter(x, psf);
A_adj = @(x) imfilter(x, rot90(psf,2));  % WARNING: 'psf' must be a (2n+1)-by-(2n+1) matrix
g.grad = @(x) A_adj(A_dir(x) - z);
g.beta = sum(abs(psf(:)));

% criteria
f.fun = @(x) indicator_box(x, 0, 255);
g.fun = @(x) sum(sum(sum((A_dir(x)-z).^2)));

%% forward finite differences (with Neumann boundary conditions)
hor_forw = @(x) [x(:,2:end,:)-x(:,1:end-1,:), zeros(size(x,1),1,size(x,3))]; % horizontal
ver_forw = @(x) [x(2:end,:,:)-x(1:end-1,:,:); zeros(1,size(x,2),size(x,3))]; % vertical

% backward finite differences (with Neumann boundary conditions)
hor_back = @(x) [-x(:,1,:), x(:,1:end-2,:)-x(:,2:end-1,:), x(:,end-1,:)];    % horizontal
ver_back = @(x) [-x(1,:,:); x(1:end-2,:,:)-x(2:end-1,:,:); x(end-1,:,:)];    % vertical

% direct and adjoint operators
h.dir_op = @(x) cat( 4, hor_forw(x), ver_forw(x) );
h.adj_op = @(y) hor_back( y(:,:,:,1) ) + ver_back( y(:,:,:,2) );

% operator norm
h.beta = 8;

%% regularization parameter
lambda = 5;

% proximity operator
h.prox = @(y,gamma) prox_L2(y, gamma*lambda, 4);

% criterion
h.fun = @(y) fun_L2(y, lambda, 4);

%% minimization
[x, it, time, crit] = FBPD(z, f, g, h);

% PSRN
psnr = 10 * log10( 255^2 / mean((x(:)-x_bar(:)).^2) );

% visualization
figure; imshow(x/255,[]); title(['Restored image - PSNR: ' num2str(round(psnr,2))])
figure; plot(time, crit); title('Convergence plot'); xlabel('seconds'); ylabel('criterion')