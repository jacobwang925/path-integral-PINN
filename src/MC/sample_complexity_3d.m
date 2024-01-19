%% sample complexity without comparison theorem

clear; clc;

dim = 3;
A = [1 0 1; 0 1 -1; 0 0 1];
B = eye(dim);

Q = 0.1*[1 1 0; 1 1 0; 0 0 1];
R = eye(dim);

T = 1; % time horizon

%%%%%%%%%%%
% change this for different MC accuracy
N = 100; % number of trajectories
%%%%%%%%%%%

dx = 0.1;
dt = 0.1;

sigma = 1;

% discrete dynamics matrices
% x_k+1 = A x_k + B u_k
A = eye(dim) + dt * A;
B = dt * B;

Nt = 10; % number of time steps
Nx = 10;


phi = zeros(Nx, Nx, Nx, N);

tic

x_0 = zeros(1,3);
for k = 1:Nx
    x_0(1) = dx*k+1;
    for j = 1:Nx
        x_0(2) = dx*j-0.1;
        for l =  1:Nx
            x_0(3) = dx*l+1;
        
            for i = 1:N
                cost = 0;
                x = zeros(Nt,dim);
                x(1,:) = x_0;
                for t = 1:Nt-1   
                    cost = cost + 0.5*x(t,:)*Q*x(t,:)' * dt;
                    x(t+1,:) = (A*x(t,:)' + sigma*sqrt(dt)*randn(dim,1))';
                end
                cost = cost + 0.5*x(Nt,:)*Q*x(Nt,:)';
                phi(k,j,l,i) = exp(-cost);
            end

        end
    end
end

toc

phi_ave = mean(phi,4);

V = -log(phi_ave);

figure
surf(squeeze(phi_ave(:,1,:)))

load('observe_x_3d_10000.mat')
figure
scatter3(observe_x(901:1000,1), observe_x(901:1000,2), observe_func(901:1000)) % data for comparison

gt = reshape(observe_func(901:1000),[Nx, Nx]);

diff = gt - squeeze(phi_ave(:,1,:));
abs_err = abs(diff);
perc_error = abs_err./gt;
perc_error = mean(mean(abs(perc_error)))
err = mean(mean(abs(diff)))

% record the results

% % error list for N = 1, 10 ,100, 1000, 10000
% sample_complexity = Nx^2 * [1, 10, 100, 1000, 10000];
% error_list = [0.1328, 0.0408, 0.0126, 0.0045, 0.0021];
% perc_error_list = [0.7675, 0.2399, 0.0680, 0.0267, 0.0132];
% computation_time_list = [0.030870, 0.263604, 2.185124, 22.293610, 224.714466];



%% sample complexity with comparison theorem

clear; clc;

dim = 3;
A = [1 0 1; 0 1 -1; 0 0 1];
B = eye(dim);

Q = 0.1*[1 1 0; 1 1 0; 0 0 1];
R = eye(dim);

T = 1; % time 

%%%%%%%%%%%
% change this for different MC accuracy
N = 100; % number of trajectories
%%%%%%%%%%%

dx = 0.1;
dt = 0.1;

sigma = 1;

% discrete dynamics matrices
% x_k+1 = A x_k + B u_k
A = eye(dim) + dt * A;
B = dt * B;

Nt = 10; % number of time steps
Nx = 10;


phi = zeros(Nx, Nx, N);

tic

x_0 = zeros(1,3);
for k = 1:Nx
    x_0(1) = 0.5*(dx*k+1);
    x_0(2) = 0.5*(dx*k+1);
    for l =  1:Nx
        x_0(3) = dx*l+1;
    
        for i = 1:N
            cost = 0;
            x = zeros(Nt,dim);
            x(1,:) = x_0;
            for t = 1:Nt-1   
                cost = cost + 0.5*x(t,:)*Q*x(t,:)' * dt;
                x(t+1,:) = (A*x(t,:)' + sigma*sqrt(dt)*randn(dim,1))';
            end
            cost = cost + 0.5*x(Nt,:)*Q*x(Nt,:)';
            phi(k,l,i) = exp(-cost);

        end
    end
end

toc

phi_ave = mean(phi,3);

V = -log(phi_ave);


figure
surf(phi_ave(:,:))

load('observe_x_3d_10000.mat')
figure
scatter3(observe_x(901:1000,1), observe_x(901:1000,2), observe_func(901:1000))

gt = reshape(observe_func(901:1000),[Nx, Nx]);

diff = gt - phi_ave;
abs_err = abs(diff);
perc_error = abs_err./gt;
perc_error = mean(mean(abs(perc_error)))
err = mean(mean(abs(diff)))

% % error list for N = 1, 10 ,100, 1000, 10000
% sample_complexity = Nx*[1, 10, 100, 1000, 10000];
% error_list = [0.1020, 0.0442, 0.0146, 0.0045, 0.0023];
% perc_error_list = [0.7392, 0.2678, 0.0797, 0.0267, 0.0117];
% computation_time_list = [0.007848, 0.031048, 0.261059, 2.317969, 21.671566];



