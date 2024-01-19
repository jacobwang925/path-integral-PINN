% probability of safety for the same 3d system without any control
% consider the state space x_i > 0
% safe set defined as max(x_1 + x_2, x_3) <= thres

%% without dimension reduction

clear; clc;

dim = 3;
A = [1 0 1; 0 1 -1; 0 0 1];
B = eye(dim);

Q = 0.5*[1 1 0; 1 1 0; 0 0 1];
R = eye(dim);

% boundary for safe set
thres = 4;

N = 100; % number of trajectories

dx = 0.1;
dt = 0.1;

sigma = 1;

% discrete dynamics matrices
% x_k+1 = A x_k + B u_k
A = eye(dim) + dt * A;
B = dt * B;

% time horizon T = 1
NT = 10; % number of time steps
Nx = 10;

safe_prob = zeros(Nx, Nx, Nx);
obs_x = zeros(Nx, Nx, Nx, 3);
obs_func = zeros(Nx, Nx, Nx);

x_0 = zeros(1,3);
for k = 1:Nx
    x_0(1) = (dx*k+1)/2;
    for j = 1:Nx
        x_0(2) = (dx*j+1)/2;
        for l =  1:Nx
            x_0(3) = dx*l+1;

            safety = ones(N,1);
        
            for i = 1:N
                cost = 0;
                x = zeros(NT,dim);
                x(1,:) = x_0;
                for t = 1:NT-1       
                    x(t+1,:) = (A*x(t,:)' + sigma*sqrt(dt)*randn(dim,1))';
                    if x(t+1,1) + x(t+1,2) > thres | x(t+1,3) > thres
                        safety(i,1) = 0;
                    end
                end
            end
            
            safe_prob(k, j, l) = mean(safety)
            obs_x(k, j, l,:) = [(dx*k+1)/2, (dx*j+1)/2, dx*l+1];
            obs_func(k, j, l) = safe_prob(k, j, l);

        end
    end
end

observe_x = reshape(obs_x, Nx*Nx*Nx, 3);
observe_func = reshape(obs_func, Nx*Nx*Nx, 1);

figure
scatter3(observe_x(:,1)+observe_x(:,2), observe_x(:,3), observe_func(:))
xlabel('$\xi_2$', 'Interpreter','latex')
ylabel('$\xi_1$', 'Interpreter','latex')
zlabel('$F$', 'Interpreter','latex')
title('Safety Probability - without Dimension Reduction')
set(gca, 'fontsize', 18)

figure
scatter3(observe_x(:,1), observe_x(:,2), observe_func(:))
xlabel('$x_2$', 'Interpreter','latex')
ylabel('$x_1$', 'Interpreter','latex')
zlabel('$F$', 'Interpreter','latex')
title('Safety Probability - without Dimension Reduction')
set(gca, 'fontsize', 18)
% we observe the symmetry between along x_1 + x_2 = constant


%% with dimension reduction

clear; clc;

dim = 3;
A = [1 0 1; 0 1 -1; 0 0 1];
B = eye(dim);


N = 100; % number of trajectories

dx = 0.1;
dt = 0.1;

sigma = 1;

% discrete dynamics matrices
% x_k+1 = A x_k + B u_k
A = eye(dim) + dt * A;
B = dt * B;

% boundary for safe set
thres = 4;

Nx = 10;
NT = 10; % number of time steps


safe_prob = zeros(Nx, Nx, NT);
obs_x = zeros(Nx, Nx, NT, 3);
obs_func = zeros(Nx, Nx, NT);


x_0 = zeros(1,3);
for k = 1:Nx
    x_0(1) = (dx*k+1)/2;
    x_0(2) = (dx*k+1)/2;
    for j = 1:Nx
        x_0(3) = dx*j+1;
        for l =  1:NT

            safety = ones(N,1);
        
            for i = 1:N
                x = zeros(NT,dim);
                x(1,:) = x_0;
                for t = 1:l 
                    x(t+1,:) = (A*x(t,:)' + sigma*sqrt(dt)*randn(dim,1))';
                    if x(t+1,1) + x(t+1,2) > thres | x(t+1,3) > thres
                        safety(i,1) = 0;
                    end
                end
            end
            
            safe_prob(k, j, l) = mean(safety)
            obs_x(k, j, l,:) = [dx*k+1, dx*j+1, dt*l];
            obs_func(k, j, l) = safe_prob(k, j, l);

        end
    end
end

figure
surf(linspace(1.1,2,Nx), linspace(1.1,2,Nx), obs_func(:,:,10))
xlabel('$\xi_2$', 'Interpreter','latex')
ylabel('$\xi_1$', 'Interpreter','latex')
zlabel('$F$', 'Interpreter','latex')
set(gca, 'fontsize', 18)

figure
surf(linspace(dt,dt*NT,NT), linspace(1.1,2,Nx), squeeze(obs_func(1,:,:)))
xlabel('$t$', 'Interpreter','latex')
ylabel('$\xi_2$', 'Interpreter','latex')
zlabel('$F$', 'Interpreter','latex')
set(gca, 'fontsize', 18)

observe_x = reshape(obs_x, Nx*Nx*NT, 3);
observe_func = reshape(obs_func, Nx*Nx*NT, 1);

figure
scatter3(observe_x(:,1), observe_x(:,2), observe_func(:))
xlabel('$\xi_2$', 'Interpreter','latex')
ylabel('$\xi_1$', 'Interpreter','latex')
zlabel('$F$', 'Interpreter','latex')
title('Safety Probability - with Dimension Reduction')
set(gca, 'fontsize', 18)

% save('observe_x_safety.mat', 'observe_x', 'observe_func')


