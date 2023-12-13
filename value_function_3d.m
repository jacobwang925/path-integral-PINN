%% Path integral MC without dimension reduction

clear; clc;

dim = 3;
A = [1 0 1; 0 1 -1; 0 0 1];
B = eye(dim);

Q = 0.1*[1 1 0; 1 1 0; 0 0 1];
R = eye(dim);


N = 100; % number of trajectories

dx = 0.1;
dt = 0.1;

sigma = 1;

% discrete dynamics matrices
% x_k+1 = A x_k + B u_k
A = eye(dim) + dt * A;
B = dt * B;


NT = 10; % number of time steps
Nx = 10;


phi = zeros(Nx, Nx, Nx, N);
obs_x = zeros(Nx, Nx, Nx, 3);
obs_func = zeros(Nx, Nx, Nx);

tic

x_0 = zeros(1,3);
for k = 1:Nx
    x_0(1) = dx*k*0.5+0.5;
    for j = 1:Nx
        x_0(2) = dx*j*0.5+0.5;
        for l =  1:Nx
            x_0(3) = dx*l+1;
        
            for i = 1:N
                cost = 0;
                x = zeros(NT,dim);
                x(1,:) = x_0;
                for t = 1:NT-1   
                    cost = cost + 0.5*x(t,:)*Q*x(t,:)' * dt;
                    x(t+1,:) = (A*x(t,:)' + sigma*sqrt(dt)*randn(dim,1))';
                end
                cost = cost + 0.5*x(NT,:)*Q*x(NT,:)';
                phi(k,j,l,i) = exp(-cost);
            end
            
            obs_x(k, j, l,:) = [dx*k*0.5+0.5, dx*j*0.5+0.5, dx*l+1];
            obs_func(k, j, l) = mean(phi(k, j, l,:));

        end
    end
end

toc

observe_x = reshape(obs_x, Nx*Nx*Nx, 3);
observe_func = reshape(obs_func, Nx*Nx*Nx, 1);

figure
scatter3(observe_x(:,1)+observe_x(:,2), observe_x(:,3), observe_func(:))


%% path integral MC with dimension reduction

clear; clc;

dim = 3;

A = [1 0 1; 0 1 -1; 0 0 1];
B = eye(dim);

Q = 0.1*[1 1 0; 1 1 0; 0 0 1];
R = eye(dim);


N = 100; % number of trajectories

dx = 0.1;
dt = 0.1;

sigma = 1;

% discrete dynamics matrices
% x_k+1 = A x_k + B u_k
A = eye(dim) + dt * A;
B = dt * B;


Nx = 10; % number of grids for state
NT = 15; % number of time steps

phi = zeros(Nx, Nx, NT, N);
obs_x = zeros(Nx, Nx, NT, 3);
obs_func = zeros(Nx, Nx, NT);


x_0 = zeros(1,3); % initial state

% path integral estimation of value function with uncontrolled process
for k = 1:Nx
    x_0(1) = (dx*k+1)/2;
    x_0(2) = (dx*k+1)/2;
    for j = 1:Nx
        x_0(3) = dx*j+1;
        for l =  1:NT

            phi = zeros(N,1);
        
            for i = 1:N
                cost = 0;
                x = zeros(NT,dim);
                x(l,:) = x_0;
                for t = l:NT-1  
                    cost = cost + 0.5*x(t,:)*Q*x(t,:)' * dt;
                    x(t+1,:) = (A*x(t,:)' + sigma*sqrt(dt)*randn(dim,1))';
                end
                cost = cost + 0.5*x(NT,:)*Q*x(NT,:)';
                phi(k, j, l, i) = exp(-cost);
            end

            obs_x(k, j, l,:) = [dx*k+1, dx*j+1, dt*l];
            obs_func(k, j, l) = mean(phi(k, j, l,:));

        end
    end
end

figure
surf(linspace(1.1,2,Nx), linspace(1.1,2,Nx), obs_func(:,:,1))
xlabel('$\xi_2$', 'Interpreter','latex')
ylabel('$\xi_1$', 'Interpreter','latex')
zlabel('$\varphi$', 'Interpreter','latex')
set(gca, 'fontsize', 18)

figure
surf(linspace(dt,dt*NT,NT), linspace(1.1,2,Nx), squeeze(obs_func(1,:,:)))
xlabel('$t$', 'Interpreter','latex')
ylabel('$\xi_2$', 'Interpreter','latex')
zlabel('$\varphi$', 'Interpreter','latex')
set(gca, 'fontsize', 18)

observe_x = reshape(obs_x, Nx*Nx*NT, 3);
observe_func = reshape(obs_func, Nx*Nx*NT, 1);

figure
scatter3(observe_x(:,1), observe_x(:,2), observe_func(:))
xlabel('$\xi_2$', 'Interpreter','latex')
ylabel('$\xi_1$', 'Interpreter','latex')
zlabel('$\varphi$', 'Interpreter','latex')
title('Value Function - Path Integral with Dimension Reduction')
set(gca, 'fontsize', 18)

% save('data_3d_MC_1000.mat', 'observe_x', 'observe_func')

%% Ground truth of the value function through Riccati equations

clear; clc;

dim = 3;
A = [1 0 1; 0 1 -1; 0 0 1];
B = eye(dim);

dx = 0.1;
dt = 0.1;

% Discrete time LQR, J = 0.5 * sum(x'Qx + u'Ru) + x'QTx
Q = 0.1*[1 1 0; 1 1 0; 0 0 1]*dt;
QT = 0.1*[1 1 0; 1 1 0; 0 0 1];
R = eye(dim);

sigma = 1;

N = 100; % number of trajectories

% discrete dynamics matrices
% x_k+1 = A x_k + B u_k
A = eye(dim) + dt * A;
B = dt * B;

Nx = 10; % number of grids for state
NT = 15; % number of time steps

% solve algebraic Riccati equation
P = zeros(dim, dim, NT);
P(:, :, NT) = QT;

for k = NT-1:-1:1
    P(:,:,k) = Q + A'*P(:,:,k+1)*A - A'*P(:,:,k+1)*B/(R+B'*P(:,:,k+1)*B)*B'*P(:,:,k+1)*A
end

K = zeros(dim, dim, NT); % optimal control gain
for k = 1:NT-1
    K(:,:,k) = (R+B'*P(:,:,k+1)*B)\B'*P(:,:,k+1)*A;
end

V = zeros(Nx, Nx, NT);
phi = zeros(Nx, Nx, NT);

x_0 = zeros(1,3);
for k = 1:Nx
    x_0(1) = (dx*k+1)/2;
    x_0(2) = (dx*k+1)/2;
    for j = 1:Nx
        x_0(3) = dx*j+1;
        for l =  1:NT
        
            V(k,j,l) = 0.5 * x_0 * P(:,:,l) * x_0'; % optimal value function
            phi(k,j,l) = exp(-V(k,j,l));
            obs_x(k, j, l,:) = [dx*k+1, dx*j+1, dt*l];
            obs_func(k, j, l) = phi(k,j,l);

        end
    end
end

observe_x = reshape(obs_x, Nx*Nx*NT, 3);
observe_func = reshape(obs_func, Nx*Nx*NT, 1);

figure
scatter3(observe_x(:,1), observe_x(:,2), observe_func(:))
xlabel('$\xi_2$', 'Interpreter','latex')
ylabel('$\xi_1$', 'Interpreter','latex')
zlabel('$\varphi$', 'Interpreter','latex')
title('Value Function - Ground Truth')
set(gca, 'fontsize', 18)

% save('data_3d_GT.mat', 'observe_x', 'observe_func')