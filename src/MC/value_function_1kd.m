%% value function through Riccati equations

clear; clc;

dim = 1000;

row_A = [1,0,1,0,1,zeros(1,dim/2-10),0,-1,0,-1,0];
quater_A = toeplitz([row_A(1) fliplr(row_A(2:end))], row_A);

A = [quater_A, zeros(size(quater_A));
    zeros(size(quater_A)), quater_A];

B = eye(dim);

Q = 0.001*[ones(size(quater_A)), zeros(size(quater_A));
    zeros(size(quater_A)), ones(size(quater_A))];

R = eye(dim);


N = 100; % number of trajectories

dx = 0.1;
dt = 0.1;

sigma = 1;

% discrete dynamics matrices
% x_k+1 = A x_k + B u_k
A = eye(dim) + dt * A;
B = dt * B;

Nx = 10;
NT = 15;  % number of time steps

P = zeros(dim, dim, NT);
P(:, :, NT) = Q;

tic

% solve algebraic Riccati equation
for k = NT-1:-1:1
    P(:,:,k) = Q + A'*P(:,:,k+1)*A - A'*P(:,:,k+1)*B/(R+B'*P(:,:,k+1)*B)*B'*P(:,:,k+1)*A;
end

K = zeros(dim, dim, NT); % control gain
for k = 1:NT-1
    K(:,:,k) = (R+B'*P(:,:,k+1)*B)\B'*P(:,:,k+1)*A;
end


V_lqg = zeros(Nx, Nx, NT);
obs_x = zeros(Nx, Nx, NT, 3);
obs_func = zeros(Nx, Nx, NT);


x_0 = zeros(1,dim);
for k = 1:Nx
    x_0(1:dim/2) = (dx*k+1)/(dim/2); 

    for j = 1:Nx
        x_0(dim/2+1:end) = (dx*j+1)/(dim/2); 

        for l =  1:NT

            phi = zeros(N,1);
        
            for i = 1:N
                cost = 0;
                x = zeros(NT,dim);
                x(l,:) = x_0;
                u = zeros(l,dim);
                for t = l:NT-1  
                    u(t,:) = -K(:,:,t)*x(t,:)';
                    cost = cost + 0.5*x(t,:)*Q*x(t,:)' * dt + 0.5 * (-K(:,:,t)*x(t,:)')'*R*(-K(:,:,t)*x(t,:)')* dt ;
                    x(t+1,:) = (A*x(t,:)' + B* (-K(:,:,t)*x(t,:)') + sigma*sqrt(dt)*randn(dim,1))';
                end
                cost = cost + 0.5*x(NT,:)*Q*x(NT,:)';
                phi(i,1) = cost;
            end
            
            V_lqg(k, j, l) = mean(phi);
            obs_x(k, j, l,:) = [dx*k+1, dx*j+1, dt*l];
            obs_func(k, j, l) = exp(-V_lqg(k, j, l));

        end
    end
end

toc

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
set(gca, 'fontsize', 18)

% save('observe_x_1kd.mat', 'observe_x', 'observe_func')

%% path integral MC with dimension reduction

clear; clc;

dim = 1000;

row_A = [1,0,1,0,1,zeros(1,dim/2-10),0,-1,0,-1,0];
quater_A = toeplitz([row_A(1) fliplr(row_A(2:end))], row_A);

A = [quater_A, zeros(size(quater_A));
    zeros(size(quater_A)), quater_A];

B = eye(dim);

Q = 0.001*[ones(size(quater_A)), zeros(size(quater_A));
    zeros(size(quater_A)), ones(size(quater_A))];

R = eye(dim);


N = 100; % number of trajectories

dx = 0.1;
dt = 0.1;

sigma = 1;

% discrete dynamics matrices
% x_k+1 = A x_k + B u_k
A = eye(dim) + dt * A;
B = dt * B;


Nx = 10;
NT = 15;  % number of time steps

phi = zeros(Nx, Nx, NT, N);
obs_x = zeros(Nx, Nx, NT, 3);
obs_func = zeros(Nx, Nx, NT);

tic

x_0 = zeros(1,dim);
for k = 1:Nx
    x_0(1:dim/2) = (dx*k+1)/(dim/2); 

    for j = 1:Nx
        x_0(dim/2+1:end) = (dx*j+1)/(dim/2); 

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

toc

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
set(gca, 'fontsize', 18)

% save('observe_x_1kd_MC.mat', 'observe_x', 'observe_func')