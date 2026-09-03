%% MIE1621_Project_Hamid_Nakhaei
%==========================================================================
% To Execute the code, just hit Run, and wait for 1 minute.
% THIS IS PROJECT A: Predictor–Corrector Primal–Dual Interior Point for QP
% By Hamid Nakhaei (Student ID: 1011678677)
%
% This script contains the implementation of predictor-corrector algorithm
% for QP and then solves two parts:
%
% Part 1: The three-asset financial optimization model in the project.
%   The model is
%       maximize    mu' * x - (delta/2)* x' * Sigma * x
%       subject to  sum(x) = 1, x >= 0.
%
%   I convert it to minimization:
%       minimize    (delta/2)* x' * Sigma * x - mu' * x.
%
%   Then I compare my predictor–corrector solver with MATLAB's quadprog.
%
% Part 2: Large-scale strictly convex QP
%   For several problem sizes, a random positive definite covariance matrix
%   is generated (using A_rand + A_rand' + nI) and solved with mu_i = 1 for
%   all i. The performance of my code is compared with quadprog.
%==========================================================================
%% predictorCorrectorQP
function [x, pi, z, iter_hist, primal_hist, dual_hist] = predictorCorrectorQP(Q, c, A, b, tol, max_iter, dampening)
% predictorCorrectorQP solves a quadratic program of the form
%
%    minimize   (1/2)* x' * Q * x + c' * x
%    subject to A*x = b,   x >= 0
%
% using a predictor-corrector primal-dual interior point method based on the
% lecture algorithm (with modifications for the quadratic case).
%
% INPUTS:
%   Q         - Hessian matrix (n x n)
%   c         - linear term (n x 1)
%   A         - constraint matrix (m x n)
%   b         - constraint right-hand side (m x 1)
%   tol       - tolerance for convergence (default 1e-6)
%   max_iter  - maximum iterations (default 50)
%   dampening - step-length dampening parameter (in [0.9, 1], default 0.95)
%
% OUTPUTS:
%   x           - primal solution (n x 1)
%   pi          - dual variables (m x 1)
%   z           - dual slack variables (n x 1)
%   iter_hist   - each row: [iter, tau, ||Ax-b||, ||(A'*pi+z-Q*x-c)||, x' * z]
%   primal_hist - each row: [iter, x (components), primal objective function]
%   dual_hist   - each row: [iter, pi (components), z (components), dual objective function]


if nargin < 5 || isempty(tol), tol = 1e-6; end
if nargin < 6 || isempty(max_iter), max_iter = 50; end
if nargin < 7 || isempty(dampening), dampening = 0.95; end

[m, n] = size(A);

%% Step 0: Initial Point Generation (Slide 40)
% Generate an initial interior point for x_bar, z_bar, and pi_bar:
% Here, instead of x_bar = A' * inv (A * A') * b; I used x_bar = A' * ((A * A') \
% b); because of numerical stability. ((A * A') \ b) solves (AA′)y=b for y
% and x_bar will be equal to A'y
x_bar = A' * ((A * A') \ b);                     %x_bar = A'(inv(AA')b)
pi_bar = (A * A') \ (A * c);                     %pi_bar = inv(AA')Ac
z_bar =  Q * x_bar + c - A' * pi_bar;            %z_bar = c-A'pi_bar

% Calculate delta_x and delta_z
delta_x = max(-1.5 * min(x_bar) , 0);
delta_z = max(-1.5 * min(z_bar) , 0);

% Calculate delta_x_bar and delta_z_bar.
delta_x_bar = delta_x + ((x_bar + delta_x * ones(size(x_bar)))' * ...
(z_bar + delta_z * ones(size(z_bar))))/(2*sum(z_bar+delta_z* ones(size(z_bar))));

delta_z_bar = delta_z + ((x_bar + delta_x * ones(size(x_bar)))' * ...
(z_bar + delta_z * ones(size(z_bar))))/(2*sum(x_bar+delta_x* ones(size(x_bar))));

% Generate the initial points.
x = x_bar + delta_x_bar;
z = z_bar + delta_z_bar;
pi = pi_bar;

% Initial duality measure:
y = (x' * z) / n;

% For "Residuals and tau": columns = [iter, tau, ||Ax-b||, ||Qx+c-A'pi-z||, x'z]
iter_hist = [];
% For "Primal Solution": columns = [iter, x(1), x(2), ... , x(n), c'*x]
primal_hist = [];
% For "Dual Solution": columns = [iter, pi(1), ..., pi(m), z(1), ..., z(n), dual objective function]
dual_hist = [];

iter_hist = [iter_hist; 0, NaN, norm(A*x - b), norm(A'*pi + z-Q*x-c), x'*z];
primal_hist = [primal_hist; 0, x', (0.5*x'*Q*x + c'*x)];
dual_hist = [dual_hist; 0, pi', z', b'*pi - 0.5*(c-A'*pi-z)' *inv(Q) *(c-A'*pi-z)];
    % Note that the last element is dual_hist is the dual objective function
    % which is b'*pi-0.5*(c-A'*pi-z)'*inv(Q)*(c-A'*pi-z)
%% Main Iteration Loop
for k = 0:max_iter
    % Compute the residuals:
    r_p = A * x - b;                 %primal residual: Ax - b.
    r_d = A'*pi + z - c - Q*x;       %dual residual: A'pi+z-c-Qx
    
    % Check the convergence criterion.
    if (norm(r_p) <= tol) && (norm(r_d) <= tol) && (y <= tol)
        fprintf('Converged at iteration %d.\n', k);
        break;
    end
    
    %% Predictor Step
    
    % For the predictor step, we drop the centering term, so r_c_aff = x .* z.
    % Define D_aff = Q + diag(z./x)
    % I implemented steps described in 2.1
    D_aff = Q + diag(z./x);
    % The reduced system is:
    %   A*(D_aff^{-1})*A' * d_pi_aff = -r_p + A*(D_aff^{-1})*(-r_d + z)
    d_pi_aff = (A * (D_aff \ A')) \ (-r_p + A * (D_aff \ (-r_d + z)));
    % Compute the affine direction for x:
    %   d_x_aff = D_aff^{-1}*(A'*d_pi_aff + r_d - z)
    d_x_aff = D_aff \ (A' * d_pi_aff + r_d - z);
    % Compute the affine direction for z
    %   d_z_aff = -X^{-1}(r_c_aff + Z*d_x_aff)   with X^{-1}Z = diag(z./x)
    d_z_aff = -z - (z./x) .* d_x_aff;
    % I've also implemented the matrix form in the block of code below. But
    % I don't recommend using it because of less computation efficiency.
    %{
    Z = diag(z);
    X = diag(x);
    KKT = [ -Q , A'       , eye(n);
        A          , zeros(m,m), zeros(m,n);
        Z          , zeros(n,m), X ];
    rhs = [-r_d;
        -r_p;
        -X*Z*ones(n,1)];
    d = KKT\rhs;
    d_x_aff = d(1:n);
    %d_pi_aff = d(n+1:n+m);
    d_z_aff = d(n+m+1:end);
    %}
    % Compute the maximum affine step lengths:
    alpha_aff_primal = stepLength(d_x_aff, x);
    alpha_aff_dual   = stepLength(d_z_aff, z);
    
    % Take the affine steps:
    x_aff = x + alpha_aff_primal * d_x_aff;
    z_aff = z + alpha_aff_dual   * d_z_aff;
    
    % Predicted duality measure:
    y_aff = (x_aff' * z_aff) / n;
    
    % Set the centering parameter tau (Slide 33):
    tau = (y_aff / y)^3;
    
    %% Corrector Step
    
    % Form the corrected residual:
    %   r_c_corr = x .* z - tau * y*e + d_x_aff .* d_z_aff.
    r_c_corr = x .* z - tau*y*ones(n,1) + d_x_aff .* d_z_aff;
    % For the corrector step we use the same D (can be updated if desired).
    % I implemented calculations described in section 2.2
    D_corr = Q + diag(z./x);
    d_pi = (A * (D_corr \ A')) \ (-r_p + A * (D_corr \ (-r_d + (r_c_corr./x))));
    d_x = D_corr \ (A' * d_pi + r_d - (r_c_corr./x));
    d_z = - (r_c_corr./x) - (z./x) .* d_x;
    % The blcok of code bellow is the implementation using the full matirx.
    %{
    D_x = diag(d_x_aff);
    D_z = diag(d_z_aff);
    rhs = [-r_d;
        -r_p;
        -X*Z*ones(n,1) - D_x*D_z*ones(n,1) + tau*y*ones(n,1)];
    d = KKT \ rhs;
    d_x = d(1:n);
    d_pi = d(n+1:n+m);
    d_z = d(n+m+1:end);
    %}
    % Compute the maximum allowable step lengths for the corrector direction:
    alpha_primal = stepLength(d_x, x) * dampening;
    alpha_dual   = stepLength(d_z, z) * dampening;
    
    % Update the iterates:
    x = x + alpha_primal * d_x;
    pi = pi + alpha_dual   * d_pi;
    z = z + alpha_dual   * d_z;
    
    % Update the duality measure:
    y = (x' * z) / n;
    
    % Save iteration history for part1.
    r_p = A*x - b;
    r_d = A'*pi + z - c - Q*x;
    if n <= 3
        iter_hist = [iter_hist; k, tau, norm(r_p), norm(r_d), x'*z];
        primal_hist = [primal_hist; k, x', (0.5*x'*Q*x + c'*x)];
        dual_hist = [dual_hist; k, pi', z', b'*pi - 0.5*(c-A'*pi-z)' *inv(Q) *(c-A'*pi-z)];
    end
end

if k == max_iter
    fprintf('Maximum iterations reached without convergence.\n');
end

end


%% Helper Function: stepLength
function alpha = stepLength(d, current)
% stepLength computes the maximum step length alpha (in [0,1])
idx = find(d < 0);
if isempty(idx)
    alpha = 1;
else
    alpha = min(1, min(-current(idx) ./ d(idx)));
end
end

% =========================================================================
%% ----------------- Part 1: Financial Optimization Model -----------------
clc; clear;

fprintf('================= Part 1: Financial Optimization Model (3 Assets) =================\n');

% Data from project part 1
mu_asset = [0.1073; 0.0737; 0.0627];
Sigma = [0.02778, 0.00387, 0.00021;
         0.00387, 0.01112, -0.00020;
         0.00021, -0.00020, 0.00115];
delta = 4;  % risk aversion parameter (choose between 3.5 and 4.5)

% Reformulate the maximization model into a minimization problem:
%   minimize   (delta/2)* x' * Sigma * x - mu' * x.
Q = delta * Sigma;
c = -mu_asset;

% Equality constraint: sum(x) = 1
A = ones(1,3);
b = 1;
lb = zeros(3,1);  % the lower bound for quadprog

fprintf('Solving the financial model with predictor-corrector interior point method...\n');
[x_pd, pi_pd, z_pd, iter_hist, primal_hist, dual_hist] = predictorCorrectorQP(Q, c, A, b, 1e-6, 50, 0.95);

fprintf('\nPredictor-Corrector Results:\n');
fprintf('Optimal x: \n'); disp(x_pd);
obj_pd = 0.5*x_pd'*Q*x_pd + c'*x_pd;
fprintf('Optimal objective value: %f\n', obj_pd);

fprintf('\nNow solving the same model using quadprog...\n');
options = optimoptions('quadprog','Display','off');
[x_qp, obj_qp] = quadprog(Q, c, [], [], A, b, lb, [], [], options);
fprintf('quadprog optimal x: \n'); disp(x_qp);
fprintf('quadprog optimal objective value: %f\n', obj_qp);

fprintf('\nIteration History (Predictor-Corrector):\n');
fprintf(' iter      tau         ||Ax-b||      ||(A''*pi+z-Q*x-c)||      (x''z)\n');
fprintf('--------------------------------------------------------------------------\n');
for i = 1:size(iter_hist,1)
    if isnan(iter_hist(i,2))
        tau_str = '   -   ';
    else
        tau_str = sprintf('%9.5e', iter_hist(i,2));
    end
    fprintf('%4d    %s    %11.5e    %11.5e    %11.5e\n', i-1, tau_str, iter_hist(i,3), iter_hist(i,4), iter_hist(i,5));
end

fprintf('\nPrimal Solution History:\n');
fprintf(' iter      x1           x2           x3        Obj = 0.5*x''*Q*x+c''*x\n');
for i = 1:size(primal_hist,1)
    fprintf('%4d    %11.5e  %11.5e  %11.5e    %11.5e\n', i-1, primal_hist(i,2), primal_hist(i,3), primal_hist(i,4), primal_hist(i,5));
end

fprintf('\nDual Solution History:\n');
fprintf(' iter      pi           z (components)                   b''*pi-0.5*(c-A''*pi-z)''*inv(Q)*(c-A''*pi-z)\n');
for i = 1:size(dual_hist,1)
    fprintf('%4d    ', i-1);
    fprintf('%11.5e ', dual_hist(i,2:2+size(A,1)-1));
    fprintf('   ');
    fprintf('%11.5e ', dual_hist(i,2+size(A,1):end-1));
    fprintf('   %11.5e\n', dual_hist(i,end));
end

%% ---------------- Part 2: Large-Scale Strictly Convex QP ----------------
fprintf(['\n==================== Part 2: Large-Scale Quadratic Programming ' ...
    '====================\n']);

% I now test my solver on a series of randomly generated covariance matrices.
% For each size n, I generate Sigma_rand = A_rand + A_rand' + n*I (which is PD),
% set mu_i = 1 (for all i), and solve the QP
%   maximize  sum(x) - (delta/2)*x'*Sigma_rand*x  (i.e., with mu_i=1)
% or equivalently,
%   minimize (delta/2)*x'*Sigma_rand*x - sum(x)
% using my predictor-corrector method and MATLAB's quadprog.

sizes = [5, 10, 20, 100, 10000];
delta_val = delta;  % use same risk aversion as in Part 1

results = struct();

for idx = 1:length(sizes)
    n = sizes(idx);
    fprintf('\nSolving QP for n = %d\n', n);
    
    % Random PD covariance matrix:
    A_rand = randn(n);
    Sigma_rand = A_rand + A_rand' + n*eye(n);
    
    % Set mu to 1
    mu_vec = ones(n,1);
    
    % Reformulate the model from max to min
    Q_rand = delta_val * Sigma_rand;
    c_rand = -ones(n,1);
    A_rand = ones(1,n);
    b_rand = 1;
    lb_rand = zeros(n,1);
    
    % Solve using my predictor-corrector solver:
    tic;
    [x_pd_rand, pi_pd_rand, z_pd_rand, ~, ~, ~] = predictorCorrectorQP(Q_rand, c_rand, A_rand, b_rand, 1e-6, 100, 0.95);
    time_pd = toc;
    
    % Solve using quadprog:
    options = optimoptions('quadprog','Display','off');
    tic;
    [x_qp_rand, obj_qp_rand] = quadprog(Q_rand, c_rand, [], [], A_rand, b_rand, lb_rand, [], [], options);
    time_qp = toc;
    
    obj_pd = 0.5*x_pd_rand'*Q_rand*x_pd_rand + c_rand'*x_pd_rand;
    
    results(idx).n = n;
    results(idx).x_pd = x_pd_rand;
    results(idx).obj_pd = obj_pd;
    results(idx).time_pd = time_pd;
    results(idx).x_qp = x_qp_rand;
    results(idx).obj_qp = obj_qp_rand;
    results(idx).time_qp = time_qp;
    
    fprintf('n = %d: Predictor-Corrector objective = %f, CPU time = %f sec\n', n, obj_pd, time_pd);
    fprintf('n = %d: quadprog objective           = %f, CPU time = %f sec\n', n, obj_qp_rand, time_qp);
end

fprintf('\nSummary for Part 2:\n');
for idx = 1:length(sizes)
    fprintf('n = %d: PD obj = %f, quadprog obj = %f, PD time = %f, quadprog time = %f\n', ...
        results(idx).n, results(idx).obj_pd, results(idx).obj_qp, results(idx).time_pd, results(idx).time_qp);
end

%% End of MIE1621_Project_Hamid_Nakhaei