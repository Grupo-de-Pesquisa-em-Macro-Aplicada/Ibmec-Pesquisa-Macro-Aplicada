function [a_prior,V_prior,Sigma_0] = Minn_prior_LITT(Y,Ylag,alpha_bar,gamma,M,p,K,t)

% 1. Minnesota Mean on VAR regression coefficients
A_prior = [zeros(1,M); 0*eye(M); zeros((p-1)*M,M)]';
%A_prior = [mean(Y(1:40,:)); 0.9*eye(M); zeros((p-1)*M,M)]';
%A_prior = [zeros(1,M) ; 0.9*repmat(eye(M),p,1)]';
a_prior = A_prior(:);

% 2. Minnesota Variance on VAR regression coefficients
% Now get residual variances of univariate p-lag autoregressions. Here
% we just run the AR(p) model on each equation, ignoring the constant
% and exogenous variables (if they have been specified for the original
% VAR model)
p_MIN = p;
sigma_sq = zeros(M,1); % vector to store residual variances
for i = 1:M
    Ylag_i = Ylag;
    X_i = [ones(t-p_MIN+p,1) Ylag_i(:,i:M:M*p_MIN)];
    Y_i = Y(:,i);
    % OLS estimates of i-th equation
    alpha_i = inv(X_i'*X_i)*(X_i'*Y_i);
    sigma_sq(i,1) = (1./(t-p_MIN+1))*(Y_i - X_i*alpha_i)'*(Y_i - X_i*alpha_i);
end

% Now define prior hyperparameters.
% Create an array of dimensions K x M, which will contain the K diagonal   
% elements of the covariance matrix, in each of the M equations.
V_i = zeros(K/M,M);
    
% index in each equation which are the own lags
ind = zeros(M,p);
for i=1:M
    ind(i,:) = i+1:M:K/M;
end
for i = 1:M  % for each i-th equation
    for j = 1:K/M   % for each j-th RHS variable   
        if j==1 % if there is constant, use this code
            V_i(j,i) = alpha_bar*sigma_sq(i,1); % variance on intercept               
        elseif find(j==ind(i,:))>0
            V_i(j,i) = gamma./((ceil(j-1)/M).^2); % variance on own lags           
            % Note: the "ceil((j-1)/M)" command finds the associated lag 
            % number for each parameter
        else
            for kj=1:M
                if find(j==ind(kj,:))>0
                    ll = kj;                   
                end
            end    % variance on other lags
            V_i(j,i) = (gamma*sigma_sq(i,1))./(((ceil(j-1)/M).^2)*sigma_sq(ll,1));           
        end        
    end
end

% Now V (MINNESOTA VARIANCE) is a diagonal matrix with diagonal elements the V_i'  
V_i_T = V_i';
V_prior = single(diag(V_i_T(:)));  % this is the prior variance of the vector alpha

Sigma_0 = single(diag(sigma_sq));
