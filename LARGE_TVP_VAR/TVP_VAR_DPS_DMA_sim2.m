% TVP_VAR_DPS_DMA_sim2.m - Forecasting with Large TVP-VAR using forgetting factors
% MULTIPLE MODEL CASE / DYNAMIC PRIOR SELECTION (DPS) AND DYNAMIC MODEL
% AVERAGING (DMA)
%-------------------------------------------------------------------------------
% The model is:
%
%	 y[t] = theta[t] x[t] + e[t]
%	 theta[t] = theta[t-1] + u[t]
%
% where x[t] = I x (y[t-1],...,y[t-p]) (Kronecker product), and e[t]~N(0,V[t])
% and u[t]~N(0,Q[t]).
%
% Additionally:
%
%  V[t] = kappa V[t-1] + (1-kappa) e[t-1]e[t-1]'
%  Q[t] = (1 - 1/lambda) S[t-1|t-1]
%
% This code estimates lambda and allows it to be time-varying. The specification is:
%
%  lambda[t] = lambda[min] + (1-lambda[min]) LL^(e[t]e[t]')
%
%-------------------------------------------------------------------------------
%  - This code allows to calculate ONLY iterated forecasts
%  - This code does predictive simulation by simulating values for theta[T+1],
%  theta[T+2],... from Normals with mean theta[T], theta[T+1],... (i.e allowing 
%  drifting parameters out-of-sample).
%  - This code does "online" forecasting, i.e. the Minnesota prior should not be
%  dependent on the data, so that the Kalman filter runs once for 1:T.
%-------------------------------------------------------------------------------
% Written by Dimitris Korobilis and Gary Koop
% University of Glasgow and University of Strathclyde
% This version: 15 January, 2012
%-------------------------------------------------------------------------------

clear all;
clc;

% Add path of data and functions
addpath('data');
addpath('functions');

%-------------------------------PRELIMINARIES--------------------------------------
forgetting = 1;    % 1: use constant factor; 2: use variable factor

lambda = 0.99;
kappa = 0.96;      % Decay factor for measurement error variance

eta = 0.99;   % Forgetting factor for DPS (dynamic prior selection) and DMA

% Please choose:
p = 4;             % p is number of lags in the VAR part
nos = 3;           % number of subsets to consider (default is 3, i.e. 3, 7, and 25 variable VARs)
% if nos=1 you might want a single model. Which one is this?
single = 1;        % 1: 3 variable VAR
                   % 2: 7 variable VAR
                   % 3: 25 variable VAR
                   
prior = 1;         % 1: Use Koop-type Minnesota prior
                   % 2: Use Litterman-type Minnesota prior
                         
% Forecasting
first_sample_ends = 1974.75; % The end date of the first sample in the 
                             % recursive exercise (default value: 1969:Q4)
nfore = 8;                   % Select forecast horizon
nsim = 5000;             % Number of times to simulate from the predictive density

% Choose which results to print
% NOTE: CHOOSE ONLY 0/1 (FOR NO/YES) VALUES!
print_fore = 1;           % summary of forecasting results
print_coefficients = 1;   % plot volatilities and lambda_t (but not theta_t which is huge)
print_pred = 1;           % plot predictive likelihoods over time
print_Min = 1;            % print the Minnesota prior over time
%----------------------------------LOAD DATA----------------------------------------
load ydata.dat;
load ynames.mat;
load tcode.dat;
load vars.mat;
%load yearlab.dat;

% Create dates variable
start_date = 1959.00; %1959.Q1
end_date = 2010.25;   %2010.Q2
yearlab = (1959:0.25:2010.25)';
T_thres = find(yearlab == first_sample_ends); % find tau_0 (first sample)

% Transform data to stationarity
% Y: standard transformations (for iterated forecasts, and RHS of direct forecasts)
[Y,yearlab] = transform(ydata,tcode,yearlab);

% Select a subset of the data to be used for the VAR
if nos>3
    error('DMA over too many models, memory concerns...')
end
Y1=cell(nos,1);
Ytemp = standardize1(Y,T_thres);
M = zeros(nos,1);
for ss = 1:nos
    if nos ~= 1
        single = ss;
    end
    select_subset = vars{single,1};
    Y1{ss,1} = Ytemp(:,select_subset);
    M(ss,1) = max(size(select_subset)); % M is the dimensionality of Y
end
t = size(Y1{1,1},1);

% The first nfocus variables are the variables of interest for forecasting
nfocus = 3;

% ===================================| VAR EQUATION |==============================
% Generate lagged Y matrix. This will be part of the X matrix
x_t = cell(nos,1);
x_f = cell(nos,1);
y_t = cell(nos,1);
K = zeros(nos,1);
for ss=1:nos
    ylag = mlag2(Y1{ss,1},p); 
    ylag = ylag(p+1:end,:);
    [temp,kk] = create_RHS(ylag,M(ss),p,t);
    x_t{ss,1} = temp;
    K(ss,1) = kk;
    x_f{ss,1} = ylag;
    y_t{ss,1} = Y1{ss,1}(p+1:end,:);    
end
yearlab = yearlab(p+1:end);
% Time series observations
t=size(y_t{1,1},1); %#ok<*NASGU>
   
%----------------------------PRELIMINARIES---------------------------------
%========= PRIORS:
% Set the alpha_bar and the set of gamma values
alpha_bar = 10;
gamma = [1e-10,1e-5,0.001,0.005,0.01,0.05,0.1];
nom = max(size(gamma));  % This variable defines the number of DPS models
%-------- Now set prior means and variances (_prmean / _prvar)
theta_0_prmean = cell(nos,1);
theta_0_prvar = cell(nos,1);
for ss=1:nos
    if prior == 1            % 1) "No dependence" prior
        for i=1:nom
            [prior_mean,prior_var] = Minn_prior_KOOP(alpha_bar,gamma(i),M(ss),p,K(ss));   
            theta_0_prmean{ss,1}(:,i) = prior_mean;
            theta_0_prvar{ss,1}(:,:,i) = prior_var;        
        end
        Sigma_0{ss,1} = cov(y_t{ss,1}(1:T_thres,:)); %#ok<*SAGROW> % Initialize the measurement covariance matrix (Important!)
    elseif prior == 2        % 2) Full Minnesota prior
        for i=1:nom
            [prior_mean,prior_var,sigma_var] = Minn_prior_LITT(y_t{ss,1}(1:T_thres,:),x_f{ss,1}(1:T_thres,:),alpha_bar,gamma(i),M(ss),p,K(ss),T_thres);   
            theta_0_prmean{ss,1}(:,i) = prior_mean;
            theta_0_prvar{ss,1}(:,:,i) = prior_var;       
        end
        Sigma_0{ss,1} = sigma_var; % Initialize the measurement covariance matrix (Important!)
    end
end

% Define forgetting factor lambda:
lambda_t = cell(nos,1);
for ss=1:nos
    if forgetting == 1
        % CASE 1: Choose the forgetting factor   
        
        inv_lambda = 1./lambda;
        lambda_t{ss,1}= lambda*ones(t,nom);
    elseif forgetting == 2
        % CASE 2: Use a variable (estimated) forgetting factor
        lambda_min = 0.97;
        inv_lambda = 1./0.99;
        alpha = 1;
        LL = 1.1;
        lambda_t{ss,1} = zeros(t,nom);
    else
        error('Wrong specification of forgetting procedure')
    end
end

% Initialize matrices
theta_pred = cell(nos,1);   
theta_update = cell(nos,1);
R_t = cell(nos,1);
S_t = cell(nos,1);
y_t_pred = cell(nos,1);
e_t =cell(nos,1);
A_t = cell(nos,1);
V_t = cell(nos,1);
y_fore = cell(nos,1);
omega_update = cell(nos,1);
omega_predict = cell(nos,1);
ksi_update = zeros(t,nos);
ksi_predict = zeros(t,nos);
w_t = cell(nos,1);
w2_t = zeros(t,nos);
f_l = zeros(nom,1);
max_prob_DMS = zeros(t,1);
index_best = zeros(t,1);
index_DMA = zeros(t,nos);


anumber = t-T_thres+1;
y_t_DMA = zeros(nfore,nfocus,anumber);
y_t_DMS = zeros(nfore,nfocus,anumber);
LOG_PL_DMA = zeros(anumber,nfore);
MSFE_DMA = zeros(anumber,nfocus,nfore);
MAFE_DMA = zeros(anumber,nfocus,nfore);
LOG_PL_DMS = zeros(anumber,nfore);
MSFE_DMS = zeros(anumber,nfocus,nfore);
MAFE_DMS = zeros(anumber,nfocus,nfore);
Minn_gams = cell(nos,1);
logpl_DMA = zeros(anumber,nfocus,nfore);
logpl_DMS = zeros(anumber,nfocus,nfore);
offset = 1e-9;  % just a constant for numerical stability

%----------------------------- END OF PRELIMINARIES ---------------------------

%======================= BEGIN KALMAN FILTER ESTIMATION =======================
tic;
for irep = 1:t
    if mod(irep,ceil(t./40)) == 0
        disp([num2str(100*(irep/t)) '% completed'])
        toc;
    end
    for ss = 1:nos  % LOOP FOR 1 TO NOS VAR MODELS OF DIFFERENT DIMENSIONS
        % Find sum of probabilities for DPS
        if irep>1
            sum_prob_omega(ss,1) = sum((omega_update{ss,1}(irep-1,:)).^eta);  % this is the sum of the nom model probabilities (all in the power of the forgetting factor 'eta')
        end
        for k=1:nom % LOOP FOR 1 TO NOM VAR MODELS WITH DIFFERENT DEGREE OF SHRINKAGE
            % Predict   
            if irep==1
                theta_pred{ss,1}(:,irep,k) = theta_0_prmean{ss,1}(:,k);         
                R_t{ss,1}(:,:,k) = theta_0_prvar{ss,1}(:,:,k);           
                omega_predict{ss,1}(irep,k) = 1./nom;
            else                
                theta_pred{ss,1}(:,irep,k) = theta_update{ss,1}(:,irep-1,k);
                R_t{ss,1}(:,:,k) = (1./lambda_t{ss,1}(irep-1,k))*S_t{ss,1}(:,:,k);
                omega_predict{ss,1}(irep,k) = ((omega_update{ss,1}(irep-1,k)).^eta + offset)./(sum_prob_omega(ss,1) + offset);
            end
            xx = x_t{ss,1}((irep-1)*M(ss)+1:irep*M(ss),:);
            y_t_pred{ss,1}(:,irep,k) = xx*theta_pred{ss,1}(:,irep,k);  % this is one step ahead prediction
       
            % Prediction error
            e_t{ss,1}(:,irep,k) = y_t{ss,1}(irep,:)' - y_t_pred{ss,1}(:,irep,k);  % this is one step ahead prediction error
    
            % Update forgetting factor
            if forgetting == 2
                lambda_t{ss,1}(irep,k) = lambda_min + (1-lambda_min)*(LL^(-round(alpha*e_t{ss,1}(1:nfocus,irep,k)'*e_t{ss,1}(1:nfocus,irep,k))));
            end
            
            % first update V[t], see the part below equation (10)
            A_t = e_t{ss,1}(:,irep,k)*e_t{ss,1}(:,irep,k)';
            if irep==1
                V_t{ss,1}(:,:,irep,k) = kappa*Sigma_0{ss,1};
            else
                V_t{ss,1}(:,:,irep,k) = kappa*squeeze(V_t{ss,1}(:,:,irep-1,k)) + (1-kappa)*A_t;
            end
            %         if all(eig(squeeze(V_t(:,:,irep,k))) < 0)
            %             V_t(:,:,irep,k) = V_t(:,:,irep-1,k);       
            %         end

            % update theta[t] and S[t]
            Rx = R_t{ss,1}(:,:,k)*xx';
            KV = squeeze(V_t{ss,1}(:,:,irep,k)) + xx*Rx;
            KG = Rx/KV;
            theta_update{ss,1}(:,irep,k) = theta_pred{ss,1}(:,irep,k) + (KG*e_t{ss,1}(:,irep,k)); %#ok<*MINV>
            S_t{ss,1}(:,:,k) = R_t{ss,1}(:,:,k) - KG*(xx*R_t{ss,1}(:,:,k));
      
            % Find predictive likelihood based on Kalman filter and update DPS weights     
            variance = V_t{ss,1}(1:nfocus,1:nfocus,irep,k) + xx(1:nfocus,:)*Rx(:,1:nfocus);
            if find(eig(variance)>0==0) > 0
                variance = abs(diag(diag(variance)));       
            end
            ymean = y_t_pred{ss,1}(1:nfocus,irep,k);        
            ytemp = y_t{ss,1}(irep,1:nfocus)';
            f_l(k,1) = mvnpdfs(ytemp,ymean,variance);       
            w_t{ss,1}(:,irep,k) = omega_predict{ss,1}(irep,k)*f_l(k,1);
        
            % Individual model forecasts for DPS
            if irep>=T_thres
                chol_var_beta = chol(S_t{ss,1}(:,:,k))';
                chol_var_y = chol(V_t{ss,1}(:,:,irep,k)); 
                % Start predictive simulation
                for sim = 1:nsim
                    Y_hat = 0;
                    % Draw from the posterior of the coefficients, theta
                    beta_sim  = theta_update{ss,1}(:,irep,k) + chol_var_beta*randn(K(ss),1);
                    bbtemp = beta_sim(M(ss)+1:K(ss));  % get the draw of B(t) at time i=1,...,T  (exclude intercept)
                    splace = 0;
                    biga=0;
                    for ii = 1:p                        
                        for iii = 1:M(ss)           
                            biga(iii,(ii-1)*M(ss)+1:ii*M(ss)) = bbtemp(splace+1:splace+M(ss),1)';
                            splace = splace + M(ss);
                        end
                    end                    
                    beta_fore = [beta_sim(1:M(ss))' ; biga'];
                    % Now create forecast for h=1
                    X_FORE = [1 y_t{ss,1}(irep,:) x_f{ss,1}(irep,1:M(ss)*(p-1))];
                    Y_hat = X_FORE*beta_fore + randn(1,M(ss))*chol_var_y;
                    y_fore{ss,1}(1,:,k,sim) = Y_hat;
                    % Now do forecasts for h>1
                    for ii = 1:nfore-1
                        beta_sim  = beta_sim + chol_var_beta*randn(K(ss),1);
                        splace = 0;
                        biga=0;
                        beta_fore=0;
                        for ig = 1:p                                               
                            for iii = 1:M(ss)           
                                biga(iii,(ig-1)*M(ss)+1:ig*M(ss)) = bbtemp(splace+1:splace+M(ss),1)';
                                splace = splace + M(ss);
                            end
                        end
                        beta_fore = [beta_sim(1:M(ss))' ; biga'];
                        if ii <= p   % if h<=p (number of lags)    
                            X_new_temp = [1 Y_hat X_FORE(:,2:M(ss)*(p-ii)+1)];                    
                            Y_temp = X_new_temp*beta_fore + randn(1,M(ss))*chol_var_y;                    
                            Y_hat = [Y_temp Y_hat];
                        else  % if h>p (number of lags) 
                            X_new_temp = [1 Y_hat(:,1:M(ss)*p)];
                            Y_temp = X_new_temp*beta_fore + randn(1,M(ss))*chol_var_y;
                            Y_hat = [Y_temp, Y_hat];
                        end
                        % This cell array saves the draws from all forecast
                        % horizons, from all model sizes.
                        y_fore{ss,1}(ii+1,:,k,sim) = Y_temp;
                    end
                end % End predictive simulation 
            end

        end % End cycling through nom models with different shrinkage factors
    
        % First calculate the denominator of Equation (19) (the sum of the w's)
        sum_w_t = 0;   
        for k_2=1:nom       
            sum_w_t = sum_w_t + w_t{ss,1}(:,irep,k_2);
        end
        
        % Then calculate the DPS probabilities  
        for k_3 = 1:nom
            omega_update{ss,1}(irep,k_3) = (w_t{ss,1}(:,irep,k_3) + offset)./(sum_w_t + offset);  % this is Equation (19)
        end
        [max_prob{ss,1},k_max{ss,1}] = max(omega_update{ss,1}(irep,:));
        index_DMA(irep,ss) = k_max{ss,1}; 
        
        % Use predictive likelihood of best (DPS) model, and fight the weight for DMA     
        w2_t(irep,ss) = omega_predict{ss,1}(irep,k_max{ss})*f_l(k_max{ss},1);

        % Find "observed" out-of-sample data for MSFE and MAFE calculations
        if irep <= t-nfore
            Yraw_f{ss,1} = y_t{ss,1}(irep+1:irep+nfore,:,:); %Pseudo out-of-sample observations                   
        else
            Yraw_f{ss,1} =[y_t{ss,1}(irep+1:t,:) ; NaN(nfore-(t-irep),M(ss))];
        end
    end
    
    % First calculate the denominator of Equation (19) (the sum of the w's)
    sum_w2_t = 0;   
    for k_2=1:nos       
        sum_w2_t = sum_w2_t + w2_t(irep,k_2);
    end
    
    % Then calculate the DPS probabilities
    for k_3 = 1:nos
        ksi_update(irep,k_3) = (w2_t(irep,k_3) + offset)./(sum_w2_t + offset);  % this is Equation (19)
    end
    
    % Find best model for DMS
    [max_prob_DMS(irep,:),ss_max] = max(ksi_update(irep,:));
    index_DMS(irep,1) = k_max{ss_max};
    
    % Now we cycled over NOM and NOS models, do DMA-over-DPS
    if irep>=T_thres
        % Now we have the predictions for each model & the associated model
        for ii = 1:nfore
            weight_pred = 0*y_fore{ss,1}(ii,1:nfocus,k_max{ss},:);
            % DPS-DMA prediction
            for ss = 1:nos
                temp_predict = y_fore{ss,1}(ii,1:nfocus,k_max{ss},:).*ksi_update(irep,ss);
                weight_pred = weight_pred + temp_predict;
                Minn_gamms{ss,1}(irep-T_thres+1,1) = gamma(1,k_max{ss,1});
            end
            
            y_t_DMA(ii,:,irep-T_thres+1) = mean(weight_pred,4);
            variance_DMA = cov(squeeze(weight_pred)');
            
            y_t_DMS(ii,:,irep-T_thres+1) = mean(y_fore{ss_max,1}(ii,1:nfocus,k_max{ss_max},:),4);
            variance_DMS = cov(squeeze(y_fore{ss_max,1}(ii,1:nfocus,k_max{ss_max},:))');
            
            
            LOG_PL_DMS(irep-T_thres+1,ii) = log(mvnpdfs(Yraw_f{ss}(ii,1:nfocus)',y_t_DMS(ii,:,irep-T_thres+1)',variance_DMS) + offset);
            MAFE_DMS(irep-T_thres+1,:,ii) = abs(Yraw_f{ss}(ii,1:nfocus) - squeeze(y_t_DMS(ii,:,irep-T_thres+1)));
            MSFE_DMS(irep-T_thres+1,:,ii) = (Yraw_f{ss}(ii,1:nfocus) - squeeze(y_t_DMS(ii,:,irep-T_thres+1))).^2;
            
            LOG_PL_DMA(irep-T_thres+1,ii) = log(mvnpdfs(Yraw_f{ss}(ii,1:nfocus)',y_t_DMA(ii,:,irep-T_thres+1)',variance_DMA) + offset);
            MAFE_DMA(irep-T_thres+1,:,ii) = abs(Yraw_f{ss}(ii,1:nfocus) - squeeze(y_t_DMA(ii,:,irep-T_thres+1)));
            MSFE_DMA(irep-T_thres+1,:,ii) = (Yraw_f{ss}(ii,1:nfocus) - squeeze(y_t_DMA(ii,:,irep-T_thres+1))).^2;
            for j=1:nfocus
                logpl_DMA(irep-T_thres+1,j,ii) = log(mvnpdfs(Yraw_f{ss}(ii,j)',y_t_DMA(ii,j,irep-T_thres+1)',variance_DMA(j,j)));
                logpl_DMS(irep-T_thres+1,j,ii) = log(mvnpdfs(Yraw_f{ss}(ii,j)',y_t_DMS(ii,j,irep-T_thres+1)',variance_DMS(j,j)));
            end      
        end
  
    end
    
end
%======================== END KALMAN FILTER ESTIMATION ========================

%===================| PRINT RESULTS |=========================
clc;
toc;
format short g;
% summary of forecasting results
if print_fore == 1;
disp('==============| DMA RESULTS |==================')
disp('MSFE for the key variables of interest')
msfe_focus = mean(MSFE_DMA(1:end-1,1:nfocus,1));
for iii = 2:nfore
    msfe_focus = [msfe_focus; mean(MSFE_DMA(1:end-iii,1:nfocus,iii))];
end
disp(['    Horizon    GDP       INFL      INTR']) %#ok<*NBRAK>
disp([(1:nfore)' msfe_focus])

disp('MAFE for the key variables of interest')
mafe_focus = mean(MAFE_DMA(1:end-1,1:nfocus,1));
for iii = 2:nfore
    mafe_focus=[mafe_focus; mean(MAFE_DMA(1:end-iii,1:nfocus,iii))];
end
disp(['    Horizon    GDP       INFL      INTR'])
disp([(1:nfore)' mafe_focus])

disp('sum of log pred likes for the key variables of interest individually')
lpls_focus = sum(logpl_DMA(1:end-1,:,1));
for iii = 2:nfore
    lpls_focus = [lpls_focus; sum(logpl_DMA(1:end-iii,:,iii))];
end
disp(['    Horizon    GDP      INFL      INTR'])
disp([(1:nfore)' lpls_focus])
disp('sum of log pred likes for the key variables of interest jointly')
lpl_focus = sum(LOG_PL_DMA(1:end-1,1));
for iii=2:nfore
    lpl_focus=[lpl_focus; sum(LOG_PL_DMA(1:end-iii,iii))];
end
disp(['    Horizon   TOTAL'])
disp([(1:nfore)' lpl_focus])
disp('                      ')
disp('                      ')

disp('==============| DMS RESULTS |==================')
disp('MSFE for the key variables of interest')
msfe_focus = mean(MSFE_DMS(1:end-1,1:nfocus,1));
for iii = 2:nfore
    msfe_focus = [msfe_focus; mean(MSFE_DMS(1:end-iii,1:nfocus,iii))];
end
disp(['    Horizon    GDP       INFL      INTR'])
disp([(1:nfore)' msfe_focus])

disp('MAFE for the key variables of interest')
mafe_focus = mean(MAFE_DMS(1:end-1,1:nfocus,1));
for iii = 2:nfore
    mafe_focus=[mafe_focus; mean(MAFE_DMS(1:end-iii,1:nfocus,iii))];
end
disp(['    Horizon    GDP       INFL      INTR'])
disp([(1:nfore)' mafe_focus])

disp('sum of log pred likes for the key variables of interest individually')
lpls_focus = sum(logpl_DMS(1:end-1,:,1));
for iii = 2:nfore
    lpls_focus = [lpls_focus; sum(logpl_DMS(1:end-iii,:,iii))];
end
disp(['    Horizon    GDP      INFL      INTR'])
disp([(1:nfore)' lpls_focus])
disp('sum of log pred likes for the key variables of interest jointly')
lpl_focus = sum(LOG_PL_DMS(1:end-1,1));
for iii=2:nfore
    lpl_focus=[lpl_focus; sum(LOG_PL_DMS(1:end-iii,iii))];
end
disp(['    Horizon   TOTAL'])
disp([(1:nfore)' lpl_focus])
end


% plot volatilities and lambda_t (but not theta_t which is huge)
if print_coefficients == 1;
    prob_pl = omega_update{1,1}(index_DMA(:,1));
    for ss=2:nos
        prob_pl = [ prob_pl omega_update{ss,1}(index_DMA(:,ss))]; %#ok<*AGROW>
    end
    final_prob = 0*prob_pl;
    for sss = 1:nos
        final_prob(:,sss) = prob_pl(:,sss)./sum(prob_pl,2);
    end
    figure
    plot(yearlab,final_prob)
    warning off;
    legend('small VAR','medium VAR','large VAR')
    title('Time-varying probabilities of small/medium/large VARs')
    if forgetting == 2;
        lam_pl = lambda_t{1,1}(index_DMA(:,1));
        for ss=2:nos
            lam_pl = [lam_pl lambda_t{ss,1}(index_DMA(:,ss))];
        end
        figure
        plot(yearlab,lam_pl)
        legend('small VAR','medium VAR','large VAR')
        title('This is the value attained by the time varying forgetting factor \lambda')
    end
end
% plot predictive likelihoods over time
if print_pred == 1;
    figure
    subplot(2,2,1)
    plot(yearlab(T_thres+1:end),cumsum(LOG_PL_DMS(1:end-1,1)))
    title('Cumulative sum of total predictive likelihood h=1')
    subplot(2,2,2)
    plot(yearlab(T_thres+2:end),cumsum(LOG_PL_DMS(1:end-2,2)))
    title('Cumulative sum of total predictive likelihood h=2')
    subplot(2,2,3)
    plot(yearlab(T_thres+3:end),cumsum(LOG_PL_DMS(1:end-3,3)))       
    title('Cumulative sum of total predictive likelihood h=3')
    subplot(2,2,4)
    plot(yearlab(T_thres+4:end),cumsum(LOG_PL_DMS(1:end-4,4)))    
    title('Cumulative sum of total predictive likelihood h=4')
end
% print the Minnesota prior over time
if print_Min == 1;
    disp('=========MINNESOTA PRIOR==========')
    disp('Best gammas for each time period')
    vars_Min = (T_thres:t)';
    for ss=1:nos
        vars_Min = [vars_Min  Minn_gamms{ss,1}];
    end
    lll={'period', 'smallVAR', 'mediumVAR', 'largeVAR'};
    disp(lll(1:nos+1));
    disp(vars_Min);
end


save(sprintf('%s_%g_%g.mat','TVP_VAR_DPS_DMA_sim2',nos,single),'-mat');