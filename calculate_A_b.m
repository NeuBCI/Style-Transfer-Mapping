function [A, b] = calculate_A_b(S, T, f, beta_coeff, gamma_coeff)
% input: source, target, confidence and two hyper-paras; output: A and b
[n, m] = size(S);      % n is the data amount, m is the feature dimision, namely 310 
s_arrow = zeros(m,1);
t_arrow = zeros(m,1);
Q_term = zeros(m,m);
P_term = zeros(m,m);

for i = 1:1:n
    s_arrow = s_arrow + f(i)*S(i,:)';
    t_arrow = t_arrow + f(i)*T(i,:)';
    Q_term = Q_term + f(i)*T(i,:)'*S(i,:);
    P_term = P_term + f(i)*S(i,:)'*S(i,:);
end

beta = beta_coeff*1/m*trace(P_term);     
gamma = gamma_coeff*sum(f);

f_arrow = sum(f)+gamma;

Q = Q_term - 1/f_arrow*(t_arrow*s_arrow') + beta*eye(m,m);
P = P_term - 1/f_arrow*(s_arrow*s_arrow') + beta*eye(m,m);

A = Q*inv(P);
b = 1/f_arrow*(t_arrow-A*s_arrow);
end

