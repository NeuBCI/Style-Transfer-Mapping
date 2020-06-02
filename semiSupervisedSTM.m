function [S_transfered_LVQ, L_B_predicted_LVQ, L_B_predicted_QDF] = semiSupervisedSTM(train_x, train_y, S_A, L_A, S_B, L_B, a, beta_coeff, gamma_coeff, iterNum)
% implement STM in semi-supervised way
% train_x & train_y: used to find prototypes and train subject-free classifiers
% S_A: labeled data in S
% L_A: labels of S_A
% S_B: unlabeled data in S
% L_B: unlabeled data's label, for test only
% a: hyper para to balance labeled data in S
% beta_coeff & gamma_coeff: two hyper paras [0 3]
% iterNum: iteration number for self_training
[nb_A, m] = size(S_A);
[nb_B, ~] = size(S_B);
S = [S_A;S_B];               % all data in S
T_A_LVQ = zeros(nb_A, m);    % target of labeled data defined by LVQ
T_A_QDF = zeros(nb_A, m);    % target of labeled data defined by QDF
T_B_LVQ = zeros(nb_B, m);    % target of unlabeled data defined by LVQ
T_B_QDF = zeros(nb_B, m);    % target of unlabeled data defined by QDF
f_A = ones(nb_A, 1);         % confidence of labeled data, initialzed as 1 (will soon be replaced by a)
f_B_LVQ = zeros(nb_B, 1);    % confidence of unlabeled data in LVQ target defination
f_B_QDF = zeros(nb_B, 1);    % confidence of unlabeled data in QDF target defination
[id_pos,~] = find(train_y(:)==1);
[id_neu,~] = find(train_y(:)==2);
[id_neg,~] = find(train_y(:)==3);
pos = train_x(id_pos,:);
neu = train_x(id_neu,:);
neg = train_x(id_neg,:);
%% -- LVQ: find prototypes(m) in each class -- %%
nb_pos = 15;
nb_neu = 15;
nb_neg = 15;
[~,m1] = kmeans(pos,nb_pos,'rep',5);
[~,m2] = kmeans(neu,nb_neu,'rep',5);
[~,m3] = kmeans(neg,nb_neg,'rep',5);
%% -- QDF: find centers(u) and segma for each class-- %%
u1 = mean(pos,1);        % center of each class
u2 = mean(neu,1);
u3 = mean(neg,1);
segma1 = cov(pos);       % convariance matrix of each class
segma2 = cov(neu);
segma3 = cov(neg);
%% --(1) learn a supervised STM {A0, b0} with labeled data-- %%
for i = 1:1:nb_A
    x = S_A(i,:);
    switch L_A(i)
        case 1
            T_A_LVQ(i,:) = findTargetLVQ(x, m1);
            T_A_QDF(i,:) = findTargetQDF(x, u1, segma1);
        case 2
            T_A_LVQ(i,:) = findTargetLVQ(x, m2);
            T_A_QDF(i,:) = findTargetQDF(x, u2, segma2);
        case 3
            T_A_LVQ(i,:) = findTargetLVQ(x, m3);
            T_A_QDF(i,:) = findTargetQDF(x, u3, segma3);
    end
end
[A0_LVQ, b0_LVQ] = calculate_A_b(S_A, T_A_LVQ, f_A, beta_coeff, gamma_coeff);
[A0_QDF, b0_QDF] = calculate_A_b(S_A, T_A_QDF, f_A, beta_coeff, gamma_coeff);
%% --(2) transform all S data by A0 and b0-- %%
S_A_LVQ = zeros(nb_A,m);    % the labeled data after transformed by A0, b0
S_A_QDF = zeros(nb_A,m);
S_B_LVQ = zeros(nb_B,m);    % the unlabeled data after transformed by A0, b0
S_B_QDF = zeros(nb_B,m);
for i = 1:1:nb_A
    S_A_LVQ(i,:) = (A0_LVQ*S_A(i,:)' + b0_LVQ)';
    S_A_QDF(i,:) = (A0_QDF*S_A(i,:)' + b0_QDF)';
end
for i = 1:1:nb_B
    S_B_LVQ(i,:) = (A0_LVQ*S_B(i,:)' + b0_LVQ)';
    S_B_QDF(i,:) = (A0_QDF*S_B(i,:)' + b0_QDF)';
end
% evaluate the supervised STM A0 and b0
svm = svmtrain(train_y, train_x);   % train subject-free classifier
[L_B_predicted_LVQ, ~, ~] = svmpredict(ones(nb_B,1), S_B_LVQ, svm);
[L_B_predicted_QDF, ~, ~] = svmpredict(ones(nb_B,1), S_B_QDF, svm);
num_lvq = 0;
num_qdf = 0;
for w = 1:1:nb_B
    if L_B_predicted_LVQ(w) == L_B(w)
        num_lvq = num_lvq + 1;
    end
    if L_B_predicted_QDF(w) == L_B(w)
        num_qdf = num_qdf + 1;
    end
end
q1 = num_lvq/nb_B;
q2 = num_qdf/nb_B;
acc_supervised = [q1,q2];
%% --(3) update T_A and reset the confidence as a-- %%
for i = 1:1:nb_A
    x_LVQ = S_A_LVQ(i,:);
    x_QDF = S_A_QDF(i,:);
    switch L_A(i)
        case 1
            T_A_LVQ(i,:) = findTargetLVQ(x_LVQ, m1);
            T_A_QDF(i,:) = findTargetQDF(x_QDF, u1, segma1);
        case 2
            T_A_LVQ(i,:) = findTargetLVQ(x_LVQ, m2);
            T_A_QDF(i,:) = findTargetQDF(x_QDF, u2, segma2);
        case 3
            T_A_LVQ(i,:) = findTargetLVQ(x_LVQ, m3);
            T_A_QDF(i,:) = findTargetQDF(x_QDF, u3, segma3);
    end
end
f_A = a*f_A;
%% --(4) begin self training-- %%
A_LVQ = eye(m,m);                   % m = 310
A_QDF = eye(m,m);
b_LVQ = zeros(m,1);
b_QDF = zeros(m,1);
acc_iter = zeros(iterNum,2);      
for iter = 1:1:iterNum
    for i = 1:1:nb_B                % in S_B
        x_LVQ = S_B_LVQ(i,:);       % a sample in S_B_LVQ
        x_QDF = S_B_QDF(i,:);       % a sample in S_B_QDF
        x_mapped_LVQ = (A_LVQ*x_LVQ' + b_LVQ)';
        x_mapped_QDF = (A_QDF*x_QDF' + b_QDF)';
        [y_arrow_LVQ, ~, decision_value_LVQ] = svmpredict(1, x_mapped_LVQ, svm);    % y_arrow is the predicted label for x, the para '1' is nonsense
        [y_arrow_QDF, ~, decision_value_QDF] = svmpredict(1, x_mapped_QDF, svm);
        switch y_arrow_LVQ
            case 1
                T_B_LVQ(i,:) = findTargetLVQ(x_LVQ, m1);
            case 2
                T_B_LVQ(i,:) = findTargetLVQ(x_LVQ, m2);
            case 3
                T_B_LVQ(i,:) = findTargetLVQ(x_LVQ, m3);
        end
        switch y_arrow_QDF
            case 1
                T_B_QDF(i,:) = findTargetQDF(x_QDF, u1, segma1);
            case 2
                T_B_QDF(i,:) = findTargetQDF(x_QDF, u2, segma2);
            case 3
                T_B_QDF(i,:) = findTargetQDF(x_QDF, u3, segma3);
        end
        % (1) the original confidence setup
        %f_B_LVQ(i) = confidence((A_LVQ*x_LVQ'+b_LVQ)', m1, m2, m3);
        %f_B_QDF(i) = f_B_LVQ(i);
        %f_B_QDF(i) = confidence((A_QDF*x_QDF'+b_QDF)', u1, u2, u3);
        %f_B_LVQ(i) = f_B_QDF(i);
        % (2) my proposed confidence setup
        Q_LVQ = 0.2*abs(decision_value_LVQ(:,1))+0.2*abs(decision_value_LVQ(:,2))+0.6*abs(decision_value_LVQ(:,3));
        Q_QDF = 0.2*abs(decision_value_QDF(:,1))+0.2*abs(decision_value_QDF(:,2))+0.6*abs(decision_value_QDF(:,3));
        f_B_LVQ(i) = 1/(1+exp(-Q_LVQ+1));
        f_B_QDF(i) = 1/(1+exp(-Q_QDF+1));
        % (3) no confidence setup, i.e., set all confidence 0.8
        %f_B_LVQ(i) = 0.8;
        %f_B_QDF(i) = 0.8;
    end
    S_LVQ = [S_A_LVQ;S_B_LVQ];
    S_QDF = [S_A_QDF;S_B_QDF];
    T_LVQ = [T_A_LVQ;T_B_LVQ];
    T_QDF = [T_A_QDF;T_B_QDF];
    f_LVQ = [f_A;f_B_LVQ];
    f_QDF = [f_A;f_B_QDF];
    % see here, the penalty in unsupervised STM should be bigger
    beta_coeff = 2;
    gamma_coeff = 2;
    % delete until here
    [A_LVQ, b_LVQ] = calculate_A_b(S_LVQ, T_LVQ, f_LVQ, beta_coeff, gamma_coeff);
    [A_QDF, b_QDF] = calculate_A_b(S_QDF, T_QDF, f_QDF, beta_coeff, gamma_coeff);
    %%%%%%  below is to embed the test procedure into the iteration, delete
    %%%%%%  after evaluating the accuracy during iteration
    %{
    A_LVQ_total = A_LVQ*A0_LVQ;
    b_LVQ_total = A_LVQ*b0_LVQ + b_LVQ;
    A_QDF_total = A_QDF*A0_QDF;
    b_QDF_total = A_QDF*b0_QDF + b_QDF;
    S_transfered_LVQ = zeros(nb_A+nb_B, m);
    S_transfered_QDF = zeros(nb_A+nb_B, m);
    for i = 1:1:nb_A+nb_B
        S_transfered_LVQ(i,:) = (A_LVQ_total*S(i,:)' + b_LVQ_total)';
        S_transfered_QDF(i,:) = (A_QDF_total*S(i,:)' + b_QDF_total)';
    end
    [L_predicted_LVQ, ~, ~] = svmpredict(ones(nb_A+nb_B,1), S_transfered_LVQ, svm);
    [L_predicted_QDF, ~, ~] = svmpredict(ones(nb_A+nb_B,1), S_transfered_QDF, svm);
    L_B_predicted_LVQ = L_predicted_LVQ(nb_A+1:nb_A+nb_B,:);
    L_B_predicted_QDF = L_predicted_QDF(nb_A+1:nb_A+nb_B,:);
    num_lvq = 0;
    num_qdf = 0;
    for w = 1:1:nb_B
        if L_B_predicted_LVQ(w) == L_B(w)
            num_lvq = num_lvq + 1;
        end
        if L_B_predicted_QDF(w) == L_B(w)
            num_qdf = num_qdf + 1;
        end
    end
    q1 = num_lvq/nb_B;
    q2 = num_qdf/nb_B;
    acc_iter(iter,:) = [q1, q2];
    %%%%%% delete till here
    %}
end
acc_all = [acc_supervised;acc_iter];
A_LVQ_total = A_LVQ*A0_LVQ;
b_LVQ_total = A_LVQ*b0_LVQ + b_LVQ;
A_QDF_total = A_QDF*A0_QDF;
b_QDF_total = A_QDF*b0_QDF + b_QDF;
S_transfered_LVQ = zeros(nb_A+nb_B, m);
S_transfered_QDF = zeros(nb_A+nb_B, m);
for i = 1:1:nb_A+nb_B
    S_transfered_LVQ(i,:) = (A0_LVQ*S(i,:)' + b0_LVQ)';
    S_transfered_QDF(i,:) = (A0_QDF*S(i,:)' + b0_QDF)';
end
[L_predicted_LVQ, ~, ~] = svmpredict(ones(nb_A+nb_B,1), S_transfered_LVQ, svm);
[L_predicted_QDF, ~, ~] = svmpredict(ones(nb_A+nb_B,1), S_transfered_QDF, svm);
L_B_predicted_LVQ = L_predicted_LVQ(nb_A+1:nb_A+nb_B,:);
L_B_predicted_QDF = L_predicted_QDF(nb_A+1:nb_A+nb_B,:);
end