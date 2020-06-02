clear all; clc;
semi = zeros(14,3);
for subjectIndex = 1:1:14
    [acc, acc_transfered_LVQ, acc_transfered_QDF] = multiSourceClassifier(subjectIndex, 1);
    semi(subjectIndex,:) = [acc, acc_transfered_LVQ, acc_transfered_QDF];
    disp('finish one');
end

