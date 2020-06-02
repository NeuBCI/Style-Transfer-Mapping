function T = findTargetLVQ(x, D)
[c, ~] = size(D); % how many prototypes, i.e., 15
dis = zeros(c,1); % 15 x 1
for j = 1:1:c
    y = D(j,:);
    dis(j) = norm(x-y);
end
[~,id] = min(dis);
T = D(id,:);
end

