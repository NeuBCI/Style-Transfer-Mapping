function f = confidence(x, u1, u2, u3)
% x is one sample, u1~u3 are prototypes belonging to the three emotion
% classes. f is a scalar, showing the confidence.
c1 = size(u1,1);
c2 = size(u2,1);
c3 = size(u3,1);
dis1 = zeros(c1,1);
dis2 = zeros(c2,1);
dis3 = zeros(c3,1);
for i = 1:1:c1
    y = u1(i,:);
    dis1(i) = norm(x-y);
end
for i = 1:1:c2
    y = u2(i,:);
    dis2(i) = norm(x-y);
end
for i = 1:1:c3
    y = u3(i,:);
    dis3(i) = norm(x-y);
end
[min_pos,~] = min(dis1);
[min_neu,~] = min(dis2);
[min_neg,~] = min(dis3);
D = [min_pos, min_neu, min_neg];
D = sort(D); % ascending order
Q = (D(2)-D(1));
f = 1/(exp(-Q+1));
end

