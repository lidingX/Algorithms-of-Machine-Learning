data = load('data.txt');
data = data';
inputs = data(2:4,:);
targets = data(1,:);
[~,n] = size(inputs);

