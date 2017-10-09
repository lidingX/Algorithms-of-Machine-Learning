data = load('data.txt');
data = data';
inputs = data(1:3,:);
targets = data(4,:) + 1;
[~,n] = size(inputs);

for t = 2:5
    display(t);
    training_inputs = [];
    training_targets = [];
    validation_inputs = [];
    validation_targets = [];
    for i = 1:n
        if mod(i,t) == 0
            validation_inputs = [validation_inputs inputs(:,i)];
            validation_targets = [validation_targets targets(:,i)];
        else
            training_inputs = [training_inputs inputs(:,i)];
            training_targets = [training_targets targets(:,i)];
        end
    end
     runkfisher(training_inputs,training_targets, validation_inputs, validation_targets,'knGauss',1,1);
     runkfisher(training_inputs,training_targets, validation_inputs, validation_targets,'knPoly',3,2);
end



function runkfisher(training_inputs,training_targets, validation_inputs, validation_targets,kernel,kpar1,kpar2);
    count = 0;
    model = kfisher(training_inputs,training_targets,1e-9,kernel,kpar1,kpar2);
    y = kfisherPred(model,validation_inputs);
    [~,n] = size(validation_targets);
    for i=1:n
         if y(i)~= validation_targets(i)
            count = count + 1;
        end
    end
    fprintf('%sfisher: rate:%f count:%d\n ',kernel,count/n,count);
end