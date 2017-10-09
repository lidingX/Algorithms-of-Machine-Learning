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
     runSVM(training_inputs,training_targets, validation_inputs, validation_targets,'knGauss',1,1);
     runSVM(training_inputs,training_targets, validation_inputs, validation_targets,'knPoly',10,2);
     runSVM(training_inputs,training_targets, validation_inputs, validation_targets,'knPoly',3,1);
     runSVM(training_inputs,training_targets, validation_inputs, validation_targets,'knPoly',1,0);
end



function runSVM(training_inputs,training_targets, validation_inputs, validation_targets,kernel,kp1,kp2)
    count = 0;
    [alpha, b, w, ~, stp, flag] = smoSVM(training_inputs,training_targets,kernel,kp1,kp2,1,1e-4,20000,1e-5,2);
    w = w/norm(w);
    [sv_n, v_n, y] = svmPred(training_inputs, training_targets,validation_inputs, alpha, b, kernel);
    [~,n] = size(validation_targets);
    for i=1:n
         if y(i)~= validation_targets(i)
            count = count + 1;
        end
    end
    fprintf('%s-%d-%d-SVM: rate:%f count:%d step:%d flag:%d\n ',kernel,kp1,kp2,count/n,count,stp,flag);
end