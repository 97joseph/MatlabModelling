function [x_res,num_sets] = splitDataSet(x,k)

    [ROWS,~]=size(x);
    rand_idx=randperm(ROWS);
    x=x(rand_idx,:);
    
    %% Split in equally sized train set & test set
    if k==2    
        training_portion=[1:round(ROWS/2)];
        testing_portion=[(round(ROWS/2)+1):ROWS];
        x_res.train{1} = x(training_portion,1:(end-1));
        x_res.test{1} = x(testing_portion,1:(end-1));
        x_res.trainlab{1} = x(training_portion,end);
        x_res.testlab{1} = x(testing_portion,end);
        num_sets=1;
    
    %% Perform leave-one-out cross validation   
    elseif k==ROWS
        for i=1:ROWS
            x_temp=x;
            x_temp(i,:)=[];
            x_res.train{i}=x_temp(:,1:(end-1));
            x_res.test{i}=x(i,1:(end-1));
            x_res.trainlab{i}=x_temp(:,end);
            x_res.testlab{i}=x(i,end);
        end
        num_sets=ROWS;
        
    %% Perform k-fold cross validation
    elseif k>2 && k<ROWS 
       x2=[x; x];
       for i=1:ROWS
            training_portion=[1:ROWS/k];
            testing_portion=[ROWS/k + 1 : ROWS];
            x_res.train{i}=x2(training_portion,1:(end-1));
            x_res.test{i}=x2(testing_portion,1:(end-1));
            x_res.trainlab{i}=x2(training_portion,end);
            x_res.testlab{i}=x2(testing_portion,end);
       end
       num_sets=k;
    
    %% Output an error message and abort the run
    elseif k<2 && k>ROWS
        disp('Error'); 
    end
    
end