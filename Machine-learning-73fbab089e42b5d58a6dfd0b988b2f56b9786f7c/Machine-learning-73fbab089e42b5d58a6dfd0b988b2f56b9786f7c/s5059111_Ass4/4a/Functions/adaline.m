function [myconfusionMatrix,w]=adaline(x_total,eta,num_sets)


    %% Init
    confusionMatrix=zeros(2,2,num_sets);

    %% For loop for each couple of train+test set 
    for sets=1:num_sets
        
        % Take the training set
        x=x_total.train{sets};
        t=x_total.trainlab{sets};

        % Create random weights
        w = rand(size(x_total.train{1},2),1);
        xl = 1;
        
        for var=1:100000
            

            %% Take pattern xl and compute y & delta
            % Compute output r = w x         
            r =  x(xl,:) * w;
            % Compute output error 
            delta = 0.5 * ( t(xl) - r );

            %% Update w
            w = w + eta * delta * x(xl,:)' * (1/size(x_total.train{1},1));

            %% Update xl
            xl=xl+1;
            if xl>=size(x_total.train{1},1)
                xl=1;
            end
            
        end
        
        %% Take the test set
        xtest=x_total.test{sets};
        xtestlab=x_total.testlab{sets};
        myPrediction=sign(xtest*w);
        
        %% Create the confusion matrix
        for i=1:size(xtest,1)
            if myPrediction(i)==1 && xtestlab(i)==1
                confusionMatrix(1,1,sets)=confusionMatrix(1,1,sets)+1;
            elseif myPrediction(i)==(-1) && xtestlab(i)==1
                confusionMatrix(1,2,sets)=confusionMatrix(1,2,sets)+1;
            elseif myPrediction(i)==1 && xtestlab(i)==(-1)
                confusionMatrix(2,1,sets)=confusionMatrix(2,1,sets)+1;
            elseif myPrediction(i)==(-1) && xtestlab(i)==(-1)
                confusionMatrix(2,2,sets)=confusionMatrix(2,2,sets)+1;
            end
        end
        
    end
    
    %% Transform matrix values in percentages
    confusionMatrix=(confusionMatrix/size(x_total.test{1},1))*100; % tests all have same size
    
    %% Average matrix
    for i=1:4
        myconfusionMatrix(1,1)=mean(confusionMatrix(1,1,:));
        myconfusionMatrix(1,2)=mean(confusionMatrix(1,2,:));
        myconfusionMatrix(2,1)=mean(confusionMatrix(2,1,:));
        myconfusionMatrix(2,2)=mean(confusionMatrix(2,2,:));
    end
    
end