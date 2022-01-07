function objective = meanSquareError(x,t,w1,w0,problem)

    N = length(x);
    if problem == 1
        y = w1*x + w0; 
        objective = (1/N)*sum((t-y).^2); 
    end
   
    if problem == 2 
        y = x*w1; 
        objective = immse(t,y); 
    end

end

