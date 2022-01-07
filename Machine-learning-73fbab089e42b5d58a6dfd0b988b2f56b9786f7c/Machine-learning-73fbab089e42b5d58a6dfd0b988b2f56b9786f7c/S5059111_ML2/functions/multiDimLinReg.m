function W = multiDimLinReg(X,t)


    W=pinv(X'*X)*X'*t;
end

