function accuracy = evaluateAccuracy(prob_c, test_res)
N_TEST = size(test_res, 1);
my_guess = zeros(N_TEST, 2); 
counter = 0;
for e = 1 : N_TEST[my_guess(e, 1), my_guess(e, 2)] = max(prob_c(e, :));  
if my_guess(e, 2) == test_res(e) 
counter = counter + 1;
end
end
accuracy = counter / N_TEST * 100;
end