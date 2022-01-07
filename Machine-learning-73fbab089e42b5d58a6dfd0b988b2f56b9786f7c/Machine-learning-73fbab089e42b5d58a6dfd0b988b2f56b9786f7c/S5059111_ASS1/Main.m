addpath('Functions')
addpath('Data')
clear all
clc 
close all
data_set = load("weather_set_numbers_S5059111.xls");
for i=1:size(data_set,1)
for j=1:size(data_set,2)
if data_set(i,j) < 1 
disp('Error:  one value is less then 1');
return;
end
end
end
[EXAMPLES, FEATURES] = size(data_set);
FEATURES=FEATURES-1;
LEVELS=zeros(1,FEATURES);
for i=1:FEATURES
temp = unique(data_set(:,i));
LEVELS(i) = length(temp);        
end
CLASSES = length(unique(data_set(:,end)));
fprintf('\n\n#1#This class has %d examples with observations divided in %d different features.\n#2#Every feature  have  numebr of levels.', EXAMPLES, FEATURES);
fprintf('\n#3# observations can be result of  %d types.\n', CLASSES)
Feature=(1:1:FEATURES)';
Level=LEVELS';
table_of_Features=table(Feature,Level)
%% dataset rows 
idx = randperm(EXAMPLES);
data_set = data_set(idx,:);
prompt = '\n?\n';
N_train = input(prompt);

if N_train >14
    disp('input number exeeds the no of lines');
    return;
elseif N_train==14
    disp('Save some lines to test your classifier!');
    return;
end
train_set = data_set(1:N_train, 1:end-1);
train_res = data_set(1:N_train, end);
% Test set
test_set = data_set(N_train+1:end, 1:end-1);
test_res = data_set(N_train+1:end, end);

%% Evaluate the model 
[p_c, p_f_c] = NaiveModel(train_set, train_res, CLASSES, LEVELS);

% Smoothed model
[p_c_smooth, p_f_c_smooth] = NaiveModelSmooth(train_set, train_res, CLASSES, LEVELS);

%% Test the model 

prob_c = NaiveClassifier(test_set, p_c, p_f_c, CLASSES);
prob_c_smooth = NaiveClassifier(test_set, p_c_smooth, p_f_c_smooth, CLASSES);
my_guess=strings(EXAMPLES-N_train,1);
test_res_word=strings(EXAMPLES-N_train,1);

% Base model
for i = 1 : (EXAMPLES-N_train)
        [unused, my_guess_val] = max(prob_c(i, :)); 
        if my_guess_val==2
          my_guess(i)='yes';
        else 
          my_guess(i)='no';
        end
end

my_guess_s=strings(EXAMPLES-N_train,1);

% Smooth model
for i = 1 : (EXAMPLES-N_train)
    
    [unused, my_guess_val] = max(prob_c_smooth(i, :)); 
    if my_guess_val==2
      my_guess_s(i)='yes';
    elseif my_guess_val==1
      my_guess_s(i)='no';
    end
    
    if test_res(i)==2
      test_res_word(i)='yes';
    elseif test_res(i)==1
      test_res_word(i)='no';
    end
    
end

    
%% CASE1: I have the results of the test set
if size(test_set, 2) > size(train_set, 2) - 1 

    accuracy = evaluateAccuracy(prob_c, test_res);
    accuracy_s = evaluateAccuracy(prob_c_smooth, test_res);
    fprintf('THe results are:');
    table(test_set,test_res_word,my_guess)
    fprintf('Accuracy of the result  is %d%%\n\n\n', accuracy);

    fprintf('The result  with smoothing are:');
    table(test_set,test_res_word,my_guess_s)
    fprintf('To result  the smoothing %d%% \n', accuracy_s);
   
elseif size(test_set, 2) == size(train_set, 2) - 1
    
    fprintf('The results are:');
    table(my_guess,test_set)

    fprintf('the results with smoothing are:');
    table(my_guess_s,test_set)
else
  disp('Error:  less features than  the data set');
  return;
end

