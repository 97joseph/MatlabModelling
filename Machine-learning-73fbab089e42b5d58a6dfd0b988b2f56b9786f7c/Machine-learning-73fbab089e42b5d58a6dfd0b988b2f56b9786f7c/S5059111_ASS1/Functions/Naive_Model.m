function [p_class, p_feature_class] = NaiveModel(my_set, my_res, CLASSES, LEVELS)
[EXAMPLES, FEATURES] = size(my_set);
max_n_levels=max(LEVELS); 
N_c = zeros(CLASSES, 1);  % To count the number of instances of classes in these
N_jc = zeros(FEATURES, max_n_levels, CLASSES);  % To count the number of instances of features for classes
    
%% Analyze the data set
for e = 1:EXAMPLES 
c=my_res(e); 
N_c(c)=N_c(c)+1;
for f = 1:FEATURES 
for l = 1:LEVELS(f) 
if my_set(e,f)==l
N_jc(f,l,c)= N_jc(f,l,c)+1; 
end
end
end
end
p_class = zeros(CLASSES, 1);
for c = 1:CLASSES
p_class(c) = N_c(c) / EXAMPLES;
end
p_feature_class = zeros(FEATURES, max_n_levels, CLASSES);
for c = 1:CLASSES 
for f = 1:FEATURES 
for l = 1:max_n_levels
p_feature_class(f, l, c) = (N_jc(f, l, c)) / ( N_c(c));
end
end
end    
for f=1:FEATURES
for c=1:CLASSES
if sum(p_feature_class(f,:,c))>=1.001 || sum(p_feature_class(f,:,c))<0.999
                disp('Probability may be wrong');
end
end
end
end
