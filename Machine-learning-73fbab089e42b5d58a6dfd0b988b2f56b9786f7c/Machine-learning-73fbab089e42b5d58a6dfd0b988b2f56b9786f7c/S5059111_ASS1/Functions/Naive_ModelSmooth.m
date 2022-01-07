function [p_class, p_feature_class] = NaiveModelSmooth(my_set, my_res, CLASSES, LEVELS)
[EXAMPLES, FEATURES] = size(my_set);
max_n_levels=max(LEVELS);
N_c = zeros(CLASSES, 1); 
N_jc = zeros(FEATURES, max_n_levels, CLASSES);
for e = 1:EXAMPLES 
c = my_res(e); 
N_c(c) = N_c(c)+1; 
for f = 1:FEATURES
for l = 1:LEVELS(f) 
if my_set(e,f)==l
N_jc(f,l,c) = N_jc(f,l,c)+1; 
end
end
end
end
p_class = zeros(CLASSES, 1);
for c = 1:CLASSES
p_class(c) = N_c(c) / EXAMPLES;
end
p_feature_class = zeros(FEATURES, max_n_levels, CLASSES);
a=1;
for c = 1:CLASSES 
for f = 1:FEATURES 
for l = 1:max_n_levels
p_feature_class(f, l, c) = (N_jc(f, l, c) + a) / ( N_c(c) + a * LEVELS(f));
end
end
end    
end
