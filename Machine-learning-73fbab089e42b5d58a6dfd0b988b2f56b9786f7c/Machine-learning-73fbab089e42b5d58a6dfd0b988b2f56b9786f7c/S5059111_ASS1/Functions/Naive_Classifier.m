function p_class_feature = NaiveClassifier(test_set, p_c, p_feature_class, CLASSES)
N_TEST=size(test_set,1);
p_class_feature = zeros(N_TEST, CLASSES); 
[N_TEST, FEATURES] = size(test_set);
prod=ones(N_TEST,CLASSES);
for e = 1:N_TEST 
for f=1:FEATURES 
l=test_set(e,f); % level that taken from data set values
for c=1:CLASSES
prod(e,c) = prod(e,c) * p_feature_class(f,l,c); 
end
end
for c = 1:CLASSES
p_class_feature(e, c) = prod(e,c) * p_c(c); 
end
end        
end
