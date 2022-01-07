function images(algorithm,datasetName,confusionMat,k)
% This function creates and saves tables in jpg form
% Inputs are used to name the image.

    f = figure('units','normalized','outerposition',[0 0 1 1], 'visible', 'off');
    sInd=[1,2;3,4];
    clrs = {'Yellow','White','White','Yellow'};
    
    % Concatenate html strings
    outHtml = strcat('<html><table border=0 width=800 bgcolor=', ...
    clrs(sInd), ... % Choose the appropriate color for each cell
    '"><TR><TD>', ...
    cellfun(@num2str,num2cell(confusionMat),'UniformOutput',false), ... % Convert num data to cell of chars
    '</TD></TR></body></html>');

    % Place this in a table
    VariableNames = {'Positive outcome','Negative outcome'};
    RowNames = {'Positive label','Negative label'}; 
    
    % Create uitable and save image
    u = uitable(f,'Data',outHtml,'RowName',RowNames,'ColumnName',VariableNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
    figName=sprintf('Results/%s/%s_with_k=%d_.jpg',datasetName, algorithm, k);
    saveas(f, figName)
    
end

