% Description: prepare data based on which investigation is taking place
%
% Inputs: x: the feature number, y: see run_investigation, im: an image, words: bag of visual words,
% optional
% 
% Outputs: f: the extracted feature
function f = get_feature(x, y, im, words)
    % make image greyscale to simplify processing complexity
    if y == 0
        im = im2gray(im);
    else
        im = my_im2gray(im);
    end
    switch x
        case 2
            % calculate average brightness of whole image
            f = sum(im(1:1:end)) / (size(im, 1) * size(im, 2) * size(im, 3));
        case 3
            % generate gradients from the image using the prewitt method
            [Gx, Gy] = imgradientxy(im, 'Prewitt');
            % apply thresholds to generated magnitudes in order to detect clear
            % edges
            Gx = abs(Gx)>=45;
            Gy = abs(Gy)>=45;
            % collect decimals representing the amount of horizontal and vertical
            % edges
            f = [(sum(sum(Gx)) / (size(Gx, 1) * size(Gx, 2))) (sum(sum(Gy)) / (size(Gy, 1) * size(Gy, 2)))];
        case 4
            % extract hog features using typical settings
            f = extractHOGFeatures(im, 'CellSize', [16 16], ...
            'BlockSize', [floor(size(im,1)/16) floor(size(im,2)/16)], ...
            'UseSignedOrientation', true);
        case 5
            f = words.encode(im);
    end
end