% TODO: what if the image already is greyscale? there's also a bit of
% variance introduced by this function, but I mean... it looks identical to
% the image produced by the official function? I don't think it's
% important. It also feels a bit slower, but that's definitely not a
% priority
% Description: convert a given image to greyscale
%
% Inputs:
% im: an image
% 
% Outputs: 
% im_g: a greyscale image consisting of uint8 values
function im_g = my_im2gray(im)
    % go through all the pixels
    for i = 1:size(im, 1)
        for j = 1:size(im, 2)
            % store the pixel converted to greyscale, then cast it to the
            % required data type (which also solves rounding). Use the
            % official matlab weightings to achieve an identical result
            im_g(i,j,:) = cast(sum([0.2989*im(i,j,1),0.587*im(i,j,2),0.114*im(i,j,3)]),'uint8');
        end
    end
end