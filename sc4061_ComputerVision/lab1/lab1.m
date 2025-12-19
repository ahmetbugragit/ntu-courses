%----------------2.1---------------------------

Pc = imread('mrt-train.jpg'); 
whos Pc
P = rgb2gray(Pc); 
%figure;imshow(P);


 min_P=min(P(:));   %min_P=13
 max_P=max(P(:));  %max_P=204

P2= imsubtract(double(P),double(min_P));
P2= uint8(immultiply(P2,255/double((max_P-min_P))));
%as learnt in lecture

%figure;imshow(P2);

max_P2=max(P2(:)); %max_P2=255
min_P2= min(P2(:)); %min_P2= 0











%---------------------------------------2.2( Histogram Equalization )--------------------------------------------

%figure; imhist(P,10);
%figure;imhist(P,256);



P3 = histeq(P,255); 
figure;imhist(P3,10);
figure;imhist(P3,256);
%imshow(P3);

P4 = histeq(P3,255); 
%figure;imhist(P4,10);
%figure;imhist(P4,256);
% Display the original and processed images for comparison
figure; 
subplot(1, 3, 1); imshow(P); title('Original Image');
subplot(1, 3, 2); imshow(P3); title('Histogram Equalized Image 1');
subplot(1, 3, 3); imshow(P4); title('Histogram Equalized Image 2');



%--------------------------------2.3(spatial)--------------------------------


size = 5;
[x,y] = meshgrid(-(size-1)/2:(size-1)/2, -(size-1)/2:(size-1)/2);

sigma1 = 1.0;
h1 = (1/(2*pi/(sigma1.^2)))*exp(-(x.^2 + y.^2) / (2*sigma1.^2));
h1 = h1 / sum(h1(:));   % normalize

sigma2 = 2.0;
h2 = (1/(2*pi*(sigma2.^2)))*exp(-(x.^2 + y.^2) / (2*sigma2.^2));
h2 = h2 / sum(h2(:));

figure; mesh(h1); title('5x5 Gaussian, sigma=1');
figure; mesh(h2); title('5x5 Gaussian, sigma=2');


Ig = imread('lib-gn.jpg');

Ig_h1 = uint8(conv2(double(Ig), h1, 'same'));
Ig_h2 = uint8(conv2(double(Ig), h2, 'same'));

%figure; montage({Ig, Ig_h1, Ig_h2}, 'Size',[1 3]);
%title('Gaussian noise: Original | σ=1 | σ=2');
figure; imshow(Ig_h1);
figure; imshow(Ig_h2);


Ig_sp = imread('lib-sp.jpg');

Igsp_h1 = uint8(conv2(double(Ig_sp), h1, 'same'));
Igsp_h2 = uint8(conv2(double(Ig_sp), h2, 'same'));

%figure; montage({Ig_sp, Igsp_h1, Igsp_h2}, 'Size',[1 3]);
%title('speckle noise: Original | σ=1 | σ=2');

figure; imshow(Igsp_h1);
figure; imshow(Igsp_h2);




%-----------------------------2.4(median_filter)----------------------------






Ig_mf = imread('lib-gn.jpg');

Ig_mf_3=uint8(medfilt2(Ig_mf,[3 3]));
Ig_mf_5=uint8(medfilt2(Ig_mf,[5 5]));

%figure; montage({Ig_mf, Ig_mf_3, Ig_mf_5}, 'Size',[1 3]);
%title('Gaussian noise: Original | 3x3 | 5x5');

figure;imshow(Ig_mf_3);
figure; imshow(Ig_mf_5);
Ig_sp_mf = imread('lib-sp.jpg');
Ig_sp_mf_3=uint8(medfilt2(Ig_sp_mf,[3 3]));
Ig_sp_mf_5=uint8(medfilt2(Ig_sp_mf,[5 5]));

%figure; montage({Ig_sp_mf, Ig_sp_mf_3, Ig_sp_mf_5}, 'Size',[1 3]);
%title('spectle noise: Original | 3x3 | 5x5');

figure;imshow(Ig_sp_mf_3);
figure; imshow(Ig_sp_mf_5);




%-------------------------2.5-----------------------------------

or_im=imread("pck-int.jpg");

orim_ft = fft2(double(or_im));    % complex-valued spectrum
orim_ft2 = abs(orim_ft).^2; 



figure;
imagesc(fftshift(orim_ft2.^0.1));
colormap default;
title('Power spectrum (fftshift, S^{0.1})');



figure;
imagesc(orim_ft2.^0.1);  colormap default;
title('Power spectrum WITHOUT fftshift (S^{0.1}) -- click the peaks');

[x, y] = ginput(2); 
x = round(x); y = round(y);

fprintf('Peak #1 at (col = %d, row = %d)\n', x(1), y(1));
fprintf('Peak #2 at (col = %d, row = %d)\n', x(2), y(2));






I = imread('pck-int.jpg');

F = fft2(double(I));             % complex spectrum (UNSHIFTED)
S = abs(F).^2;                   % power spectrum (for reference)

pe_po1 = [241,   9];
pe_po2 = [17, 249 ];
pe_po3=  [1,      1];


rad = 2;


%[H, W] = size(F);
H=256;
W=256;

r1_a = max(1,     pe_po1(1)-rad);
r1_b = min(H,     pe_po1(1)+rad);
c1_a = max(1,     pe_po1(2)-rad);
c1_b = min(W,     pe_po1(2)+rad);


r2_a = max(1, pe_po2(1)-rad);
r2_b = min(H, pe_po2(1)+rad);
c2_a = max(1, pe_po2(2)-rad);
c2_b = min(W, pe_po2(2)+rad);


r3_a = max(1, pe_po3(1)-rad);
r3_b = min(H, pe_po3(1)+rad);
c3_a = max(1, pe_po3(2)-rad);
c3_b = min(W, pe_po3(2)+rad);


%UNSHIFTED
F(r1_a:r1_b, c1_a:c1_b) = 0;
F(r2_a:r2_b, c2_a:c2_b) = 0;
%F(r3_a:r3_b, c3_a:c3_b) = 0;


S2 = abs(F).^2;

figure;
imagesc(fftshift(S2.^0.1));  colormap default;
title('Power spectrum after zeroing 5x5 around peaks (shifted view)');


Irec = real(ifft2(F));
Irec2=uint8(Irec);
figure; montage({I,Irec2}, 'Size',[1 2]);
title('Interference removal: Original | Filtered (5x5 notches)');







prim=imread("primate-caged.jpg");

prim=rgb2gray(prim);
 A= fft2(double(prim));    % complex-valued spectrum
B = abs(A).^2; 
figure;
imagesc(B.^0.1);  colormap default;
title('Power spectrum WITHOUT fftshift (S^{0.1}) -- click the peaks');
%{
[x, y] = ginput(2);  
x = round(x); y = round(y);

fprintf('Peak #1 at (col = %d, row = %d)\n', x(1), y(1));
fprintf('Peak #2 at (col = %d, row = %d)\n', x(2), y(2));

%}

pe_po1=[252,11];
pe_po2=[10,236];

% Define the radius for zeroing around peaks
rad = 2;

r1_a = max(1, pe_po1(1)-rad);
r1_b = min(256, pe_po1(1)+rad);
c1_a = max(1, pe_po1(2)-rad);
c1_b = min(256, pe_po1(2)+rad);

r2_a = max(1, pe_po2(1)-rad);
r2_b = min(256, pe_po2(1)+rad);
c2_a = max(1, pe_po2(2)-rad);
c2_b = min(256, pe_po2(2)+rad);


% 6) Zero out the 5x5 neighborhoods around the detected peaks
A(r1_a:r1_b, c1_a:c1_b) = 0;
A(r2_a:r2_b, c2_a:c2_b) = 0;


% 7) Recalculate the power spectrum after zeroing the peaks
B2 = abs(A).^2;

figure;
imagesc(fftshift(B2.^0.1));  colormap default;
title('Power spectrum after zeroing 5x5 around peaks (shifted view)');


Irec = real(ifft2(A));
Irec2=uint8(Irec);
figure; montage({prim,Irec2}, 'Size',[1 2]);
title('Interference removal: Original | Filtered (5x5 notches)');















%------------------------------2.6-----------------------------------


% Read the image
I = imread('book.jpg');

% Show and pick 4 corners (clockwise)
imshow(I);
title('Click 4 corners');
[x, y] = ginput(4);    

% Target rectangle size (e.g. A4: 210x297)
X = [0 210 210 0];   
Y = [0 0 297 297];   

% Build A matrix and v vector
A = zeros(8, 8); 
v = zeros(8, 1);

for i = 1:4
    A(2*i-1, :) = [x(i) y(i) 1 0 0 0 -X(i)*x(i) -X(i)*y(i)];
    v(2*i-1)   = X(i);

    A(2*i, :)  = [0 0 0 x(i) y(i) 1 -Y(i)*x(i) -Y(i)*y(i)];
    v(2*i)     = Y(i);
end

% Solve A * u = v
u = A \ v;

% Get 3x3 homography matrix
U = reshape([u;1], 3, 3)';

% Transform clicked points (for check)
w = U * [x'; y'; ones(1,4)];
w = w ./ (ones(3,1) * w(3,:));
disp(w);

% Apply transform to image
T = maketform('projective', U'); 
I2 = imtransform(I, T, 'XData', [0 210], 'YData', [0 297]);

% Show result
figure; imshow(I2);
title('Corrected Image');



% Read the image
img = imread('corrected_book.jpg');

% Show the image
imshow(img);
title('Original Image');

% Convert RGB image to HSV
hsv_img = rgb2hsv(img);

% Split HSV channels
h = hsv_img(:,:,1);  % hue
s = hsv_img(:,:,2);  % saturation
v = hsv_img(:,:,3);  % brightness

% Create a mask for pink color (hue around 0 or 1)
mask1 = h < 0.06;
mask2 = h > 0.85;
mask = (mask1 | mask2);

% Check saturation and brightness levels
mask = mask & s > 0.32;
mask = mask & v > 0.32;

% Remove noise from the borders
mask(1:20, :) = 0;
mask(end-20:end, :) = 0;
mask(:, 1:20) = 0;
mask(:, end-20:end) = 0;

% Show the pink area mask
figure;
imshow(mask);
title('Pink Area Mask');

% Label connected regions in the mask
[label_img, num_labels] = bwlabel(mask);

% Find the biggest region
max_area = 0;
biggest_label = 0;

for i = 1:num_labels
    current_area = sum(label_img(:) == i);
    if current_area > max_area
        max_area = current_area;
        biggest_label = i;
    end
end

% Keep only the largest region
clean_mask = (label_img == biggest_label);

% Find bounding box of the region
[rows, cols] = find(clean_mask);
xmin = min(cols);
xmax = max(cols);
ymin = min(rows);
ymax = max(rows);

% Draw rectangle on the original image
figure;
imshow(img);
title('Detected Pink Area');
hold on;
rectangle('Position', [xmin ymin xmax - xmin ymax - ymin], ...
          'EdgeColor', 'g', 'LineWidth', 2);
hold off;





%----------------------------------------------2.7----------------------------------


clc; clear;

% Learning rate
alpha = 1;

% Sample 1 → belongs to Class +1
x1 = [3 3 1];    % Bias term added
r1 = 1;

% Sample 2 → belongs to Class -1
x2 = [1 1 1];    % Bias term added
r2 = -1;

%% ---------------- Algorithm 1: Perceptron Learning Rule ---------------- %%
w1 = [0 0 0];    % Initial weights for Algorithm 1

for step = 1:4
    if mod(step, 2) == 1
        xk = x1;
        rk = r1;
    else
        xk = x2;
        rk = r2;
    end

    yk = dot(w1, xk);    % Prediction

    % Apply PLA rules
    if (rk == 1 && yk <= 0)
        w1 = w1 + alpha * xk;  % Misclassified as -1, correct to +1
    elseif (rk == -1 && yk >= 0)
        w1 = w1 - alpha * xk;  % Misclassified as +1, correct to -1
    end
end

%% ---------------- Algorithm 2: Gradient-based Update (Slide 5.3) ---------------- %%
w2 = [0 0 0];    % Initial weights for Algorithm 2

for step = 1:4
    if mod(step, 2) == 1
        xk = x1;
        rk = r1;
    else
        xk = x2;
        rk = r2;
    end

    yk = dot(w2, xk);    % Prediction

    % Update using error signal (r - y) * x
    w2 = w2 + alpha * (rk - yk) * xk;
end

%% ---------------- Display Results ---------------- %%
disp('Final weights using Algorithm 1 (PLA):');
disp(w1);

disp('Final weights using Algorithm 2 (Gradient-based from Slide):');
disp(w2);
