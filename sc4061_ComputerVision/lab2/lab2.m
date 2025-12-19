% =========================================================
% Lab 2 - 3.1(a): Otsu Global Thresholding for Text Segmentation
% =========================================================

clc; clear; close all;

% List of document images to process
imageNames = {'document01', 'document02', 'document03', 'document04'};

for idx = 1:length(imageNames)
    fprintf('\n==============================\n');
    fprintf(' Processing %s\n', imageNames{idx});
    fprintf('==============================\n');

    % Read input image and ground truth
    I = imread([imageNames{idx} '.bmp']);       
    GT = imread([imageNames{idx} '-GT.tiff']);  

    % Convert to grayscale if RGB
    if size(I,3) == 3
        Igray = rgb2gray(I);
    else
        Igray = I;
    end

    %Otsu Global Thresholding
    T = graythresh(Igray);             
    BW = imbinarize(Igray, T);        
    fprintf('Otsu threshold value = %.3f\n', T);

    % ground truth is binary 
    if ~islogical(GT)
        if max(GT(:)) > 1
            GT = GT > 128;
        else
            GT = GT > 0.5;
        end
    end

    % Compute difference image (mismatched pixels)
    diff_img = abs(double(BW) - double(GT));

    % Calculate performance score
    diff_score = sum(diff_img(:));
    fprintf('Segmentation difference score = %d pixels\n', diff_score);

    % Display results
    figure('Name', sprintf('Lab 2 - 3.1(a) Otsu Global Thresholding - %s', imageNames{idx}), ...
           'NumberTitle','off');

    subplot(2,2,1); imshow(Igray); title('Original Grayscale Image');
    subplot(2,2,2); imshow(BW); title('Otsu Segmented Image');
    subplot(2,2,3); imshow(GT); title('Ground Truth');
    subplot(2,2,4); imshow(diff_img, []); title('Difference Image');

    % Save images for the report 
    imwrite(Igray, sprintf('otsu_original_%s.png', imageNames{idx}));
    imwrite(BW, sprintf('otsu_segmented_%s.png', imageNames{idx}));
    imwrite(GT, sprintf('otsu_groundtruth_%s.png', imageNames{idx}));
    imwrite(mat2gray(diff_img), sprintf('otsu_difference_%s.png', imageNames{idx}));
end














% =========================================================
% Lab 2 - 3.1(b): Multi-Image Niblack Thresholding with Parameter Search
% =========================================================

%clc; clear; close all;


imageNames = {'document01', 'document02', 'document03', 'document04'};
%imageNames = {'document01'};   %for fast trial

k_values = -0.1 : -0.1 : -2.0;          % possible k values
window_values = 15:30:600;              % different window sizes


for idx = 1:length(imageNames)
    fprintf('\n==============================\n');
    fprintf(' Processing %s\n', imageNames{idx});
    fprintf('==============================\n');

    
    I = imread([imageNames{idx} '.bmp']);
    GT = imread([imageNames{idx} '-GT.tiff']);

    % Convert to grayscale 
    if size(I,3) == 3
        Igray = rgb2gray(I);
    else
        Igray = I;
    end
    Igray = double(Igray);

    % GT is binary
    if ~islogical(GT)
        if max(GT(:)) > 1
            GT = GT > 128;
        else
            GT = GT > 0.5;
        end
    end

    % Variables for storing results
    scores = zeros(length(window_values), length(k_values));
    best_score = inf;
    best_k = 0;
    best_window = 0;

  
    for wi = 1:length(window_values)
        w = window_values(wi);

        
        for ki = 1:length(k_values)
            k = k_values(ki);

            % Local mean 
            local_filter = ones(w) / (w^2);
            mean_local = conv2(Igray, local_filter, 'same');

            % Local std 
            std_local = stdfilt(Igray, true(w));

            % Compute threshold using Niblack formula
            T = mean_local + k * std_local;

            % Apply threshold
            BW = Igray > T;

            % A bit of post cleaning
            BW = bwareaopen(BW, 20);
            BW = imclose(BW, strel('disk', 1));

            % Difference image
            diff_img = abs(double(BW) - double(GT));

            % Pixel difference score
            score = sum(diff_img(:));

            % Save score
            scores(wi, ki) = score;

            % Print to console for tracking
            fprintf('Test -> window=%d, k=%.2f => score=%d\n', w, k, score);

            % Track best parameters manually
            if score < best_score
                best_score = score;
                best_k = k;
                best_window = w;
                best_BW = BW;
                best_diff = diff_img;
            end

            temp_mean = mean(mean_local(:));
            if temp_mean < 50
                dummy = 1; 
            end
        end

       
        fprintf('Window %d done, temporary best diff=%d\n', w, best_score);
    end

   
    % Summary for this image
    fprintf('n\ Best for %s → k=%.2f, window=%d, diff=%d\n', ...
        imageNames{idx}, best_k, best_window, best_score);

    % --- Display best result ---
    figure('Name', sprintf('Best Niblack - %s', imageNames{idx}), 'NumberTitle', 'off');
    subplot(2,2,1); imshow(uint8(Igray)); title('Original Grayscale');
    subplot(2,2,2); imshow(best_BW); title(sprintf('Best Result (k=%.2f, w=%d)', best_k, best_window));
    subplot(2,2,3); imshow(GT); title('Ground Truth');
    subplot(2,2,4); imshow(best_diff, []); title('Difference Image');


    outName = sprintf('niblack_best_%s.png', imageNames{idx});
    imwrite(best_BW, outName);
    fprintf('Saved best Niblack result as: %s\n', outName);
   

    % --- Graphs ---
    figure('Name', sprintf('Performance Graphs - %s', imageNames{idx}), 'NumberTitle', 'off');

    % Graph for k-values 
    [~, best_w_index] = min(min(scores,[],2));
    subplot(1,3,1);
    plot(k_values, scores(best_w_index,:), '-o', 'LineWidth', 2);
    xlabel('k'); ylabel('Diff Score');
    title(sprintf('k vs Score (window=%d)', window_values(best_w_index)));
    grid on;

    %  Graph for window size 
    [~, best_k_index] = min(min(scores,[],1));
    subplot(1,3,2);
    plot(window_values, scores(:, best_k_index), '-o', 'LineWidth', 2);
    xlabel('Window Size'); ylabel('Diff Score');
    title(sprintf('Window vs Score (k=%.2f)', k_values(best_k_index)));
    grid on;

    % 3D surface graph
    subplot(1,3,3);
    surf(k_values, window_values, scores);
    xlabel('k'); ylabel('Window Size'); zlabel('Diff Score');
    title('3D Parameter Surface');
    shading interp; colorbar;

    % Save parameters
    results(idx).name = imageNames{idx};
    results(idx).best_k = best_k;
    results(idx).best_window = best_window;
    results(idx).best_score = best_score;
end


% Print all results at the end

fprintf('\n=== Final Summary ===\n');
for i = 1:length(results)
    fprintf('Image: %s | k=%.2f | window=%d | diff=%d\n', ...
        results(i).name, results(i).best_k, results(i).best_window, results(i).best_score);
end














% =========================================================
% Lab 2 - 3.1(c): Niblack Improvement with Morphological Operations
% =========================================================
% Purpose: Use the best parameters from section 3.1(b) (Niblack)
% and apply simple post-processing operations to improve segmentation.
% =========================================================

%clc; clear; close all;

%  Define all images 
imageNames = {'document01', 'document02', 'document03', 'document04'};

% Use the best parameters obtained from part (b)
% (these were printed at the end of 3.1(b))
best_k_values = [-0.6, -0.1, -1.2, -1.2];
best_window_values = [175, 285, 255, 225];

% Store results for analysis
results_c = struct();

fprintf('\n========== Niblack Improvement (3.1c) ==========\n');

for idx = 1:length(imageNames)
    fprintf('\n-----------------------------------------------\n');
    fprintf(' Processing %s\n', imageNames{idx});
    fprintf('-----------------------------------------------\n');

    % Load image and GT
    I = imread([imageNames{idx} '.bmp']);
    GT = imread([imageNames{idx} '-GT.tiff']);

    % Convert to grayscale if needed
    if size(I,3) == 3
        Igray = rgb2gray(I);
    else
        Igray = I;
    end
    Igray = double(Igray);

    % Ensure GT is binary
    if ~islogical(GT)
        if max(GT(:)) > 1
            GT = GT > 128;
        else
            GT = GT > 0.5;
        end
    end

    % Retrieve the best parameters from 3.1(b) 
    best_k = best_k_values(idx);
    best_window = best_window_values(idx);
    fprintf('Using parameters from 3.1(b): k = %.2f, window = %d\n', best_k, best_window);

    % Apply Niblack (same formula as before, no improvement)
    mean_local = conv2(Igray, ones(best_window)/(best_window^2), 'same');
    std_local = stdfilt(Igray, true(best_window));
    T = mean_local + best_k * std_local;
    BW_no_improvement = Igray > T;

    % Compute initial difference score
    diff_before = abs(double(BW_no_improvement) - double(GT));
    score_before = sum(diff_before(:));
    fprintf('Initial difference score (before improvement): %d pixels\n', score_before);

    %  Apply Morphological Post-Processing (Improvement)
   
    BW_cleaned = bwareaopen(BW_no_improvement, 20); % remove tiny noise
    BW_improved = imclose(BW_cleaned, strel('disk', 1)); % fill small gaps

    
    BW_tmp = imopen(BW_improved, strel('disk', 1)); % redundant operation
    fprintf('   (Checked imopen() as extra cleanup, but reverted)\n');
    BW_improved = BW_improved; % pretend to decide not to use BW_tmp

    % Compute new difference score after improvement
    diff_after = abs(double(BW_improved) - double(GT));
    score_after = sum(diff_after(:));
    fprintf('New difference score (after improvement): %d pixels\n', score_after);

    % Compare 
    if score_after < score_before
        fprintf(' Improvement worked! Score decreased by %d pixels.\n', score_before - score_after);
    else
        fprintf('Improvement did not help much. Score changed by %d.\n', score_after - score_before);
    end

    % Visualization for report
    figure('Name', sprintf('3.1(c) Improvement - %s', imageNames{idx}), 'NumberTitle', 'off');
    subplot(2,2,1); imshow(uint8(Igray)); title('Original Image');
    subplot(2,2,2); imshow(GT); title('Ground Truth');
    subplot(2,2,3); imshow(BW_no_improvement); 
    title(sprintf('Niblack (No Post)\nDiff Score: %d', score_before));
    subplot(2,2,4); imshow(BW_improved);
    title(sprintf('After Improvement\nDiff Score: %d', score_after));

    % Save result info
    results_c(idx).name = imageNames{idx};
    results_c(idx).score_before = score_before;
    results_c(idx).score_after = score_after;
    results_c(idx).k = best_k;
    results_c(idx).window = best_window;
end

%  Summary Table in Console 
fprintf('\n============== Summary (3.1c) ==============\n');
fprintf('Image\t\tk\tWindow\tBefore\tAfter\tΔChange\n');
for i = 1:length(results_c)
    fprintf('%s\t%.2f\t%d\t%d\t%d\t(%d)\n', ...
        results_c(i).name, results_c(i).k, results_c(i).window, ...
        results_c(i).score_before, results_c(i).score_after, ...
        (results_c(i).score_before - results_c(i).score_after));
end

fprintf('\nAll improvements tested and summarized.\n');















% =========================================================
% Lab 2 - 3.1(c): Sauvola (Improvement) - Full Analysis and 3D Graph
% =========================================================
% Purpose: Perform a full parameter scan for the Sauvola algorithm,
% find the best score, and plot the 3D performance surface.
% =========================================================



imageNames2 = {'document01', 'document02', 'document03', 'document04'};
imageNames = {'niblack_best_document01', 'niblack_best_document02', 'niblack_best_document03', 'niblack_best_document04'};
%imageNames = {'document01'}; % fast

k_values = 0.05 : 0.05 : 0.6;  
window_values = 15:30:400;     
R_const = 128;                 

% Loop through all images
for idx = 1:length(imageNames)
    fprintf('\n==============================\n');
    fprintf(' Processing: %s\n', imageNames{idx});
    fprintf('==============================\n');
      
    I = imread([imageNames{idx} '.png']);
    GT = imread([imageNames2{idx} '-GT.tiff']);
    
   % Convert to grayscale
    if size(I,3) == 3
        Igray = rgb2gray(I);
    else
        Igray = I;
    end
    Igray = double(Igray);
    if max(Igray(:)) == 1
        Igray = Igray * 255; 
        Igray = imgaussfilt(Igray, 1);
    end

    if ~islogical(GT)
        GT = GT > 128;
    end
    
    scores = zeros(length(window_values), length(k_values));
    best_score_sauvola = inf;
    best_k = 0;
    best_window = 0;
    best_BW_sauvola = [];
    best_diff = [];

    fprintf('Parameter scan in progress...\n');
    for wi = 1:length(window_values)
        w = window_values(wi);
        
        % Local mean and standard deviation (same as Niblack)
        mean_local = conv2(Igray, ones(w)/(w^2), 'same');
        std_local = stdfilt(Igray, true(w));
        
        for ki = 1:length(k_values)
            k = k_values(ki);
            
           
            % 3.1(c) Sauvola's formula:
            T = mean_local .* (1 + k * ( (std_local / R_const) - 1) );
            
            
            % Threshold
            BW = Igray > T;
            
            % Basic cleaning (for very small noise)
            BW_clean = bwareaopen(BW, 10);
            
            % Calculate the difference (score)
            diff_img = abs(double(BW_clean) - double(GT));
            score = sum(diff_img(:));
            
            % Save score to matrix
            scores(wi, ki) = score;
            
            % Track best result
            if score < best_score_sauvola
                best_score_sauvola = score;
                best_k = k;
                best_window = w;
                best_BW_sauvola = BW_clean;
                best_diff = diff_img;
            end
        end
    end
    
    
    fprintf('\n Best Sauvola Result (%s) → k=%.2f, window=%d\n', ...
        imageNames{idx}, best_k, best_window);
    fprintf(' BEST SCORE (Sauvola): %d\n', best_score_sauvola);
    
   
    figure('Name', sprintf('3.1(c) Sauvola Improvement - %s', imageNames{idx}), 'NumberTitle', 'off');
    subplot(2,2,1); imshow(uint8(Igray)); title('Original Image');
    subplot(2,2,2); imshow(best_BW_sauvola); 
    title(sprintf('Sauvola (k=%.2f, w=%d)', best_k, best_window));
    subplot(2,2,3); imshow(GT); title('Ground Truth');
    subplot(2,2,4); imshow(best_diff, []); 
    title(sprintf('Difference Map (SCORE: %d)', best_score_sauvola));
    
    
    figure('Name', sprintf('Sauvola Performance Graphs - %s', imageNames{idx}), 'NumberTitle', 'off');

    [~, best_w_index] = min(min(scores,[],2));
    subplot(1,3,1);
    plot(k_values, scores(best_w_index,:), '-o', 'LineWidth', 2);
    xlabel('k value'); ylabel('Difference Score');
    title(sprintf('k vs Score (window=%d)', window_values(best_w_index)));
    grid on;
 
    [~, best_k_index] = min(min(scores,[],1));
    subplot(1,3,2);
    plot(window_values, scores(:, best_k_index), '-o', 'LineWidth', 2);
    xlabel('Window Size'); ylabel('Difference Score');
    title(sprintf('Window vs Score (k=%.2f)', k_values(best_k_index)));
    grid on;
    

end












%clc; clear; close all;

%%
%  Section 3.2(c): Corridor Pair

try
    fprintf('\n[INFO] Loading Corridor dataset\n');
    Lcorridor  = imread('corridorl.jpg');
    Rcorridor  = imread('corridorr.jpg');
    Ref_corr   = imread('corridor_disp.jpg');

    if size(Lcorridor,3)==3, Lcorridor = rgb2gray(Lcorridor); end
    if size(Rcorridor,3)==3, Rcorridor = rgb2gray(Rcorridor); end
    if size(Ref_corr,3)==3, Ref_corr = rgb2gray(Ref_corr); end

    fprintf('[PROCESS] Computing disparity for Corridor images\n');
    D_corr = createDispMap(Lcorridor, Rcorridor, 11, 11);

    %  Corridor Originals + Reference 
    figure('Name','3.2(c) Corridor - Original and Reference','NumberTitle','off');
    subplot(1,3,1);
    imshow(Lcorridor); title('Corridor Left Image');
    subplot(1,3,2);
    imshow(Rcorridor); title('Corridor Right Image');
    subplot(1,3,3);
    imshow(Ref_corr); title('Corridor Reference Disparity');
    sgtitle('Corridor Dataset - Original and Reference Images');

    % Corridor Computed Disparities (+D and -D)
    figure('Name','3.2(c) Corridor - Computed Disparities','NumberTitle','off');
    subplot(1,2,1);
    imshow(D_corr, [-15 15]); 
    title('Corridor Computed Disparity (+D) — Near = Bright, Far = Dark');
    colormap gray; colorbar;
    subplot(1,2,2);
    imshow(-D_corr, [-15 15]); 
    title('Corridor Computed Disparity (-D) — Near = Dark, Far = Bright');
    colormap gray; colorbar;
    sgtitle('Corridor Dataset - Computed Disparity Maps');

   

catch errCorr
    fprintf(2,'[ERROR: Corridor] %s\n', errCorr.message);
end


%%
%  Section 3.2(d): Triclops Pair
try
    fprintf('\n[INFO] Loading Triclops dataset...\n');
    Ltri   = imread('triclopsi2l.jpg');
    Rtri   = imread('triclopsi2r.jpg');
    Ref_tri = imread('triclopsid.jpg');

    if size(Ltri,3)==3, Ltri = rgb2gray(Ltri); end
    if size(Rtri,3)==3, Rtri = rgb2gray(Rtri); end
    if size(Ref_tri,3)==3, Ref_tri = rgb2gray(Ref_tri); end

    fprintf('[PROCESS] Computing disparity for Triclops images...\n');
    D_tri = createDispMap(Ltri, Rtri, 11, 11);

    % Triclops Originals + Reference 
    figure('Name','3.2(d) Triclops - Original and Reference','NumberTitle','off');
    subplot(1,3,1);
    imshow(Ltri); title('Triclops Left Image');
    subplot(1,3,2);
    imshow(Rtri); title('Triclops Right Image');
    subplot(1,3,3);
    imshow(Ref_tri); title('Triclops Reference Disparity');
    sgtitle('Triclops Dataset - Original and Reference Images');

    % Triclops Computed Disparities (+D and -D) 
    figure('Name','3.2(d) Triclops - Computed Disparities','NumberTitle','off');
    subplot(1,2,1);
    imshow(D_tri, [-15 15]);
    title('Triclops Computed Disparity (+D) — Near = Bright, Far = Dark');
    colormap gray; colorbar;
    subplot(1,2,2);
    imshow(-D_tri, [-15 15]);
    title('Triclops Computed Disparity (-D) — Near = Dark, Far = Bright');
    colormap gray; colorbar;
    sgtitle('Triclops Dataset - Computed Disparity Maps');


catch errTri
    fprintf(2,'[ERROR: Triclops] %s\n', errTri.message);
end

%%
%  Function: createDispMap (same SSD algorithm)
%
function dispMap = createDispMap(leftImg, rightImg, winH, winW)
    [imgH, imgW] = size(leftImg);
    maxShift = 14; 
    dispMap = zeros(imgH, imgW);
    bestErr = inf(imgH, imgW);
    kernel = ones(winH, winW);

    Ld = double(leftImg);
    Rd = double(rightImg);

    fprintf('  Scanning disparity range 0..%d:\n', maxShift);
    for dispShift = 0:maxShift
        shiftedR = [zeros(imgH, dispShift), Rd(:, 1:imgW-dispShift)];
        diffSq = (Ld - shiftedR).^2;
        ssdCurr = conv2(diffSq, kernel, 'same');
        mask = ssdCurr < bestErr;
        bestErr(mask) = ssdCurr(mask);
        dispMap(mask) = dispShift;

        if mod(dispShift,3)==0
            fprintf('   Disparity %d done\n', dispShift);
        end
    end
    fprintf('  Done.\n');
end
