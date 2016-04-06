
function [seg_out, C, probability_maps] = rfcm(input_image, c, options)
                                    
% ************************************************************************
% Robust fuzzy-c-means segmentation
%  -inputs:
%   -> input_image (2d or 3d input image) 
%   -> c: number of classes 
%   -> options:
%             options.weighting = fuzzy factor exponent in FCM (default 2)
%             options.maxiter = Number of maximum iterations during energy minimization FCM (default 200)
%             options.num_neigh = Radius of the neighborhood used in spatial contraint (default 1)
%             options.dim = Dimension of the neighborhood (default 2)
%             options.term = Maximum error in energy minimization (default 1E-3)
%             options.gpu = Use GPU (default 0)
%             options.info = Show information during tissue segmentation (default 0)
%
% - outputs:
%   seg_out = hard labelled segmentation
%   C = final class intensity centers
%   probability_maps = probabilistic segmentation for each class probability_maps = [input_size, num_of_classes]
%
%
% svalverde@eia.udg.edu 2016
% NeuroImage Computing Group. Vision and Robotics Insititute (University of Girona)
% ***************************************************************************************************




    
    switch nargin
      case 1
        error('Incorrect number of parameters');
      case 2
        options = struct;
      otherwise
        
    end

    
    options = parse_options(options);
    
    m = options.weighting;             % fuzziness exponent
    max_iter = options.maxiter;        % Max. iteration
    n_neigh = options.num_neigh;       % Number of neighbors used in the penalized function.
    neigh_dim = options.dim;           % Dimension of the neighborhood
    beta = options.beta;               % Beta parameter controling the strenght of the penalized function
    term_thr = options.term;           % Termination threshold
    use_gpu = options.gpu;             % Use GPU for compputation
    display = options.info;            % Display info or not

     
    if m <= 1,
        error('The weighting exponent should be greater than 1!');
    end
    
    % input Data

    input_image = double(input_image);
    Y = (find(input_image > 0));            % Indexed positions of each voxel
    X = (input_image(Y));                   % 1D reshaped input image
    [n,d] = size(X);                        % vector size
    W = (zeros(c,numel(X)));                % Weighting vectors for each class
    WP = (zeros(c,numel(X)));               % Weighting penalty vectors for each class
    WP_notmembers = (zeros(c, numel(X)));   % Weighting penalty for non class members
    error_v = (zeros(max_iter, 1));         % Array for termination measure values
    XC= (repmat(X',c,1));                   % precomputed duplicated input data vector X to increase the speed of the distance function
    seg_out = zeros(size(input_image));

    if use_gpu
        % pass to CUDA arrays
        Y = gpuArray(double(Y));
        X = gpuArray((X));
        W = gpuArray(double(W));
        WP = gpuArray(double(WP));
        WP_notmembers = gpuArray(double(WP_notmembers));
        XC = gpuArray(double(XC));
        seg_out = gpuArray(seg_out);
        if display
            disp(['GPU based computation. Data have been transformed to GPUArrays']);
        end
    end

    % (3). Find the initial cluster centers. Based on the histogram. By default,
    %C = zeros(c,1);
    C = rand(c,1);
    % (4). Precompute neighbor voxel position indices.
    neighbor_voxels = (compute_neighborhoods(Y, size(input_image), n_neigh, neigh_dim));

    current_weight = (zeros(size(input_image)));
    class_vector = 1:c;
    not_member_class = repmat(class_vector,c,1) ~= repmat(class_vector',1,c);
    if use_gpu
        neighbor_voxels = gpuArray(neighbor_voxels);
        current_weight = gpuArray(current_weight);
    end


    % (5). Minimize the objective function
    if display
        disp('Minimizing the objective function......');
    end

    for i = 1:max_iter,

        % (5.1) penalty function: for each voxel and class compute the sum
        % of the weights of their neighbors
        for class=1:c
            current_weight(Y) = W(class,:);
            WP(class,:) = sum(current_weight(neighbor_voxels),2);
        end

        % for each voxel, sum non-class member neighbors.
        for class=1:c
             not_members = class_vector(not_member_class(class,:));
             WP_notmembers(class,:) = sum(WP(not_members,:));
        end
        
        % (5.2) new weights W
        dist = abs(repmat(C,1,n) - XC);
        denom = (dist + (beta.*(WP_notmembers.^m))).^(-2/(m-1));
        W = denom./ (ones(c, 1)*(sum(denom)));

        % Correct the situation of "singularity" (one of the data points is
        % exactly the same as one of the cluster centers).
        if use_gpu
            si = gather(find (denom == Inf));
        else
            si = find (denom == Inf);
        end

        
        if si > 0
            W(si) = 1;
            if display
                disp('singularity');
            end
        end

        % Check constraint
        tmp = find ((sum (W) - ones (1, n)) > 0.0001);
        if (size(tmp,2) ~= 0)
            disp('RFCM, Warning: Constraint for U is not hold.');
        end

        % (5.3) calculate new centers C and update the error
        C_old = C;
        mf = W.^m;
        C = mf*X./((ones(d, 1)*sum(mf'))');

        error_v(i) = norm (C - C_old, 1);
        if display
            disp(['Iteration: ', num2str(i), ' Estimated error: ', num2str(error_v(i))]);
        end
        % check termination condition
        if error_v(i) <= term_thr, break; end,
    end

    iter_n = i;	% Actual number of iterations
    error_v(iter_n+1:max_iter) = [];

    % (8). compute binary segmentation.
    
    [C, index] = sort(C);
    [~, segmentation] = max(W(index,:));
    seg_out(Y) = segmentation;
    
    % gather tissue segmentation from the GPU back to the CPU
    if use_gpu
        seg_out = gather(seg_out);
        W = gather(W);
    end

    % probability classes
    probability_maps = zeros([size(seg_out), c]);
    for cl=1:c
        tmp_prob = zeros(size(seg_out));
        tmp_prob(Y) = W(index(cl),:);
        probability_maps(:,:,:,cl) = tmp_prob;
    end
        
end




function [neighbor_positions] = compute_neighborhoods(target_positions, original_size, n, neighbors_dimension)
    % ------------------------------------------------------------------------
    % [neighbor_voxels] = compute_neighborhoods(target_positions, original_size, n, neighbors_dimension) 
    % 
    % Compute the positions of the {n x n x n} neighbors voxel given a  list of target
    % positions. 
    %
    %  -target_positions: vector containing the positions of the 3D matrix
    %                     expressed as linear indices
    %  -original_size: Size of the original 3D matrix
    %  -n: radius of the number of neighbors (total neigh: (2*n +1)^2 in 2D
    %      or (2*n+1)^3 for 3D matrices.
    %  -neighbors_dimension: Compute either the neighbor positions in 2D or 
    %                        3D.   
    %
    %  neighbor_voxels: returns a matrix with size [num voxels, n^neighbor_dimension] 
    %                   with the positions ocompute_neighborhoods3.mf all the adjacent neighbors.
    %
    % 
    % svalverde@eia.udg.edu 2016
    % NeuroImage Computing Group. Vision and Robotics Insititute (University of Girona)
    % ***************************************************************************************************

        
    if neighbors_dimension ~= 2 && neighbors_dimension ~= 3
        error([num2str(neighbors_dimension), ' appears not a valid neighbor dimensionality argument']);
    end
    
    diameter = (2*n+1);
    
    original_r = original_size(1);
    original_c = original_size(2);

    if size(original_size,2) > 2
        original_s = original_size(3);
        [rv,cv,sv] = ind2sub(original_size, target_positions); 

        % compute based on expanded matrix. We generate all the
        % neighbor positions by expanding them (exclude the center voxel)
        expansor = repmat(-n:n, numel(rv),1);
        N_r = repmat(rv,1,diameter) + expansor;
        N_c = repmat(cv,1,diameter) + expansor;

        % check dimensionality 
        if neighbors_dimension == 2
            N_s = repmat(sv,1,diameter); 
            % rows, cols and slices are concatenated in the column dimension to
            % generate all possible neighbor combinations for each voxel
            expanded_rows = repmat(N_r,1, diameter);  
            expanded_cols = kron(N_c, ones(1,diameter)); 
            expanded_slices = kron(N_s, ones(1,diameter));
            neighbor_positions = sub2ind(original_size, max(1, min(original_r, expanded_rows)), ...
                                         max(1, min(original_c, expanded_cols)), ...
                                         max(1, min(original_s, expanded_slices)));

        else
            N_s = repmat(sv,1,diameter) + expansor; 
            expanded_rows = repmat(N_r,1, diameter^2);  
            expanded_cols = repmat(kron(N_c, ones(1,diameter)),1,diameter); 
            expanded_slices = kron(N_s, ones(1,diameter^2));
            neighbor_positions = sub2ind(original_size, max(1, min(original_r, expanded_rows)), ...
                                         max(1, min(original_c, expanded_cols)), ...
                                         max(1, min(original_s, expanded_slices)));
        end
    else
        [rv,cv] = ind2sub(original_size, target_positions); 

        % compute based on expanded matrix. We generate all the
        % neighbor positions by expanding them (exclude the center voxel)
        expansor = repmat(-n:n, numel(rv),1);
        N_r = repmat(rv,1,diameter) + expansor;
        N_c = repmat(cv,1,diameter) + expansor;

        % rows, cols and slices are concatenated in the column dimension to
        % generate all possible neighbor combinations for each voxel
        expanded_rows = repmat(N_r,1, diameter);  
        expanded_cols = kron(N_c, ones(1,diameter)); 
        neighbor_positions = sub2ind(original_size, max(1, min(original_r, expanded_rows)), ...
                                     max(1, min(original_c, expanded_cols)));;
    end
    
end 


function options = parse_options(options)

% ********************************************************************************
% function to parse the mandatory options for the method
%
% ********************************************************************************

    % Beta parameter (afterwards it is updated)
     if ~isfield(options,'beta')
         options.beta = 0.1;
     end
     % fuzzy factor exponent
     if ~isfield(options,'weighting')
         options.weighting = 2;
     end
     % number of maximum iterations
     if ~isfield(options,'maxiter')
         options.maxiter = 200;
     end
     % Number of neighbors in the spatial constraint
     if ~isfield(options,'num_neigh')
         options.num_neigh  = 1;
     end
     % Number of dimensions of the neighborhood
     if ~isfield(options,'dim')
         options.dim  = 2;
     end
     % Number of maximum iterations for the FCM clustering
     if ~isfield(options,'term')
         options.term  = 1E-3;
     end
     % Use GPU
     if ~isfield(options,'gpu')
         options.gpu = 0;
     end
     % Show information
     if ~isfield(options,'info')
         options.info = 0;
     end
     %options
end