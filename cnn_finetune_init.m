function net = cnn_finetune_init(imdb, net)

opts.weightInitMethod = 'xavierimproved' ;
opts.scale = 1; 

if ~exist('net', 'var') || isempty(net), 
  net = 'imagenet-matconvnet-vgg-m';
end

if  ischar(net), 
  net_path = fullfile('data','models',[net '.mat']);
  if ~exist(net_path,'file'), 
    fprintf('Downloading model (%s) ...', net) ;
    vl_xmkdir(fullfile('data','models')) ;
    urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', ...
      [net '.mat']), net_path) ;
    fprintf(' done!\n');
  end
  net = load(net_path);
end

net.layers{end} = struct('name','loss','type','softmaxloss'); 

if ~isfield(net.meta, 'pretrain'), 
  net.meta.pretrain = {};
end
net.meta.pretrain = [net.meta.pretrain net.meta];
net.meta.pretrain{end} = rmfield(net.meta.pretrain{end}, 'pretrain'); 
net.meta.classes.name = imdb.meta.classes;
net.meta.classes.description = imdb.meta.classes;

[h,w,in,out] = size(net.layers{end-1}.weights{1});
out = numel(net.meta.classes.name); 
net.layers{end-1}.weights = {init_weight(opts, h, w, in, out, 'single'), zeros(out, 1, 'single')};

end

% -------------------------------------------------------------------------
function weights = init_weight(opts, h, w, in, out, type)
% -------------------------------------------------------------------------
% See K. He, X. Zhang, S. Ren, and J. Sun. Delving deep into
% rectifiers: Surpassing human-level performance on imagenet
% classification. CoRR, (arXiv:1502.01852v1), 2015.

switch lower(opts.weightInitMethod)
  case 'gaussian'
    sc = 0.01/opts.scale ;
    weights = randn(h, w, in, out, type)*sc;
  case 'xavier'
    sc = sqrt(3/(h*w*in)) ;
    weights = (rand(h, w, in, out, type)*2 - 1)*sc ;
  case 'xavierimproved'
    sc = sqrt(2/(h*w*out)) ;
    weights = randn(h, w, in, out, type)*sc ;
  otherwise
    error('Unknown weight initialization method''%s''', opts.weightInitMethod) ;
end

end
