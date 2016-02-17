function [net, info] = cnn_finetune(datasetName, varargin)

opts.expDir     = fullfile('data','exp') ;
opts.baseNet    = 'imagenet-matconvnet-vgg-m';
opts.numEpochs  = [5 5 10]; 
opts.numFetchThreads = 12 ;
opts.imdb       = [];
opts.aug 	= 'stretch'; 
opts.pad 	= 0; 
[opts,varargin] = vl_argparse(opts, varargin) ;

opts.train = struct() ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
if isempty(opts.imdb), 
  imdb = get_imdb(datasetName); 
else
  imdb = opts.imdb;
end
net = cnn_finetune_init(imdb,opts.baseNet); 

net.meta.trainOpts.learningRate = [0.05*ones(1,5) 0.01*ones(1,5) 0.001*ones(1,5) 0.0001*ones(1,5)]; 
net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 64;
net.meta.trainOpts.gpus = [];
net.meta.trainOpts = vl_argparse(net.meta.trainOpts, varargin) ;

net.meta.trainOpts.sessions{2}.startEpoch = opts.numEpochs(2) + 1;

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
trainable_layers = find(cellfun(@(l) isfield(l,'weights'),net.layers)); 
fc_layers = find(cellfun(@(s) numel(s.name)>=2 && strcmp(s.name(1:2),'fc'),net.layers));
fc_layers = intersect(fc_layers, trainable_layers);
lr = cellfun(@(l) l.learningRate, net.layers(trainable_layers),'UniformOutput',false); 
layers_for_update = {trainable_layers(end), fc_layers, trainable_layers}; 

% tune last layer --> tune fc layers --> tune all layers
for s = 1:numel(opts.numEpochs), 
  if opts.numEpochs(s)<1, continue; end
  for i = 1:numel(trainable_layers), 
    l = trainable_layers(i); 
    if ismember(l,layers_for_update{s}), 
      net.layers{l}.learningRate = lr{i}; 
    else
      net.layers{l}.learningRate = lr{i}*0; 
    end
  end
  [net, info] = cnn_train(net, imdb, getBatchFn(opts, net.meta), ...
                          'expDir', opts.expDir, ...
                          net.meta.trainOpts, ...
                          'numEpochs', opts.numEpochs(s)) ;
end

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------
net = cnn_imagenet_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat')
save(modelPath, '-struct', 'net') ;


% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
bopts.numThreads = opts.numFetchThreads ;
bopts.pad = opts.pad ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
% bopts.transformation = meta.augmentation.transformation ;
bopts.transformation = opts.aug ;

fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;


% -------------------------------------------------------------------------
function [im,labels] = getSimpleNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  labels = imdb.images.class(batch) ;
end


