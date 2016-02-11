function [net, info] = cnn_vehicle(varargin)

opts.expDir = fullfile('data','exp') ;
opts.numFetchThreads = 12 ;
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

imdb = get_imdb('bing7'); 
net = cnn_finetune_init(imdb); 
net.meta.trainOpts.learningRate = [0.05*ones(1,10) 0.01*ones(1,10) 0.001*ones(1,10) 0.0001*ones(1,10)]; 
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate); 
net.meta.trainOpts.batchSize = 64;
net.meta.trainOpts.sessions{2}.startEpoch = 11;

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

trainable_layers = find(cellfun(@(l) isfield(l,'weights'),net.layers)); 
lr = cellfun(@(l) l.learningRate, net.layers(trainable_layers),'UniformOutput',false); 
trainOpts = rmfield(net.meta.trainOpts,'sessions'); 

% session 1: finetune the last layer only
for i = 1:numel(lr), 
  l = trainable_layers(i); 
  if ismember(l,net.meta.trainOpts.sessions{1}.layers), 
    net.layers{l}.learningRate = lr{i}; 
  else
    net.layers{l}.learningRate = lr{i}*0; 
  end
end
trainOpts.numEpochs = net.meta.trainOpts.sessions{2}.startEpoch - 1;
[net, info] = cnn_train(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      trainOpts, ...
                      opts.train) ;

% session 2: finetune all layers jointly
for i = 1:numel(lr), 
  l = trainable_layers(i); 
  if ismember(l,net.meta.trainOpts.sessions{2}.layers), 
    net.layers{l}.learningRate = lr{i}; 
  else
    net.layers{l}.learningRate = lr{i}*0; 
  end
end
trainOpts.numEpochs = net.meta.trainOpts.numEpochs;
[net, info] = cnn_train(net, imdb, getBatchFn(opts, net.meta), ...
                      'expDir', opts.expDir, ...
                      trainOpts, ...
                      opts.train) ;

% -------------------------------------------------------------------------
%                                                                    Deploy
% -------------------------------------------------------------------------

net = cnn_deploy(net) ;
modelPath = fullfile(opts.expDir, 'net-deployed.mat')
save(modelPath, '-struct', 'net') ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;

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


