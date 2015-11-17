--[[
This file is a means to refactor command line argument parsing outside of the
main nn files.

This was loosely adapted from Andrej Karpathy's char-rnn train.lua file
source: https://github.com/karpathy/char-rnn/blob/master/train.lua

Major changes made were:
1. Refactoring to new file for readbility
2. Changing names to match rest of project's naming scheme
3. Adding relevent commandline args for this project.
]]--
local CommandLineArgs = {}

function CommandLineArgs.sampleCmdArgs()
  print("hey")
end

function CommandLineArgs.trainCmdArgs()
  print("Running trainCmdArgs")
  cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Train a character-level language model')
  cmd:text()
  cmd:text('Options')
  -- data
  cmd:option('-embeddingSize',300,'Word embedding size')
  cmd:option('-dataDir','data/','Data directory. Should point to tokenized input file')
  cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
  cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
  cmd:option('-batchSize',50,'number of sequences to train on in parallel')
  cmd:option('-trainFrac',0.95,'fraction of data that goes into train set')
  cmd:option('-testFrac',0.05,'fraction of data that goes into test set')
  cmd:option('-evalFrac',0,'fraction of data that goes into eval set')
  cmd:option('-maxEpochs',50,'number of full passes through the training data')
  cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
  cmd:option('-maxSeqLength',90,'Max sentence length.')



  -- model params
  cmd:option('-rnnSize', 128, 'size of LSTM internal state')
  cmd:option('-numLayers', 2, 'number of layers in the LSTM')
  -- optimization
  cmd:option('-learningRate',2e-3,'learning rate')
  cmd:option('-learning_rate_decay',0.97,'learning rate decay')
  cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
  cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
  cmd:option('-grad_clip',5,'clip gradients at this value')
              -- test_frac will be computed as (1 - train_frac - val_frac)
  cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
  -- bookkeeping
  cmd:option('-seed',123,'torch manual random number generator seed')
  cmd:option('-print_every',1,'how many steps/minibatches between printing out the loss')
  cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
  cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
  cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
  -- GPU/CPU
  cmd:text()
-- parse input params
  return cmd:parse(arg)
end

function CommandLineArgs.generalCmdArgs()
  print("hello world")
end

return CommandLineArgs
