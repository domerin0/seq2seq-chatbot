--This file is heavily influenced by:
-- Andrej Karpathy's https://github.com/karpathy/char-rnn/blob/master/train.lua


local CommandLineArgs = require "Util.CommandLineArgs"
local Preprocessor = require "Util.Preprocessor"
local MiniBatchLoader = require "Util.MiniBatchLoader"
local VerifyGPU = require "Util.VerifyGPU"
local Seq2Seq = require "Models.Seq2Seq"

local options = CommandLineArgs.trainCmdArgs()
torch.manualSeed(options.seed)

if Preprocessor.shouldRun(options.dataDir) then
  print("Starting pre-processor")
  Preprocessor.start(options.dataDir)
  collectgarbage()
else
  print("Preprocessor doesn't need to be run, moving on...")
end

--prepare data for training with (input, output) pairs

if MiniBatchLoader.shouldRun(options.dataDir) then
  print("Creating minibatches...")
  MiniBatchLoader.createMiniBatches(options.dataDir, options.batchSize,
    options.maxSeqLength)
  collectgarbage()
else
  print("Minibatches already created before, moving on...")
end

--Now, check and enable GPU usage:

local cuid = VerifyGPU.checkCuda(options.gpuid, options.seed)


local clid = VerifyGPU.checkOpenCl(options.gpuid, options.seed)

if clid == -1 and cuid == -1 then
  options.gpuid = -1
end



--Load minibatches into memory!

local batchLoader = MiniBatchLoader.loadMiniBatches(options.dataDir, options.batchSize, trainFrac,
  options.evalFrac, options.testFrac)







--perform training of n minibatches of m epochs over bs backsteps
