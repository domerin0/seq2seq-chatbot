local CommandLineArgs = require "Util.CommandLineArgs"
local Preprocessor = require "Util.Preprocessor"
local MiniBatchLoader = require "Util.MiniBatchLoader"
local VerifyGPU = require "Util.VerifyGPU"

local options = CommandLineArgs.trainCmdArgs()

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

--VerifyGPU.checkCuda(options.gpuid, options.seed)


--VerifyGPU.checkOpenCl(options.gpuid, options.seed)

--Load minibatches into memory!

local batchLoader = MiniBatchLoader.loadMiniBatches(options.dataDir, options.batchSize, trainFrac,
  options.evalFrac, options.testFrac)

--Create model, or load from checkpoint


--perform training of n minibatches of m epochs over bs backsteps
