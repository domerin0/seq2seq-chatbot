local CommandLineArgs = require "Util.CommandLineArgs"
local Preprocessor = require "Util.Preprocessor"
local MiniBatchLoader = require "Util.MiniBatchLoader"
local VerifyGPU = require "Util.VerifyGPU"

local options = CommandLineArgs.trainCmdArgs()

--decide whether to use CPU or GPU, and if GPU whether cuda or opencl
local useGPU = false
local useCuda = false

if options.opencl == 1 or options.cuda == 1 then
  useCuda = options.opencl == 1
  useGPU = true
end

if options.gpuid > -1 then
  useGPU = useGPU and true
end

print("Starting pre-processor")

local dataFiles = Preprocessor.start(options.dataDir)

--prepare data for training with (input, output) pairs

print("Creating minibatches...")

MiniBatchLoader.createMiniBatches(dataFiles, options.batchSize, options.trainFrac,
  options.evalFrac, options.testFrac)

--perform training of n minibatches of m epochs over bs backsteps
