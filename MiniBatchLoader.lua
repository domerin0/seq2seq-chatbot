--[[
The purpose of this file is to enable the train.lua file to
'efficiently' create, load, and save the minibatches
]]
local TableUtils = require "Util.TableUtils"
local StringUtils = require "Util.StringUtils"
local Preprocessor = require "Preprocessor"
local Constants = require "Util.Constants"
require 'lfs'

local MiniBatchLoader = {}
MiniBatchLoader.__index = MiniBatchLoader

function MiniBatchLoader.createMiniBatches(options)

  print("Max sequence length is ... ".. options.maxSeqLength)
  local batchFiles  = {}

  local dataFiles = {}
  local processedDir = path.join(options.dataDir, Constants.processedFolder)
  print("Loading data...")

  for file in lfs.dir(processedDir) do
    if not StringUtils.startsWith(file,".") then
      local filePath = path.join(processedDir, file)
      table.insert(dataFiles, filePath)
    end
  end

  local sourceTargetPairs = {}
  for key, value in ipairs(dataFiles) do
    print(value)
    local data = torch.load(value)
    print("Loaded data...")
    --local sourceTargetPairs = torch.IntTensor(#data-1, 2, options.maxSeqLength)
    -- Reverse sequences to introduce short-term dependency's (Google's result)
    --Insert training pairs into tensor
    local index = 1
    for i=1,#data-1 do
      if not (data[i]:size(1) > options.maxSeqLength) and
        not (data[i + 1]:size(1) > options.maxSeqLength) then
          sourceTargetPairs[index] = {}
          sourceTargetPairs[index][1] = data[i]
          sourceTargetPairs[index][2] = data[i + 1]
          index = index + 1
      end
    end
  end
  print("Creating minibatch files...")

  local batchFile = path.join(options.dataDir,Constants.rawBatchesFolder..Constants.rawBatchesFile)
  torch.save(batchFile, sourceTargetPairs)

end

--[[Reads in tensor files, and gets max seqence length.
While this is inneficient to do so, this method was made to prevent dependency
on the preprocesor.lua file to pass max sequence length forward. Making this design
decision allows for modularity between minibatch creation, and preprocessing the data.
]]
function MiniBatchLoader.getMaxSequenceLength(dataFile)
  local data = torch.load(dataFile)
  local maxLength = 0
  for key, vec in ipairs(data) do
    if vec:size(1) > maxLength then
      maxLength = vec:size(1)
    end
  end
  return maxLength
end

--[[
This function check if we even need to run the minibatchmaker
It assumed if there are batches in the train folder that it
does not need to be run
]]
function MiniBatchLoader.shouldRun(dataDir)
  local batchDataDir = path.join(dataDir, Constants.rawBatchesFolder)
  local batchFile = path.join(batchDataDir, Constants.rawBatchesFile)
  return not path.exists(batchFile)
end

function MiniBatchLoader.loadBatches(dataDir, batchSize, trainFrac, evalFrac, testFrac)
  local self = {}
  setmetatable(self, MiniBatchLoader)
  assert(evalFrac >= 0 and evalFrac < 1, "evalFrac not between 0 and 1...")
  assert(trainFrac > 0 and trainFrac <= 1, "trainFrac not between 0 and 1...")
  assert(testFrac >= 0 and testFrac < 1, "testFrac not between 0 and 1...")
  assert(testFrac + evalFrac + trainFrac == 1, "eval, train and test don't add up to 1!")

  print("Using ".. trainFrac .. " As percentage of data to train on...")
  print("Using ".. evalFrac .. " As percentage of data to validate on...")
  print("Using " .. testFrac .. " As percentage of data to test on...")
  --Checks to ensure user isn't testing us
  self.trainFrac = trainFrac
  self.evalFrac = evalFrac
  self.testFrac = testFrac
  self.batchSize = batchSize
  self.batchPointer = 1

  --1 = train set, 2 = test set, 3 = cross val set
  self.splitIndex = 1
  local batchDataDir = path.join(dataDir, Constants.rawBatchesFolder)
  local batchFile = path.join(batchDataDir, Constants.rawBatchesFile)
  self.batches = torch.load(batchFile)
  local counter = 0
  for key, value in pairs(self.batches) do
    counter = counter + 1
  end
  self.numBatches = math.floor(counter / self.batchSize)
  self.batchLimits = {
    {1,math.floor(self.numBatches * self.trainFrac)},
    {math.floor(self.numBatches * self.trainFrac)+1, math.floor(self.numBatches * (self.trainFrac + self.testFrac))},
    {math.floor(self.numBatches * (self.trainFrac + self.testFrac)) + 1, self.numBatches}
  }
  print('Loading previously allocated minibatches...')
  return self
end

function MiniBatchLoader.nextBatch(self)
  local batch = {}
  for i=1,self.batchSize do
    batch[i] = self.batches[((self.batchPointer - 1) * self.batchSize) + i]
  end
  if (self.batchPointer*self.batchSize) >= self.batchLimits[self.splitIndex][2] then
    self.splitIndex = 1
    self.batchPointer = 1
  else
    self.batchPointer = self.batchPointer + 1
  end
  return batch
end

function MiniBatchLoader.resetPointer(self, splitIndex)
  self.splitIndex = splitIndex
  self.batchPointer = self.batchLimits[self.splitIndex][1]
end

return MiniBatchLoader

--code I will probably need to refer to later:
--local trainFiles = {}
--local testFiles = {}
--local evalFiles = {}
--local trainFile = path.join(dataDir,Constants.trainFolder..Constants.trainFile)
--local testFile = path.join(dataDir,Constants.testFolder..Constants.testFile)
--local evalFile = path.join(dataDir,Constants.evalFolder..Constants.evalFile)
--miniBatches, testBatches, evalBatches = miniBatches:sub(1, numTrain),
--  miniBatches:sub(numTrain + 1, numTrain + numTest),
--  miniBatches:sub(numTrain + numTest + 1,
--   numTrain + numTest + numEval)
