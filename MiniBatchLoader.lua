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


--[[
This method now stores each batch in a separate file... Yes this will probably be slower,
but I'm trying it out to see if it alleviates memory issues.
]]
function MiniBatchLoader.createMiniBatches(options)

  print("Max sequence length is ... ".. options.maxSeqLength)
  local dataFiles = {}
  local batchMetaData = {}
  local processedDir = path.join(options.dataDir, Constants.processedFolder)
  print("Loading data...")

  for file in lfs.dir(processedDir) do
    if not StringUtils.startsWith(file,".") then
      local filePath = path.join(processedDir, file)
      table.insert(dataFiles, filePath)
    end
  end

  local sourceTargetPairs = {}
  local batchFiles  = {}
  local index = 1
  local fileCounter = 1
  local batchCounter = 0

  for key, value in ipairs(dataFiles) do
    print(value)
    local data = torch.load(value)
    print("Loaded data...")
    --local sourceTargetPairs = torch.IntTensor(#data-1, 2, options.maxSeqLength)
    -- Reverse sequences to introduce short-term dependency's (Google's result)
    --Insert training pairs into tensor
    for i=1,#data-1 do
      --right now we remove sequences longer than the max length,
      --in the future it may be wiser to just truncate them?
      if not (data[i]:size(1) > options.maxSeqLength) and
        not (data[i + 1]:size(1) > options.maxSeqLength) then
          sourceTargetPairs[index] = {}
          --reverse source sentence
          sourceTargetPairs[index][1] = TableUtils.reverseTensor(data[i])
          sourceTargetPairs[index][2] = data[i + 1]
          index = index + 1
      end
      if #sourceTargetPairs > 0 and #sourceTargetPairs % options.batchSize == 0 then
        batchCounter = batchCounter + 1
        local batchFile = path.join(options.dataDir,Constants.rawBatchesFolder..fileCounter..Constants.rawBatchesFile)
        table.insert(batchFiles, batchFile)
        torch.save(batchFile, sourceTargetPairs)
        sourceTargetPairs = {}
        index = 1
        fileCounter = fileCounter + 1
      end
    end
    collectgarbage()
  end
  local batchMetaFile = path.join(options.dataDir, Constants.rawBatchesFolder..Constants.metaBatchInfo)
  batchMetaData.batchFiles = batchFiles
  batchMetaData.numBatchs = batchCounter
  batchMetaData.batchSize = options.batchSize
  batchMetaData.maxSeqLength = options.maxSeqLength
  torch.save(batchMetaFile, batchMetaData)
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
  return not path.exists(batchDataDir)
end

function MiniBatchLoader.loadBatches(dataDir,trainFrac)
  local self = {}
  setmetatable(self, MiniBatchLoader)
  assert(trainFrac > 0 and trainFrac <= 1, "trainFrac not between 0 and 1...")

  print("Using ".. trainFrac .. " As percentage of data to train on...")
  print("Using " .. 1 - trainFrac .. " As percentage of data to test on...")
  --Checks to ensure user isn't testing us
  self.trainFrac = trainFrac
  self.testFrac =  1 - trainFrac
  self.trainBatchPointer = 0
  self.testBatchPointer = 0
  --flag will be set to 1 when test set has been entirely iterated over
  self.testSetFlag = 0
  --1 = train set, 2 = test set
  self.splitIndex = 1
  self.batchFiles = {}
  local batchDataDir = path.join(dataDir, Constants.rawBatchesFolder)
  local batchMetaData = torch.load(path.join(batchDataDir, Constants.metaBatchInfo))
  self.batchSize = batchMetaData.batchSize
  self.numBatches = batchMetaData.numBatchs
  self.batchFiles = batchMetaData.batchFiles
  self.maxSeqLength = batchMetaData.maxSeqLength
  --shuffling batches
  TableUtils.shuffleTable(self.batchFiles)

  self.batchLimits = {
    {1,math.floor(self.numBatches * self.trainFrac)},
    {math.floor(self.numBatches * self.trainFrac)+1, self.numBatches}
  }
  return self
end

--index of 1 indicates train set being drawn from
--index of 2 indicates test set being drawn from
function MiniBatchLoader.nextBatch(self, index)
  if index == 1 then
    local batch = torch.load(self.batchFiles[self.trainBatchPointer + 1])
    self.trainBatchPointer = (self.trainBatchPointer + 1) % self.batchLimits[1][2]
    return batch
  end
  if index ==2 then
    if self.testBatchPointer == self.batchLimits[2][2] then return nil end
    -- 1-based indexing...
    local batch = torch.load(self.batchFiles[self.testBatchPointer + 1])
    self.testBatchPointer = (self.testBatchPointer + 1) % (self.batchLimits[2][2] + 1)
    return batch
  end
--return nil if we get down here...
  return nil
end

--[[function MiniBatchLoader.resetPointer(self, splitIndex)
  self.splitIndex = splitIndex
  self.batchPointer = self.batchLimits[self.splitIndex][1]
end]]--

return MiniBatchLoader
