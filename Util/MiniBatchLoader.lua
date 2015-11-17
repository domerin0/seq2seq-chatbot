--[[
The purpose of this file is to enable the train.lua file to
'efficiently' create, load, and save the minibatches
]]
local TableUtils = require "Util.TableUtils"
local Preprocessor = require "Util.Preprocessor"
require 'lfs'
local MiniBatchLoader = {}

function MiniBatchLoader.createMiniBatches(dataDir, batchSize, trainFrac,
  evalFrac, testFrac, maxSequenceLength)

  --Checks to ensure user isn't testing us
  assert(evalFrac >= 0 and evalFrac < 1, "evalFrac not between 0 and 1...")
  assert(trainFrac > 0 and trainFrac <= 1, "trainFrac not between 0 and 1...")
  assert(testFrac >= 0 and testFrac < 1, "testFrac not between 0 and 1...")
  assert(testFrac + evalFrac + trainFrac == 1, "eval, train and test don't add up to 1!")

  print("Using ".. trainFrac .. " As percentage of data to train on...")
  print("Using ".. evalFrac .. " As percentage of data to validate on...")
  print("Using " .. testFrac .. " As percentage of data to test on...")
  print("Max sequence length is ... ".. maxSequenceLength)
  local batchFiles  = {}
  local trainFiles = {}
  local testFiles = {}
  local evalFiles = {}
  local dataFiles = {}
  local processedDir = path.join(dataDir, "processed/")
  print("Loading data...")

  local trainingPairSize = {}

  for file in lfs.dir(processedDir) do
    table.insert(dataFiles, path.join(processedDir, file))
  end

  local miniBatches = torch.Tensor(0,2,maxSequenceLength))

  for key, value in ipairs(dataFiles) do
    local data = torch.load(value)
    print("Loaded data...")

    local sourceTargetPairs = torch.Tensor(#data-1, 2, maxSequenceLength)

    -- Reverse sequences to introduce short-term dependency's (Google's result)
    --Insert training pairs into table
    for i=1,#data-1 do
      local source = TableUtils.padTable(TableUtils.reverseTable(data[i]),
      maxSequenceLength)
      data[i] = nil
      local target = TableUtils.padTable(data[i + 1], maxSequenceLength)
      if not sourceTargetPairs[i][1]:size(1) > maxSequenceLength and
        not sourceTargetPairs[i][1]:size(1) > maxSequenceLength then
          sourceTargetPairs[i] = torch.Tensor({source, target})
      end
    end

    miniBatches =  torch.cat(miniBatches, sourceTargetPairs, 1)

  end
    --save batch sets in appropriate t7 files
    print("Creating minibatch files...")
  --Due to memory constraints of data I have decided to split everything
  --into multiple files

  local batchFile = path.join(dataDir, "rawbatches/batch.t7")
  local trainFile = path.join(dataDir, "train/train.t7")
  local testFile = path.join(dataDir, "test/test.t7")
  local evalFile = path.join(dataDir, "eval/eval.t7")
  local miniBatches = torch.load(file)
  local totalNum = miniBatches:size(1)

  local numTrain = math.floor(trainFrac * totalNum)
  local numTest = math.floor(testFrac * totalNum)
  local numEval = totalNum - numTrain - numTest
  torch.save(batchFile, miniBatches)
  torch.save(trainFile, miniBatches:sub(1, numTrain))
  torch.save(testFile, miniBatches:sub(numTrain + 1, numTrain + numTest))
  torch.save(evalFile, miniBatches:sub(numTrain + numTest + 1, numTrain + numTest + numEval))

end

--[[Reads in tensor files, and gets max seqence length.
While this is inneficient to do so, this method was made to prevent dependency
on the preprocesor.lua file to pass max sequence length forward. Making this design
decision allows for modularity between minibatch creation, and preprocessing the data.
]]
function MiniBatchLoader.getMaxSequenceLength(dataFile)
  local data = torch.load(dataFile)
  local maxLength = 0
  for key vec in ipairs(data) do
    if vec:size(1) > maxLength then
      maxLength = vec:size(1)
    end
  end
  return maxLength
end

function MiniBatchLoader.loadBatches(batchFile)
  local self = {}
  setmetatable(self, MiniBatchLoader)

  print('Loading previously allocated minibatches...')

end

function MiniBatchLoader.nextBatch()
end

function MiniBatchLoader.resetPointer()
end

return MiniBatchLoader
