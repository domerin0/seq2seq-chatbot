--[[
The purpose of this file is to enable the train.lua file to
'efficiently' create, load, and save the minibatches
]]
local TableUtils = require "Util.TableUtils"
local Preprocessor = require "Util.Preprocessor"
require 'lfs'
local MiniBatchLoader = {}
math.randomseed( os.time() )

function MiniBatchLoader.createMiniBatches(dataDir, batchSize, trainFrac,
  evalFrac, testFrac)

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

  local miniBatches = {}

  for key, value in pairs(dataFiles) do
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
      sourceTargetPairs[i] = torch.Tensor({source, target})
    end


    table.insert(miniBatches, sourceTargetPairs)
  end
    --save batch sets in appropriate t7 files
    print("Creating minibatch files...")
  for key, file in pairs(batchFiles) do
  --Due to memory constraints of data I have decided to split everything
  --into multiple files

    local batchFile = path.join(dataDir, "rawbatches/batch"..key..".t7")
    table.insert(batchFiles, batchFile)

    local trainFile = path.join(dataDir, "train/train"..key..".t7")
    table.insert(trainFiles, trainFile)

    local testFile = path.join(dataDir, "test/test"..key..".t7")
    table.insert(testFiles, testFile)

    local evalFile = path.join(dataDir, "eval/eval"..key..".t7")
    table.insert(evalFiles, evalFile)

    local miniBatches = torch.load(file)
    local totalNum = #miniBatches

    local numTrain = math.floor(trainFrac * totalNum)
    local numTest = math.floor(testFrac * totalNum)
    local numEval = totalNum - numTrain - numTest
    torch.save(batchFile, miniBatches)
    torch.save(trainFile, table.unpack(miniBatches, 1, numTrain))
    torch.save(testFile, table.unpack(miniBatches, numTrain + 1, numTrain + numTest))
    torch.save(evalFile, table.unpack(miniBatches, numTrain + numTest + 1, numTrain + numTest + numEval))
  end

end

function MiniBatchLoader.loadPreviousBatches(batchFile)
  local self = {}
  setmetatable(self, MiniBatchLoader)

  print('Loading previously allocated minibatches...')

end

function MiniBatchLoader.nextBatch()
end

function MiniBatchLoader.resetPointer()
end

return MiniBatchLoader
