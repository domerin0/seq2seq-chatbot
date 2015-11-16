--[[
The purpose of this file is to enable the train.lua file to
'efficiently' create, load, and save the minibatches
]]
local TableUtils = require "Util.TableUtils"
local MiniBatchLoader = {}
math.randomseed( os.time() )

function MiniBatchLoader.createMiniBatches(dataFiles, batchSize, trainFrac,
  evalFrac, testFrac)
  --Checks to ensure user isn't testing us
  assert(evalFrac >= 0 and evalFrac < 1, "evalFrac not between 0 and 1...")
  assert(trainFrac > 0 and trainFrac <= 1, "trainFrac not between 0 and 1...")
  assert(testFrac >= 0 and testFrac < 1, "testFrac not between 0 and 1...")
  assert(testFrac + evalFrac + trainFrac == 1, "eval, train and test don't add up to 1!")

  print("Using ".. trainFrac .. " As percentage of data to train on...")
  print("Using ".. evalFrac .. " As percentage of data to validate on...")
  print("Using " .. testFrac .. " As percentage of data to test on...")

  local batchFiles  = {}
  local trainFiles = {}
  local testFiles = {}
  local evalFiles = {}

  print("Loading data...")

  local trainingPairSize = {}

  for key, value in pairs(dataFiles) do
    --Due to memory constraints of data I have decided to split everything
    --into multiple files
    local batchFile = path.join(dataDir, "rawbatches/batch"..key..".t7")
    table.insert(batchFiles, batchFile)

    local trainFile = path.join(dataDir, "train/train.t7")
    table.insert(trainFiles, trainFile)

    local testFile = path.join(dataDir, "test/test.t7"))
    table.insert(testFiles, testFile)

    local evalFile = path.join(dataDir, "eval/eval.t7")
    table.insert(evalFiles, evalFile)

    local data = torch.load(value)
    print(#data)
    --Sort sequences by length!
    print("Finding longest sequence length...")
    table.sort(data,
      function (a,b)
        return (#a > #b)
      end
      )

    local maxSequenceLength = #data[1]
    --reload values since order of them does matter for training purposes
    data = torch.load(value)
    print("Length of longest sequence is " .. maxSequenceLength .. "...")
    local sourceTargetPairs = {}

    -- Reverse sequences to introduce short-term dependency's (Google's result)
    --Insert training pairs into table
    for i=1,#data-1 do
      local source = TableUtils.padTable(TableUtils.reverseTable(data[i]),
      maxSequenceLength)
      data[i] = nil
      local target = TableUtils.padTable(data[i + 1], maxSequenceLength)
      table.insert(sourceTargetPairs, {source, target})
    end
    local miniBatches = {}
    local numBatches = math.floor(#sourceTargetPairs / batchSize)
    local counter = 0
    for i=1,#numBatches do
      local batch = {}
      for j=1,batchSize do
        counter = counter + 1
        local torchifiedBatch = sourceTargetPairs[counter]
        table.insert(batch, torchifiedBatch)
      end
      table.insert(miniBatches, torch.Tensor(batch))
    end

      --shuffle the minibatches in place
    TableUtils.shuffleTable(miniBatches)
    table.insert(trainingPairSize, #miniBatches)
    torch.save(batchFile, miniBatches)
  end

      --create train/test/eval batch sets
  local totalNum = TableUtils.sum(trainingPairSize)
  local numTrain = math.floor(trainFrac * totalNum)
  local numTest = math.floor(testFrac * totalNum)
  local numEval = totalNum - numTrain - numTest

    --save batch sets in appropriate t7 files
  for key, file in pairs(batchFiles)
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
