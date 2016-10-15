--This file is influenced by:
-- Andrej Karpathy's https://github.com/karpathy/char-rnn/blob/master/train.lua


local VocabMapping = require "Util.VocabMapping"
local CommandLineArgs = require "Util.CommandLineArgs"
local Preprocessor = require "Preprocessor"
local MiniBatchLoader = require "MiniBatchLoader"
local VerifyGPU = require "Util.VerifyGPU"
require "seq2seq"

--------Preprocessing functions-------------------

function cpuPrepro(input, target)
  input = input:contiguous()
  target = target:contiguous()
  return input, target
end

function clPrepro(input, target)
  input = input:contiguous():cl()
  target = target:contiguous():cl()
  return input, target
end

function cudaPrepro(input, target)
  input = input:contiguous():cuda()
  target = target:contiguous():cuda()
  return input, target
end

-------end preprocessing functions------------------



local options = CommandLineArgs.trainCmdArgs()
torch.manualSeed(options.seed)
--catch this issue early
assert(options.printFreq <= options.maxEpochs, "Must have printFreg <= max epochs")
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
  MiniBatchLoader.createMiniBatches(options)
  collectgarbage()
else
  print("Minibatches already created before, moving on...")
end

local vMap = VocabMapping.create(options.dataDir)
options.vocabSize = vMap.size

print("Vocab size is: "..options.vocabSize)
--Now, check and enable GPU usage:
--set preprocessor funtion here to prevent call if on every tick
local prepro = cudaPrepro
local cuid = VerifyGPU.checkCuda(options.gpuid, options.seed)

if cuid < 0 then
  prepro = clPrepro
  local clid = VerifyGPU.checkOpenCl(options.gpuid, options.seed)
end

if (clid == -1 or clid == nil) and cuid == -1 then
  prepro = cpuPrepro
  options.gpuid = -1
end

--Load minibatches into memory!
local checkpointDir = path.join(options.dataDir, options.checkpoints)
lfs.mkdir(checkpointDir)
local batchLoader = MiniBatchLoader.loadBatches(options.dataDir, options.trainFrac)

local chatbot = nil
if options.startFrom then
  chatbot = seq2seq.Seq2Seq(options.numLayers, options.hiddenSize,
  options.vocabSize, options.dropout, options.dataDir)
else
  local checkpoint = torch.load(options.startFrom)
  chatbot = checkpoint.chatbot
  options = checkpoint.options
end

if options.gpuid > -1 then
  chatbot:cuda()
end
local optimState = {learningRate = options.learningRate, momentum = options.momentum}
local iteration = 0
local maxIterations = options.maxEpochs * batchLoader.numBatches * batchLoader.batchSize
local printEvery =  math.floor(options.printFreq * batchLoader.numBatches)
print(string.format("There are %s batches per epoch: ", batchLoader.numBatches))
local testLosses = {}
local trainLosses = {}

for epoch=1,options.maxEpochs do

  print("\n-- Epoch " .. epoch .. " / " .. options.maxEpochs)
  print("")
    local losses = 0
    for batch=1,batchLoader.numBatches do
      local trainBatch = batchLoader:nextBatch(1)

-------------I am unrolling the for loop here in an attempt to speed up the code--------------

      local b = batchLoader.batchSize % 4
      if b ~= 0 then
        for i=1,b do
          trainBatch[i][1], trainBatch[i][2] = prepro(trainBatch[i][1], trainBatch[i][2])
        end
      end
      for i=1,batchLoader.batchSize,4 do
        trainBatch[i][1], trainBatch[i][2] = prepro(trainBatch[i][1], trainBatch[i][2])
        trainBatch[i + 1][1], trainBatch[i + 1][2] = prepro(trainBatch[i + 1][1], trainBatch[i + 1][2])
        trainBatch[i + 2][1], trainBatch[i + 2][2] = prepro(trainBatch[i + 2][1], trainBatch[i + 2][2])
        trainBatch[i + 3][1], trainBatch[i + 3][2] = prepro(trainBatch[i + 3][1], trainBatch[i + 3][2])
      end

-------------end of for loop unroll------------------------------------------------------

      local timer = torch.Timer()
      local loss = 0
      for example=1,batchLoader.batchSize do
        iteration = iteration + 1
        local input, target = trainBatch[example][1], trainBatch[example][2]
        loss = chatbot:train(input, target, optimState)

        losses = losses + (loss / batchLoader.batchSize)

      end

      local time = timer:time().real
--    Do this stuff (run test set, print some output to console, etc..)
--  Every so often
    if math.floor(iteration / batchLoader.batchSize) % 10 == 0 then
      print(string.format("Batch took: %.4fs, percent of epoch done: %.4f, loss: %.4f",time, batch / batchLoader.numBatches, losses / iteration))
    end
      if  batch %  printEvery == 0  then
        table.insert(trainLosses, losses / printEvery)
        losses = 0
        loss = 0
        --decay learning rate if no improvement in 3 steps
        if(#trainLosses > 2) then
          if(trainLosses[#trainLosses]  > trainLosses[#trainLosses - 2]) then
            print('decayed learning rate by a factor ' .. decayFactor .. ' to ' .. optimState.learningRate)
            optimState.learningRate = optimState.learningRate * options.lrDecay
            local decayFactor = options.lrDecay
          end
        end
        print("Evaluating test set: ")
        local batch = batchLoader:nextBatch(2)
        local counter = 1
        while batch do
          batch = batchLoader:nextBatch(2)
          for i=1,batchLoader.batchSize do
            loss = chatbot:eval(batch[i][1], batch[i][2])
            losses = losses + (loss / batchLoader.batchSize)
          end
          counter = counter + 1
          batch = batchLoader:nextBatch(2)
        end
        table.insert(testLosses, losses / counter)
        print("Train loss: "..trainLosses[#trainLosses] .. " Test Loss: " ..testLosses[#testLosses])
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, time/batch = %.4fs", iteration, maxIterations, epoch, trainLosses[#trainLosses], time))
        print("\n(Saving model ...)")
        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', checkpointDir, options.savefile, iteration / maxIterations, testLosses[#testLosses])
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.vocabSize = chatbot.vocabSize
        checkpoint.options = options
        checkpoint.trainLosses = trainLosses
        checkpoint.valLossess = valLosses
        checkpoint.epoch = epoch
        checkpoint.model = chatbot
        torch.save(savefile, checkpoint)
      end

  end
  collectgarbage()

end
