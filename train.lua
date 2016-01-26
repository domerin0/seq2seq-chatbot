--This file is influenced by:
-- Andrej Karpathy's https://github.com/karpathy/char-rnn/blob/master/train.lua


local VocabMapping = require "Util.VocabMapping"
local CommandLineArgs = require "Util.CommandLineArgs"
local Preprocessor = require "Preprocessor"
local MiniBatchLoader = require "MiniBatchLoader"
local VerifyGPU = require "Util.VerifyGPU"
require "seq2seq"

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

local cuid = VerifyGPU.checkCuda(options.gpuid, options.seed)

if cuid < 1 then
  local clid = VerifyGPU.checkOpenCl(options.gpuid, options.seed)
end

if (clid == -1 or clid == nil) and cuid == -1 then
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
    for _=1,batchLoader.numBatches do
      local trainBatch = batchLoader:nextBatch(1)
      local timer = torch.Timer()
      local losses = 0
      local loss = 0
      for batch=1,batchLoader.batchSize do
        iteration = iteration + 1

        local input, target = trainBatch[batch][1], trainBatch[batch][2]
        if options.gpuid > -1 then
          input = input:contiguous():cuda()
          target = target:contiguous():cuda()
        end
        loss = chatbot:train(input, target, optimState)

        --Check for NaN
        if loss ~= loss then
          print("Critical error, stopping early!")
          break
        end

        losses = losses + (loss / printEvery)

      end

      local time = timer:time().real
--    Do this stuff (run test set, print some output to console, etc..)
--  Every so often
    if math.floor(iteration / batchLoader.numBatches) % 4 == 0 then
      print(string.format("Percentage of Epoch done: %d", math.floor(iteration / batchLoader.numBatches)))
    end
      if  math.floor((iteration / batchLoader.batchSize)) %  printEvery == 0  then
        table.insert(trainLosses, losses)
        losses = 0
        loss = 0
        --decay learning rate if no improvement in 3 steps
        if(#trainLosses > 2) then
          if(loss  > unpack(trainLosses)) then
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
            loss = chatbot:eval(miniBatch[i][1], miniBatch[i][2])
            counter = counter + 1
            losses = loss + losses
          end
          batch = batchLoader:nextBatch(2)
        end
        table.insert(testLosses, losses / counter)
        print("Train loss: "..trainLosses[#trainLosses] .. " Test Loss: " ..testLosses[#testLosses])
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, time/batch = %.4fs", iteration, maxIterations, epoch, trainLosses[#trainLosses], time))
        print("\n(Saving model ...)")
        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', checkpointDir, options.savefile, iteration / maxIterations, testLoss)
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
