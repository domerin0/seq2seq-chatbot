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
options.vocabSize = vMap:size()

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
local batchLoader = MiniBatchLoader.loadBatches(options.dataDir, options.batchSize,
  options.trainFrac, options.evalFrac, options.testFrac)

local chatbot = nil
if options.startFrom then
  chatbot = seq2seq.Seq2Seq(options.numLayers, options.hiddenSize, options.vocabSize, options.dropout)
else
  local checkpoint = torch.load(options.startFrom)
  chatbot = checkpoint.chatbot
  options = checkpoint.options
end

if options.gpuid > -1 then
  chatbot:cuda()
end
chatbot.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

local optimState = {learningRate = options.learningRate, momentum = options.momentum}
local iteration = 0
local maxIterations = options.maxEpochs * batchLoader.numBatches * batchLoader.batchSize
for epoch=1,options.maxEpochs do

  print("\n-- Epoch " .. epoch .. " / " .. options.maxEpochs)
  print("")
  local miniBatch = batchLoader:nextBatch()
  valLimit = (batchLoader.batchLimits[2][2] - batchLoader.batchLimits[2][1]) * batchLoader.batchSize
  local valLosses = torch.Tensor(valLimit):fill(0)
  local trainLosses = torch.Tensor(batchLoader.numBatches * batchLoader.batchSize):fill(0)
  local timer = torch.Timer()
  for batch=1,options.batchSize do
    iteration = iteration + 1
    --x, y = prepro(miniBatch[batch])
    local loss = chatbot:train(miniBatch[batch][1], miniBatch[batch][2], optimState)

    --Check for NaN
    if loss ~= loss then
      print("Critical error, stopping early!")
      break
    end

    trainLosses[iteration % (batchLoader.numBatches * batchLoader.batchSize)] = loss

    if iteration % 50 == 0 and options.lrDecay < 1 then
      if epoch >= options.lrDecayAfter then
        local decayFactor = options.lrDecay
        optimState.learningRate = optimState.learningRate * decay_factor -- decay it
        print('decayed learning rate by a factor ' .. decayFactor .. ' to ' .. optimState.learningRate)
      end
    end

  end

  local time = timer:time().real

    -- every now and then or on last iteration
    if minMeanError == nil or trainLosses:mean() < minMeanError then
      print("Performing test on cross validation set: ")
      batchLoader:resetPointer(2)
      local counter = 1
      while batchLoader.splitIndex == 2 do
        local miniBatch = batchLoader:nextBatch()
        for i=1,options.batchSize do
          local loss = chatbot:eval(miniBatch[i][1], miniBatch[i][2])
          valLosses[counter] = loss
          counter = counter + 1
        end
      end

      print("\n(Saving model ...)")
      minMeanError = trainLosses:mean()

      local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', checkpointDir, options.savefile, iteration / batchLoader.numBatches, valLosses:mean())
      print('saving checkpoint to ' .. savefile)
      local checkpoint = {}
      checkpoint.vocabSize = chatbot.vocabSize
      checkpoint.options = options
      checkpoint.trainLosses = trainLosses
      checkpoint.valLossess = valLosses
      checkpoint.batch =  batch
      checkpoint.epoch = epoch
      checkpoint.model = chatbot
      torch.save(savefile, checkpoint)
  end

  if iteration % options.printFreq == 0 then
    print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, time/batch = %.4fs", iteration, maxIterations, epoch, trainLosses:mean(), time))
  end

  if iteration % 10 == 0 then collectgarbage() end

  if trainLosses[1] < trainLosses:mean()*3 then
      print('Loss is exploding, aborting.')
      break
  end
end
