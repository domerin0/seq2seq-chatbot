--This file is influenced by:
-- Andrej Karpathy's https://github.com/karpathy/char-rnn/blob/master/train.lua


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

local VocabMapping = require "Util.VocabMapping"
vMap = VocabMapping.create(options.dataDir)
print(vMap:index2Token(607))
options.vocabSize = vMap:size()
print(vMap:size())
--Now, check and enable GPU usage:

local cuid = VerifyGPU.checkCuda(options.gpuid, options.seed)

if cuid < 1 then
  local clid = VerifyGPU.checkOpenCl(options.gpuid, options.seed)
end

if clid == -1 and cuid == -1 then
  options.gpuid = -1
end

--Split batches to test/train/cross val sets!
if MiniBatchLoader.shouldSplit(options.dataDir) then
  print("Splitting batches into proper train sets!")
  MiniBatchLoader.splitBatches(options.trainFrac, options.evalFrac, options.testFrac, options.dataDir)
  collectgarbage()
else
  print("Batches already split, moving them into memory!")
end

--Load minibatches into memory!

local batchLoader = MiniBatchLoader.loadBatches(options.dataDir, options.batchSize)

chatbot = seq2seq.Seq2Seq(options)

if options.gpuid > -1 then
  chatbot:cuda()
end

trainLosses = {}
valLosses = {}

local optimState = {learningRate = options.learningRate, alpha = options.decayRate}

for epoch=1,options.maxEpochs do

  print("\n-- Epoch " .. epoch .. " / " .. options.maxEpoch)
  print("")
  local miniBatch = batchLoader:nextBatch()

  for batch=1,options.batchSize do
    local timer = torch.Timer()
    x, y = prepro(miniBatch[batch])
    local loss, _, __ = chatbot:train(miniBatch[batch][1], miniBatch[batch][2], optimState)

    --Check for NaN
    if loss ~= loss then
      print("Critical error, stopping early!")
      break
    end

    trainLosses[#trainLosses + 1] = loss
  end

    local time = timer:time().real


    if epoch*batch % 50 == 0 and options.learningRateDecay < 1 then
        if epoch >= options.learningRateDecayAfter then
            local decayFactor = options.learningRateDecay
            optimState.learningRate = optimState.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decayFactor .. ' to ' .. optimState.learningRate)
        end
    end

    -- every now and then or on last iteration
    if epoch*batch % options.evalEvery == 0 or i == options.maxEpochs*options.batchSize then

      --TODO implement evaluation data set

      local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', options.checkpointDir, options.savefile, epoch, valLoss)
      print('saving checkpoint to ' .. savefile)
      local checkpoint = {}
      checkpoint.protos = chatbot.protos
      checkpoint.options = options
      checkpoint.trainLosses = trainLosses
      checkpoint.batch =  batch
      checkpoint.epoch = epoch
      checkpoint.vocab = loader.vocabMapping
      torch.save(options.savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.4fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if epoch*batch % 10 == 0 then collectgarbage() end

    if loss0 == nil then loss0 = loss end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break
    end
end

--[[remove -1 padding, and make contiguous in memory
]]
function prepro(source, target)
end
