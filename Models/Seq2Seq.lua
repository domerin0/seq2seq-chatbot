--[[
This is based roughly off:
https://github.com/macournoyer/neuralconvo/blob/master/seq2seq.lua
and:

The main differences are not using rnn or nnx package,
and having dropout/customizable layers.
]]


local Seq2Seq = torch.class("seq2seq.Seq2Seq")
local model_utils = require "Util.model_utils"
local Constants = require "Util.Constants"
--[[
Build a conversational model using encoder-decoder structure,
this assumes source vocab size = target vocab size
(this couldn't be used for translation yet)
]]
function Seq2Seq:__init(options)
  self.options = options
  buildModel()

end

function Seq2Seq:buildModel()
  if not path.exists(self.options.checkpointDir) then lfs.mkdir(self.options.checkpointDir) end

  -- define the model: prototypes for one timestep, then clone them in time
  local do_random_init = true
  if string.len(self.options.initFrom) > 0 then
      print('loading a model from checkpoint ' .. self.options.initFrom)
      local checkpoint = torch.load(self.options.initFrom)
      self.protos = checkpoint.protos
      -- make sure the vocabs are the same
      local vocab_compatible = true
      local checkpoint_vocab_size = 0
      for c,i in pairs(checkpoint.vocab) do
          if not (vocab[c] == i) then
              vocab_compatible = false
          end
          checkpoint_vocab_size = checkpoint_vocab_size + 1
      end
      if not (checkpoint_vocab_size == vocab_size) then
          vocab_compatible = false
          print('checkpoint_vocab_size: ' .. checkpoint_vocab_size)
      end
      assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
      -- overwrite model settings based on checkpoint to ensure compatibility
      print('overwriting rnnSize=' .. checkpoint.opt.rnnSize .. ', numLayers=' .. checkpoint.opt.numLayers .. ' based on the checkpoint.')
      self.options.rnnSize = checkpoint.opt.rnnSize
      self.options.numLayers = checkpoint.opt.numLayers
      self.options.model = checkpoint.opt.model
      do_random_init = false
  else
      print('creating an ' .. self.options.model .. ' with ' .. self.options.numLayers .. ' layers')
      self.protos = {}
      self.protos.encoder = LSTM.lstm(self.options.embeddingSize, self.options.vocabSize,
        self.options.rnnSize, self.options.n, self.options.dropout, true)
      self.protos.decoder = LSTM.lstm(self.options.embeddingSize, self.options.vocabSize,
        self.options.rnnSize,self.options.n, self.options.dropout, false)
      self.protos.criterion = nn.ClassNLLCriterion()
  end

  self.encoderInitState = {}
  self.decoderInitState = {}
  for L=1,self.options.numLayers do
    local h_init = torch.zeros(self.options.batchSize,
      self.options.rnnSize)
    if self.options.gpuid >=0 and self.options.opencl == 0
      then h_init = h_init:cuda() end
    if self.options.gpuid >=0 and self.options.opencl == 1
      then h_init = h_init:cl() end
    table.insert(self.encoderInitState, h_init:clone())
    table.insert(self.encoderInitState, h_init:clone())

    table.insert(self.decoderInitState, h_init:clone())
    table.insert(self.decoderInitState, h_init:clone())
  end

  self.encoderInitStateGlobal = TableUtils.cloneList(encoderInitState)

  encoderParams, encoderGradParams = model_utils.combine_all_parameters(self.protos.encoder)
  decoderParams, decoderGradParams = model_utils.combine_all_parameters(self.protos.decoder)


  if do_random_init then
    encoderParams:uniform(-0.08, 0.08)
    decoderParams:uniform(-0.08, 0.08)
  end

  self.clones = {}
  for name,proto in pairs(self.protos) do
    print('cloning ' .. name)
    self.clones[name] = model_utils.clone_many_times(self.proto,
      self.options.maxSeqLength, not self.proto.parameters)
  end

  self.clones.encoder:zeroGradParameters()
  self.clones.decoder:zeroGradParameters()

end

function Seq2Seq:cuda()
  if self.options.gpuid >= 0 and self.options.opencl == 0 then
    for k,v in pairs(self.protos) do v:cuda() end
  end
  if self.options.gpuid >= 0 and self.options.opencl == 1 then
    for k,v in pairs(self.protos) do v:cl() end
  end
end

--[[
TODO clean up function (maybe remove some code duplication)
]]
function Seq2Seq:train(input, target)
  local encoderInput = input
  local decoderInput = target:sub(1,-2)
  local decoderTarget = target:sub(2,-1)

  local loss = 0
  local predictions  = {}

  --forward pass--------------------------------------

  local rnnEncoderState = {[0] = self.encoderInitStateGlobal}

  for i=1,encoderInput:size(1) do
    self.clones.encoder[i]:training()
    local encoderOut = self.clones.encoder:forward{
      encoderInput[i],
      unpack(rnnEncoderState[i-1])
    }
    rnnEncoderState[i] = {}

    for j=1,#encoderInitState do
      table.insert(rnnEncoderState[i], encoderOut[i])
    end
  end

--Pass encoder hidden state to decoder

  local rnnDecoderState = {[0] = rnnEncoderState[input:size(1)]}

  local decoderOut = {}
  for i=1,decoderInput:size(1) do
    self.clones.decoder[i]:training()
    decoderOut = self.clones.decoder:forward{
      decoderInput[i],
      unpack(rnnDecoderState[i-1])
    }

    for j=1,#decoderInitState do
      table.insert(rnnDecoderState[j], decoderOut[j])
    end
    predictions[i] = decoderOut[#decoderOut]
    self.loss = self.loss + self.clones.criterion[i]:forward(
      predictions[i],
      decoderTarget[i])
  end


  --backward pass---------------------------------------------------

  local drnnState = {[decoderInput:size(1)] = clone_list(
    decoderInitStateGlobal,
    true)}
    for t=decoderInput:size(1),1,-1 do
        -- backprop through loss, and softmax/linear
        local decoderGrad = self.clones.criterion[t]:backward(
          predictions[t],
          decoderTarget[t])
        table.insert(drnn_state[t], decoderGrad)

        local dlst = self.clones.decoder[t]:backward({
          decoderInput[t],
          unpack(rnnDecoderState[t-1])
          }, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then
                drnn_state[t-1][k-1] = v
            end
        end
    end

    --copy decoder gradients to encoder

    local ernnState = {[encoderInput:size(1)] = drnnState[1]}

      for t=encoderInput:size(1),1,-1 do

          local dlst = self.clones.decoder[t]:backward({
            encoderInput[t],
            unpack(rnnEncoderState[t-1])
            }, ernnState[t])
          ernnState[t-1] = {}
          for k,v in pairs(dlst) do
              if k > 1 then
                  ernnState[t-1][k-1] = v
              end
          end
      end

    encoderInitStateGlobal = decoderInitState[#decoderInitState]

    self.clones.encoder:updateParameters(self.options.learningRate)
    self.clones.decoder:updateParameters(self.options.learningRate)

    self.clones.encoder:zeroGradParameters()
    self.clones.decoder:zeroGradParameters()

    encoderGradParams:clamp(-self.options.gradClip,
      self.options.gradClip)
    decoderGradParams:clamp(-self.options.gradClip,
      self.options.gradClip)

    return loss
end

--[[
prediction
]]
function Seq2Seq:eval(input)

  local predictions = {}
  local probabilities = {}
  --will probably refactor this to command line arg
  local maxOutputSize = 100


  local rnnEncoderState = {[0] = self.encoderInitStateGlobal}

  for i=1,encoderInput:size(1) do
    self.clones.encoder[i]:evaluate()
    local encoderOut = self.clones.encoder:forward{
      encoderInput[i],
      unpack(rnnEncoderState[i-1])
    }
    rnnEncoderState[i] = {}

    for j=1,#encoderInitState do
      table.insert(rnnEncoderState[i], encoderOut[i])
    end
  end

--Pass encoder hidden state to decoder

  local rnnDecoderState = {[0] = rnnEncoderState[input:size(1)]}

--Keep going until EOS is reached

  local decoderOutput = Constants.GO

  for i=1,maxOutputSize do
    self.clones.decoder[i]:evaluate()
    local prediction = self.clones.decoder:forward{
      decoderOutput,
      unpack(rnnDecoderState[i-1])
    }

    for j=1,#decoderInitState do
      table.insert(rnnDecoderState[j], prediction[j])
    end

    local prob, wordId = prediction:sort(1, true)

    decoderOutput = prediction[1]

    if output == Constants.EOS then
      break
    end

    table.insert(predictions, wordId)
    table.insert(probabilities, prob)
  end

  return predictions, prbabilities

end
