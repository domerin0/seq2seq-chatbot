--[[
This is based roughly off:
https://github.com/macournoyer/neuralconvo/blob/master/seq2seq.lua
and:

The main differences are not using rnn or nnx package,
and having dropout/customizable layers.
]]


local Seq2Seq = torch.class("Models.Seq2Seq")
local model_utils = require "Util.model_utils"
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
      end
      self.protos.criterion = nn.ClassNLLCriterion()
  end

  self.init_state = {}
  for L=1,self.options.numLayers do
    local h_init = torch.zeros(self.options.batchSize,
      self.options.rnnSize)
    if self.options.gpuid >=0 and self.options.opencl == 0
      then h_init = h_init:cuda() end
    if self.options.gpuid >=0 and self.options.opencl == 1
      then h_init = h_init:cl() end
    table.insert(self.init_state, h_init:clone())
    table.insert(self.init_state, h_init:clone())
  end

  cuda()

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

end

function Seq2Seq:cuda()
  if self.options.gpuid >= 0 and self.options.opencl == 0 then
    for k,v in pairs(self.protos) do v:cuda() end
  end
  if self.options.gpuid >= 0 and self.options.opencl == 1 then
    for k,v in pairs(self.protos) do v:cl() end
  end
end

--Copy hidden state from encoder to decoder
function Seq2Seq:forwardConnect()

end

function Seq2Seq:backwardConnect()
end

function Seq2Seq:train(input, target)
  local encoderInput = input
  local decoderInput = target:sub(1,-2)
  local decoderTarget = target:sub(2,-1)

  --forward pass

  local rnnState = {[0] = initStateGlobal}
  local predictions = {}
  local loss = 0


  self.protos.encoder:forward(encoderInput)

  --backward pass


end

function Seq2Seq:eval(input)
end
