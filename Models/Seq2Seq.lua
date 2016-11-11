
--modified: https://github.com/macournoyer/neuralconvo/blob/master/seq2seq.lua
--added multiple layers and dropout
local Seq2Seq = torch.class("seq2seq.Seq2Seq")

function Seq2Seq:__init(numLayers, hiddenSize, vocabSize, dropout, dataDir)
  self.dropout = dropout or 0
  self.vocabSize = vocabSize
  self.hiddenSize = hiddenSize
  self.numLayers = numLayers
  local Constants = require "Util.Constants"
  local vocabPath = path.join(dataDir, Constants.vocabFile)
  local vocabFile = torch.load(vocabPath)
  self.goToken = vocabFile[Constants.GO]
  self.eosToken = vocabFile[Constants.EOS]
  self:buildModel()
end

function Seq2Seq:buildModel()

  print("Building model with:")
  print("Hidden size of: ".. self.hiddenSize)
  print("Number of layers: "..self.numLayers)
  print("Dropout: ".. self.dropout)
  self.encoder = nn.Sequential()
  self.encoder:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  self.encoder:add(nn.SplitTable(1,2))
  self.encoderHidden = {}
  for i=1,self.numLayers do
    self.encoderHidden[i] = nn.LSTM(self.hiddenSize, self.hiddenSize)
    self.encoder:add(nn.Sequencer(self.encoderHidden[i]))
    self.encoder:add(nn.Sequencer(nn.Dropout(self.dropout)))
  end
  self.encoder:add(nn.SelectTable(-1))

  self.decoder = nn.Sequential()
  self.decoder:add(nn.LookupTable(self.vocabSize, self.hiddenSize))
  self.decoder:add(nn.SplitTable(1,2))
  self.decoderHidden = {}
  for i=1,self.numLayers do
    self.decoderHidden[i] = nn.LSTM(self.hiddenSize, self.hiddenSize)
    self.decoder:add(nn.Sequencer(self.decoderHidden[i]))
    self.decoder:add(nn.Sequencer(nn.Dropout(self.dropout)))
  end
  self.decoder:add(nn.Sequencer(nn.Linear(self.hiddenSize, self.vocabSize)))
  self.decoder:add(nn.Sequencer(nn.LogSoftMax()))

  self.criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

end

function Seq2Seq:cuda()
  self.encoder:cuda()
  self.decoder:cuda()

  if self.criterion then
    self.criterion:cuda()
  end
end

function Seq2Seq:forwardConnect(inputSeqLength)
    self.decoderHidden[1].userPrevOutput =
      nn.rnn.recursiveCopy(
        self.decoderHidden[1].userPrevOutput,
        self.encoderHidden[self.numLayers].outputs[inputSeqLength]
      )

    self.decoderHidden[1].userPrevCell =
      nn.rnn.recursiveCopy(
        self.decoderHidden[1].userPrevCell,
        self.encoderHidden[self.numLayers].cells[inputSeqLength]
      )
end

function Seq2Seq:backwardConnect()
    self.encoderHidden[self.numLayers].userNextGradCell =
      nn.rnn.recursiveCopy(
        self.encoderHidden[self.numLayers].userNextGradCell,
        self.decoderHidden[1].userGradPrevCell
      )

    self.encoderHidden[self.numLayers].gradPrevOutput =
      nn.rnn.recursiveCopy(
        self.encoderHidden[self.numLayers].gradPrevOutput,
        self.decoderHidden[1].userGradPrevOutput
      )
end


function Seq2Seq:eval(input, target)
  local encoderInput = input
  local decoderInput = target:sub(1, -2)
  local decoderTarget = target:sub(2, -1)

  -- Forward pass
  self.encoder:forward(encoderInput)
  self:forwardConnect(encoderInput:size(1))
  local decoderOutput = self.decoder:forward(decoderInput)

  -- convert to table
  decoderTargetTable = {}
  for i=1,decoderTarget:size(1) do
    decoderTargetTable[i] = decoderTarget[i]
  end

  local loss = self.criterion:forward(decoderOutput, decoderTargetTable)

  if loss ~= loss then
    return loss
  end
  return loss
end

function Seq2Seq:train(input, target, optimState)
  local encoderInput = input
  local decoderInput = target:sub(1, -2)
  local decoderTarget = target:sub(2, -1)

  -- Forward pass
  encoderOutput = self.encoder:forward(encoderInput)
  self:forwardConnect(encoderInput:size(1))
  local decoderOutput = self.decoder:forward(decoderInput)

  -- convert to table
  decoderTargetTable = {}
  for i=1,decoderTarget:size(1) do
    decoderTargetTable[i] = decoderTarget[i]
  end

  local loss = self.criterion:forward(decoderOutput, decoderTargetTable)

  if loss ~= loss then
    return loss
  end

-- Backward

  local gradLoss = self.criterion:backward(decoderOutput, decoderTargetTable)

  self.decoder:backward(decoderInput, gradLoss)
  self:backwardConnect()
  self.encoder:backward(encoderInput, encoderOutput)

  self.encoder:updateGradParameters(optimState.momentum)
  self.decoder:updateGradParameters(optimState.momentum)
  self.decoder:updateParameters(optimState.learningRate)
  self.encoder:updateParameters(optimState.learningRate)
  self.encoder:zeroGradParameters()
  self.decoder:zeroGradParameters()

  --self.decoder:forget()
  --self.encoder:forget()

  return loss
end

--This function is heavily borrowed from:
--https://github.com/macournoyer/neuralconvo/blob/master/seq2seq.lua
function Seq2Seq:predict(input)
  self.encoder:forward(input)
  self:forwardConnect(input:size(1))

  local predictions = {}
  local probabilities = {}

  local output = self.goToken
  --limit output to 50 for now
  for i = 1, 50 do
    local prediction = self.decoder:forward(torch.Tensor{output})[1]

    local prob, wordIds = prediction:sort(1, true)

    output = wordIds[1]

    if output == self.eosToken then
      break
    end

    table.insert(predictions, wordIds)
    table.insert(probabilities, prob)
  end

  --self.decoder:forget()
  --self.encoder:forget()

  return predictions, probabilities
end
