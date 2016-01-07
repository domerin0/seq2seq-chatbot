--[[
This is based off:
https://github.com/macournoyer/neuralconvo/blob/master/seq2seq.lua
The main differences are not using rnn or nnx package,
and having dropout/customizable layers.
]]


local Seq2Seq = torch.class("Models.Seq2Seq")

--[[
Build a conversational model using encoder-decoder structure,
this assumes source vocab size = target vocab size
(this couldn't be used for translation yet)
]]
function Seq2Seq:__init(embeddingSize, vocabSize,rnnSize, n, dropout)
  self.vocabSize = assert(vocabSize, "vocabSize required at arg #1")
  self.hiddenSize = assert(rnnSize, "hiddenSize required at arg #2")
  self.encoder = LSTM.lstm(embeddingSize, vocabSize, rnnSize, n, dropout, true)
  self.decoder = LSTM.lstm(embeddingSize, vocabSize, rnnSize, n, dropout, false)

end

function Seq2Seq:forwardConnect()
end

function Seq2Seq:backwardConnect()
end

function Seq2Seq:train(input, target)
end

function Seq2Seq:eval(input)
end
