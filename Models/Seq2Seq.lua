--[[
It may have been better to create a custom container (or module),
but for now I'll use this hacky solution to build the seq2seq model.
]]
local Seq2Seq = {}

--[[
Build a conversational model using encoder-decoder structure,
this assumes source vocab size = target vocab size
]]
function Seq2Seq.conversationalModel(embeddingSize, vocabSize,rnn_size, n, dropout)
  local encoder = LSTM.lstm(embeddingSize, vocabSize, rnn_size, n, dropout, true, true)
  local decoder = LSTM.lstm(embeddingSize, vocabSize, rnn_size, n, dropout, false, false)
  return {encoder, decoder}
end

return Seq2Seq
