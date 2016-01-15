--[[
Taken and adapted from  from : https://github.com/karpathy/char-rnn/blob/master/model/LSTM.lua
Main changes are removing the one-hot encoding in favor of lookup table
]]

--[[
vocabSize: size of vocabulary
rnnSize: number of neurons in each hidden layer
n: number of hidden layers
dropout: add a dropout layer to prevent overfitting
]]
require 'nn'
local LSTM = {}
function LSTM.lstm(vocabSize, rnnSize, n, dropout, isEncoder)
  dropout = dropout or 0

  -- there will be 2*n+1 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n do
    -- c,h from previos timesteps
    local prev_h = inputs[L*2+1]
    local prev_c = inputs[L*2]
    -- the input to this layer
    if L == 1 then
        x = nn.LookupTable(vocabSize, rnnSize)(inputs[1])
        input_size_L = vocabSize
    else
      x = outputs[(L-1)*2]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnnSize
    end
    -- evaluate the input sums at once for efficiency
    local i2h = nn.Linear(input_size_L, 4 * rnnSize)(x):annotate{name='i2h_'..L}
    local h2h = nn.Linear(rnnSize, 4 * rnnSize)(prev_h):annotate{name='h2h_'..L}
    local all_input_sums = nn.CAddTable()({i2h, h2h})

    local reshaped = nn.Reshape(4, rnnSize)(all_input_sums)
    local n1, n2, n3, n4 = nn.SplitTable(2)(reshaped):split(4)
    -- decode the gates
    local in_gate = nn.Sigmoid()(n1)
    local forget_gate = nn.Sigmoid()(n2)
    local out_gate = nn.Sigmoid()(n3)
    -- decode the write inputs
    local in_transform = nn.Tanh()(n4)
    -- perform the LSTM update
    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
      })
    -- gated cells form the output
    local next_h = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    table.insert(outputs, next_c)
    table.insert(outputs, next_h)
  end

  -- set up the decoder
  local top_h = outputs[#outputs]
  if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
  local proj = nn.Linear(rnnSize, vocabSize)(top_h):annotate{name='decoder'}
  if not isEncoder then
    local logsoft = nn.LogSoftMax()(proj)
    table.insert(outputs, logsoft)
  else
    local selection = nn.SelectTable(-1)(proj)
    table.insert(outputs, selection)
  end

  return nn.gModule(inputs, outputs)
end

return LSTM
