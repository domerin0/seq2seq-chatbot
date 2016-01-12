

local TableUtils = {}

--[[
If a sequence has n elements, this function reverses the first n-1 elements.
This is a function to try out the method Google proposed of reversing the source
sentence for a better source-target dependency relationship.
]]
function TableUtils.reverseTable(sequence)
  local reversed = {}
  local counter = 1
  for index=#sequence - 1,1,-1 do
    reversed[counter] = sequence[index]
    counter = counter + 1
  end
  --Add EOS token to end (don't swap nth element)
  reversed[#reversed + 1] = sequence[#sequence]
  return reversed
end

--[[
Another function that doesn't quite fit here, but is here temporarily
returns a tensor where tensor[i] = i. (takes length as argument)
]]
function TableUtils.indexTensor(indexLength)
  local temp = torch.LongTensor(indexLength)
  for i=1,indexLength do
    temp[i] = i
  end
  return temp
end

--[[Not really a table function, but I'll leave this here for now
swap tensor in place. Assumes 1d tensor (otherwise this doesn't make sense)
]]
function TableUtils.reverseTensor(tensor)
  counter = 1
  for i=tensor:size(1),1,-1 do
    tensor[i], tensor[counter] = tensor[counter],tensor[i]
    counter = counter + 1
  end
  return tensor
end

--Need to add check for numerical types!!
function TableUtils.sum(t)
  local sum = 0
  for key, value in ipairs(t) do
    sum = sum + value
  end
  return sum
end

--This function pads the table with -1
function TableUtils.padTable(t, maxLength)
  if #t < maxLength then
    for i=#t + 1, maxLength do
      t[i] = -1
    end
  end
  return t
end

--function taken from https://coronalabs.com/blog/2014/09/30/tutorial-how-to-shuffle-table-items/
function TableUtils.shuffleTable(t)
    local rand = math.random
    assert( t, "shuffleTable() expected a table, got nil" )
    local iterations = #t
    local j

    for i = iterations, 2, -1 do
        j = rand(i)
        t[i], t[j] = t[j], t[i]
    end
end


--[[
Taken from: https://github.com/karpathy/char-rnn/blob/4297a9bf69726823d944ad971555e91204f12ca8/util/misc.lua
]]
function cloneList(tensor_list, zero_too)
    -- utility function. todo: move away to some utils file?
    -- takes a list of tensors and returns a list of cloned tensors
    local out = {}
    for k,v in pairs(tensor_list) do
        out[k] = v:clone()
        if zero_too then out[k]:zero() end
    end
    return out
end

function TableUtils.makeContextWindow(sentence, n)
  assert(n%2==1, "Context windows size must be odd..")
  assert(n>1, "Context window must be greater than 1...")
  local temp1 = (torch.ones(math.floor(n) / 2) * -1)
  local paddedSentence = torch.cat(temp1, sentence, 1):cat(temp1, 1)
  local context = torch.Tensor(sentence:size(1),n)
  for i=1,sentence:size(1) do
    local ind = 1
    for j=(i),(i+n - 1) do
      context[i][ind] = paddedSentence[j]
      ind = ind + 1
    end
  end
  print(context)
  assert(#sentence == #context, "Something funny with sentence length...")
  return context
end

return TableUtils
