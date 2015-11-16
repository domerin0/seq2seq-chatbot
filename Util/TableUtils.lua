

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
