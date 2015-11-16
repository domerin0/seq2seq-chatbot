--[[
sentence is a torch tensor of indices
n is the context window size
]]--
t = torch.Tensor(5)
for i=1,5 do
  t[i] = i
end

function makeContextWindow(sentence, n)
  assert(n%2==1, "Context windows size must be odd..")
  assert(n>1, "Context window must be greater than 1...")
  local temp1 = (torch.ones(math.floor(n) / 2) * -1)
  local paddedSentence = torch.cat(temp1, sentence, 1):cat(temp1, 1)
  local context = torch.Tensor(sentence:size(1),n)
  for i=1,sentence:size(1) do
    local ind = 1
    --print("i= " .. i .. " j= "..i + n - 1)
    for j=(i),(i+n - 1) do
      context[i][ind] = paddedSentence[j]
      ind = ind + 1
    end
  end
  print(context)
  assert(#sentence == #context, "Something funny with sentence length...")
  return context
end

makeContextWindow(t, 11)
