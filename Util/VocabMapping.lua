--[[
This file provides utility functions to go between words and indices and back
It also provides utility to go between words and vector embeddings and back
]]
local Constants = require "Util.Constants"
local VocabMapping = {}
VocabMapping.__index = VocabMapping

function VocabMapping.create(dataDir)
  local self = {}
  setmetatable(self, VocabMapping)

  local vocabPath = path.join(dataDir, Constants.vocabFile)
  local dicPath = path.join(dataDir, Constants.dicFile)
  local vocabFile = torch.load(vocabPath)
  local dicFile = torch.load(dicPath)
  self.vocab = vocabFile
  self.dictionary = dicFile
  return self
end

function VocabMapping:index2Token(index)
  local token = self.dictionary[index]
  if not token then
    local eos = self.vocab[Constants.EOS]
    token = self.dictionary[eos]
  end
  return token
end


function VocabMapping:token2Index(token)
  local index = self.vocab[token]
  if not index then
    index = self.vocab[Constants.EOS]
  end
  return index
end

function VocabMapping:size()
  local count = 0
  for key, value in pairs(self.vocab) do
    count = count + 1
  end
  return count
end

return VocabMapping
