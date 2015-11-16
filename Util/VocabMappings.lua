--[[
This file provides utility functions to go between words and indices and back
It also provides utility to go between words and vector embeddings and back
]]
local w2vutils = require "Util.w2vutils"
local VocabMapping = {}
VocabMapping.__index = VocabMapping

function VocabMapping.new(vocabFile, dicFile)
  local self = setmetatable({}, VocabMapping)
  self.vocabFile = torch.load(vocabFile)
  self.dicFile = torch.load(dicFile)
  return self
end

function VocabMapping.index2Token(index)
  for token, i in ipairs(self.vocabFile) do
    if i == index then
      return token
    end
  end
  --return UNK token if we somehow get to here
  return "UNK"
end

function VocabMapping.token2Index(token)
  return self.vocabFile[token]
end

function VocabMapping.token2Vec(token)
  return self.dicFile[self.vocabFile[token]]
end

function VocabMapping.vec2Token(vec, searchWidth)
  return value w2vutils:distance(vec,k)
end

return VocabMapping
