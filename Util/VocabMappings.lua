--[[
This file provides utility functions to go between words and indices and back
It also provides utility to go between words and vector embeddings and back
]]
local w2vutils = require "Util.w2vutils"
local Constants = require "Util.Constants"
local VocabMapping = {}
VocabMapping.__index = VocabMapping

function VocabMapping.create(vocabFile, indexFile)
  local self = setmetatable({}, VocabMapping)
  self.vocabFile = torch.load(vocabFile)
  self.indexFile = torch.load(indexFile)
  return self
end

function VocabMapping.index2Token(index)
  return self.indexFile[token]
end


function VocabMapping.token2Index(token)
  return self.vocabFile[token]
end
--[[
function VocabMapping.token2Vec(token)
  return self.dicFile[self.vocabFile[token]]
--[[end

function VocabMapping.vec2Token(vec, searchWidth)
  return value w2vutils:distance(vec,k)
end]]

return VocabMapping
