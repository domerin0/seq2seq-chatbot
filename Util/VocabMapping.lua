--[[
This file provides utility functions to go between words and indices and back
It also provides utility to go between words and vector embeddings and back
]]
local Constants = require "Util.Constants"
local VocabMapping = {}
VocabMapping.__index = VocabMapping

function VocabMapping.create(dataDir)
  local self = setmetatable({}, VocabMapping)
  self.vocabFile = torch.load(path.join(dataDir, Constants.vocabFile))
  self.dicFile = torch.load(path.join(dataDir, Constants.dicFile))
  return self
end

function VocabMapping.index2Token(self, index)
  return self.dicFile[index]
end


function VocabMapping.token2Index(self, token)
  return self.vocabFile[token]
end

return VocabMapping
