--[[
This file provides utility functions to go between words and indices and back
It also provides utility to go between words and vector embeddings and back
]]
local Constants = require "Util.Constants"
local VocabMapping = {}

function VocabMapping.create(dataDir)
  local self = {}
  local vocabPath = path.join(dataDir, Constants.vocabFile)
  local dicPath = path.join(dataDir, Constants.dicFile)
  local vocabFile = torch.load(vocabPath)
  local dicFile = torch.load(dicPath)
  local count = 0
  for key, value in pairs(vocabFile) do
    count = count + 1
  end
  self.vocabMapping = vocabFile
  self.dicMapping = dicFile
  self.size = count
  return self
end

return VocabMapping
