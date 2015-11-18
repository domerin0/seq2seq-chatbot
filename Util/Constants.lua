--[[
This file contains global constants. This was done to refactor
most constants from other files.
]]

local Constants = {}
--File names
local Constants.vocabFile = "vocab.t7"
local Constants.dicFile = "index2vec.t7"
local Constants.trainFile = "train.t7"
local Constants.evalFile = "eval.t7"
local Constants.testFile = "test.t7"
local Constants.rawBatchesFile = "batches.t7"
--Folder names
local Constants.rawFolder = "raw/"
local Constants.processedFolder = "processed/"
local Constants.rawBatchesFolder = "rawbatches/"
local Constants.trainFolder = "train/"
local Constants.testFolder = "test/"
local Constants.evalFolder = "eval/"
--Reserved tokens
local Constants.EOS = "EOS"
local Constants.UNK = "UNK"
local Constants.PAD = "$PAD$"

return Constants
