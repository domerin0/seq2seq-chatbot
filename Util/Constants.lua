--[[
This file contains global constants. This was done to refactor
most constants from other files.
]]

local Constants = {}
--File names
Constants.vocabFile = "vocab.t7"
Constants.dicFile = "index2vec.t7"
Constants.trainFile = "train.t7"
Constants.evalFile = "eval.t7"
Constants.testFile = "test.t7"
Constants.rawBatchesFile = "batches.t7"
--folder names
Constants.saveFolder = "save/"
Constants.rawFolder = "raw/"
Constants.processedFolder = "processed/"
Constants.rawBatchesFolder = "rawbatches/"
Constants.trainFolder = "train/"
Constants.testFolder = "test/"
Constants.evalFolder = "eval/"
--reserved tokens
Constants.EOS = "$EOS$"
Constants.UNK = "$UNK$"
Constants.PAD = "$PAD$"
Constants.GO = "$GO$"
return Constants
