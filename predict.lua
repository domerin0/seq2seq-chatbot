local VocabMapping = require "Util.VocabMapping"
local CommandLineArgs = require "Util.CommandLineArgs"
require 'seq2seq'


local options = CommandLineArgs.sampleCmdArgs()

--I realize the chainis silly, I should fix this later TODO
local vMap = VocabMapping.create(options.checkpoint.options.dataDir)

local chatbot = options.checkpoint.model
