local VocabMapping = require "Util.VocabMapping"
local CommandLineArgs = require "Util.CommandLineArgs"
require 'seq2seq'


local options = CommandLineArgs.sampleCmdArgs()

local dataDir = options.checkpoint.options.dataDir

local chatbot = options.checkpoint.model
