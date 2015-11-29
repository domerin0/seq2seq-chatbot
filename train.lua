local CommandLineArgs = require "Util.CommandLineArgs"
local Preprocessor = require "Util.Preprocessor"
local MiniBatchLoader = require "Util.MiniBatchLoader"
local VerifyGPU = require "Util.VerifyGPU"
local Seq2Seq = require "Models.Seq2Seq"

local options = CommandLineArgs.trainCmdArgs()
torch.manualSeed(options.seed)

if Preprocessor.shouldRun(options.dataDir) then
  print("Starting pre-processor")
  Preprocessor.start(options.dataDir)
  collectgarbage()
else
  print("Preprocessor doesn't need to be run, moving on...")
end

--prepare data for training with (input, output) pairs

if MiniBatchLoader.shouldRun(options.dataDir) then
  print("Creating minibatches...")
  MiniBatchLoader.createMiniBatches(options.dataDir, options.batchSize,
    options.maxSeqLength)
  collectgarbage()
else
  print("Minibatches already created before, moving on...")
end

--Now, check and enable GPU usage:

local cuid = VerifyGPU.checkCuda(options.gpuid, options.seed)


local clid = VerifyGPU.checkOpenCl(options.gpuid, options.seed)

if clid == -1 and cuid == -1 then
  options.gpuid = -1
end



--Load minibatches into memory!

local batchLoader = MiniBatchLoader.loadMiniBatches(options.dataDir, options.batchSize, trainFrac,
  options.evalFrac, options.testFrac)

--Create model, or load from checkpoint
if not path.exists(options.checkpoints) then
  lfs.mkdir(options.checkpoints)
end

local fromCheckpoint = false
if(string.len(options.startFrom) > 0) then
  print("Loading network parameters from checkpoint... "..options.startFrom)
  local checkpoint = torch.load(options.startFrom)
  protos = checkpoint.protos
  --Maybe check vocab here TODO?
  print("rnnSize= "..checkpoint.options.rnnSize.." numLayers= "..checkpoint.options.numLayers..)
  options.rnnSize = checkpoint.options.rnnSize
  options.numLayers = checkpoint.options.numLayers
  fromCheckpoint = true
else
  print('Creating a chatbot with ' .. options.num_layers .. ' layers')
  protos = {}
  local model = Seq2Seq.conversationalModel(options.embeddingSize,
    options.vocabSize,options.rnnSize, options.numLayers, options.dropout)
  protos.encoder = model.encoder
  protos.decoder = model.decoder
  protos.criterion = nn.CrossEntropyCriterion()
end

local initialState = {}
for i=1,options.numLayers do
  local hInit = torch.zeroes(options.batchSize, options.rnnSize)
  if options.gpuid >=0 and options.opencl == 0 then hInit = hInit:cuda() end
  if options.gpuid >=0 and options.opencl == 1 then hInit = hInit:cl() end
  table.insert(initialState, initialState:clone())
  table.insert(initialState, initialState:clone())
end

if options.gpuid >= 0 and options.opencl == 0 then
  if options.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
  end
  if options.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
  end
end

encoderParams, encoderGradParams = model_utils.combine_all_parameters(protos.encoder)
decoderParams, decoderGradParams = model_utils.combine_all_parameters(protos.decoder)

if fromCheckpoint == false then
  encoderParams:uniform(-0.08, 0.08)
  decoderParams:uniform(-0.08, 0.08)
end





--perform training of n minibatches of m epochs over bs backsteps
