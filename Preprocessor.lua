 --[[
The main goal of this file is to run the preprocessor if necessary:
1. Create vocab file
2. Create minibatches save to tensor file
]]--
require "lfs"

local Preprocessor = {}
local StringUtils = require "Util.StringUtils"
local Constants = require "Util.Constants"
--trainFrac is percentage of data to use for training
-- validation data is computed as (1 - trainFrac)
function Preprocessor.start(dataDir)
  lfs.mkdir (dataDir)
  local vocabFile = path.join(dataDir, Constants.vocabFile)
  local dicFile = path.join(dataDir, Constants.dicFile  )
  --Make directories we will need just incase they don't exist
  local inputFilesDir = path.join(dataDir, Constants.rawFolder)
  local batchesDataDir = path.join(dataDir, Constants.rawBatchesFolder )
  local processedDir = path.join(dataDir, Constants.processedFolder)
  local batchFile = path.join(batchesDataDir, Constants.rawBatchesFile)

  lfs.mkdir(processedDir)
  lfs.mkdir(batchesDataDir)
  lfs.mkdir(inputFilesDir)

  local rawFiles = {}
  local dataFiles = {}

  local numLines = 0
  local maxSequenceLength = 0

  for file in lfs.dir(inputFilesDir) do
    if not StringUtils.startsWith(file,".") then
      table.insert(rawFiles, path.join(inputFilesDir, file))
    end
  end
    -- Should take in all files and make one vocab mapping
    Preprocessor.createVocabFile(rawFiles, vocabFile)
    Preprocessor.createDicFile(vocabFile, dicFile)

    --Not very helpful, but for debugging purposes
    Preprocessor.createDataFile(vocabFile, rawFiles, processedDir)
end

--[[ This function takes in a list of tokenized input files
it creates a single vocab mapping]]--
function Preprocessor.createVocabFile(inputFiles, vocabFile)
  print("Creating token frequency list...")
  local mostCommonTokens = {}
  local tokenFreq = {}
  for key, file in ipairs(inputFiles) do
    local numLines = 0
    for line in io.lines(file) do
      numLines = numLines + 1
        local tokens = StringUtils.split(line, ' ')
        for key, token in pairs(tokens) do
          if not tokenFreq[token] then
              tokenFreq[token] = 1
          else
              tokenFreq[token] = tokenFreq[token] + 1
          end
        end
      end
    end
  print("Removing least common tokens...")
  local numTokens = 1
  for tok, count in pairs(tokenFreq) do
    if count > 2 then
      mostCommonTokens[tok] = numTokens
      numTokens = numTokens + 1
    else
      print("Removing uncommon token " .. tok)
    end
  end
  --Add <EOS> and <UNK> and $PAD$ tokens to vocab
  print("Adding special end of sentence and out of vocab tokens...")
  mostCommonTokens[Constants.EOS] = numTokens
  mostCommonTokens[Constants.UNK] = numTokens + 1
  mostCommonTokens[Constants.PAD] = numTokens + 2
  mostCommonTokens[Constants.GO] = numTokens + 3
  numTokens = numTokens + 3
  print("Number of tokens in vocab... "..numTokens)
  print("Saving vocab mapping...")
  torch.save(vocabFile, mostCommonTokens)
end

function Preprocessor.createDicFile(vocabFile, dicFile)
  print("Creating dictionary mapping...")
  local vocabMapping = torch.load(vocabFile)
  local indexMapping = {}
  for key, value in pairs(vocabMapping) do
    indexMapping[value] = key
  end

  print("Saving dictionary mapping...")
  torch.save(dicFile, indexMapping)
end

--[[
This function will write the contents of the inputfile to a more easily
machine readable lua table to be later made into minibatches
]]
function Preprocessor.createDataFile(vocabFile, dataFiles, outDir)
  print("Creating data file...")
  local vocabMapping = torch.load(vocabFile)
  local dataset = {}
  local counter = 0
  for _, file in pairs(dataFiles) do
    for line in io.lines(file) do
      local sequence = {}
      local tempStr = StringUtils.split(line, ' ')
      for key, tok in pairs(tempStr) do
        if not vocabMapping[tok] then
          sequence[#sequence + 1] = vocabMapping[Constants.UNK]
        else
          sequence[#sequence + 1] = vocabMapping[tok]
        end
      end
      --Must indicate end of sentence
        sequence[#sequence + 1] = vocabMapping[Constants.EOS]
        table.insert(dataset, torch.IntTensor(sequence))
    end
    local outFile = path.join(outDir, "out"..counter..".t7")
    counter = counter + 1
    print("Saving... "..outFile)
    torch.save(outFile, dataset)
    dataset = {}
  end
end

--[[
This function determines whether or not the preprocessor
should run. It makes a (possibly poor) assumption that if
the vocab and dictionary files exist then it does not need
to be run.
TODO find more stringent shouldRun criteria to make it user-proof
]]
function Preprocessor.shouldRun(dataDir)
  local vocabFile = path.join(dataDir, Constants.vocabFile)
  local dicFile = path.join(dataDir, Constants.dicFile)

  local runPreprocessor = false

  if not path.exists(vocabFile) then
    print('vocab.t7 does not exist... Creating it...')
    f=io.open(vocabFile,"w")
    f:close()
    runPreprocessor = true
  end

  if not path.exists(dicFile) then
    print("No dictionary file found.. Creating it...")
    f=io.open(dicFile,"w")
    f:close()
    runPreprocessor = true
  end
  return runPreprocessor
end

return Preprocessor
