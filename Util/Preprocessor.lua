 --[[
The main goal of this file is to run the preprocessor if necessary:
1. Create vocab file
2. Create minibatches save to tensor file
]]--
require "lfs"

local Preprocessor = {}
local StringUtils = require "Util.StringUtils"

--trainFrac is percentage of data to use for training
-- validation data is computed as (1 - trainFrac)
function Preprocessor.start(dataDir)
  lfs.mkdir (dataDir)
  local vocabFile = path.join(dataDir, "vocab.t7")
  local dicFile = path.join(dataDir, "index2Vec.t7")
  --Make directories we will need just incase they don't exist
  local inputFilesDir = path.join(dataDir, "raw/")
  local processedDataDir  = path.join(dataDir, "processed/")
  local batchesDataDir = path.join(dataDir, "rawbatches/")
  local trainDataDir = path.join(dataDir, "train/")
  local testDataDir = path.join(dataDir, "test/")
  local evalDataDir = path.join(dataDir, "eval/")
  lfs.mkdir(batchesDataDir)
  lfs.mkdir(trainDataDir)
  lfs.mkdir(testDataDir)
  lfs.mkdir(evalDataDir)
  lfs.mkdir(processedDataDir)

  local rawFiles = {}
  local dataFiles = {}


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

  print(runPreprocessor)
  local numLines = 0
  local maxSequenceLength = 0

  if runPreprocessor then

    for file in lfs.dir(inputFilesDir) do
      if not StringUtils.startsWith(file,".") then
        table.insert(rawFiles, path.join(inputFilesDir, file))
        table.insert(dataFiles, path.join(processedDataDir, file))
      end
    end
      -- Should take in all files and make one vocab mapping
      Preprocessor.createVocabFile(rawFiles, vocabFile)
      Preprocessor.createDicFile(vocabFile, dicFile)

      --Not very helpful, but for debugging purposes
    assert(#rawFiles == #dataFiles, "Something went wrong...")

    for key, file in ipairs(rawFiles) do
      Preprocessor.createDataFile(file, vocabFile, dataFiles[key])
    end

    collectgarbage()
  end
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
  mostCommonTokens["EOS"] = numTokens
  mostCommonTokens["UNK"] = numTokens + 1
  mostCommonTokens["$PAD$"] = numTokens + 2
  numTokens = numTokens + 2
  print("Number of tokens in vocab... "..numTokens)
  print("Saving vocab mapping...")
  torch.save(vocabFile, mostCommonTokens)
end

function Preprocessor.createDicFile(vocabFile, dicFile)
  local w2vutils = require "Util.w2vutils"
  print("Creating dictionary mapping...")
  local vocabMapping = torch.load(vocabFile)
  local count = 0
  for _ in pairs(vocabMapping) do count = count + 1 end
  --for now constant 300 since that is embedding size
  local indexMapping = torch.FloatTensor(count, 300)
  print(count)
  --local indexMapping = {}
  for token, index in pairs(vocabMapping) do
      local t = w2vutils:word2vec(token)
      if t ~= nil then
        local vec = torch.FloatTensor(t)
        indexMapping[math.floor(index)] = vec
      else
        --Accountingfor out of vocab tokens in w2v
      indexMapping[math.floor(index)] = torch.FloatTensor(w2vutils:word2vec("UNK"))
    end
    --indexMapping[index] = table.insert(indexMapping, w2vutils:word2vec(token))
  end
  print("Saving dictionary mapping...")
  torch.save(dicFile, indexMapping)
end

--[[
This function will write the contents of the inputfile to a more easily
machine readable lua table to be later made into minibatches
]]
function Preprocessor.createDataFile(inputFile, vocabFile, dataFile)
  print("Creating data file...")
  local vocabMapping = torch.load(vocabFile)
  local dataset = {}
  local maxSequenceLength = 0
  for line in io.lines(inputFile) do
    local sequence = {}
    local tempStr = StringUtils.split(line, ' ')
    for key, tok in pairs(tempStr) do
      if not vocabMapping[tok] then
        sequence[#sequence + 1] = vocabMapping["UNK"]
      else
        sequence[#sequence + 1] = vocabMapping[tok]
      end
    end
      --Must indicate end of sentence
      sequence[#sequence + 1] = vocabMapping["EOS"]
      if maxSequenceLength < #sequence then
        maxSequenceLength = #sequence
      end
      table.insert(dataset, torch.ShortTensor(sequence))
  end
  print("Saving... "..dataFile)
  torch.save(dataFile, torch.ShortTensor(dataset))
  dataset = {}
end

return Preprocessor
