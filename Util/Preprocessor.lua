 --[[
The main goal of this file is to run the preprocessor if necessary:
1. Create vocab file
2. Create minibatches save to tensor file
]]--
require "lfs"

local Preprocessor = {}
local maxFileSize = 666569
local StringUtils = require "Util.StringUtils"

--trainFrac is percentage of data to use for training
-- validation data is computed as (1 - trainFrac)
function Preprocessor.start(dataDir)
  lfs.mkdir (dataDir)
  local vocabFile = path.join(dataDir, "vocab.t7")
  local dicFile = path.join(dataDir, "index2Vec.t7")
  local inputFile = path.join(dataDir, "raw.txt")
  --Make directories we will need just incase they don't exist
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
  local dataFiles = {}


  local runPreprocessor = false


  if not path.exists(inputFile) then
    print("No input file detected.. make sure you have a raw.txt in your data directory")
    return
  end

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

  local vocabAttributes = lfs.attributes(vocabFile)
  local dicAttributes = lfs.attributes(dicFile)
  local inputAttributes = lfs.attributes(inputFile)
  --[Iterate through all the text in file to tokenize it]

  if inputAttributes.modification > vocabAttributes.modification or
    dicAttributes.modification < inputAttributes.modification then
      print("Vocab and or data files are old, re-processing...")
      runPreprocessor = true
  end
  print(runPreprocessor)
  local numLines = 0
  local maxSequenceLength = 0

  if runPreprocessor then
    numLines = Preprocessor.createVocabFile(inputFile, vocabFile)

    --1333139 was chosen because it represents about half
    --the dataset
    for i=1,math.floor(numLines/maxFileSize) do
      local p = path.join(processedDataDir, "data"..i..".t7")
      table.insert(dataFiles, p)
      if not path.exists(p) then
        f=io.open(p,"w")
        f:close()
      end
    end

    Preprocessor.createDicFile(vocabFile, dicFile)
    collectgarbage()
    maxSequenceLength = Preprocessor.createDataFile(inputFile, vocabFile, dataFiles, numLines)
  end
  return maxSequenceLength
end

--[[ This function assumes that the raw.txt file has already been tokenized,
where each sentence has been tokenized on a line
This function returns the vocab size]]--
function Preprocessor.createVocabFile(inputFile, vocabFile)
  print("Creating token frequency list...")
  local tokenFreq = {}
  local numLines = 0
  for line in io.lines(inputFile) do
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
  print("Removing least common tokens...")
  local mostCommonTokens = {}
  local numTokens = 1
  for tok, count in pairs(tokenFreq) do
      if count > 1 then
        mostCommonTokens[tok] = numTokens
        numTokens = numTokens + 1
      else
        print("Removing uncommon token " .. tok)
      end
  end
  --Add <EOS> and <UNK> and $PAD$ tokens to vocab
  print("Adding special end of sentence and out of vocab tokens...")
  mostCommonTokens["EOS"] = numTokens + 1
  mostCommonTokens["UNK"] = numTokens + 2
  mostCommonTokens["$PAD$"] = numTokens + 3
  numTokens = numTokens + 3

  print("Saving vocab mapping...")
  torch.save(vocabFile, mostCommonTokens)
  return numLines
end

function Preprocessor.createDicFile(vocabFile, dicFile)
  local w2vutils = require "Util.w2vutils"
  print("Creating dictionary mapping...")
  local vocabMapping = torch.load(vocabFile)
  local indexMapping = {}
  for token, index in pairs(vocabMapping) do
    indexMapping[index] = w2vutils:word2vec(token)
  end
  print("Saving dictionary mapping...")
  torch.save(dicFile, indexMapping)
end

function Preprocessor.createDataFile(inputFile, vocabFile, dataFiles, numLines)
  print("Creating data file...")
  local vocabMapping = torch.load(vocabFile)
  local dataset = {}
  local lineCounter = 0
  local fileCounter = 1
  local maxSequenceLength = 0
  for line in io.lines(inputFile) do
    lineCounter = lineCounter + 1
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
      table.insert(dataset, sequence)
      --Use magic number determined above (or if < magic number use max lines)
      if lineCounter % maxFileSize == 0 or lineCounter == numLines then
        torch.save(dataFiles[fileCounter], dataset)
        fileCounter = fileCounter  + 1
        dataset = {}
        --the stop condition (we lose at most one data point doing this)
        if fileCounter == #dataFiles + 1 then break end
      end
  end
  return maxSequenceLength
end

return Preprocessor
