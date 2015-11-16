 --[[
The main goal of this file is to run the preprocessor if necessary:
1. Create vocab file
2. Create minibatches save to tensor file
]]--
require "lfs"
local ffi = require('ffi')

local Preprocessor = {}

local StringUtils = require "Util.StringUtils"

--trainFrac is percentage of data to use for training
-- validation data is computed as (1 - trainFrac)
function Preprocessor.start(dataDir)

  local vocabFile = path.join(dataDir, "vocab.t7")
  local inputFile = path.join(dataDir, "raw.txt")
  local processedDataDir  = path.join(dataDir, "processed/")
  local dataFile1  = path.join(processedDataDir, "data1.t7")
  local dataFile2  = path.join(processedDataDir, "data2.t7")

  local dicFile = path.join(dataDir, "index2Vec.t7")

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

  if not path.exists(dataFile1) or not path.exists(dataFile2) then
    print('data.t7 does not exist... Creating it...')
    f=io.open(dataFile1,"w")
    f:close()
    f=io.open(dataFile2,"w")
    f:close()
  end

  dataFiles = {dataFile1, dataFile2}

  local vocabAttributes = lfs.attributes(vocabFile)
  local inputAttributes = lfs.attributes(inputFile)
  local tensorAttributes = lfs.attributes(dataFile)
  --[Iterate through all the text in file to tokenize it]

  if inputAttributes.modification > vocabAttributes.modification or
    tensorAttributes.modification < inputAttributes.modification then
      print("Vocab and or data files are old, re-processing...")
      runPreprocessor = true
  end
  print(runPreprocessor)
  local numLines = 0
  if runPreprocessor then
    numLines = Preprocessor.createVocabFile(inputFile, vocabFile)
    Preprocessor.createDicFile(vocabFile, dicFile)
    collectgarbage()
    Preprocessor.createDataFile(inputFile, vocabFile, dataFiles, numLines)
  end
  return dataFiles
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
      table.insert(dataset, torch.Tensor(sequence))
      if fileCounter < #dataFiles and
      lineCounter % math.floor(numLines / #dataFiles) == 0 then
        torch.save(dataFiles[fileCounter], dataset)
        fileCounter = fileCounter  + 1
        dataset = {}
      end
  end
  torch.save(dataFile, dataset)
end

return Preprocessor
