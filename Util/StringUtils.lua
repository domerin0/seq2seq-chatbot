--[[
A bunch of static re-usable lua functions for string manipulation
]]--
local StringUtils = {}


--Taken from http://stackoverflow.com/questions/1426954/split-string-in-lua
function StringUtils.split(inputstr, sep)
        if sep == nil then
                sep = "%s"
        end
        local t={}
        local i=1
        for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
                t[i] = str
                i = i + 1
        end
        return t
end

-- Taken from http://stackoverflow.com/questions/22831701/lua-read-beginning-of-a-string
function StringUtils.startsWith(String,Start)
   return string.sub(String,1,string.len(Start))==Start
end

-- This function returns the input string as a tokenized table
function StringUtils.tokenize(text)
  local tokenList = {}
  local punc = {"," , ".", "!", "?"}

  for mark in punc do
    text = text:gsub("%" .. mark .. "", " " .. mark .." ")
  end
  text = string.lower(text)
  tokenList = split(text, " ")
  --Replace new line character at end with EOS token
  tokenList[#tokenList] = "<EOS>"
  return Preprocessor.lemmatize(tokenList)
end

--This function lemmitizes a tokenized list
--TODO come up with lemmatize list to actually use here
function StringUtils.lemmatize(tokenList)
    return tokenList
end

return StringUtils
