--[[
Verifies that there is a GPU, the right libraries are installed, and that it can be used.
If the GPU can't be used, the CPU will be fallen back onto. These functions only
return the gpuid.

This code was adapted from portions of the code found at:
https://github.com/karpathy/char-rnn/blob/master/train.lua

]]--

local VerifyGPU = {}

function VerifyGPU.checkCuda(gpuid, seed)
    local okCunn, cunn = pcall(require, 'cunn')
    local okCutorch, cutorch = pcall(require, 'cutorch')
    if not okCunn then print('package cunn not found!') end
    if not okCutorch then print('package cutorch not found!') end
    if okCunn and okCutorch then
        print('using CUDA on GPU ' .. gpuid .. '...')
        cutorch.setDevice(gpuid + 1)
        cutorch.manualSeed(seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        gpuid = -1 -- overwrite user setting
    end
    return gpuid
end

function VerifyGPU.checkOpenCl(gpuid, seed)
  local okClnn, clnn = pcall(require, 'clnn')
  local okCltorch, cltorch = pcall(require, 'cltorch')
  if not okCunn then print('package clnn not found!') end
  if not okCutorch then print('package cltorch not found!') end
  if okClnn and okCltorch then
      print('using CUDA on GPU ' .. gpuid .. '...')
      cltorch.setDevice(gpuid + 1)
      cltorch.manualSeed(seed)
  else
      print('openCL initialization failed...')
      print('Falling back on CPU mode')
      gpuid = -1 -- overwrite user setting
  end
  return gpuid
end

return VerifyGPU
