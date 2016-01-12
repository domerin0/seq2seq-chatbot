require 'torch'

seq2seq = {}
torch.include('seq2seq', 'Models/Seq2Seq.lua')

return seq2seq
