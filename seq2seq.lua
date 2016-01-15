require 'torch'
require 'nn'
require 'rnn'

seq2seq = {}
torch.include('seq2seq', 'Models/Seq2Seq.lua')

return seq2seq
