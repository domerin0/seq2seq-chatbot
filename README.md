# seq2seq-chatbot
An (in progress) implementation of Google's seq2seq architecture.

##A few notes:

- This is based off of [Sutskever et al., 2014.](http://arxiv.org/abs/1409.3215) and [Vinyals & Le, 2015.](http://arxiv.org/pdf/1506.05869v1.pdf)
- The word embeddings were obtained from [rotmanmi's WORD2VEC wrapper for Torch7](https://github.com/rotmanmi/word2vec.torch)
- The data this is being tested on is the OpenSubtitles dataset, I used [a script I made](https://github.com/inikdom/opensubtitles-parser) to tokenize and create the input output sequences
 

##Todo

- Finish implentation of train.lu, LSTM.lua
- Look into solving memory issue (minibatch table exceeds luajit 1gb limit.) Currently work around is to make multiple minibatch files (not sure how good this approach is)
