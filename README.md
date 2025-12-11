This is an LLM powered by a decoder-only Transformer + RNN with a ROSA mechanism.
It can be used out of the box to train models on the mini-pile or full-pile datasets (See the training folder).

Would not have been possible without previous work by @BlinkDL

The core idea behind this project is to use recursive graphs instead of attention in inference while we have an existing reference of the group of tokens in the conversation
