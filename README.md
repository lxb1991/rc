#cnn model
ref: Relation Classification via Convolutional Deep Neural Network

paper: http://www.aclweb.org/anthology/C14-1220

| model | acc | f1 score |
| :--- | :----: | ----: |
| cnn  | 65.3% | 69.05 |
| cnn (lexical feature) | 74.7% | 78.42 |
| cnn (pos feature) | 77.6% | 81.88 |
| cnn (pos feature) (lexical feature) | 77.8% | 81.72 |

Not contained lexical feature
()  not contained pos feature

| model | acc | f1 score |
| :--- | :----: | ----: |
| cnn + entity_attention  | 58.7% (49.3)(no dropout 53.1%) | 63.65 (51.78) (no dropout 56.71)|
| cnn + self_attention  | 63.7% (45.6%)(no dropout 46.7%) | 68.93 (48.13) (no dropout49.81)|

#rnn model
ref: Relation Classification via Recurrent Neural Network

paper: https://arxiv.org/pdf/1508.01006.pdf

| model | acc | f1 score |
| :--- | :----: | ----: |
| rnn  | 73.7% | 77.92 |
| lstm | 76.4% | 80.77 |
