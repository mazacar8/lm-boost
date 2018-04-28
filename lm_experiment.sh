bazel-bin/lm_1b/lm_1b_eval --mode eval \
                             --pbtxt data/graph-2016-09-10.pbtxt \
                             --vocab_file data/vocab-2016-09-10.txt  \
                             --input_data ../data/word2vec/genderedaa \
                             --ckpt 'data/ckpt-*'

bazel-bin/lm_1b/lm_1b_eval --mode eval \
                             --pbtxt data/graph-2016-09-10.pbtxt \
                             --vocab_file data/vocab-2016-09-10.txt  \
                             --input_data ../data/word2vec/1-billion-word-language-modeling-benchmark-r13output/split_gender/addedv200 \
                             --ckpt 'data/ckpt-*'