# MOGANED-Implementation
The code is an **unofficial** implementation of [Event Detection with Multi-order Graph Convolution and Aggregated Attention](https://www.aclweb.org/anthology/D19-1582/) (EMNLP19 paper). (The official code is not released, and this code is for a reference.)

# Requirments
tensorflow-gpu==1.10

stanfordcorenlp (see https://github.com/Lynten/stanford-corenlp for detail)

numpy

tqdm

# Usage
To run this code, you need to:
1. put ```English``` folder of ACE05 dataset into ```./```, or you can modify path in ```constant.py```. (You can get ACE2005 dataset here: https://catalog.ldc.upenn.edu/LDC2006T06)
2. put stanford language model into ```./```, or you can modify path in ```constant.py```. (You can download here: https://stanfordnlp.github.io/CoreNLP/history.html)
3. put GloVe embedding file into ```./glove``` folder, or you can modify path in ```constant.py```. (You can download GloVe embedding here: https://nlp.stanford.edu/projects/glove/)
4. Run ```python train.py --gpu 0 --mode MOGANED``` to run with MOGANED model.  Run ```python train.py --gpu 0 --mode DMCNN``` to run with DMCNN model.

All parameters are in ```constant.py```, you can modify them as you wish.

# Dataset
Due to license limitation, we can't distribute datasets directly, please download the dataset by yourself. The download link is given in **Usage** part.

The code will automatically extract information of ACE2005 dataset and dumps them into json format(```train.json``` ,```dev.json``` and ```test.json```) into path ```ACE_DUMP``` in ```constant.py```. This is implented in class ```Extractor``` in ```utils.py```.

Each file is composed of a list, which elements are instances with following format:
```
{
    "tokens": XX,           #tokens of a sentence, a list with string elements
    "start": XX,            #starting offsets of the sentence in original files, an integer
    "end": XX,              #ending offsets of the sentence in original files, an integer
    "offsets":XX,           #offsets of each tokens, a list with tuple elements
    "trigger_tokens":XX,    #tokens of trigger words, a list with string elements
    "trigger_start":XX,     #start index of trigger words of tokens, an integer
    "trigger_end":XX,       #end index of trigger words of tokens, an integer
    "trigger_offsets":XX,   #offsets of trigger words, a list with tuple elements
    "event_type":XX,        #event type of tokens with given triggers, a string
    "ner":XX,               #ner tag of each token, a list
    "pos":XX,               #pos tag of each token, a list
    "dependency":XX,        #dependency parsing results of tokens with StanfordCoreNLP format, a list
    "file":XX,              #file name without suffix
    "dir":XX,               #dir name
    "entities":XX           #entitie in this sentencem, a list with entity elements
}
```


Each entity is a dictionary with following format:
```
{
    "token":XX,             #tokens of the entity, a list with string elements
    "role":XX,              #role of the entity when trigger is given, a string
    "offsets":XX,           #offsets of entity, a list with tuple elements
    "start":XX,             #start offset of entity, an integer
    "end":XX,               #snd offset of entity, an integer
    "idx_start":XX,         #start index in tokens, an integer
    "idx_end":XX            #end index in tokens, an integer
}
```
# Results
Depends on the split of train/dev/test, results will have some change accordingly, but won't change much.
I use [this split](https://github.com/thunlp/HMEAE/blob/master/logs/split.json) and get following results:
|Method|Precision|Recall|F1|
|--|--|--|--|
|MOGANED(Paper)|79.5|72.3|75.7|
|MOGANED(This code)|72.4|71.0|71.7|

# Note
There are some differences on training strategy between this code and the original paper:
1. The code doesn't use BIO schema. This is because trigger words are usually a single word rather than a phrase in ACE05, this won't affect results in ACE05.
2. The code doesn't use L2-norm, only use dropout. From my personal experience, this won't affect results much. 
3. The code uses AdamOptimizer rather than AdadeltaOptimizer. During experiments, I found Adadelta can't train a good classifier, however, Adam can. 
4. This code sets bias loss lambda to 1 rather than 5 since I found this will make F1 score higher.

# TODO
The code structure is quite like [another repo](https://github.com/thunlp/HMEAE), I will merge these codes if I have time in future.
