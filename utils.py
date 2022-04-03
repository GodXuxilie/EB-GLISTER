from nltk.tokenize import word_tokenize
from nltk import data
data.path.append('/home/x/xuxilie/nltk_data')
import torch
import torchtext.vocab as vocab
from torch.utils.data import DataLoader
from torch.utils.data.dataset import TensorDataset
from torch.nn.utils.rnn import pad_sequence

def sentence_to_id(sentence_list, max_len=36, dim=300):
    vocab.pretrained_aliases.keys()
    cache_dir = "/home/x/xuxilie/glove"
    # glove = vocab.pretrained_aliases["glove.6B.50d"](cache=cache_dir)
    glove = vocab.GloVe(name='6B', dim="300", cache=cache_dir) # 与上面等价
    phrase_sentence = []
    for sentence in sentence_list:

        split_phrase_list = []
        tmp_sentence = word_tokenize(sentence)
        # print(tmp_sentence)
        for sentence_id in tmp_sentence:
            try:
                split_phrase_list.append(glove.stoi[sentence_id])
            except:
                pass
        # print(split_phrase_list)
        phrase_sentence.append(torch.LongTensor(split_phrase_list))
    
     # Desired max length
    # max_len = 36
    # pad first seq to desired length
    phrase_sentence[0] = torch.nn.ConstantPad1d((0, max_len - phrase_sentence[0].shape[0]), 0)(phrase_sentence[0])
    # pad all seqs to desired length
    phrase_sentence = pad_sequence(phrase_sentence, batch_first=True)
    phrase_sentence = phrase_sentence[:,:max_len]

    return phrase_sentence
    

def create_dataset(sentence_list1, target, batch_size=32, shuffle=True, sentence_list2=None, max_len=36, fraction=1, begin=0):
    inputs = sentence_to_id(sentence_list1, max_len=max_len)
    
    if sentence_list2 is not None:
        inputs_append = sentence_to_id(sentence_list2, max_len=max_len)
        inputs = torch.cat((inputs, inputs_append), dim=1)
        # print(inputs.shape)
    
    target = torch.LongTensor(target)

    # if fraction < 1:
        # inputs = inputs[int(len(inputs) * begin):int(len(inputs) * (begin+fraction))]
        # target = target[int(len(target) * begin):int(len(target) * (begin+fraction))]

    print(inputs.shape, target.shape)
    dataset = TensorDataset(inputs, target)
    dataloader = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size,  # mini batch size
            shuffle=shuffle,  # 要不要打乱数据 (打乱比较好)
            num_workers=0,  # 多线程来读数据
        )

    return dataloader