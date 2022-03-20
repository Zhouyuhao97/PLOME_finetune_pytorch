import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import numpy as np

def read_train_ds(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            tokens, labels = [], []
            source, target = line.strip('\n').split('\t')[0:2]
            segs1 = source.strip().split(' ')
            segs2 = target.strip().split(' ')
            for x, y in zip(segs1, segs2):
                tokens.append(x)
                labels.append(y)
            yield {'source': tokens, 'target': labels}

# input_ids, input_mask, pinyin_ids, stroke_ids, _lmask, label_ids
def collate_fn(examples, tokenizer, pytool):
    input_ids = torch.LongTensor([example[0] for example in examples])
    input_mask = torch.LongTensor([example[1] for example in examples])
    pinyin_ids = torch.LongTensor([example[2] for example in examples])
    stroke_ids = torch.LongTensor([example[3] for example in examples])
    _lmask = torch.LongTensor([example[4] for example in examples])
    label_ids = torch.LongTensor([example[5] for example in examples])
    tokenid_pyid = {}
    for key in tokenizer.vocab:
        tokenid_pyid[tokenizer.vocab[key]] = pytool.get_pinyin_id(key)
    def get_py_seq(token_seq):
        ans = []
        for t in list(token_seq):
            pyid = tokenid_pyid.get(t, 1)
            ans.append(pyid)
        ans = np.asarray(ans, dtype=np.int32)
        return ans
    py_labels = torch.LongTensor([get_py_seq(example[5]) for example in examples])
    return input_ids, input_mask, pinyin_ids, stroke_ids, _lmask, label_ids, py_labels

def get_zi_py_matrix(pytool, tokenizer):
    pysize = 430
    matrix = []
    for k in range(len(tokenizer.vocab)):
        matrix.append([0] * pysize)

    for key in tokenizer.vocab:
        tokenid = tokenizer.vocab[key]
        pyid = pytool.get_pinyin_id(key)
        matrix[tokenid][pyid] = 1.
    return np.asarray(matrix, dtype=np.float32)

def convert_single_example(example, max_sen_len, tokenizer, pytool, sktool):
    label_map = tokenizer.vocab
    tokens = example["source"]
    labels = example["target"]
    # Account for [CLS] and [SEP] with "- 2"

    if len(tokens) > max_sen_len - 2:
        tokens = tokens[0:(max_sen_len - 2)]
        labels = labels[0:(max_sen_len - 2)]

    _tokens = []
    _labels = []
    _lmask = []
    segment_ids = []
    stroke_ids = []
    _tokens.append("[CLS]")
    _lmask.append(0)
    _labels.append(labels[0])
    segment_ids.append(0)
    stroke_ids.append(0)
    for token, label in zip(tokens, labels):
        _tokens.append(token)
        _labels.append(label)
        _lmask.append(1)
        segment_ids.append(pytool.get_pinyin_id(token))
        stroke_ids.append(sktool.get_pinyin_id(token))
    _tokens.append("[SEP]")
    segment_ids.append(0)
    stroke_ids.append(0)
    _labels.append(labels[0])
    _lmask.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_sen_len:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        stroke_ids.append(0)
        _labels.append(labels[0])
        _lmask.append(0)

    assert len(input_ids) == max_sen_len
    assert len(input_mask) == max_sen_len
    assert len(segment_ids) == max_sen_len
    assert len(stroke_ids) == max_sen_len

    label_ids = [label_map.get(l, label_map['UNK']) for l in _labels]
    # print(input_ids, input_mask, segment_ids, stroke_ids, _lmask, label_ids)
    return input_ids, input_mask, segment_ids, stroke_ids, _lmask, label_ids
def is_chinese_char(cp):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #     https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
        (cp >= 0x3400 and cp <= 0x4DBF) or  #
        (cp >= 0x20000 and cp <= 0x2A6DF) or  #
        (cp >= 0x2A700 and cp <= 0x2B73F) or  #
        (cp >= 0x2B740 and cp <= 0x2B81F) or  #
        (cp >= 0x2B820 and cp <= 0x2CEAF) or
        (cp >= 0xF900 and cp <= 0xFAFF) or  #
        (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
        return True

    return False


class MyData(Dataset):
    def __init__(self, data):
        super(MyData, self).__init__()
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)
