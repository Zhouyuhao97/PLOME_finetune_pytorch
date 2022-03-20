import argparse
import os.path
import random
from functools import partial

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

import train
from model import Bert_PLOME
from utils import read_train_ds, collate_fn, MyData#, convert_example
from vocab import Vocab
from utils import get_zi_py_matrix, convert_single_example
from pinyin_tool import PinyinTool
import tokenization


def config():
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=16, type=int, help="Batch size per GPU/CPU for training.")
	parser.add_argument("--model_path", type=str, default="datas/pretrained_plome/",
						help="Pretraining model name or path.")
	parser.add_argument("--max_seq_length", type=int, default=180,
						help="The maximum total input sequence length after SentencePiece tokenization.")
	parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate used to train.")
	parser.add_argument("--dropout", default=0.1, type=float, help="dropout ratio.")
	parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
	parser.add_argument("--epochs", type=int, default=10, help="Number of epoches for training.")

	parser.add_argument("--seed", type=int, default=1, help="Random seed for initialization.")
	parser.add_argument("--warmup_proportion", default=0.1, type=float,
						help="Linear warmup proption over the training process.")
	parser.add_argument("--max_steps", default=-1, type=int,
						help="If > 0: set total number of training steps to perform. Override num_train_epochs.", )
	parser.add_argument("--pinyin_vocab_file_path", type=str, default="pinyin_vocab.txt", help="pinyin vocab file path.")
	parser.add_argument("--pinyin_dim", type=int, default=32, help="pinyin dim.")
	parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
	parser.add_argument("--ignore_label", default=-100, type=int, help="Ignore label for CrossEntropyLoss.")
	parser.add_argument("--train_file", type=str, default="datas/train.txt", help="train file path.")
	parser.add_argument("--test_file", type=str, default="pinyin_vocab.txt", help="test file path.")
	parser.add_argument("--output_dir", type=str, default="plome_output", help="output file path.")
	args = parser.parse_args()

	return args


def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
	setup_seed(20)
	args = config()
	if not os.path.exists(args.output_dir):
		os.mkdir(args.output_dir)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	# 数据读取部分
	train_list = list(read_train_ds('datas/train.txt'))
	test_List = list(read_train_ds('datas/test.txt'))

	pytool = PinyinTool(py_dict_path='pinyin_data/zi_py.txt', py_vocab_path='pinyin_data/py_vocab.txt', py_or_sk='py')
	sktool = PinyinTool(py_dict_path='stroke_data/zi_sk.txt', py_vocab_path='stroke_data/sk_vocab.txt', py_or_sk='sk')
	PYID2SEQ = pytool.get_pyid2seq_matrix()  # pinyin_id到zimu_id的映射
	SKID2SEQ = sktool.get_pyid2seq_matrix() # 汉字到笔画的映射
	vocab_file = args.model_path + 'vocab.txt'
	tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=False)
	label_list = {}
	for key in tokenizer.vocab:
		label_list[tokenizer.vocab[key]] = key
	py_label_list = {v: k for k, v in pytool.vocab.items()}
	# 这里偷懒了，有些地方直接复用了原作者的代码
	zi_py_matrix = get_zi_py_matrix(pytool, tokenizer)
	# input_ids, input_mask, segment_ids(pinyin), stroke_ids, _lmask, label_ids
	train_ids = [convert_single_example(example, args.max_seq_length, tokenizer, pytool, sktool) for example in train_list]
	test_ids = [convert_single_example(example, args.max_seq_length, tokenizer, pytool, sktool)for example in test_List]

	train_x = MyData(train_ids)
	test_x = MyData(test_ids)
	collate = partial(collate_fn, tokenizer=tokenizer, pytool=pytool)
	train_data_loader = DataLoader(train_x, batch_size=args.batch_size, collate_fn=collate, shuffle=True)
	test_data_loader = DataLoader(test_x, batch_size=args.batch_size, collate_fn=collate, shuffle=False)

	# 模型建立
	model = Bert_PLOME(args.model_path, num_class=len(label_list), pyid2seq=PYID2SEQ, skid2seq=SKID2SEQ, py_dim=args.pinyin_dim,
				 max_sen_len=args.max_seq_length, zi_py_matrix=zi_py_matrix, dropout=args.dropout)
	# 加载预训练权重
	print("start to load weight")
	model.load_state_dict(torch.load(args.model_path + 'pytorch_model.bin'))
	# 训练
	print('start to train')
	train.trainer(model, train_data_loader, test_data_loader, args, device, label_list, py_label_list)
