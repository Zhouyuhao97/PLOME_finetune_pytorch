import torch
from torch import nn
from transformers import BertModel

"""
tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-1.0")
model = AutoModel.from_pretrained("nghuyong/ernie-1.0")
"""
class GRU(nn.Module):
	def __init__(self, input_num, hidden_num):
		super(GRU, self).__init__()
		self.hidden_size = hidden_num
		# 这里设置了 batch_first=True, 所以应该 inputs = inputs.view(inputs.shape[0], -1, inputs.shape[1])
		# 针对时间序列预测问题，相当于将时间步（seq_len）设置为 1。
		self.GRU_layer = nn.GRU(input_size=input_num, hidden_size=hidden_num, batch_first=True)
		# self.output_linear = nn.Linear(hidden_num, output_num)
		self.hidden = None

	def forward(self, x):
		# h_n of shape (num_layers * num_directions, batch, hidden_size)
		# 这里不用显式地传入隐层状态 self.hidden
		x, self.hidden = self.GRU_layer(x)
		# x = self.output_linear(x)
		return x, self.hidden

def weights_init(m):
	for name, param in m.named_parameters():
		if "weight" in name:
			print(name)
			torch.nn.init.normal_(param, mean=0, std=0.02)
		else:
			nn.init.zeros_(param)
class Bert_PLOME(nn.Module):
	def __init__(self, modelPath,
				 num_class,
				 pyid2seq,
				 skid2seq,
				 py_dim,
				 max_sen_len,
				 zi_py_matrix,
				 dropout):
		super(Bert_PLOME, self).__init__()
		self.model = BertModel.from_pretrained('bert-base-chinese')
		self.num_class = num_class  # 21128
		self.embed_size = self.model.config.hidden_size
		self.hidden_size = self.model.config.hidden_size
		self.vocab_size = self.model.config.vocab_size
		self.dropout = nn.Dropout(dropout)
		self.pyid2seq = pyid2seq
		self.skid2seq = skid2seq
		self.PYLEN = 4
		self.SKLEN = 10
		self.PYDIM = py_dim
		self.MAX_SEN_LEN = max_sen_len
		self.zi_py_matrix = zi_py_matrix

		self.py_ebd = torch.nn.Embedding(30, self.PYDIM)
		self.sk_ebd = torch.nn.Embedding(7, self.PYDIM)
		# self.py_ebd.apply(weights_init)
		# self.sk_ebd.apply(weights_init)
		self.gru_py = GRU(input_num=self.PYDIM, hidden_num=self.hidden_size)
		self.gru_sk = GRU(input_num=self.PYDIM, hidden_num=self.hidden_size)
		# self.gru_py.apply(weights_init)
		# self.gru_sk.apply(weights_init)
		self.hanzi_linear = nn.Linear(self.hidden_size, self.num_class)
		self.pinyin_linear = nn.Linear(self.hidden_size, 430)
		self.log_softmax = nn.LogSoftmax(dim=-1)
		self.softmax = nn.Softmax(dim=-1)

	def lookup_py(self, ID2SEQ, sen_pyids):
		py_seqs = [ID2SEQ[py_id] for py_id in sen_pyids]
		return py_seqs

	def forward(self, input_ids, pinyin_ids, stroke_ids, device, position_ids=None, attention_mask=None, is_training=False):

		# input_ids : [batch size, sequence length]
		# attention_mask : [batch size, sequence length]

		# 对应 BertModel 中的 get_extended_attention_mask(attention_mask, input_shape, device)
		attention_mask = attention_mask[:, None, None, :]
		attention_mask = attention_mask.to(torch.float32)
		attention_mask = (1.0 - attention_mask) * -10000.0

		# pinyin_stroke Embedding
		pinyin_ids = torch.reshape(pinyin_ids, shape=[-1])
		stroke_ids = torch.reshape(stroke_ids, shape=[-1])
		py_seqs = torch.LongTensor(self.lookup_py(self.pyid2seq, pinyin_ids)).to(device)
		sk_seqs = torch.LongTensor(self.lookup_py(self.skid2seq, stroke_ids)).to(device)
		py_seq_emb = self.py_ebd(py_seqs)
		py_seq_emb = torch.reshape(py_seq_emb, shape=[-1, self.PYLEN, self.PYDIM])
		sk_seq_emb = self.sk_ebd(sk_seqs)
		sk_seq_emb = torch.reshape(sk_seq_emb, shape=[-1, self.SKLEN, self.PYDIM])
		_, output_py = self.gru_py(py_seq_emb)
		_, output_sk = self.gru_sk(sk_seq_emb)
		py_sk_embs = output_py + output_sk
		py_sk_embs = torch.reshape(py_sk_embs, [-1, self.MAX_SEN_LEN, self.hidden_size])
		# embedding_output : [batch size, sequence length, hidden size]
		embedding_output = self.model.embeddings(input_ids=input_ids, position_ids=position_ids, py_sk_embs=py_sk_embs)
		# BERT output : last-layer hidden state, (all hidden states), (all attentions)
		outputs = self.model.encoder(embedding_output, attention_mask)
		outputs = self.dropout(outputs[0])

		# hanzi output
		logits_hanzi = self.hanzi_linear(outputs)
		prob_hanzi = self.log_softmax(logits_hanzi)
		# pinyin output
		logits_pinyin = self.pinyin_linear(outputs)
		prob_pinyin = self.log_softmax(logits_pinyin)
		#fudion output
		prob_fusion = torch.zeros(1, 1).to(device)
		if not is_training:
			prob_fusion = torch.matmul(self.softmax(logits_pinyin), torch.t(torch.FloatTensor(self.zi_py_matrix).to(device)))
			prob_fusion = prob_fusion * self.softmax(logits_hanzi)
		return prob_hanzi, prob_pinyin, prob_fusion


if __name__ == "__main__":
	pass
	# model = Bert('./chinese_roberta_wwm_ext', 1000, 1, tie_weight=True)
	#
	# inputs_id = torch.LongTensor([[1,2,3,4,5,6,0,0]
	# 							  ,[2,3,4,6,5,6,0,0]])
	#
	# output = model(inputs_id, inputs_id)
	#
	# print(output)

	# model = AutoModel.from_pretrained('ernie')
	# attention_mask = (inputs_id != 0).long()
	# tmp = model(inputs_id, attention_mask)
	#
	# print(tmp[0].shape)

