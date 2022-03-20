import random
import numpy as np
import torch
from transformers import BertModel
import tensorflow as tf
from model import Bert_PLOME

def setup_seed(seed):
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
	# bert = BertModel.from_pretrained('bert-base-chinese')
	bert = Bert_PLOME(None,21128,None,None,32,180,None,0.1)
	print("torch:")
	for name,parameters in bert.named_parameters():
		print(name, parameters.shape)
	reader = tf.compat.v1.train.NewCheckpointReader("./tf_model/bert_model.ckpt")
	variables = reader.get_variable_to_shape_map()

	# gru
	kernel1 = reader.get_tensor('py_emb/GRU/rnn/gru_cell/candidate/kernel')
	kernel2 = reader.get_tensor('py_emb/GRU/rnn/gru_cell/gates/kernel')
	bias1 = reader.get_tensor('py_emb/GRU/rnn/gru_cell/candidate/bias')
	bias2 = reader.get_tensor('py_emb/GRU/rnn/gru_cell/gates/bias')
	B = 32
	H = 768
	all_kernel = np.concatenate([kernel2, kernel1], axis=1)  # shape (B+H, 3 * H)
	kernel_ih_py = np.transpose(all_kernel[:B])  # shape(B, 3 * H)
	kernel_hh_py = np.transpose(all_kernel[B:])  # shape (H, 3 * H)
	bias = np.concatenate([bias2, bias1], axis=0)  # shape (B, 3 * H)
	zero_bias = np.zeros([3 * H])
	bert.state_dict()['gru_py.GRU_layer.weight_ih_l0'].copy_(torch.Tensor(kernel_ih_py))
	bert.state_dict()['gru_py.GRU_layer.weight_hh_l0'].copy_(torch.Tensor(kernel_hh_py))
	bert.state_dict()['gru_py.GRU_layer.bias_ih_l0'].copy_(torch.Tensor(bias))
	bert.state_dict()['gru_py.GRU_layer.bias_hh_l0'].copy_(torch.Tensor(zero_bias))

	kernel1 = reader.get_tensor('sk_emb/GRU/rnn/gru_cell/candidate/kernel')
	kernel2 = reader.get_tensor('sk_emb/GRU/rnn/gru_cell/gates/kernel')
	bias1 = reader.get_tensor('sk_emb/GRU/rnn/gru_cell/candidate/bias')
	bias2 = reader.get_tensor('sk_emb/GRU/rnn/gru_cell/gates/bias')
	B = 32
	H = 768
	all_kernel = np.concatenate([kernel2, kernel1], axis=1)  # shape (B+H, 3 * H)
	kernel_ih_sk = np.transpose(all_kernel[:B])  # shape(B, 3 * H)
	kernel_hh_sk = np.transpose(all_kernel[B:])  # shape (H, 3 * H)
	bias = np.concatenate([bias2, bias1], axis=0)  # shape (B, 3 * H)
	zero_bias = np.zeros([3 * H])
	bert.state_dict()['gru_sk.GRU_layer.weight_ih_l0'].copy_(torch.Tensor(kernel_ih_sk))
	bert.state_dict()['gru_sk.GRU_layer.weight_hh_l0'].copy_(torch.Tensor(kernel_hh_sk))
	bert.state_dict()['gru_sk.GRU_layer.bias_ih_l0'].copy_(torch.Tensor(bias))
	bert.state_dict()['gru_sk.GRU_layer.bias_hh_l0'].copy_(torch.Tensor(zero_bias))
	#py and sk ebd
	zimu_ebd = reader.get_tensor('py_emb/zimu_emb')
	zisk_ebd = reader.get_tensor('sk_emb/zisk_emb')
	bert.state_dict()['py_ebd.weight'].copy_(torch.Tensor(zimu_ebd))
	bert.state_dict()['sk_ebd.weight'].copy_(torch.Tensor(zisk_ebd))

	# linear
	linear_w = reader.get_tensor('loss/output_weights')
	linear_b = reader.get_tensor('loss/output_bias')
	bert.state_dict()['hanzi_linear.weight'].copy_(torch.Tensor(linear_w))
	bert.state_dict()['hanzi_linear.bias'].copy_(torch.Tensor(linear_b))

	#gamma 和kernel是权重矩阵，bias和beta是偏置项
	for name in variables:
		if ('adam_v' not in name and 'adam_m' not in name) and 'bert' in name:
			if 'embeddings' in name:
				level1, level2 = name.split("/")[2:4]
				if level2 == 'word_embeddings' or level2 == 'position_embeddings':
					torch_layer_name = level1 + '.' + level2 + '.weight'
				else: # embeddings/LayerNorm
					level3 = name.split("/")[-1]
					if level3 == 'gamma':
						torch_layer_name = level1 + '.' + level2 + '.weight'
					else:
						torch_layer_name = level1 + '.' + level2 + '.bias'
			elif 'pooler' in name:
				level1, level2, level3 = name.split("/")[2:]
				if level3 == 'kernel':
					torch_layer_name = level1 + '.' + level2 + '.weight'
				else:
					torch_layer_name = level1 + '.' + level2 + '.bias'

			else:
				# encoder
				level1, level2, level3, level4 = name.split("/")[2:6]
				level2 = '.'.join(level2.split('_'))
				if level3 == 'attention':
					level5, level6 = name.split("/")[6:]
					if level6 == 'gamma' or level6 == 'kernel':
						torch_layer_name = level1 + '.' + level2 + '.' + level3 + '.' + level4 + '.' + level5 + '.weight'
					else:
						torch_layer_name = level1 + '.' + level2 + '.' + level3 + '.' + level4 + '.' + level5 + '.bias'
				else:
					level5 = name.split("/")[-1]
					if level5 == 'gamma' or level5 == 'kernel':
						torch_layer_name = level1 + '.' + level2 + '.' + level3 + '.' + level4 + '.weight'
					else:
						torch_layer_name = level1 + '.' + level2 + '.' + level3 + '.' + level4 + '.bias'
			torch_layer_name = 'model.' + torch_layer_name
			w = reader.get_tensor(name)
			if 'kernel' in name:
				w = np.transpose(w)
			w = torch.Tensor(w)
			bert.state_dict()[torch_layer_name].copy_(w)
	# 其他原模型中没有的，随机初始化 pinyin linear weight(bias初始化为0)
	# randn_tensor = torch.normal(mean=0, std=torch.ones((430, 768))*0.02)
	# bert.state_dict()['pinyin_linear.weight'].copy_(randn_tensor)
	# zero_tensor = torch.zeros(430)
	# bert.state_dict()['pinyin_linear.bias'].copy_(zero_tensor)
	# 参数更新完成，保存模型
	torch.save(bert.state_dict(), './torch_model/pytorch_model.bin')
