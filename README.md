# PLOME_finetune_pytorch
最近在学习PLOME:Pre-training with Misspelled Knowledge for Chinese Spelling Correction (ACL2021)，由于对tensorflow代码不熟悉，加上项目中的其他代码都是pytorch写的，因此尝试在pytorch中实现了论文的finetune部分。  
  
请先运行change_model_tf2torch/tf2torch.py转换tensorflow权重放置到datas\pretrained_plome下，权重下载见[tensorflow代码](https://github.com/liushulinle/PLOME)。
  
然后运行main.py开始训练。
  
tensorflow模型权重中有部分权重缺失，没有转换为pytorch权重，最后结果和原文相比稍低。  
  
个人能力有限，如有错误请指出。  
  
[原文](https://aclanthology.org/2021.acl-long.233.pdf)  
[tensorflow代码](https://github.com/liushulinle/PLOME)

