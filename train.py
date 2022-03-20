import os
import sys
import time

import torch
import logging
from torch import nn
from transformers import AdamW, WarmupLinearSchedule

from tagging_eval import *

logger = logging.getLogger(__name__)


def trainer(model, train_data_loader, eval_data_loader, args, device, label_list, py_label_list):
    num_training_steps = args.max_steps if args.max_steps > 0 else len(train_data_loader
                                                                       ) * args.epochs

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=int(num_training_steps * args.warmup_proportion),
                                     t_total=num_training_steps)
    global_steps = 0
    best_score = 0.0
    tic_train = time.time()

    model = model.to(device)
    nll_loss = nn.NLLLoss(reduction='sum')

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_data_loader, start=1):
            model.train()
            input_ids, input_mask, pinyin_ids, stroke_ids, _lmask, label_ids, py_labels = tuple(t.to(device) for t in batch)
            prob_hanzi, prob_pinyin, _ = model(input_ids, pinyin_ids, stroke_ids, device, None, input_mask, True)

            # label_onehot = nn.functional.one_hot(label_ids, num_classes=21128)
            # py_onehot = nn.functional.one_hot(py_labels, num_classes=430)
            # print(prob_hanzi.shape, input_mask)
            # 计算loss
            loss = 0
            hanzi_num_sum = 0
            for i in range(len(input_ids)):
                hanzi_num = sum(_lmask[i])
                hanzi_num_sum += hanzi_num
                loss = loss + nll_loss(prob_hanzi[i][1:hanzi_num+1], label_ids[i][1:hanzi_num+1])
                loss = loss + nll_loss(prob_pinyin[i][1:hanzi_num+1], py_labels[i][1:hanzi_num+1])
            # 每个字Loss求平均
            loss = loss / hanzi_num_sum
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if global_steps % args.save_steps == 0:
                print(
                    "global step %d, epoch: %d, batch: %d, loss: %f, speed: %.2f step/s"
                    % (global_steps, epoch, step, loss,
                       args.save_steps / (time.time() - tic_train)))
                tic_train = time.time()
                print("Eval:")
                f1 = evaluate(model, eval_data_loader, device, label_list, py_label_list, args.output_dir)
                if f1 > best_score:
                    # save best model
                    torch.save(model.state_dict(),
                               os.path.join(args.output_dir,
                                            "best_model.pth"))
                    print("Save best model at {} step.".format(
                        global_steps))
                    best_score = f1
            if 0 < args.max_steps <= global_steps:
                sys.exit(0)
            global_steps += 1

def evaluate(model, eval_data_loader, device, label_list, py_label_list, output_dir):
    model.eval()
    all_inputs, all_golds, all_preds = [], [], []
    all_py_golds, all_py_preds = [], []
    all_fusino_preds = []
    all_inputs_sent, all_golds_sent, all_preds_sent = [], [], []
    all_py_pred_sent, all_py_gold_sent, all_fusion_sent = [], [], []
    all_py_inputs, all_py_inputs_sent = [], []
    for step, batch in enumerate(eval_data_loader, start=1):
        input_ids, input_mask, pinyin_ids, stroke_ids, _lmask, label_ids, py_labels = tuple(t.to(device) for t in batch)
        batch_size = len(input_ids)
        max_sen_len = len(input_ids[0])
        prob_hanzi, prob_pinyin, prob_fusion = model(input_ids, pinyin_ids, stroke_ids, device, None, input_mask, False)

        prob_hanzi = np.argmax(prob_hanzi.cpu().detach().numpy(), axis=2)
        prob_pinyin = np.argmax(prob_pinyin.cpu().detach().numpy(), axis=2)
        prob_fusion = np.argmax(prob_fusion.cpu().detach().numpy(), axis=2)
        label_ids = label_ids.cpu().detach().numpy()
        py_labels = py_labels.cpu().detach().numpy()
        input_ids = input_ids.cpu().detach().numpy()
        pinyin_ids = pinyin_ids.cpu().detach().numpy()
        for k in range(batch_size):
            tmp1, tmp2, tmp3, tmps4, tmps5, tmps6, tmps7 = [], [], [], [], [], [], []
            for j in range(max_sen_len):
                if _lmask[k][j] == 0: continue
                # 汉字
                all_golds.append(label_ids[k][j])
                all_preds.append(prob_hanzi[k][j])
                all_inputs.append(input_ids[k][j])
                tmp1.append(label_list[label_ids[k][j]])
                tmp2.append(label_list[prob_hanzi[k][j]])
                tmp3.append(label_list[input_ids[k][j]])
                # 拼音
                all_py_inputs.append(pinyin_ids[k][j])
                all_py_golds.append(py_labels[k][j])
                all_py_preds.append(prob_pinyin[k][j])
                all_fusino_preds.append(prob_fusion[k][j])
                tmps4.append(str(py_labels[k][j]))
                tmps5.append(str(prob_pinyin[k][j]))
                tmps6.append(label_list[prob_fusion[k][j]])
                tmps7.append(str(pinyin_ids[k][j]))
            all_golds_sent.append(tmp1)
            all_preds_sent.append(tmp2)
            all_inputs_sent.append(tmp3)
            all_py_pred_sent.append(tmps4)
            all_py_gold_sent.append(tmps5)
            all_fusion_sent.append(tmps6)
            all_py_inputs_sent.append(tmps7)
    all_golds = [label_list[k] for k in all_golds]
    all_preds = [label_list[k] for k in all_preds]
    all_inputs = [label_list[k] for k in all_inputs]
    all_fusino_preds = [label_list[k] for k in all_fusino_preds]
    all_py_inputs = [py_label_list.get(int(k), k) for k in all_py_inputs]
    all_py_golds = [py_label_list.get(int(k), k) for k in all_py_golds]
    all_py_preds = [py_label_list.get(int(k), k) for k in all_py_preds]
    print('pinyin result:')
    score_f_py((all_py_inputs, all_py_golds, all_py_preds), (all_inputs, all_golds, all_preds), output_dir, False)
    print('fusion result:')
    p, r, f = score_f((all_inputs, all_golds, all_fusino_preds), out_dir=output_dir)
    score_f_sent(all_inputs_sent, all_golds_sent, all_fusion_sent)
    return f