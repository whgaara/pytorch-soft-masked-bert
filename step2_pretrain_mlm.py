import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.nn as nn

from torch.optim import Adam
from smbert.data.smbert_dataset import *
from smbert.layers.SM_Bert_mlm import SMBertMlm


if __name__ == '__main__':
    best_top1 = 0

    if Debug:
        print('开始训练 %s' % get_time())
    soft_masked_bert = SMBertMlm().to(device)
    if Debug:
        print('Total Parameters:', sum([p.nelement() for p in soft_masked_bert.parameters()]))

    if os.path.exists(FinetunePath):
        print('开始加载本地预训练模型！')
        soft_masked_bert.load_pretrain(FinetunePath)
        print('完成加载本地预训练模型！')

    dataset = SMBertDataSet(CorpusPath)
    testset = SMBertEvalSet(TestPath)

    optim = Adam(soft_masked_bert.parameters(), lr=MLMLearningRate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(MLMEpochs):
        # train
        if Debug:
            print('第%s个Epoch %s' % (epoch, get_time()))
        soft_masked_bert.train()
        data_iter = tqdm(enumerate(dataset),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataset),
                         bar_format='{l_bar}{r_bar}')
        print_loss = 0.0
        for i, data in data_iter:
            if Debug:
                print('生成数据 %s' % get_time())
            data = {k: v.to(device) for k, v in data.items()}
            batch_input = data['batch_inputs']
            batch_position = data['batch_position']
            batch_segments = data['batch_segments']
            batch_labels = data['batch_labels']
            if Debug:
                print('获取数据 %s' % get_time())
            mlm_output = soft_masked_bert(batch_input, batch_position, batch_segments).permute(0, 2, 1)
            if Debug:
                print('完成前向 %s' % get_time())
            mask_loss = criterion(mlm_output, batch_labels)
            print_loss = mask_loss.item()

            mask_loss.backward()
            optim.step()
            optim.zero_grad()

            if Debug:
                print('完成反向 %s\n' % get_time())

        print('EP_%d mask loss:%s' % (epoch, print_loss))

        # test
        with torch.no_grad():
            soft_masked_bert.eval()
            test_count = 0
            top1_count = 0
            top5_count = 0
            for test_data in testset:
                eval_token = test_data['eval_token'].unsqueeze(0).to(device)
                eval_position = test_data['eval_position'].unsqueeze(0).to(device)
                eval_segment = test_data['eval_segment'].unsqueeze(0).to(device)
                label_list = test_data['eval_label'].tolist()

                eval_token_list = eval_token.tolist()
                input_len = len([x for x in eval_token_list[0] if x]) - 2

                mlm_output = soft_masked_bert(eval_token, eval_position, eval_segment)[:, 1:input_len + 1, :]
                output_tensor = torch.nn.Softmax(dim=-1)(mlm_output)
                output_topk = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()

                # 累计数值
                test_count += input_len
                for i in range(input_len):
                    batch_labels = label_list[i + 1]
                    if batch_labels == output_topk[i][0]:
                        top1_count += 1
                    if batch_labels in output_topk[i]:
                        top5_count += 1

            if test_count:
                top1_acc = float(top1_count) / float(test_count)
                print('top1纠正正确率：%s' % round(top1_acc, 2))
                top5_acc = float(top5_count) / float(test_count)
                print('top5纠正正确率：%s' % round(top5_acc, 2))

                # save
                if top1_acc > best_top1:
                    best_top1 = top1_acc
                    torch.save(soft_masked_bert.cpu(), FinetunePath)
                    soft_masked_bert.to(device)
                    print('EP:%d Model Saved on:%s\n' % (epoch, FinetunePath))
