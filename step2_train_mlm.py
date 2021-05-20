import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.nn as nn

from torch.optim import Adam
from smbert.data.smbert_dataset import *
from smbert.layers.SM_Bert_mlm import SMBertMlm


if __name__ == '__main__':
    # gama选用0.8是依赖论文中的数据
    gama = 0.8
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
    evalset = SMBertEvalSet(TestPath)

    optim = Adam(soft_masked_bert.parameters(), lr=MLMLearningRate)
    criterion_c = nn.CrossEntropyLoss().to(device)
    criterion_d = nn.BCELoss().to(device)

    for epoch in range(MLMEpochs):
        # train
        if Debug:
            print('第%s个Epoch %s' % (epoch, get_time()))
        soft_masked_bert.train()
        data_iter = tqdm(enumerate(dataset),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataset),
                         bar_format='{l_bar}{r_bar}')
        print_c_loss = 0.0
        print_d_loss = 0.0
        print_loss = 0.0
        for i, data in data_iter:
            if Debug:
                print('生成数据 %s' % get_time())
            data = {k: v.to(device) for k, v in data.items()}
            batch_input = data['batch_inputs']
            batch_position = data['batch_position']
            batch_segments = data['batch_segments']
            batch_labels = data['batch_labels']
            batch_isG = data['batch_isG']
            if Debug:
                print('获取数据 %s' % get_time())
            mlm_isG, mlm_output = soft_masked_bert(batch_input, batch_position, batch_segments)
            mlm_output = mlm_output.permute(0, 2, 1)
            mlm_isG = mlm_isG.squeeze(-1)
            if Debug:
                print('完成前向 %s' % get_time())

            c_loss = criterion_c(mlm_output, batch_labels)
            d_loss = criterion_d(mlm_isG, batch_isG)
            loss = gama * c_loss + (1 - gama) * d_loss

            print_c_loss = c_loss.item()
            print_d_loss = d_loss.item()
            print_loss = loss.item()

            loss.backward()
            optim.step()
            optim.zero_grad()

            if Debug:
                print('完成反向 %s\n' % get_time())

        print('EP_%d mask c loss:%s' % (epoch, print_c_loss))
        print('EP_%d mask d loss:%s' % (epoch, print_d_loss))
        print('EP_%d mask total loss:%s' % (epoch, print_loss))

        # eval
        with torch.no_grad():
            soft_masked_bert.eval()
            eval_count = 0
            top1_count = 0
            top5_count = 0
            for eval_data in evalset:
                eval_token = eval_data['eval_token'].unsqueeze(0).to(device)
                eval_position = eval_data['eval_position'].unsqueeze(0).to(device)
                eval_segment = eval_data['eval_segment'].unsqueeze(0).to(device)
                label_list = eval_data['eval_label'].tolist()

                eval_token_list = eval_token.tolist()
                input_len = len([x for x in eval_token_list[0] if x]) - 2

                mlm_isG, mlm_output = soft_masked_bert(eval_token, eval_position, eval_segment)
                mlm_output = mlm_output[:, 1:input_len + 1, :]
                output_tensor = torch.nn.Softmax(dim=-1)(mlm_output)
                output_topk = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()

                # 累计数值
                eval_count += input_len
                for j in range(input_len):
                    batch_labels = label_list[j + 1]
                    if batch_labels == output_topk[j][0]:
                        top1_count += 1
                    if batch_labels in output_topk[j]:
                        top5_count += 1

            if eval_count:
                top1_acc = float(top1_count) / float(eval_count)
                print('top1纠正正确率：%s' % round(top1_acc, 2))
                top5_acc = float(top5_count) / float(eval_count)
                print('top5纠正正确率：%s' % round(top5_acc, 2))

                # save
                if top1_acc > best_top1:
                    best_top1 = top1_acc
                    torch.save(soft_masked_bert.cpu(), FinetunePath)
                    soft_masked_bert.to(device)
                    print('EP:%d Model Saved on:%s\n' % (epoch, FinetunePath))
