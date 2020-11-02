import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch.nn as nn

from torch.optim import Adam
from torch.utils.data import DataLoader
from smbert.data.mlm_dataset import *
from smbert.layers.SM_Bert_mlm import SMBertMlm


if __name__ == '__main__':
    if Debug:
        print('开始训练 %s' % get_time())
    onehot_type = False
    soft_masked_bert = SMBertMlm().to(device)
    if Debug:
        print('Total Parameters:', sum([p.nelement() for p in soft_masked_bert.parameters()]))

    if os.path.exists(PretrainPath):
        print('开始加载预训练模型！')
        soft_masked_bert.load_pretrain(PretrainPath)
        print('完成加载预训练模型！')

    dataset = SMBertDataSet(CorpusPath, onehot_type)
    dataloader = DataLoader(dataset=dataset, batch_size=BatchSize, shuffle=True, drop_last=True)
    testset = RobertaTestSet(TestPath)

    optim = Adam(soft_masked_bert.parameters(), lr=MLMLearningRate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(MLMEpochs):
        # train
        if Debug:
            print('第%s个Epoch %s' % (epoch, get_time()))
        soft_masked_bert.train()
        data_iter = tqdm(enumerate(dataloader),
                         desc='EP_%s:%d' % ('train', epoch),
                         total=len(dataloader),
                         bar_format='{l_bar}{r_bar}')
        print_loss = 0.0
        for i, data in data_iter:
            if Debug:
                print('生成数据 %s' % get_time())
            data = {k: v.to(device) for k, v in data.items()}
            input_token = data['input_token_ids']
            segment_ids = data['segment_ids']
            label = data['token_ids_labels']
            if Debug:
                print('获取数据 %s' % get_time())
            mlm_output = soft_masked_bert(input_token, segment_ids).permute(0, 2, 1)
            if Debug:
                print('完成前向 %s' % get_time())
            mask_loss = criterion(mlm_output, label)
            print_loss = mask_loss.item()
            optim.zero_grad()
            mask_loss.backward()
            optim.step()
            if Debug:
                print('完成反向 %s\n' % get_time())

        print('EP_%d mask loss:%s' % (epoch, print_loss))

        # save
        output_path = FinetunePath + '.ep%d' % epoch
        torch.save(soft_masked_bert.cpu(), output_path)
        soft_masked_bert.to(device)
        print('EP:%d Model Saved on:%s' % (epoch, output_path))

        # test
        with torch.no_grad():
            soft_masked_bert.eval()
            test_count = 0
            top1_count = 0
            top5_count = 0
            for test_data in testset:
                input_token = test_data['input_token_ids'].unsqueeze(0).to(device)
                segment_ids = test_data['segment_ids'].unsqueeze(0).to(device)
                input_token_list = input_token.tolist()
                input_len = len([x for x in input_token_list[0] if x]) - 2
                label_list = test_data['token_ids_labels'].tolist()
                mlm_output = soft_masked_bert(input_token, segment_ids)[:, 1:input_len + 1, :]
                output_tensor = torch.nn.Softmax(dim=-1)(mlm_output)
                output_topk = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()

                # 累计数值
                test_count += input_len
                for i in range(input_len):
                    label = label_list[i + 1]
                    if label == output_topk[i][0]:
                        top1_count += 1
                    if label in output_topk[i]:
                        top5_count += 1

            if test_count:
                top1_acc = float(top1_count) / float(test_count)
                acc = round(top1_acc, 2)
                print('top1纠正正确率：%s' % acc)
                top5_acc = float(top5_count) / float(test_count)
                acc = round(top5_acc, 2)
                print('top5纠正正确率：%s\n' % acc)
