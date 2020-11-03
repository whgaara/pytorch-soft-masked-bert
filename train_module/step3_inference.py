# coding: utf-8
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch

from tqdm import tqdm
from char_sim import CharFuncs
from pretrain_config import PretrainPath, device, PronunciationPath, SentenceLength
from smbert.data.mlm_dataset import DataFactory


def curve(confidence, similarity):
    flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
    flag2 = 0.1 * confidence + similarity - 0.6 > 0
    if flag1 or flag2:
        return True
    return False


class Inference(object):
    def __init__(self, mode='s'):
        self.char_count = 0
        self.top1_acc = 0
        self.top5_acc = 0
        self.sen_count = 0
        self.sen_acc = 0
        self.sen_invalid = 0
        self.sen_wrong = 0
        self.mode = mode
        self.model = torch.load(PretrainPath).to(device).eval()
        self.char_func = CharFuncs(PronunciationPath)
        self.smbert_data = DataFactory()
        print('加载模型完成！')

    def get_id_from_text(self, text):
        assert isinstance(text, str)
        inputs = []
        segments = []
        text = [text]
        ids = self.smbert_data.texts_to_ids(text)
        inputs.append(self.smbert_data.token_cls_id)
        segments.append(1)
        for id in ids:
            if len(inputs) < SentenceLength - 1:
                if isinstance(id, list):
                    for x in id:
                        inputs.append(x)
                        segments.append(1)
                else:
                    inputs.append(id)
                    segments.append(1)
            else:
                inputs.append(self.smbert_data.token_sep_id)
                segments.append(1)
                break

        if len(inputs) != len(segments):
            print('len error!')
            return None

        if len(inputs) < SentenceLength - 1:
            inputs.append(self.smbert_data.token_sep_id)
            segments.append(1)
            for i in range(SentenceLength - len(inputs)):
                inputs.append(self.smbert_data.token_pad_id)
                segments.append(self.smbert_data.token_pad_id)

        inputs = torch.tensor(inputs).unsqueeze(0).to(device)
        segments = torch.tensor(segments).unsqueeze(0).to(device)
        return inputs, segments

    def get_topk(self, text):
        input_len = len(text)
        text2id, segments = self.get_id_from_text(text)
        with torch.no_grad():
            result = []
            output_tensor = self.model(text2id, segments)[:, 1:input_len + 1, :]
            output_tensor = torch.nn.Softmax(dim=-1)(output_tensor)
            output_topk_prob = torch.topk(output_tensor, 5).values.squeeze(0).tolist()
            output_topk_indice = torch.topk(output_tensor, 5).indices.squeeze(0).tolist()
            for i, words in enumerate(output_topk_indice):
                tmp = []
                for j, candidate in enumerate(words):
                    word = self.smbert_data.tokenizer.id_to_token(candidate)
                    tmp.append(word)
                result.append(tmp)
        return result, output_topk_prob

    def inference_single(self, text, gt=''):
        candidates, probs = self.get_topk(text)
        text_list = list(text)
        correct_sentence = []
        result = {
            '原句': text,
            '纠正': '',
            '纠正数据': [
            ]
        }

        for i, ori in enumerate(text_list):
            if ori == candidates[i][0]:
                correct_sentence.append(ori)
                self.top1_acc += 1
                self.top5_acc += 1
                self.char_count += 1
                continue
            correct = {}
            correct['原字'] = ori
            candidate = candidates[i]
            confidence = probs[i]

            # 统计正确率数据
            if gt:
                gt = list(gt)
                self.char_count += 1
                if gt[i] == candidate[0]:
                    self.top1_acc += 1
                if gt[i] in candidate:
                    self.top5_acc += 1

            if self.mode == 'p':
                if ori in candidate:
                    correct_sentence.append(ori)
                    continue
                else:
                    max_can = ''
                    max_sim = 0
                    max_conf = 0
                    for j, can in enumerate(candidate):
                        similarity = self.char_func.similarity(ori, can)
                        if similarity > max_sim:
                            max_can = can
                            max_sim = similarity
                            max_conf = confidence[j]
                    # if max_sim > 0.5:
                    if curve(max_conf, max_sim):
                        correct['新字'] = max_can
                        correct['相似度'] = max_sim
                        result['纠正数据'].append(correct)
                        correct_sentence.append(max_can)
                    else:
                        correct_sentence.append(ori)
            else:
                tmp_can = []
                tmp_cof = []
                for index, score in enumerate(confidence):
                    if score > 0.001:
                        tmp_can.append(candidate[index])
                        tmp_cof.append(confidence[index])
                if ori in tmp_can:
                    correct_sentence.append(ori)
                    continue
                if confidence[0] > 0.99:
                    correct['新字'] = candidate[0]
                    correct['候选字'] = candidate
                    correct['置信度'] = confidence
                    result['纠正数据'].append(correct)
                    correct_sentence.append(candidate[0])
                else:
                    correct_sentence.append(ori)

        result['纠正'] = ''.join(correct_sentence)
        return result

    def inference_batch(self, file_path):
        f = open(file_path, 'r', encoding='utf-8')
        for line in tqdm(f):
            if line:
                line = line.strip()
                self.sen_count += 1
                line = line.split('-***-')
                src = line[0]
                target = line[1]
                result = self.inference_single(target, src)
                if src == result['纠正']:
                    self.sen_acc += 1
                else:
                    self.sen_invalid += 1
                    if target != result['纠正']:
                        self.sen_wrong += 1
                    print(src, result['纠正'])
        print('句子正确个数：%s，句子总共个数：%s，句子正确率：%s' %
              (self.sen_acc, self.sen_count, round(float(self.sen_acc) / float(self.sen_count), 2)))
        print('句子纠错个数：%s，句子未纠正个数：%s，句子纠错率：%s' %
              (self.sen_wrong, self.sen_invalid, round(float(self.sen_wrong) / float(self.sen_invalid), 2)))
        print('top1正确个数：%s，top1总共个数：%s，top1正确率：%s' %
              (self.top1_acc, self.char_count, round(float(self.top1_acc) / float(self.char_count), 2)))
        print('top5正确个数：%s，top5总共个数：%s，top5正确率：%s' %
              (self.top5_acc, self.char_count, round(float(self.top5_acc) / float(self.char_count), 2)))


if __name__ == '__main__':
    bert_infer = Inference()
    bert_infer.inference_batch('../data/test_data/test.txt')
