import time
import torch

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

VocabPath = 'checkpoint/pretrain/vocab.txt'

# ## mlm模型文件路径 ## #
SourcePath = 'data/src_data/src_data.txt'
CorpusPath = 'data/train_data/train.txt'
TestPath = 'data/test_data/test.txt'
PronunciationPath = 'data/char_meta.txt'

# Debug开关
Debug = False

# attention_mask开关
AttentionMask = True

# 使用预训练模型开关
UsePretrain = True

# mask方式
AllMask = True

# ## MLM训练调试参数开始 ## #
MLMEpochs = 16
WordGenTimes = 3
if WordGenTimes > 1:
    RanWrongDivisor = 1.0
else:
    RanWrongDivisor = 0.15
MLMLearningRate = 1e-4
BatchSize = 16
SentenceLength = 512
FinetunePath = 'checkpoint/finetune/mlm_trained_%s.model' % SentenceLength
# ## MLM训练调试参数结束 ## #

# ## MLM通用参数 ## #
DropOut = 0.1
MaskRate = 0.15
VocabSize = len(open(VocabPath, 'r', encoding='utf-8').readlines())
HiddenSize = 768
HiddenLayerNum = 12
IntermediateSize = 3072
AttentionHeadNum = 12


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
