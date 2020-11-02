import time
import torch

cuda_condition = torch.cuda.is_available()
device = torch.device('cuda:0' if cuda_condition else 'cpu')

VocabPath = '../../checkpoint/pretrain/vocab.txt'

# ## mlm模型文件路径 ## #
SourcePath = '../../data/src_data/src_data.txt'
CorpusPath = '../../data/train_data/train.txt'
TestPath = '../../data/test_data/test.txt'
PronunciationPath = '../../data/char_meta.txt'

# Debug开关
Debug = False

# 使用预训练模型开关
UsePretrain = True

# 任务模式
ModelClass = 'Bert'

# ## MLM训练调试参数开始 ## #
MLMEpochs = 16
WordGenTimes = 10
if WordGenTimes > 1:
    RanWrongDivisor = 1.0
else:
    RanWrongDivisor = 0.15
MLMLearningRate = 1e-4
if ModelClass == 'Bert':
    RepeatNum = 1
    BatchSize = 16
    SentenceLength = 128
    PretrainPath = '../../checkpoint/finetune/mlm_trained_%s.model' % SentenceLength
if ModelClass == 'SMBertMlm':
    RepeatNum = 10
    BatchSize = 1
    SentenceLength = 512
    PretrainPath = '../../checkpoint/pretrain/pytorch_model.bin'
FinetunePath = '../../checkpoint/finetune/mlm_trained_%s.model' % SentenceLength
# ## MLM训练调试参数结束 ## #

# ## MLM通用参数 ## #
DropOut = 0.1
MaskRate = 0.15
VocabSize = len(open(VocabPath, 'r', encoding='utf-8').readlines())
HiddenSize = 768
HiddenLayerNum = 12
IntermediateSize = 3072
AttentionHeadNum = 12

# 参数名配对
local2target_emb = {
    'roberta_emd.token_embeddings.weight': 'bert.embeddings.word_embeddings.weight',
    'roberta_emd.type_embeddings.weight': 'bert.embeddings.token_type_embeddings.weight',
    'roberta_emd.position_embeddings.weight': 'bert.embeddings.position_embeddings.weight',
    'roberta_emd.emb_normalization.weight': 'bert.embeddings.LayerNorm.weight',
    'roberta_emd.emb_normalization.bias': 'bert.embeddings.LayerNorm.bias'
}

local2target_transformer = {
    'transformer_blocks.%s.multi_attention.q_dense.weight': 'bert.encoder.layer.%s.attention.self.query.weight',
    'transformer_blocks.%s.multi_attention.q_dense.bias': 'bert.encoder.layer.%s.attention.self.query.bias',
    'transformer_blocks.%s.multi_attention.k_dense.weight': 'bert.encoder.layer.%s.attention.self.key.weight',
    'transformer_blocks.%s.multi_attention.k_dense.bias': 'bert.encoder.layer.%s.attention.self.key.bias',
    'transformer_blocks.%s.multi_attention.v_dense.weight': 'bert.encoder.layer.%s.attention.self.value.weight',
    'transformer_blocks.%s.multi_attention.v_dense.bias': 'bert.encoder.layer.%s.attention.self.value.bias',
    'transformer_blocks.%s.multi_attention.o_dense.weight': 'bert.encoder.layer.%s.attention.output.dense.weight',
    'transformer_blocks.%s.multi_attention.o_dense.bias': 'bert.encoder.layer.%s.attention.output.dense.bias',
    'transformer_blocks.%s.attention_layernorm.weight': 'bert.encoder.layer.%s.attention.output.LayerNorm.weight',
    'transformer_blocks.%s.attention_layernorm.bias': 'bert.encoder.layer.%s.attention.output.LayerNorm.bias',
    'transformer_blocks.%s.feedforward.dense1.weight': 'bert.encoder.layer.%s.intermediate.dense.weight',
    'transformer_blocks.%s.feedforward.dense1.bias': 'bert.encoder.layer.%s.intermediate.dense.bias',
    'transformer_blocks.%s.feedforward.dense2.weight': 'bert.encoder.layer.%s.output.dense.weight',
    'transformer_blocks.%s.feedforward.dense2.bias': 'bert.encoder.layer.%s.output.dense.bias',
    'transformer_blocks.%s.feedforward_layernorm.weight': 'bert.encoder.layer.%s.output.LayerNorm.weight',
    'transformer_blocks.%s.feedforward_layernorm.bias': 'bert.encoder.layer.%s.output.LayerNorm.bias',
}

# ## NER训练调试参数开始 ## #
NEREpochs = 8
NERLearningRate = 1e-3
NerBatchSize = 8
MedicineLength = 32
# ## NER训练调试参数结束 ## #

# ## ner模型文件路径 ## #
NerSourcePath = '../../data/src_data/ner_src_data.txt'
NerCorpusPath = '../../data/train_data/ner_train.txt'
NerTestPath = '../../data/test_data/ner_test.txt'
Class2NumFile = '../../data/train_data/c2n.pickle'
NerFinetunePath = '../../checkpoint/finetune/ner_trained_%s.model' % MedicineLength

# ## NER通用参数 ## #
NormalChar = 'ptzf'


def get_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
