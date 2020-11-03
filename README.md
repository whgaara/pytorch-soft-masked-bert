# 复现：Soft-Masked-Bert
- 更清晰的torch版Soft-Masked-Bert
- csdn：https://blog.csdn.net/BmwGaara
- 知乎：https://zhuanlan.zhihu.com/c_1264223811643252736

## 说明
前段时间本人复现了bert和roberta两片文章，正好工作中遇到错别字纠正的场景，于是看到了Soft-Masked-Bert这片文章，出于一个nlper的专业心里，毫不犹豫复现了它。
先说结论，相比于bert而言，roberta和smbert都使用更多的技巧，但是就是论模型的效果和便利，我还是站bert。
知乎专栏和CSDN专栏会有详细的解读。

## 错别字纠错的模型使用
本模型没有设置预训练模型的加载模块：
- 第一步，将训练文本添加到data/src_data中，文本内容就是一行行的句子即可。
- 第二步，进入train_module，运行stp1_gen_train_test.py生成对应的训练和测试集。
- 第三步，打开根目录的pretrain_config.py设置你需要的参数。
- 第四步，修改好参数后，即可运行python3 step2_pretrain_mlm.py来训练了，这里训练的只是掩码模型。训练生成的模型保存在checkpoint/finetune里。
- 第五步，如果你需要预测并测试你的模型，则需要运行根目录下的step3_inference.py。需要注意的事，你需要将训练生成的模型改名成：mlm_trained_xx.model，xx是设置的句子最大长度，或者自行统一模型名称。
预测中有一个参数：mode，值为'p'或者's'，前者表示按拼音相似性纠错，后者表示按字形相似性纠错。遗憾的是汉字的笔画数据本人没空准备，因此自形相似性的纠正字是候选字中的top1。

我是隔壁小王，欢迎大家留言了。
