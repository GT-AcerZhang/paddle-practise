以手写体数字识别为例，演示动态图的构建、训练、模型保存与加载、图片预测等过程
1. 训练
python train.py cpu ../data/dygraph
python train.py onegpu ../data/dygraph
python -m paddle.distributed.launch --selected_gpus=0,1,2,3 --log_dir . train.py multigpu ../data/dygraph
2. 预测
python predict.py ../data/dygraph
3. 一些特性演示
python demo.py
