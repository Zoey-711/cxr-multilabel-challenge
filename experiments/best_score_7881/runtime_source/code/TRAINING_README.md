训练来源保留于scripts/train.py及scripts/batch_train.py、src/cxr/和train_records/。
各模型原训练参数和路径见train_records；推理包不启动训练。
code/predict.py为最终T4入口；scripts/predict.py和其他历史生成脚本只作训练/历史流程来源，不作为本次正式推理入口。
