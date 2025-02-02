# Flower-Classification
Version 1  ----关键优化点说明

混合精度训练（AMP）：
使用 torch.cuda.amp.autocast 和 GradScaler，显著减少显存占用并加速计算。
适用场景：P100支持FP16计算，适合大规模矩阵运算。

数据加载优化：
增大 num_workers=8，充分利用CPU多线程预加载数据。
启用 pin_memory=True 和 non_blocking=True，减少CPU到GPU的数据传输延迟。
使用 persistent_workers=True，避免重复初始化数据加载线程。

批量大小调整：
将 BATCH_SIZE 从128增加到256，充分利用P100的16GB显存。

学习率调度策略：
使用逐batch更新的余弦退火（T_max=NUM_EPOCHS * len(train_loader)），更精细调整学习率。

模型训练加速：
启用 torch.backends.cudnn.benchmark = True，自动优化卷积算法。
使用异步数据传输（non_blocking=True），覆盖计算与数据传输时间。

验证过程优化：
批量计算预测结果（避免逐样本处理），减少循环开销。

日志与调试：
添加训练耗时统计，监控GPU利用率。

性能提升预估
优化项	预期加速比	显存占用降低

混合精度训练（AMP）	2-3x	40%-50%

增大Batch Size	1.5-2x	-

数据加载优化	1.2-1.5x	-
