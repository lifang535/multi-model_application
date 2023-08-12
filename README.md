# multi-model_application
This is a test of multi-model_application.

# multi-model_app
This is a test of multi-model_app.

## 代码逻辑

`'pipeline.py'` 代码共分成六个模块，`Loader` 模块向 `Model_1` 以一定的速率传输视频路径；

`Model_1` 加载视频，将视频拆解成视频帧，再对视频帧使用模型 ''hustvl/yolos-tiny'' 处理，将检测出 `'car'` 的视频帧发送给 `Model_2`，将检测出 `'person'` 的视频帧发送给 `Model_3`；

`Model_2` 使用模型 `'facebook/detr-resnet-50'`，`Model_3` 使用模型 `'facebook/detr-resnet-101'`，分别处理各自接收到的视频帧，并将绘图信息（label、score、box...）发送给 `Model_4`；

`Model_4` 接收绘图信息后处理视频帧，单个视频处理结束后向 `Model_5` 发送信息；

`Model_5` 将处理后的视频帧重组成视频并存储。

![Image](https://github.com/lifang535/multi-model_app/blob/main/multi-model_application/modules/multi-model_structure.png)

各个模块检测接收上级模块请求的速率，并将结果储存在 `'/multi-model_app/modules/logs_rate'` 文件夹中，运行 `'draw_data.py'` 绘制各模块检测的请求到达速率折线图：

![Image](https://github.com/lifang535/multi-model_app/blob/main/multi-model_application/modules/multi-model_curve_graph.png)

## 问题

`to_monitor_rate` 应该不用作为进程共享变量，只在 `monitor_rate(self)` 中定义即可，但是调整之后速率计算的误差会变大

处理队列阻塞主要发生在 `Model_1`，直接增多数量会造成 cuda 内存不足，考虑调整模型数量比与发送视频速率


