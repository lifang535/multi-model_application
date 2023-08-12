# multi-model_application
This is a test of multi-model_application.

## 代码逻辑

`'pipeline.py'` 代码共分成六个模块，`Loader` 模块向 `Model_1` 以一定的速率传输视频路径；

`Model_1` 加载视频，将视频拆解成视频帧，再对视频帧使用模型 ''hustvl/yolos-tiny'' 处理，将检测出 `'car'` 的视频帧发送给 `Model_2`，将检测出 `'person'` 的视频帧发送给 `Model_3`；

`Model_2` 使用模型 `'facebook/detr-resnet-50'`，`Model_3` 使用模型 `'facebook/detr-resnet-101'`，分别处理各自接收到的视频帧，并将绘图信息（label、score、box...）发送给 `Model_4`；

`Model_4` 接收绘图信息后处理视频帧，单个视频处理结束后向 `Model_5` 发送信息；

`Model_5` 将处理后的视频帧重组成视频并存储。

![Image](https://github.com/lifang535/multi-model_application/blob/main/multi-model_application/modules/multi-model_structure.png)

各个模块检测接收上级模块请求的速率，并将结果储存在 `'/multi-model_app/modules/logs_rate'` 文件夹中，运行 `'draw_data.py'` 绘制各模块检测的请求到达速率折线图：

### 1. same rate and different video
```
time.sleep(10) # 发送视频的时间间隔相同
input_video_dir = 'input_videos' # 存储了 15 个不同的视频
```

![Image](https://github.com/lifang535/multi-model_application/blob/main/multi-model_application/modules/same_rate_and_different_video.png)

速率较大的位置大致即处理数据量高的视频和视频帧时的位置；后边 `Model_2` 速率为 0，而 `Model_3` 不为 0，说明视频的数据量影响了各模块接收请求的速率

### 2. different rate and same video
```
sleep_time = [5, 10, 15, 20, 25, 25, 20, 15, 10, 5, 10, 15, 20, 25, 30] # 发送视频的时间间隔不同
input_video_dir = 'input_videos_2' # 存储了 15 个相同的视频
```

![Image](https://github.com/lifang535/multi-model_application/blob/main/multi-model_application/modules/different_rate_and_same_video.png)

前部分 `Model_1` 速率较低时，其它模块部分速率也较低；后部分速率高峰和低谷不明显可能是由于队列阻塞

### 3. same rate and same video
```
time.sleep(10) # 发送视频的时间间隔相同
input_video_dir = 'input_videos_2' # 存储了 15 个相同的视频
```

![Image](https://github.com/lifang535/multi-model_application/blob/main/multi-model_application/modules/same_rate_and_same_video.png)

相较于前两次实验，整体速率较为稳定，说明输入速率和视频不变时，处理速率较为稳定（上下波动是由于同一视频的不同段连续的视频帧数据量不同导致的）

## 问题

`to_monitor_rate` 应该不用作为进程共享变量，只在 `monitor_rate(self)` 中定义即可，但是调整之后速率计算的误差会变大

`'different_rate_and_same_video.png'` 中，前方模块速率增大对后方影响不明显的原因可能是队列阻塞，后方速率达到最大值

处理队列阻塞主要发生在 `Model_1`，直接增多数量会造成 cuda 内存不足，考虑调整模型数量比与和发送视频速率
