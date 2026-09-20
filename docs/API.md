# KineJing 本地研究 API

状态：**真实权重已通过 HTTP 调用；本机研究预览，未开放公网、未开通收费。**
输出未来视觉特征，不是视频，不提供机器人控制指令。当前检查点的适用范围与限制见[模型卡](DYNAMICS_MODEL_CARD.md)。

## 启动

使用已有神经网络环境（Python 3.10+、PyTorch、torchvision、NumPy、OpenCV、h5py），另装 `pip install -r requirements-api.txt`。

在本机环境变量中设置以下值，不将密钥提交到 Git：

```text
KINEJING_API_KEY=<至少24字符的随机密钥>
KINEJING_CHECKPOINT=<kinejing-dynamics.pt 的本地路径>
KINEJING_ENCODER_WEIGHTS=<官方 dinov2_vits14_pretrain.pth 的本地路径>
KINEJING_DINO_SOURCE=<模型卡指定版本的 DINOv2 本地源码目录>
KINEJING_DEVICE=cuda
KINEJING_PORT=8765
```

```bash
python -m kinejing.api
```

服务固定绑定 `127.0.0.1`，单进程加载权重。访问 `http://127.0.0.1:8765/docs` 查看接口结构。

## 接口

- `GET /health`：启动后的就绪状态。
- `GET /v1/models`：需要 `Authorization: Bearer <密钥>`，返回真实支持的模型与输出类型。
- `POST /v1/predictions`：同样需要密钥。

请求 JSON 字段：

- `action_schema`：必须为 `triworldbench-joint-action-vector-v1`。
- `images`：恰好包含 `head`、`left`、`right`，值为 JPEG/PNG 文件的 base64 字符串。每张解码后不超过 1 MiB，最长边不超过 2048 像素；只接收单帧。
- `actions`：9～4096 行，每行 14 个有限数值，保持训练数据的原始列顺序和单位。服务均匀取 9 个时刻。

返回 `predicted_features`，形状为 `[9,3,768]`，并包含抽样帧索引、模型检查点哈希、请求编号和推理耗时。特征不是像素或任务成功概率。

对已有的合法输入目录，可直接运行：

```bash
python scripts/call_api.py --inputs data/test_dataset --episode 1
```

服务在读取请求体前鉴权，请求体上限 4 MiB；拒绝额外字段、错误动作约定、无效图片与非有限动作。GPU 忙时返回 429；内部推理失败返回 503 并释放锁。默认不记录请求图像、动作或访问日志。

## 已验证与未验证

已执行真实本地 HTTP 冒烟测试：加载正式研究检查点，对测试输入首帧及动作进行预测；未读取未来图像。测试了未授权请求和无效图片。接口专项测试另覆盖输入限制、并发拒绝及失败后恢复。

尚未进行多客户负载测试、长时间稳定性测试或部署容量评估。现有共享密钥仅适合本机开发；不得据此宣称生产级隔离或可收费交付。

## 从演示到收费服务

1. 明确客户需要的视频、动作预测或其他输出，以及可接受的质量和耗时。当前接口只能交付特征。
2. 在目标客户数据上验证价值；视频 API 需要接入并真正运行视频生成权重。
3. 测量每次成功任务的 GPU 时间、失败重试、存储与网络成本，再确定价格和配额。
4. 部署持久在线服务，添加 HTTPS、每客户密钥、配额、持久任务队列、幂等提交与用量记录；长视频任务采用异步任务接口。
5. 核对实际采用的模型、权重、训练数据和服务用途的许可。JEPA-WMs 的非商用研究代码不默认纳入收费产品。
6. 小范围试用，验证失败处理与容量后再开始收费。

开源代码无需购买不等于托管服务没有计算、存储或运维成本。这个本地原型没有账单、支付或外部云资源。
