### 快速启动

#### 环境准备

请确保已经安装python环境，并下载好docker-desktop（如果你在Windows系统上）。
你可以使用uv管理依赖也可以使用conda 创建虚拟环境。

#### 依赖安装

如果使用uv管理依赖

```
uv sync 
```

如果使用conda 创建虚拟环境

```
conda create -n pytorch_env python=3.7
conda activate pytorch_env
pip install -r requirements.txt
```

#### 下载模型

```
python Scripts/download_model.py
```

#### 启动docker镜像中的数据库

```
docker compose up -d --build
```

#### 启动检索示例

```
python main.py
```
