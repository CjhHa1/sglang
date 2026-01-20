# 复现 Issue #17359: Qwen3-VL-235B PP-size 2 启动失败

这个目录包含了用于复现 SGLang Issue #17359 的脚本和工具。

## 问题描述

**Issue**: [#17359](https://github.com/sgl-project/sglang/issues/17359)

**标题**: [Bug] Qwen-3-VL-235B won't start with pp-size 2 (KeyError: 'model.layers.15.mlp.experts.w2_weight')

**症状**:
- Qwen3-VL-235B-A22B-Instruct-FP8 模型无法以 `--pp-size 2` 启动
- 报错: `KeyError: 'model.layers.15.mlp.experts.w2_weight'`
- 环境: Docker v0.5.7, 4x H200 GPUs

**根本原因**:
在 Pipeline Parallelism (PP) 模式下，模型层被分配到不同的 GPU ranks。MoE 模型的专家权重在加载时没有正确处理 PP 分片，导致某些层的权重在 `params_dict` 中找不到。

## 文件说明

### 1. `reproduce_issue_17359.py` (Python 复现脚本)

功能完整的 Python 脚本，可以:
- 检查环境(GPU、SGLang 版本)
- 调试模式分析模型配置和层分配
- 启动服务器以复现问题
- 捕获并显示错误信息

**使用方法**:

```bash
# 基础用法
python reproduce_issue_17359.py

# 指定模型路径
python reproduce_issue_17359.py --model-path Qwen/Qwen3-VL-235B-A22B-Instruct-FP8

# 修改 PP/TP 配置
python reproduce_issue_17359.py --pp-size 2 --tp-size 1

# 仅运行调试分析（不启动完整服务器）
python reproduce_issue_17359.py --debug-only

# 跳过环境检查
python reproduce_issue_17359.py --skip-env-check
```

**参数说明**:
- `--model-path`: 模型路径(默认: Qwen/Qwen3-VL-235B-A22B-Instruct-FP8)
- `--pp-size`: Pipeline parallelism size (默认: 2)
- `--tp-size`: Tensor parallelism size (默认: 1)
- `--debug-only`: 仅运行调试模式
- `--skip-env-check`: 跳过环境检查

### 2. `reproduce_issue_17359.sh` (Bash 复现脚本)

简单的 Bash 脚本，用于快速复现问题。

**使用方法**:

```bash
# 添加执行权限
chmod +x reproduce_issue_17359.sh

# 基础用法
./reproduce_issue_17359.sh

# 指定模型路径
./reproduce_issue_17359.sh Qwen/Qwen3-VL-235B-A22B-Instruct-FP8

# 指定 PP size, TP size, 端口
./reproduce_issue_17359.sh Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 2 1 30000
```

**功能**:
- 自动检查 GPU 环境
- 检查端口占用
- 实时显示服务器日志
- 检测特定错误并高亮显示
- 保存完整日志到 `logs_issue_17359/` 目录

### 3. `analyze_qwen3_vl_weights.py` (权重分析工具)

深入分析模型结构和权重分布的工具。

**使用方法**:

```bash
# 分析模型配置和 PP 层分配
python analyze_qwen3_vl_weights.py

# 指定不同的 PP size
python analyze_qwen3_vl_weights.py --pp-size 4

# 分析实际权重文件（需要下载模型）
python analyze_qwen3_vl_weights.py --analyze-weights

# 指定模型路径
python analyze_qwen3_vl_weights.py --model-path Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
```

**输出信息**:
- 模型配置详情(层数、专家数等)
- PP 模式下的层分配方案
- 权重命名模式
- 第 15 层的专家权重信息
- 修复建议

## 环境要求

### 硬件
- **最少**: 2 个 GPU (用于 pp-size=2)
- **推荐**: 4 个 H200/H100 GPU (与原始 issue 环境一致)

### 软件
- Python 3.8+
- SGLang (最好是 v0.5.7 与原始 issue 一致)
- PyTorch
- transformers
- CUDA 驱动

### 安装依赖

```bash
# 安装 SGLang
pip install sglang

# 或者从源码安装
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e "python[all]"

# 安装其他依赖
pip install transformers safetensors huggingface_hub
```

## 复现步骤

### 方法 1: 使用 Python 脚本 (推荐)

```bash
# 1. 检查环境并运行完整复现
python reproduce_issue_17359.py

# 输出会显示:
# - GPU 检测结果
# - 模型层分配详情
# - 服务器启动过程
# - 错误信息（如果复现成功）
```

### 方法 2: 使用 Bash 脚本

```bash
# 1. 添加执行权限
chmod +x reproduce_issue_17359.sh

# 2. 运行脚本
./reproduce_issue_17359.sh

# 日志会保存在 logs_issue_17359/ 目录
```

### 方法 3: 手动复现

```bash
# 直接启动服务器
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
    --pp-size 2 \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port 30000 \
    --trust-remote-code \
    --log-level debug
```

## 预期结果

如果成功复现 Issue #17359，你应该看到类似以下的错误:

```
KeyError: 'model.layers.15.mlp.experts.w2_weight'
```

完整的错误堆栈应该指向权重加载相关的代码，特别是：
- `python/sglang/srt/models/qwen3_vl_moe.py` 中的 `load_weights` 方法
- 专家权重的映射和加载逻辑

## 问题分析

### 层分配示例 (假设 160 层, pp-size=2)

```
Rank 0: layers 0-79   (80 层)
Rank 1: layers 80-159 (80 层)
```

第 15 层属于 Rank 0，但权重加载时可能：
1. `params_dict` 中不包含该层的参数(因为 PP 分片)
2. 专家权重的映射逻辑没有正确处理跨 rank 的情况
3. 权重名称转换时出现错误

### 相关代码位置

1. **PP 层分配**: `python/sglang/srt/distributed/utils.py:63-99`
   - `get_pp_indices()` 函数

2. **权重加载**: `python/sglang/srt/models/qwen3_vl_moe.py:183-350`
   - `load_weights()` 方法
   - 专家权重映射逻辑

3. **层创建**: `python/sglang/srt/utils/common.py:588-629`
   - `make_layers()` 函数

## 修复方向

基于代码分析，可能的修复方向：

1. **检查层范围**: 在加载专家权重前，检查该层是否属于当前 PP rank
   ```python
   # 在 load_weights 中添加检查
   layer_id = extract_layer_id(name)
   if layer_id < self.start_layer or layer_id >= self.end_layer:
       continue  # 跳过不属于当前 rank 的层
   ```

2. **修复参数查找逻辑**: 确保 `params_dict` 查找时正确处理 PPMissingLayer
   ```python
   if name not in params_dict:
       # 检查是否是因为 PP 分片导致的缺失
       if is_pp_missing_layer(name, self.start_layer, self.end_layer):
           continue
       else:
           logger.warning(f"Parameter {name} not found")
   ```

3. **参考其他 MoE 模型**: 查看 DeepSeek-V2 等其他 MoE 模型在 PP 模式下的权重加载实现

## 调试技巧

### 1. 查看层分配

```bash
python analyze_qwen3_vl_weights.py --pp-size 2
```

### 2. 启用详细日志

```bash
export SGLANG_LOG_LEVEL=debug
python -m sglang.launch_server ... --log-level debug
```

### 3. 打印权重加载过程

在 `qwen3_vl_moe.py:load_weights` 中添加调试输出:

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
    for name, loaded_weight in weights:
        print(f"Loading weight: {name}, shape: {loaded_weight.shape}")
        # ... 原有逻辑
```

## 验证修复

修复后，使用以下命令验证:

```bash
# 应该能够成功启动
python -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
    --pp-size 2 \
    --tp-size 1 \
    --host 127.0.0.1 \
    --port 30000 \
    --trust-remote-code

# 测试推理
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## 相关资源

- **原始 Issue**: https://github.com/sgl-project/sglang/issues/17359
- **Pipeline Parallelism 文档**: `docs/advanced_features/pipeline_parallelism.md`
- **Qwen3-VL 使用文档**: `docs/basic_usage/qwen3_vl.md`
- **模型页面**: https://huggingface.co/Qwen/Qwen3-VL-235B-A22B-Instruct-FP8

## 贡献

如果你：
- 成功复现了问题
- 找到了解决方案
- 有其他发现

欢迎在原始 issue 中评论或提交 PR！
