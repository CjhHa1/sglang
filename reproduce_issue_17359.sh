#!/bin/bash
# 复现 Issue #17359: Qwen3-VL-235B 在 pp-size=2 时无法启动
#
# 使用方法:
#   ./reproduce_issue_17359.sh [MODEL_PATH]
#
# 示例:
#   ./reproduce_issue_17359.sh Qwen/Qwen3-VL-235B-A22B-Instruct-FP8
#

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "========================================================================"
echo "SGLang Issue #17359 复现脚本"
echo "========================================================================"

# 模型路径
MODEL_PATH=${1:-"Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"}
PP_SIZE=${2:-2}
TP_SIZE=${3:-1}
PORT=${4:-30000}

echo "模型路径: $MODEL_PATH"
echo "PP Size: $PP_SIZE"
echo "TP Size: $TP_SIZE"
echo "端口: $PORT"
echo ""

# 检查 GPU
echo "========================================================================"
echo "检查 GPU 环境..."
echo "========================================================================"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}✗ nvidia-smi 未找到，请确保安装了 NVIDIA 驱动${NC}"
    exit 1
fi

GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo -e "${GREEN}✓ 检测到 $GPU_COUNT 个 GPU${NC}"

if [ $GPU_COUNT -lt $PP_SIZE ]; then
    echo -e "${RED}✗ 错误: pp-size=$PP_SIZE 需要至少 $PP_SIZE 个 GPU，但只检测到 $GPU_COUNT 个${NC}"
    exit 1
fi

nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# 检查端口
echo ""
echo "========================================================================"
echo "检查端口 $PORT..."
echo "========================================================================"
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠ 警告: 端口 $PORT 已被占用${NC}"
    read -p "是否终止占用该端口的进程? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        PID=$(lsof -Pi :$PORT -sTCP:LISTEN -t)
        kill -9 $PID
        echo -e "${GREEN}✓ 已终止进程 $PID${NC}"
    else
        echo "请使用不同的端口或手动终止占用进程"
        exit 1
    fi
fi

# 创建日志目录
LOG_DIR="./logs_issue_17359"
mkdir -p $LOG_DIR
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/reproduce_${TIMESTAMP}.log"

echo ""
echo "========================================================================"
echo "启动服务器 (日志: $LOG_FILE)"
echo "========================================================================"

# 启动命令
CMD="python3 -m sglang.launch_server \
    --model-path $MODEL_PATH \
    --pp-size $PP_SIZE \
    --tp-size $TP_SIZE \
    --host 127.0.0.1 \
    --port $PORT \
    --trust-remote-code \
    --log-level debug"

echo "执行命令:"
echo "$CMD"
echo ""
echo "开始启动..."
echo ""

# 执行并捕获输出
set +e
$CMD 2>&1 | tee $LOG_FILE &
SERVER_PID=$!

# 等待服务器启动或失败
MAX_WAIT=300  # 最多等待 5 分钟
WAIT_TIME=0
SUCCESS=0
ERROR_FOUND=0

echo "等待服务器启动 (PID: $SERVER_PID)..."

while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo -e "${RED}✗ 服务器进程已退出${NC}"
        ERROR_FOUND=1
        break
    fi

    # 检查日志中的错误
    if grep -q "KeyError.*w2_weight" $LOG_FILE 2>/dev/null; then
        echo -e "${RED}✗ 检测到目标错误: KeyError: w2_weight${NC}"
        ERROR_FOUND=1
        kill $SERVER_PID 2>/dev/null
        break
    fi

    # 检查日志中的其他错误
    if grep -q "Error\|Exception\|Traceback" $LOG_FILE 2>/dev/null; then
        if ! grep -q "Started server" $LOG_FILE 2>/dev/null; then
            echo -e "${YELLOW}⚠ 检测到错误，查看日志获取详情${NC}"
        fi
    fi

    # 检查服务器是否成功启动
    if grep -q "Started server\|Application startup complete" $LOG_FILE 2>/dev/null; then
        echo -e "${GREEN}✓ 服务器启动成功！${NC}"
        SUCCESS=1
        kill $SERVER_PID 2>/dev/null
        break
    fi

    sleep 2
    WAIT_TIME=$((WAIT_TIME + 2))
    echo -n "."
done

echo ""
echo ""
echo "========================================================================"
echo "复现结果"
echo "========================================================================"

if [ $ERROR_FOUND -eq 1 ]; then
    echo -e "${RED}✗ 成功复现 Issue #17359${NC}"
    echo ""
    echo "错误详情:"
    grep -A 10 "KeyError\|Traceback" $LOG_FILE | head -20 || grep -A 10 "Error" $LOG_FILE | head -20
    echo ""
    echo "完整日志: $LOG_FILE"
    echo ""
    echo "问题描述:"
    echo "  - Qwen3-VL-235B-A22B-Instruct-FP8 模型无法以 pp-size=2 启动"
    echo "  - 错误: KeyError: 'model.layers.15.mlp.experts.w2_weight'"
    echo "  - 原因: MoE 模型的专家权重在 Pipeline Parallelism 分片时映射错误"
    exit 1
elif [ $SUCCESS -eq 1 ]; then
    echo -e "${GREEN}✓ 服务器启动成功，问题可能已修复${NC}"
    echo "日志: $LOG_FILE"
    exit 0
else
    echo -e "${YELLOW}⚠ 超时或未知状态${NC}"
    echo "日志: $LOG_FILE"
    if ps -p $SERVER_PID > /dev/null 2>&1; then
        kill $SERVER_PID 2>/dev/null
    fi
    exit 1
fi
