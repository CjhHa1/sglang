#!/usr/bin/env python3
"""
复现 Issue #17359: Qwen3-VL-235B won't start with pp-size 2
KeyError: 'model.layers.15.mlp.experts.w2_weight'

该脚本尝试启动 Qwen3-VL-235B-A22B-Instruct-FP8 模型并设置 pp-size=2，
以复现权重加载错误。
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path


def check_environment():
    """检查环境要求"""
    print("=" * 80)
    print("环境检查")
    print("=" * 80)

    # 检查 GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        gpus = result.stdout.strip().split('\n')
        print(f"✓ 检测到 {len(gpus)} 个 GPU:")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu}")

        if len(gpus) < 2:
            print(f"⚠ 警告: pp-size=2 至少需要 2 个 GPU，当前只有 {len(gpus)} 个")
            return False
    except Exception as e:
        print(f"✗ GPU 检测失败: {e}")
        return False

    # 检查 SGLang 版本
    try:
        import sglang
        print(f"✓ SGLang 版本: {sglang.__version__}")
    except ImportError:
        print("✗ SGLang 未安装")
        return False

    return True


def reproduce_with_minimal_config(model_path: str, pp_size: int = 2, tp_size: int = 1):
    """
    使用最小配置复现问题

    Args:
        model_path: 模型路径，例如 "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8"
        pp_size: Pipeline parallelism size (默认 2，会触发 bug)
        tp_size: Tensor parallelism size (默认 1)
    """
    print("\n" + "=" * 80)
    print(f"复现配置: pp_size={pp_size}, tp_size={tp_size}")
    print("=" * 80)

    # 构建启动命令
    cmd = [
        sys.executable,
        "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--pp-size", str(pp_size),
        "--tp-size", str(tp_size),
        "--host", "127.0.0.1",
        "--port", "30000",
        "--trust-remote-code",
        # 添加调试选项
        "--log-level", "debug",
    ]

    print(f"\n启动命令:")
    print(" ".join(cmd))
    print("\n" + "=" * 80)
    print("开始启动服务器...")
    print("=" * 80)

    # 启动服务器
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        # 实时打印输出
        error_found = False
        error_lines = []

        for line in process.stdout:
            print(line, end='')

            # 检测错误
            if "KeyError" in line or "w2_weight" in line:
                error_found = True
                error_lines.append(line)

            # 如果服务器成功启动
            if "Started server process" in line or "Application startup complete" in line:
                print("\n" + "=" * 80)
                print("✓ 服务器启动成功！（这意味着问题可能已经修复）")
                print("=" * 80)
                process.terminate()
                process.wait(timeout=10)
                return True

        # 等待进程结束
        return_code = process.wait(timeout=300)

        if error_found or return_code != 0:
            print("\n" + "=" * 80)
            print("✗ 复现成功！检测到错误:")
            print("=" * 80)
            for line in error_lines:
                print(line, end='')
            print(f"\n返回码: {return_code}")
            return False

    except subprocess.TimeoutExpired:
        print("\n进程超时")
        process.kill()
        return False
    except KeyboardInterrupt:
        print("\n用户中断")
        process.terminate()
        process.wait(timeout=10)
        return False
    except Exception as e:
        print(f"\n✗ 启动失败: {e}")
        return False


def reproduce_with_debug_mode(model_path: str):
    """
    使用调试模式深入分析问题
    在权重加载时打印详细信息
    """
    print("\n" + "=" * 80)
    print("调试模式: 分析权重加载过程")
    print("=" * 80)

    # 创建一个临时的调试脚本
    debug_script = """
import os
os.environ['SGLANG_LOG_LEVEL'] = 'debug'

import sys
import torch
from transformers import AutoConfig
from sglang.srt.model_loader.loader import DefaultModelLoader
from sglang.srt.server_args import ServerArgs

# 模型配置
model_path = sys.argv[1]
pp_size = int(sys.argv[2])

print("=" * 80)
print(f"加载模型配置: {model_path}")
print("=" * 80)

# 加载配置
config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
print(f"\\n模型类型: {config.architectures}")
print(f"文本配置: {config.text_config if hasattr(config, 'text_config') else 'N/A'}")

if hasattr(config, 'text_config'):
    text_config = config.text_config
    num_layers = text_config.num_hidden_layers
    num_experts = getattr(text_config, 'num_experts', 0)

    print(f"\\n总层数: {num_layers}")
    print(f"专家数量: {num_experts}")

    # 计算 PP 层分配
    if pp_size == 2:
        # 模拟 get_pp_indices 的逻辑
        base_layers = num_layers // pp_size
        remainder = num_layers % pp_size

        print(f"\\nPP 层分配 (pp_size={pp_size}):")
        for rank in range(pp_size):
            if rank >= pp_size - remainder:
                partitions_without_extra_layer = pp_size - remainder
                start_layer = rank * (base_layers + 1) - partitions_without_extra_layer
                end_layer = start_layer + (base_layers + 1)
            else:
                start_layer = rank * base_layers
                end_layer = start_layer + base_layers

            print(f"  Rank {rank}: layers {start_layer} - {end_layer-1} (共 {end_layer - start_layer} 层)")

            # 检查第 15 层属于哪个 rank
            if start_layer <= 15 < end_layer:
                print(f"    ⚠ 第 15 层 (报错的层) 属于 Rank {rank}")

print("\\n" + "=" * 80)
print("尝试初始化模型...")
print("=" * 80)

# 设置分布式环境
os.environ['WORLD_SIZE'] = str(pp_size)
os.environ['RANK'] = '0'
os.environ['LOCAL_RANK'] = '0'
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '29500'

try:
    # 初始化模型加载器
    server_args = ServerArgs(
        model_path=model_path,
        pp_size=pp_size,
        tp_size=1,
        trust_remote_code=True,
    )

    print(f"\\n服务器参数: {server_args}")

except Exception as e:
    print(f"\\n✗ 错误: {e}")
    import traceback
    traceback.print_exc()
"""

    debug_file = "/tmp/debug_qwen3_vl_pp.py"
    with open(debug_file, "w") as f:
        f.write(debug_script)

    # 运行调试脚本
    try:
        result = subprocess.run(
            [sys.executable, debug_file, model_path, "2"],
            capture_output=True,
            text=True,
            timeout=120
        )
        print(result.stdout)
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("调试脚本超时")
        return False
    except Exception as e:
        print(f"调试失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="复现 SGLang Issue #17359: Qwen3-VL-235B PP-size 2 启动失败"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        help="模型路径 (默认: Qwen/Qwen3-VL-235B-A22B-Instruct-FP8)"
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=2,
        help="Pipeline parallelism size (默认: 2)"
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Tensor parallelism size (默认: 1)"
    )
    parser.add_argument(
        "--debug-only",
        action="store_true",
        help="仅运行调试模式，不启动完整服务器"
    )
    parser.add_argument(
        "--skip-env-check",
        action="store_true",
        help="跳过环境检查"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SGLang Issue #17359 复现脚本")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"配置: pp_size={args.pp_size}, tp_size={args.tp_size}")
    print("=" * 80)

    # 环境检查
    if not args.skip_env_check:
        if not check_environment():
            print("\n环境检查失败。使用 --skip-env-check 跳过检查。")
            return 1

    # 运行调试模式
    if args.debug_only:
        reproduce_with_debug_mode(args.model_path)
        return 0

    # 完整复现
    print("\n步骤 1: 运行调试分析...")
    reproduce_with_debug_mode(args.model_path)

    print("\n步骤 2: 尝试启动服务器以复现问题...")
    success = reproduce_with_minimal_config(args.model_path, args.pp_size, args.tp_size)

    if not success:
        print("\n" + "=" * 80)
        print("复现总结")
        print("=" * 80)
        print("✗ 成功复现 Issue #17359")
        print("问题: Qwen3-VL-235B 模型在 pp-size=2 时无法加载")
        print("错误: KeyError: 'model.layers.15.mlp.experts.w2_weight'")
        print("\n可能的原因:")
        print("1. MoE 模型的专家权重在 PP 分片时没有正确映射")
        print("2. 第 15 层的专家权重在 params_dict 中找不到")
        print("3. 权重加载逻辑没有正确处理跨 PP rank 的层")
        return 1
    else:
        print("\n" + "=" * 80)
        print("✓ 服务器启动成功，问题可能已修复")
        print("=" * 80)
        return 0


if __name__ == "__main__":
    sys.exit(main())
