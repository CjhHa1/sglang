#!/usr/bin/env python3
"""
最简单的 Issue #17359 分析脚本
无需任何外部依赖，只分析 PP 层分配和问题成因
"""

import argparse


class SimpleConfig:
    """简化的模型配置"""
    def __init__(self):
        # Qwen3-VL-235B 的简化配置
        self.num_hidden_layers = 32  # 实际是 160+，这里用 32 演示
        self.num_experts = 128
        self.hidden_size = 4096
        self.intermediate_size = 3072


def get_pp_indices(num_hidden_layers: int, pp_rank: int, pp_size: int):
    """
    计算 PP 层分配
    复制自 sglang/srt/distributed/utils.py:get_pp_indices
    """
    base_layers = num_hidden_layers // pp_size
    remainder = num_hidden_layers % pp_size

    if pp_rank >= pp_size - remainder:
        partitions_without_extra_layer = pp_size - remainder
        start_layer = pp_rank * (base_layers + 1) - partitions_without_extra_layer
        end_layer = start_layer + (base_layers + 1)
    else:
        start_layer = pp_rank * base_layers
        end_layer = start_layer + base_layers

    return start_layer, end_layer


def analyze_pp_distribution(num_layers: int, pp_size: int, target_layer: int = 15):
    """分析 PP 层分配"""
    print("=" * 80)
    print(f"Pipeline Parallelism 层分配分析")
    print("=" * 80)
    print(f"总层数: {num_layers}")
    print(f"PP Size: {pp_size}")
    print(f"目标层: {target_layer} (Issue #17359 报错的层)")
    print("=" * 80)

    base_layers = num_layers // pp_size
    remainder = num_layers % pp_size

    print(f"\n分配策略:")
    print(f"  基础层数/rank: {base_layers}")
    print(f"  余数: {remainder}")
    print(f"  最后 {remainder} 个 rank 各多分配 1 层\n")

    target_rank = None

    for rank in range(pp_size):
        start, end = get_pp_indices(num_layers, rank, pp_size)
        layer_count = end - start

        has_target = start <= target_layer < end
        marker = "⚠" if has_target else " "

        print(f"{marker} Rank {rank}: 层 {start:3d} - {end-1:3d}  ({layer_count:2d} 层)", end="")

        if has_target:
            print(f"  ← 第 {target_layer} 层在这里 (会触发 bug)")
            target_rank = rank
        else:
            print()

    return target_rank


def explain_bug(target_layer: int, target_rank: int, pp_size: int):
    """解释 bug 的成因"""
    print("\n" + "=" * 80)
    print("Bug 成因分析")
    print("=" * 80)

    print(f"""
问题描述:
  当使用 pp-size={pp_size} 启动 Qwen3-VL-235B 模型时，报错:
  KeyError: 'model.layers.{target_layer}.mlp.experts.w2_weight'

发生位置:
  - PP Rank {target_rank} 负责第 {target_layer} 层
  - 在加载该层的 MoE 专家权重时失败

权重加载流程:
  1. 模型初始化时，根据 PP 分配创建各层
     - Rank {target_rank}: 创建第 {target_layer} 层的实际参数
     - 其他 Rank: 使用 PPMissingLayer 占位

  2. load_weights() 遍历所有权重文件:
     for name, weight in weights:
         # name = "model.layers.{target_layer}.mlp.experts.w2_weight"
         # 尝试在 params_dict 中查找对应参数
         param = params_dict[name]  ← KeyError 在这里!

  3. 问题根源:
     - 融合专家权重(fused expert weights)使用特殊命名:
       * "experts.w13_weight" (gate_up_proj 融合)
       * "experts.w2_weight" (down_proj)

     - 但 params_dict 中的参数名可能是:
       * "experts.gate_up_proj"
       * "experts.down_proj"

     - 权重映射逻辑在 PP 模式下没有正确转换这些名称

代码位置:
  python/sglang/srt/models/qwen3_vl_moe.py:183-350
    - load_weights() 方法
    - 第 215-218 行: fused_expert_params_mapping
    - 第 267-303 行: 专家权重加载循环

修复思路:
  1. 在处理融合专家权重前，检查该层是否属于当前 PP rank
  2. 如果不属于，跳过权重加载
  3. 修改权重名称映射逻辑，正确处理 PP 模式
""")


def simulate_weight_loading(num_layers: int, pp_size: int, target_layer: int = 15):
    """模拟权重加载过程"""
    print("\n" + "=" * 80)
    print("模拟权重加载过程")
    print("=" * 80)

    # 找到包含 target_layer 的 rank
    target_rank = None
    for rank in range(pp_size):
        start, end = get_pp_indices(num_layers, rank, pp_size)
        if start <= target_layer < end:
            target_rank = rank
            break

    print(f"\nRank {target_rank} 的权重加载:")

    start, end = get_pp_indices(num_layers, target_rank, pp_size)

    print(f"  负责层: {start} - {end-1}")
    print(f"\n  params_dict 包含的参数:")

    # 模拟 params_dict 的内容
    for layer_id in range(start, end):
        if layer_id == target_layer:
            print(f"    ✓ model.layers.{layer_id}.self_attn.*")
            print(f"    ✓ model.layers.{layer_id}.mlp.gate.weight")
            print(f"    ✓ model.layers.{layer_id}.mlp.experts.gate_up_proj")
            print(f"    ✓ model.layers.{layer_id}.mlp.experts.down_proj")
            print(f"    ...")
        else:
            print(f"    ✓ model.layers.{layer_id}.*")

    print(f"\n  权重文件中的名称:")
    print(f"    • model.layers.{target_layer}.mlp.experts.w13_weight")
    print(f"    • model.layers.{target_layer}.mlp.experts.w2_weight  ← 这个!")
    print(f"    ...")

    print(f"\n  映射过程:")
    print(f"    1. 读取: 'model.layers.{target_layer}.mlp.experts.w2_weight'")
    print(f"    2. 应该映射到: 'model.layers.{target_layer}.mlp.experts.down_proj'")
    print(f"    3. 但是在 params_dict 查找 'model.layers.{target_layer}.mlp.experts.w2_weight'")
    print(f"    4. 找不到! → KeyError")

    print(f"\n  ✗ 问题: 权重名称没有正确映射就直接查找 params_dict")


def generate_test_commands():
    """生成测试命令"""
    print("\n" + "=" * 80)
    print("测试命令")
    print("=" * 80)

    print("""
# 1. 使用完整复现脚本（需要模型文件）
python reproduce_issue_17359.py --debug-only

# 2. 使用轻量级测试（创建 mock 模型，需要 torch）
python test_issue_17359_lightweight.py --pp-size 2

# 3. 仅分析层分配（当前脚本）
python test_issue_17359_simple.py --pp-size 2 --num-layers 32

# 4. 不同的 PP 配置
python test_issue_17359_simple.py --pp-size 4
python test_issue_17359_simple.py --pp-size 8
""")


def main():
    parser = argparse.ArgumentParser(
        description="简单分析 Issue #17359 - 无需任何依赖"
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=2,
        help="Pipeline parallelism size (默认: 2)"
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="模型层数 (默认: 32, 实际 Qwen3-VL-235B 有 160+ 层)"
    )
    parser.add_argument(
        "--target-layer",
        type=int,
        default=15,
        help="目标层 (Issue #17359 报错的层，默认: 15)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Issue #17359 简单分析工具")
    print("=" * 80)
    print(f"配置:")
    print(f"  - PP Size: {args.pp_size}")
    print(f"  - 总层数: {args.num_layers}")
    print(f"  - 目标层: {args.target_layer}")
    print("=" * 80)

    # 分析 PP 层分配
    target_rank = analyze_pp_distribution(args.num_layers, args.pp_size, args.target_layer)

    if target_rank is not None:
        # 解释 bug
        explain_bug(args.target_layer, target_rank, args.pp_size)

        # 模拟权重加载
        simulate_weight_loading(args.num_layers, args.pp_size, args.target_layer)

    # 生成测试命令
    generate_test_commands()

    print("\n" + "=" * 80)
    print("分析完成!")
    print("=" * 80)


if __name__ == "__main__":
    main()
