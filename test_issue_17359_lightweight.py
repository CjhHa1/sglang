#!/usr/bin/env python3
"""
轻量级复现 Issue #17359 - 无需下载完整模型权重
使用 meta device 或 mock 权重来测试 PP 模式下的权重加载逻辑
"""

import os
import sys
import torch
import argparse
from typing import Dict, Iterator, Tuple
from pathlib import Path


def setup_distributed_env(pp_size: int, tp_size: int, rank: int = 0):
    """设置分布式环境变量"""
    os.environ['WORLD_SIZE'] = str(pp_size * tp_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # PP 相关
    if pp_size > 1:
        os.environ['SGLANG_PP_SIZE'] = str(pp_size)


def create_mock_config(model_type: str = "qwen3_vl_moe"):
    """创建模拟的模型配置"""
    print("=" * 80)
    print("创建模拟模型配置")
    print("=" * 80)

    from transformers import PretrainedConfig

    if model_type == "qwen3_vl_moe":
        # 模拟 Qwen3-VL-235B 的配置（简化版）
        # 使用较小的层数来加快测试
        class MockTextConfig(PretrainedConfig):
            model_type = "qwen3_moe"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.hidden_size = 4096
                self.intermediate_size = 3072
                self.num_hidden_layers = 32  # 简化版，原版是 160+
                self.num_attention_heads = 32
                self.num_key_value_heads = 8
                self.num_experts = 128  # MoE 专家数
                self.num_experts_per_tok = 8
                self.num_shared_expert = 2
                self.rope_theta = 10000.0
                self.rms_norm_eps = 1e-6
                self.vocab_size = 152064
                self.max_position_embeddings = 32768

        class MockVisionConfig(PretrainedConfig):
            model_type = "qwen3_vl_vision"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.hidden_size = 1536
                self.num_hidden_layers = 24

        class MockQwen3VLConfig(PretrainedConfig):
            model_type = "qwen3_vl_moe"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.text_config = MockTextConfig()
                self.vision_config = MockVisionConfig()
                self.architectures = ["Qwen3VLMoeForConditionalGeneration"]

        config = MockQwen3VLConfig()

        print(f"✓ 模拟配置创建成功")
        print(f"  - 文本层数: {config.text_config.num_hidden_layers}")
        print(f"  - 专家数: {config.text_config.num_experts}")
        print(f"  - 每token激活专家数: {config.text_config.num_experts_per_tok}")

        return config

    raise ValueError(f"Unsupported model type: {model_type}")


def generate_mock_weights(config, include_layer: int = 15) -> Iterator[Tuple[str, torch.Tensor]]:
    """
    生成模拟的权重数据
    重点包含会触发 bug 的第 15 层 MoE 专家权重
    """
    print("\n" + "=" * 80)
    print("生成模拟权重数据")
    print("=" * 80)

    text_config = config.text_config
    num_layers = text_config.num_hidden_layers
    num_experts = text_config.num_experts
    hidden_size = text_config.hidden_size
    intermediate_size = text_config.intermediate_size

    weights_generated = []

    # 生成 embedding 权重
    weight_name = "model.embed_tokens.weight"
    weights_generated.append(weight_name)
    yield (weight_name, torch.randn(text_config.vocab_size, hidden_size, dtype=torch.float16))

    # 只生成关键层的权重（包括第 15 层）
    # 为了加快测试，只生成部分层
    layers_to_generate = [0, 1, 15, 16, num_layers - 1] if include_layer == 15 else [0, 1, num_layers - 1]

    for layer_idx in layers_to_generate:
        if layer_idx >= num_layers:
            continue

        prefix = f"model.layers.{layer_idx}"

        # Attention 权重
        weights_generated.append(f"{prefix}.self_attn.q_proj.weight")
        yield (f"{prefix}.self_attn.q_proj.weight", torch.randn(hidden_size, hidden_size, dtype=torch.float16))

        weights_generated.append(f"{prefix}.self_attn.k_proj.weight")
        yield (f"{prefix}.self_attn.k_proj.weight", torch.randn(hidden_size, hidden_size, dtype=torch.float16))

        weights_generated.append(f"{prefix}.self_attn.v_proj.weight")
        yield (f"{prefix}.self_attn.v_proj.weight", torch.randn(hidden_size, hidden_size, dtype=torch.float16))

        weights_generated.append(f"{prefix}.self_attn.o_proj.weight")
        yield (f"{prefix}.self_attn.o_proj.weight", torch.randn(hidden_size, hidden_size, dtype=torch.float16))

        # LayerNorm
        weights_generated.append(f"{prefix}.input_layernorm.weight")
        yield (f"{prefix}.input_layernorm.weight", torch.randn(hidden_size, dtype=torch.float16))

        weights_generated.append(f"{prefix}.post_attention_layernorm.weight")
        yield (f"{prefix}.post_attention_layernorm.weight", torch.randn(hidden_size, dtype=torch.float16))

        # MoE 专家权重 - 这是关键部分！
        # 测试融合专家权重格式 (fused expert weights)
        # 这种格式会触发 Issue #17359
        expert_intermediate = intermediate_size // 2  # 简化

        # 融合的 gate_up_proj (w13_weight)
        fused_weight_name = f"{prefix}.mlp.experts.w13_weight"
        weights_generated.append(fused_weight_name)
        # shape: [num_experts, 2 * intermediate_size, hidden_size]
        fused_w13 = torch.randn(num_experts, 2 * expert_intermediate, hidden_size, dtype=torch.float16)
        yield (fused_weight_name, fused_w13)

        # 融合的 down_proj (w2_weight) - 这个会触发 KeyError!
        fused_weight_name = f"{prefix}.mlp.experts.w2_weight"
        weights_generated.append(fused_weight_name)
        # shape: [num_experts, hidden_size, intermediate_size]
        fused_w2 = torch.randn(num_experts, hidden_size, expert_intermediate, dtype=torch.float16)
        yield (fused_weight_name, fused_w2)

        # Gate 权重
        weights_generated.append(f"{prefix}.mlp.gate.weight")
        yield (f"{prefix}.mlp.gate.weight", torch.randn(num_experts, hidden_size, dtype=torch.float16))

    # 最后的 norm
    yield ("model.norm.weight", torch.randn(hidden_size, dtype=torch.float16))
    weights_generated.append("model.norm.weight")

    # LM head
    yield ("lm_head.weight", torch.randn(text_config.vocab_size, hidden_size, dtype=torch.float16))
    weights_generated.append("lm_head.weight")

    print(f"✓ 生成了 {len(weights_generated)} 个权重张量")
    print(f"  - 包含关键层: {layers_to_generate}")
    print(f"  - 每层专家数: {num_experts}")

    if include_layer == 15:
        print(f"\n⚠ 特别注意: 已生成第 15 层的融合专家权重")
        print(f"  - model.layers.15.mlp.experts.w13_weight")
        print(f"  - model.layers.15.mlp.experts.w2_weight ← 这个会触发 KeyError!")


def simulate_pp_layer_distribution(num_layers: int, pp_size: int, pp_rank: int):
    """模拟 PP 层分配"""
    # 复制 sglang/srt/distributed/utils.py:get_pp_indices 的逻辑
    base_layers = num_layers // pp_size
    remainder = num_layers % pp_size

    if pp_rank >= pp_size - remainder:
        partitions_without_extra_layer = pp_size - remainder
        start_layer = pp_rank * (base_layers + 1) - partitions_without_extra_layer
        end_layer = start_layer + (base_layers + 1)
    else:
        start_layer = pp_rank * base_layers
        end_layer = start_layer + base_layers

    return start_layer, end_layer


def test_weight_loading_with_pp(pp_size: int = 2, pp_rank: int = 0):
    """
    测试 PP 模式下的权重加载
    """
    print("\n" + "=" * 80)
    print(f"测试 PP 模式下的权重加载 (pp_size={pp_size}, rank={pp_rank})")
    print("=" * 80)

    # 设置环境
    setup_distributed_env(pp_size, tp_size=1, rank=pp_rank)

    # 创建配置
    config = create_mock_config("qwen3_vl_moe")
    num_layers = config.text_config.num_hidden_layers

    # 计算层分配
    start_layer, end_layer = simulate_pp_layer_distribution(num_layers, pp_size, pp_rank)

    print(f"\nPP Rank {pp_rank}:")
    print(f"  - 负责层: {start_layer} - {end_layer - 1}")
    print(f"  - 层数量: {end_layer - start_layer}")

    # 检查第 15 层是否在当前 rank
    if start_layer <= 15 < end_layer:
        print(f"  ⚠ 第 15 层在当前 rank 的范围内")
        layer_15_in_rank = True
    else:
        print(f"  ✓ 第 15 层不在当前 rank 的范围内")
        layer_15_in_rank = False

    # 尝试加载权重
    print("\n" + "=" * 80)
    print("模拟权重加载过程")
    print("=" * 80)

    try:
        # 导入必要的模块
        from sglang.srt.models.qwen3_vl_moe import Qwen3VLMoeForConditionalGeneration

        print("✓ 成功导入 Qwen3VLMoeForConditionalGeneration")

        # 尝试创建模型（使用 meta device 避免实际分配内存）
        print("\n尝试创建模型...")

        with torch.device('meta'):
            model = Qwen3VLMoeForConditionalGeneration(
                config=config,
                quant_config=None,
            )

        print("✓ 模型结构创建成功")

        # 检查模型是否有 start_layer 和 end_layer
        if hasattr(model.model, 'start_layer'):
            print(f"\n模型 PP 层分配:")
            print(f"  - start_layer: {model.model.start_layer}")
            print(f"  - end_layer: {model.model.end_layer}")

            if model.model.start_layer != start_layer or model.model.end_layer != end_layer:
                print(f"  ⚠ 警告: 层分配不匹配!")

        # 生成模拟权重
        mock_weights = list(generate_mock_weights(config, include_layer=15))

        print(f"\n开始加载权重...")
        print(f"  - 权重总数: {len(mock_weights)}")

        # 尝试加载权重
        errors_found = []
        loaded_count = 0

        try:
            model.load_weights(mock_weights)
            loaded_count = len(mock_weights)
            print(f"✓ 权重加载成功! (已加载 {loaded_count} 个)")

        except KeyError as e:
            error_msg = str(e)
            print(f"\n✗ 捕获到 KeyError: {error_msg}")
            errors_found.append(("KeyError", error_msg))

            if "w2_weight" in error_msg:
                print(f"\n{'='*80}")
                print(f"🎯 成功复现 Issue #17359!")
                print(f"{'='*80}")
                print(f"错误: KeyError: {error_msg}")
                print(f"\n问题分析:")
                print(f"  1. 第 15 层位于 Rank {pp_rank} (层 {start_layer}-{end_layer-1})")
                print(f"  2. 权重名称: model.layers.15.mlp.experts.w2_weight")
                print(f"  3. 在 PP 模式下，该权重无法被正确加载到 params_dict")
                print(f"\n根本原因:")
                print(f"  - 融合的专家权重(fused expert weights)在 PP 分片时")
                print(f"  - 没有正确映射到对应 rank 的 params_dict")
                print(f"  - 导致在 load_weights 中找不到对应的参数")
                return False

        except Exception as e:
            error_msg = str(e)
            print(f"\n✗ 捕获到其他错误: {type(e).__name__}: {error_msg}")
            errors_found.append((type(e).__name__, error_msg))
            import traceback
            traceback.print_exc()
            return False

        if not errors_found:
            print(f"\n✓ 未复现错误 - 可能问题已修复或配置不同")
            return True

    except ImportError as e:
        print(f"\n✗ 无法导入模块: {e}")
        print(f"这可能是因为 SGLang 环境未正确设置")
        return None

    except Exception as e:
        print(f"\n✗ 测试过程中出错: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_simple_analysis():
    """
    运行简单的分析，不实际创建模型
    """
    print("=" * 80)
    print("简单分析模式 - 不创建实际模型")
    print("=" * 80)

    # 创建配置
    config = create_mock_config("qwen3_vl_moe")
    num_layers = config.text_config.num_hidden_layers

    # 模拟不同的 PP 配置
    for pp_size in [2, 4, 8]:
        print(f"\n{'='*80}")
        print(f"PP Size = {pp_size}")
        print(f"{'='*80}")

        for rank in range(pp_size):
            start, end = simulate_pp_layer_distribution(num_layers, pp_size, rank)

            has_layer_15 = start <= 15 < end
            marker = "⚠" if has_layer_15 else " "

            print(f"{marker} Rank {rank}: 层 {start:2d}-{end-1:2d} ({end-start:2d} 层)", end="")
            if has_layer_15:
                print(f"  ← 包含第 15 层 (会触发 bug)")
            else:
                print()


def main():
    parser = argparse.ArgumentParser(
        description="轻量级测试 Issue #17359 - 无需下载完整模型"
    )
    parser.add_argument(
        "--pp-size",
        type=int,
        default=2,
        help="Pipeline parallelism size (默认: 2)"
    )
    parser.add_argument(
        "--pp-rank",
        type=int,
        default=0,
        help="PP rank to simulate (默认: 0)"
    )
    parser.add_argument(
        "--simple-analysis",
        action="store_true",
        help="仅运行简单分析，不创建模型"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("轻量级 Issue #17359 复现测试")
    print("=" * 80)
    print(f"配置: pp_size={args.pp_size}, pp_rank={args.pp_rank}")
    print("=" * 80)

    if args.simple_analysis:
        run_simple_analysis()
        return 0

    # 运行完整测试
    result = test_weight_loading_with_pp(args.pp_size, args.pp_rank)

    if result is False:
        print("\n" + "=" * 80)
        print("测试结果: 成功复现 Issue #17359 ✗")
        print("=" * 80)
        return 1
    elif result is True:
        print("\n" + "=" * 80)
        print("测试结果: 未复现问题 ✓")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("测试结果: 无法完成测试")
        print("=" * 80)
        return 2


if __name__ == "__main__":
    sys.exit(main())
