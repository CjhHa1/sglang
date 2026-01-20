#!/usr/bin/env python3
"""
分析 Qwen3-VL-235B 模型权重结构
帮助理解 Issue #17359 中的权重加载问题
"""

import argparse
import sys
from pathlib import Path


def analyze_model_config(model_path: str):
    """分析模型配置"""
    print("=" * 80)
    print("分析模型配置")
    print("=" * 80)

    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        print(f"\n模型架构: {config.architectures}")
        print(f"\n完整配置:")
        print(f"  - 模型类型: {config.model_type}")

        # VL 模型通常有 text_config 和 vision_config
        if hasattr(config, "text_config"):
            text_config = config.text_config
            print(f"\n文本模型配置:")
            print(f"  - 隐藏层数: {text_config.num_hidden_layers}")
            print(f"  - 隐藏大小: {text_config.hidden_size}")
            print(f"  - 注意力头数: {text_config.num_attention_heads}")
            print(f"  - 中间层大小: {text_config.intermediate_size}")

            # MoE 相关配置
            if hasattr(text_config, "num_experts"):
                print(f"\n混合专家(MoE)配置:")
                print(f"  - 专家数量: {text_config.num_experts}")
                print(f"  - 激活专家数: {getattr(text_config, 'num_experts_per_tok', 'N/A')}")
                print(f"  - 共享专家数: {getattr(text_config, 'num_shared_expert', 'N/A')}")

            return text_config.num_hidden_layers, getattr(text_config, "num_experts", 0)
        else:
            print(f"\n基础配置:")
            print(f"  - 隐藏层数: {config.num_hidden_layers}")
            print(f"  - 隐藏大小: {config.hidden_size}")
            return config.num_hidden_layers, 0

    except Exception as e:
        print(f"错误: 无法加载配置: {e}")
        import traceback

        traceback.print_exc()
        return None, None


def simulate_pp_layer_distribution(num_layers: int, pp_size: int):
    """
    模拟 Pipeline Parallelism 的层分配
    基于 sglang/srt/distributed/utils.py:get_pp_indices
    """
    print("\n" + "=" * 80)
    print(f"Pipeline Parallelism 层分配 (pp_size={pp_size})")
    print("=" * 80)

    base_layers = num_layers // pp_size
    remainder = num_layers % pp_size

    print(f"\n总层数: {num_layers}")
    print(f"PP Size: {pp_size}")
    print(f"基础层数/rank: {base_layers}")
    print(f"余数: {remainder}")
    print()

    layer_distribution = []
    for rank in range(pp_size):
        # 计算该 rank 的层范围
        if rank >= pp_size - remainder:
            # 最后 remainder 个 rank 各多分配一层
            partitions_without_extra_layer = pp_size - remainder
            start_layer = rank * (base_layers + 1) - partitions_without_extra_layer
            end_layer = start_layer + (base_layers + 1)
        else:
            # 前面的 rank 只分配基础层数
            start_layer = rank * base_layers
            end_layer = start_layer + base_layers

        layer_count = end_layer - start_layer
        layer_distribution.append((rank, start_layer, end_layer))

        print(f"Rank {rank}:")
        print(f"  层范围: {start_layer} - {end_layer - 1}")
        print(f"  层数量: {layer_count}")

        # 标注特殊层
        if start_layer <= 15 < end_layer:
            print(f"  ⚠ 包含第 15 层 (Issue #17359 报错的层)")
        print()

    return layer_distribution


def analyze_weight_names(model_path: str, sample_size: int = 50):
    """分析模型权重名称"""
    print("=" * 80)
    print("分析模型权重结构")
    print("=" * 80)

    try:
        from transformers import AutoModelForCausalLM
        from safetensors import safe_open
        import os

        # 尝试找到模型文件
        model_dir = Path(model_path)
        if not model_dir.exists():
            # 尝试从 HuggingFace cache 中找
            from huggingface_hub import snapshot_download

            print(f"下载模型文件索引...")
            cache_dir = snapshot_download(
                model_path, allow_patterns=["*.safetensors.index.json", "*.json"]
            )
            model_dir = Path(cache_dir)

        # 查找权重文件
        safetensors_files = list(model_dir.glob("*.safetensors"))
        index_file = model_dir / "model.safetensors.index.json"

        if index_file.exists():
            import json

            with open(index_file) as f:
                index = json.load(f)

            print(f"\n找到权重索引文件")
            print(f"权重文件数量: {len(index.get('weight_map', {}).values())}")

            # 分析权重名称
            weight_map = index.get("weight_map", {})
            all_weights = list(weight_map.keys())

            print(f"总权重数量: {len(all_weights)}")

            # 查找与 issue 相关的权重
            print(f"\n搜索 MoE 专家权重...")
            expert_weights = [w for w in all_weights if "expert" in w.lower()]
            print(f"专家权重数量: {len(expert_weights)}")

            # 查找第 15 层的权重
            layer_15_weights = [w for w in all_weights if "layers.15." in w]
            print(f"\n第 15 层权重:")
            for w in layer_15_weights[:20]:
                print(f"  - {w}")
            if len(layer_15_weights) > 20:
                print(f"  ... (共 {len(layer_15_weights)} 个)")

            # 查找 w2_weight
            w2_weights = [w for w in all_weights if "w2_weight" in w]
            print(f"\n包含 'w2_weight' 的权重:")
            for w in w2_weights[:20]:
                print(f"  - {w}")
            if len(w2_weights) > 20:
                print(f"  ... (共 {len(w2_weights)} 个)")

            # 分析第 15 层的专家权重
            layer_15_expert = [w for w in all_weights if "layers.15.mlp.experts" in w]
            if layer_15_expert:
                print(f"\n第 15 层的 MoE 专家权重:")
                for w in layer_15_expert:
                    print(f"  - {w}")
            else:
                print(f"\n⚠ 未找到第 15 层的 MoE 专家权重")

        elif safetensors_files:
            print(f"\n找到 {len(safetensors_files)} 个 safetensors 文件")
            # 读取第一个文件的元数据
            first_file = safetensors_files[0]
            print(f"分析文件: {first_file.name}")

            with safe_open(first_file, framework="pt") as f:
                keys = f.keys()
                print(f"权重数量: {len(keys)}")
                print(f"\n前 {sample_size} 个权重:")
                for i, key in enumerate(list(keys)[:sample_size]):
                    print(f"  {i + 1}. {key}")

        else:
            print("未找到权重文件")

    except ImportError as e:
        print(f"缺少依赖: {e}")
        print("请安装: pip install transformers safetensors huggingface_hub")
    except Exception as e:
        print(f"错误: {e}")
        import traceback

        traceback.print_exc()


def analyze_pp_weight_loading(num_layers: int, pp_size: int):
    """
    分析在 PP 模式下权重加载的逻辑
    """
    print("\n" + "=" * 80)
    print("Pipeline Parallelism 权重加载分析")
    print("=" * 80)

    layer_dist = simulate_pp_layer_distribution(num_layers, pp_size)

    print("\n权重加载策略:")
    for rank, start, end in layer_dist:
        print(f"\nRank {rank}:")
        print(f"  应该加载层 {start}-{end - 1} 的权重")
        print(f"  其他层使用 PPMissingLayer 占位")

        if start <= 15 < end:
            print(f"  ⚠ Rank {rank} 需要加载第 15 层的权重")
            print(f"     包括: model.layers.15.mlp.experts.w2_weight")


def generate_fix_suggestions():
    """生成修复建议"""
    print("\n" + "=" * 80)
    print("修复建议")
    print("=" * 80)

    print("""
问题分析:
1. 在 Pipeline Parallelism 模式下，模型层被分配到不同的 rank
2. 每个 rank 只应该加载属于自己的层的权重
3. MoE 模型的专家权重可能在权重加载时没有正确处理 PP 分片

可能的问题点:
1. qwen3_vl_moe.py:load_weights() 方法中的专家权重映射逻辑
2. PP 模式下，params_dict 可能不包含其他 rank 的层参数
3. 权重加载时没有正确检查层是否属于当前 rank

建议的修复方案:
1. 在加载专家权重前，检查该层是否属于当前 PP rank
2. 修改 load_weights 中的专家权重加载逻辑，跳过不属于当前 rank 的层
3. 参考其他 MoE 模型(如 DeepSeek-V2)的 PP 实现

相关代码位置:
- python/sglang/srt/models/qwen3_vl_moe.py:183-350
- python/sglang/srt/distributed/utils.py:63-99
- python/sglang/srt/utils/common.py:588-629
""")


def main():
    parser = argparse.ArgumentParser(
        description="分析 Qwen3-VL-235B 模型权重结构 (Issue #17359)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        help="模型路径",
    )
    parser.add_argument(
        "--pp-size", type=int, default=2, help="Pipeline parallelism size"
    )
    parser.add_argument(
        "--analyze-weights",
        action="store_true",
        help="分析实际权重文件(需要下载模型)",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("Qwen3-VL 模型权重分析工具 (Issue #17359)")
    print("=" * 80)
    print(f"模型: {args.model_path}")
    print(f"PP Size: {args.pp_size}")
    print("=" * 80)

    # 分析配置
    num_layers, num_experts = analyze_model_config(args.model_path)

    if num_layers:
        # 模拟 PP 层分配
        simulate_pp_layer_distribution(num_layers, args.pp_size)

        # 分析权重加载
        analyze_pp_weight_loading(num_layers, args.pp_size)

    # 分析实际权重
    if args.analyze_weights:
        analyze_weight_names(args.model_path)

    # 生成修复建议
    generate_fix_suggestions()


if __name__ == "__main__":
    main()
