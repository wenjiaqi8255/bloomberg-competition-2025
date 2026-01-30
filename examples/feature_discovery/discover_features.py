#!/usr/bin/env python3
"""
特征发现工具 - 扫描并显示系统中所有可用的特征

这个脚本会扫描特征工程代码，提取所有具体特征名称，
并按类型分组显示，让用户知道有哪些特征可以选择。
"""

import sys
import os
import re
from pathlib import Path
from typing import Dict, List, Set
import pandas as pd

def extract_feature_names_from_file(file_path: Path) -> Set[str]:
    """从文件中提取所有特征名称"""
    features = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找 features['feature_name'] 模式
        pattern = r"features\['([^\']+)'\]"
        matches = re.findall(pattern, content)
        features.update(matches)

        # 查找 return pd.DataFrame(columns=[...]) 模式
        pattern = r"return\s+pd\.DataFrame\(columns=\[([^\]]+)\]\)"
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            # 清理并分割列名
            columns = re.findall(r"'([^']+)'", match)
            features.update(columns)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return features

def extract_cross_sectional_features(file_path: Path) -> Set[str]:
    """从截面特征文件中提取特征名"""
    features = set()

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 查找特征计算返回的列名
        if 'cross_sectional_features' in str(file_path):
            # 从文档字符串和返回值中提取
            doc_pattern = r"Returns:\s*\n\s*DataFrame with columns:\s*\n.*?- ([^\n]+)"
            matches = re.findall(doc_pattern, content, re.DOTALL)
            for match in matches:
                # 提取特征名
                feature_match = re.match(r"-\s*([a-zA-Z_][a-zA-Z0-9_]*)", match.strip())
                if feature_match:
                    features.add(feature_match[0])

            # 查找配置中的特征名称映射
            config_mapping = {
                'market_cap': 'market_cap_proxy',
                'book_to_market': 'book_to_market_proxy',
                'size': 'size_factor',
                'value': 'value_factor',
                'momentum': 'momentum_12m',
                'volatility': 'volatility_60d'
            }

            # 查找代码中的具体特征赋值
            feature_patterns = [
                r"symbol_features\['([a-zA-Z_][a-zA-Z0-9_]*)'\]\s*=",  # symbol_features['feature'] =
                r"'([a-zA-Z_][a-zA-Z0-9_]*)'\s*:\s*.*?\._calculate_",  # 'feature': self._calculate_
            ]

            for pattern in feature_patterns:
                matches = re.findall(pattern, content)
                features.update(matches)

        # 查找 DataFrame 返回值
        pattern = r"return\s+pd\.DataFrame\(\s*\{([^}]+)\}"
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            # 查找字典中的键
            keys = re.findall(r"'([^']+)':", match)
            features.update(keys)

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

    return features

def scan_all_features() -> Dict[str, Set[str]]:
    """扫描所有特征工程文件，提取特征名"""
    base_path = Path("src/trading_system/feature_engineering")

    feature_categories = {
        "technical_features": set(),
        "cross_sectional_features": set(),
        "momentum_features": set(),
        "volatility_features": set(),
        "other_features": set()
    }

    # 扫描所有相关文件
    feature_files = [
        "utils/technical_features.py",
        "utils/cross_sectional_features.py",
        "utils/validation.py",
        "pipeline.py"
    ]

    for file_path in feature_files:
        full_path = base_path / file_path
        if full_path.exists():
            print(f"Scanning {full_path}...")

            # 提取特征名
            features = extract_feature_names_from_file(full_path)
            cross_features = extract_cross_sectional_features(full_path)
            features.update(cross_features)

            # 按类型分类
            for feature in features:
                if any(x in feature.lower() for x in ['momentum', 'trend', 'exp_momentum']):
                    feature_categories["momentum_features"].add(feature)
                elif any(x in feature.lower() for x in ['volatility', 'vol', 'parkinson', 'garman_klass', 'range_vol']):
                    feature_categories["volatility_features"].add(feature)
                elif any(x in feature.lower() for x in ['rsi', 'macd', 'bollinger', 'stochastic', 'williams', 'mfi', 'adx', 'cci']):
                    feature_categories["technical_features"].add(feature)
                elif any(x in feature.lower() for x in ['market_cap', 'book_to_market', 'size_factor', 'value_factor', 'cross_section']):
                    feature_categories["cross_sectional_features"].add(feature)
                else:
                    feature_categories["other_features"].add(feature)

    return feature_categories

def generate_feature_report(feature_categories: Dict[str, Set[str]]) -> str:
    """生成特征报告"""
    report = []

    report.append("# 系统可用特征清单")
    report.append("=" * 50)
    report.append("")

    total_features = sum(len(features) for features in feature_categories.values())
    report.append(f"总计发现 {total_features} 个特征")
    report.append("")

    for category, features in feature_categories.items():
        if features:
            category_name = category.replace("_features", "").title()
            report.append(f"## {category_name} 特征 ({len(features)} 个)")
            report.append("")

            # 按字母顺序排序
            sorted_features = sorted(features)
            for feature in sorted_features:
                report.append(f"- `{feature}`")
            report.append("")

    # 添加配置示例
    report.append("## 配置示例")
    report.append("")
    report.append("```yaml")
    report.append("# 选择具体特征")
    report.append("feature_engineering:")
    report.append("  specific_features:")
    report.append("    momentum:")
    report.append("      - momentum_21d")
    report.append("      - momentum_63d")
    report.append("      - exp_momentum_21d")
    report.append("    volatility:")
    report.append("      - volatility_20d")
    report.append("      - garman_klass_volatility")
    report.append("    technical:")
    report.append("      - rsi_14")
    report.append("      - macd_line")
    report.append("      - bollinger_position")
    report.append("    cross_sectional:")
    report.append("      - market_cap_proxy")
    report.append("      - book_to_market_proxy")
    report.append("      - size_factor")
    report.append("```")
    report.append("")

    return "\n".join(report)

def main():
    """主函数"""
    print("🔍 正在扫描系统中的特征...")
    print("")

    # 扫描所有特征
    feature_categories = scan_all_features()

    # 生成报告
    report = generate_feature_report(feature_categories)

    # 保存到文件
    with open("FEATURES.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("✅ 特征扫描完成！")
    print(f"📄 报告已保存到: FEATURES.md")
    print("")

    # 显示摘要
    for category, features in feature_categories.items():
        if features:
            category_name = category.replace("_features", "").title()
            print(f"📊 {category_name}: {len(features)} 个特征")

    print("")
    print("💡 提示: 查看 FEATURES.md 文件获取完整的特征列表和配置示例")

if __name__ == "__main__":
    main()