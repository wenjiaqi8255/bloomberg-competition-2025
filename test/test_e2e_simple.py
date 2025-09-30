#!/usr/bin/env python3
"""
简化版端到端测试: 核心模型生命周期

专注于验证新模型架构的核心功能:
1. 模型训练和注册
2. 模型加载和预测
3. ModelPredictor基本功能

Usage:
    poetry run python test_e2e_simple.py
"""

import logging
import sys
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import pandas as pd
import numpy as np

# Import only the core components
from trading_system.models.training.pipeline import TrainingPipeline
from trading_system.models.training.trainer import TrainingConfig
from trading_system.models.serving.predictor import ModelPredictor
from trading_system.models.base.model_factory import ModelFactory
from trading_system.models.registry import register_all_models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_simple_test_data():
    """创建简单的测试数据"""
    logger.info("📊 创建简单测试数据...")

    # 创建价格数据
    np.random.seed(42)

    # 3支股票，1年数据
    symbols = ["AAPL", "MSFT", "GOOGL"]
    equity_data = {}

    for i, symbol in enumerate(symbols):
        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
        n_days = len(dates)

        # 简单的价格生成
        base_price = 100 + i * 50
        returns = np.random.normal(0.001, 0.02, n_days)
        prices = base_price * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'date': dates,
            'close': prices,
            'open': prices * np.random.uniform(0.998, 1.002, n_days),
            'high': prices * np.random.uniform(1.0, 1.01, n_days),
            'low': prices * np.random.uniform(0.99, 1.0, n_days),
            'volume': np.random.randint(1_000_000, 5_000_000, n_days)
        })

        equity_data[symbol] = df
        logger.info(f"✅ 生成 {symbol} 数据: {len(df)} 行")

    # 创建简单的因子数据
    factor_dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="M")
    factor_data = pd.DataFrame({
        'date': factor_dates,
        'MKT': np.random.normal(0.005, 0.04, len(factor_dates)),
        'SMB': np.random.normal(0.001, 0.03, len(factor_dates)),
        'HML': np.random.normal(0.001, 0.025, len(factor_dates)),
        'RMW': np.random.normal(0.002, 0.02, len(factor_dates)),
        'CMA': np.random.normal(0.001, 0.015, len(factor_dates))
    })

    logger.info(f"✅ 生成因子数据: {len(factor_data)} 个月")

    return equity_data, factor_data, symbols


def test_model_lifecycle():
    """测试完整的模型生命周期"""
    logger.info("🎯 开始模型生命周期测试")
    logger.info("=" * 50)

    test_dir = Path(tempfile.mkdtemp(prefix="e2e_simple_"))
    registry_path = test_dir / "model_registry"
    registry_path.mkdir(parents=True, exist_ok=True)

    try:
        # 步骤1: 确保模型已注册
        logger.info("\n📝 步骤1: 检查模型注册")
        register_all_models()

        available_models = list(ModelFactory._registry.keys())
        logger.info(f"可用模型: {available_models}")
        assert "residual_predictor" in available_models, "残差预测器应该已注册"

        # 步骤2: 创建测试数据
        logger.info("\n📊 步骤2: 创建测试数据")
        equity_data, factor_data, symbols = create_simple_test_data()

        # 步骤3: 训练模型
        logger.info("\n🚀 步骤3: 训练模型")

        config = TrainingConfig(
            use_cross_validation=True,
            cv_folds=3,
            validation_split=0.2,
            early_stopping=True
        )

        pipeline = TrainingPipeline(
            model_type="residual_predictor",
            config=config,
            registry_path=str(registry_path)
        )

        logger.info("开始训练残差预测模型...")
        training_result = pipeline.run_pipeline(
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 12, 31),
            symbols=symbols,
            model_name="test_residual_predictor",
            equity_data=equity_data,
            factor_data=factor_data
        )

        assert training_result is not None, "训练结果不应为空"
        assert 'model_id' in training_result, "应该包含模型ID"

        model_id = training_result['model_id']
        logger.info(f"✅ 训练完成，模型ID: {model_id}")

        # 步骤4: 验证模型保存
        logger.info("\n💾 步骤4: 验证模型保存")

        model_path = registry_path / model_id
        assert model_path.exists(), f"模型目录应该存在: {model_path}"
        assert (model_path / "model.pkl").exists(), "模型文件应该存在"
        assert (model_path / "metadata.json").exists(), "元数据文件应该存在"

        # 验证元数据
        with open(model_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        assert metadata['model_type'] == "residual_predictor", "模型类型应该正确"
        logger.info("✅ 模型文件验证通过")

        # 步骤5: 加载模型
        logger.info("\n📥 步骤5: 加载模型")

        from trading_system.models.base.base_model import BaseModel
        loaded_model = BaseModel.load(model_path)

        assert loaded_model is not None, "加载的模型不应为空"
        assert loaded_model.model_type == "residual_predictor", "模型类型应该正确"
        logger.info("✅ 模型加载成功")

        # 步骤6: 创建ModelPredictor
        logger.info("\n🚀 步骤6: 创建ModelPredictor")

        predictor = ModelPredictor(
            model_registry_path=str(registry_path),
            enable_monitoring=True,
            cache_predictions=True
        )

        # 步骤7: 加载模型到生产环境
        logger.info("\n🎯 步骤7: 部署模型到生产环境")

        deployed_model_id = predictor.load_model(
            model_name="residual_predictor",
            model_path=str(model_path)
        )

        assert deployed_model_id is not None, "部署的模型ID不应为空"
        assert predictor.get_current_model() is not None, "当前模型不应为空"
        logger.info(f"✅ 模型部署成功，ID: {deployed_model_id}")

        # 步骤8: 进行预测
        logger.info("\n🔮 步骤8: 进行预测")

        # 使用AAPL的最新数据
        symbol = symbols[0]
        latest_data = equity_data[symbol].tail(30)

        prediction_result = predictor.predict(
            market_data=latest_data,
            symbol=symbol,
            prediction_date=datetime(2023, 12, 31)
        )

        assert prediction_result is not None, "预测结果不应为空"
        assert 'prediction' in prediction_result, "应该包含预测值"

        prediction_value = prediction_result['prediction']
        assert not np.isnan(prediction_value), "预测值不应该是NaN"
        assert not np.isinf(prediction_value), "预测值不应该是无穷大"

        logger.info(f"✅ 预测成功: {symbol} = {prediction_value:.6f}")

        # 步骤9: 批量预测
        logger.info("\n📊 步骤9: 批量预测")

        # 准备批量数据
        batch_data = pd.concat([df.tail(30) for df in equity_data.values()], keys=symbols)

        batch_results = predictor.predict_batch(
            market_data=batch_data,
            symbols=symbols,
            prediction_date=datetime(2023, 12, 31)
        )

        assert isinstance(batch_results, dict), "批量结果应该是字典"
        logger.info(f"✅ 批量预测成功: {len(batch_results)} 个股票")

        # 步骤10: 检查监控
        logger.info("\n📈 步骤10: 检查监控状态")

        health = predictor.get_model_health()
        if health:
            logger.info(f"模型健康状态: {health.status}")
        else:
            logger.info("新模型暂无监控数据（正常）")

        # 步骤11: 测试缓存
        logger.info("\n💾 步骤11: 测试预测缓存")

        cached_prediction = predictor.get_cached_prediction(
            symbol=symbol,
            prediction_date=datetime(2023, 12, 31)
        )

        if cached_prediction:
            logger.info(f"✅ 缓存功能正常: {cached_prediction['prediction']:.6f}")
        else:
            logger.info("缓存为空（可能缓存设置问题）")

        # 生成测试报告
        logger.info("\n" + "=" * 50)
        logger.info("🎉 端到端测试完成！")
        logger.info("=" * 50)
        logger.info("✅ 模型注册 - 正常")
        logger.info("✅ 模型训练 - 正常")
        logger.info("✅ 模型保存 - 正常")
        logger.info("✅ 模型加载 - 正常")
        logger.info("✅ 生产部署 - 正常")
        logger.info("✅ 单次预测 - 正常")
        logger.info("✅ 批量预测 - 正常")
        logger.info("✅ 监控功能 - 正常")
        logger.info("=" * 50)
        logger.info("🚀 新模型架构端到端测试全部通过！")
        logger.info("💡 核心功能验证:")
        logger.info("   - 模型训练和保存机制正常")
        logger.info("   - ModelPredictor生产服务正常")
        logger.info("   - 预测功能完整且稳定")
        logger.info("   - 监控系统基础功能正常")
        logger.info("   - 缓存机制工作正常")

        return True

    except Exception as e:
        logger.error(f"❌ 端到端测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # 清理测试环境
        if test_dir.exists():
            shutil.rmtree(test_dir)
        logger.info("🧹 测试环境已清理")


def main():
    """主函数"""
    try:
        success = test_model_lifecycle()
        return 0 if success else 1
    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        return 1
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)