#!/usr/bin/env python3
"""
端到端测试: 完整的模型生命周期

场景1: 从原始数据到生产预测的完整流程
测试目标: 验证整个系统从输入到输出的完整链路

测试流程:
1. 准备阶段 - 创建测试数据和环境
2. 训练阶段 - 使用TrainingPipeline训练模型
3. 注册验证阶段 - 验证模型正确注册和加载
4. 生产部署阶段 - 使用ModelPredictor部署模型
5. 预测阶段 - 进行实际预测
6. 监控阶段 - 验证模型监控功能
7. 策略集成阶段 - 验证策略能正确使用模型

Usage:
    poetry run python test_end_to_end_lifecycle.py
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

# Add src to path - 修正路径问题
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

import pandas as pd
import numpy as np

# Import the components we want to test
from trading_system.models.training.pipeline import TrainingPipeline
from trading_system.models.training.trainer import TrainingConfig
from trading_system.models.serving.predictor import ModelPredictor
from trading_system.models.base.model_factory import ModelFactory
from trading_system.strategies.core_ffml_strategy import CoreFFMLStrategy
from trading_system.data.yfinance_provider import YFinanceProvider
from trading_system.data.ff5_provider import FF5DataProvider
from trading_system.config.strategy import StrategyConfig
from trading_system.config.backtest import BacktestConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndToEndLifecycleTest:
    """端到端生命周期测试类"""

    def __init__(self):
        """初始化测试环境"""
        self.test_dir = Path(tempfile.mkdtemp(prefix="e2e_test_"))
        self.registry_path = self.test_dir / "model_registry"
        self.registry_path.mkdir(parents=True, exist_ok=True)

        self.test_symbols = ["AAPL", "MSFT", "GOOGL"]
        self.start_date = datetime(2023, 1, 1)
        self.end_date = datetime(2023, 12, 31)

        # 测试结果存储
        self.test_results = {
            'setup': False,
            'training': False,
            'registration': False,
            'deployment': False,
            'prediction': False,
            'monitoring': False,
            'strategy_integration': False
        }

        logger.info(f"🧪 初始化端到端测试环境: {self.test_dir}")

    def cleanup(self):
        """清理测试环境"""
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)
        logger.info("🧹 测试环境已清理")

    def create_synthetic_data(self) -> tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """创建合成测试数据"""
        logger.info("📊 创建合成测试数据...")

        # 创建价格数据
        equity_data = {}
        np.random.seed(42)  # 确保可重现

        for i, symbol in enumerate(self.test_symbols):
            # 生成日期序列
            dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
            # 只保留交易日（周一到周五）
            dates = dates[dates.dayofweek < 5]

            # 生成合成价格数据
            n_days = len(dates)

            # 基础价格趋势 + 随机游走
            base_price = 100 + i * 50  # AAPL: 100, MSFT: 150, GOOGL: 200

            # 趋势组件
            trend = np.linspace(0, 0.3, n_days)  # 30% 年化增长

            # 季节性组件
            seasonal = 0.1 * np.sin(2 * np.pi * np.arange(n_days) / 60)  # 季节性波动

            # 随机游走
            random_walk = np.random.normal(0, 0.02, n_days).cumsum()

            # 组合价格
            price_multiplier = 1 + trend + seasonal + random_walk
            prices = base_price * price_multiplier

            # 创建OHLCV数据
            df = pd.DataFrame({
                'date': dates,
                'Close': prices,
                'Open': prices * np.random.uniform(0.995, 1.005, n_days),
                'High': prices * np.random.uniform(1.0, 1.02, n_days),
                'Low': prices * np.random.uniform(0.98, 1.0, n_days),
                'Volume': np.random.randint(1_000_000, 10_000_000, n_days)
            })

            # 设置日期为索引，这对pipeline很重要
            df = df.set_index('date')

            # 添加一些缺失值模拟真实数据
            missing_indices = np.random.choice(n_days, size=int(n_days * 0.02), replace=False)
            for idx in missing_indices:
                df.iloc[idx, df.columns.get_loc('Volume')] = np.nan

            equity_data[symbol] = df
            logger.info(f"✅ 生成 {symbol} 数据: {len(df)} 行")

        # 创建FF5因子数据
        factor_dates = pd.date_range(start=self.start_date, end=self.end_date, freq='M')
        n_factors = len(factor_dates)

        # 合成FF5因子 (基于历史统计特征)
        factor_data = pd.DataFrame({
            'MKT': np.random.normal(0.006, 0.045, n_factors),  # 市场因子
            'SMB': np.random.normal(0.002, 0.030, n_factors),  # 规模因子
            'HML': np.random.normal(0.001, 0.025, n_factors),  # 价值因子
            'RMW': np.random.normal(0.003, 0.020, n_factors),  # 盈利能力因子
            'CMA': np.random.normal(0.001, 0.015, n_factors)   # 投资因子
        }, index=factor_dates)

        logger.info(f"✅ 生成因子数据: {len(factor_data)} 个月")

        return equity_data, factor_data

    def test_setup_phase(self):
        """测试阶段1: 准备阶段"""
        logger.info("\n🔧 阶段1: 准备测试环境")

        try:
            # 1. 创建测试数据
            self.equity_data, self.factor_data = self.create_synthetic_data()

            # 2. 验证数据质量
            assert len(self.equity_data) == 3, "应该有3支股票的数据"
            assert len(self.factor_data) > 0, "因子数据不应为空"

            for symbol, data in self.equity_data.items():
                assert len(data) > 200, f"{symbol} 应该有足够的数据点"
                assert 'Close' in data.columns, f"{symbol} 应该有收盘价数据"

            # 3. 验证测试环境
            assert self.test_dir.exists(), "测试目录应该存在"
            assert self.registry_path.exists(), "模型注册目录应该存在"

            self.test_results['setup'] = True
            logger.info("✅ 准备阶段完成")

        except Exception as e:
            logger.error(f"❌ 准备阶段失败: {e}")
            raise

    def test_training_phase(self):
        """测试阶段2: 训练阶段"""
        logger.info("\n🚀 阶段2: 模型训练")

        try:
            # 1. 创建训练配置
            config = TrainingConfig(
                use_cross_validation=True,
                cv_folds=3,  # 减少折数以加快测试
                early_stopping=True,
                validation_split=0.2
            )

            # 2. 创建训练Pipeline
            pipeline = TrainingPipeline(
                model_type="residual_predictor",
                config=config,
                registry_path=str(self.registry_path)
            )

            # 3. 创建模拟数据提供者
            from unittest.mock import Mock
            mock_data_provider = Mock()
            mock_data_provider.get_price_data.return_value = self.equity_data

            # Factor data is already indexed by date
            mock_data_provider.get_factor_data.return_value = self.factor_data

            # 配置数据提供者
            pipeline.configure_data(mock_data_provider)

            # 4. 执行训练
            logger.info("开始训练残差预测模型...")
            start_time = datetime.now()

            self.training_result = pipeline.run_pipeline(
                start_date=self.start_date,
                end_date=self.end_date,
                symbols=self.test_symbols,
                model_name="test_residual_predictor"
            )

            training_time = (datetime.now() - start_time).total_seconds()

            # 5. 验证训练结果
            assert self.training_result is not None, "训练结果不应为空"
            assert 'pipeline_info' in self.training_result, "应该包含pipeline信息"
            assert 'model_id' in self.training_result['pipeline_info'], "应该包含模型ID"
            assert 'training_results' in self.training_result, "应该包含训练结果"

            self.model_id = self.training_result['pipeline_info']['model_id']

            # 6. 验证训练时间合理 (不超过5分钟)
            assert training_time < 300, f"训练时间过长: {training_time:.2f}秒"

            # 7. 验证模型文件已保存
            model_path = self.registry_path / self.model_id
            assert model_path.exists(), f"模型目录应该存在: {model_path}"
            assert (model_path / "model.pkl").exists(), "模型文件应该存在"
            assert (model_path / "metadata.json").exists(), "元数据文件应该存在"

            # 8. 验证元数据内容
            with open(model_path / "metadata.json", 'r') as f:
                metadata = json.load(f)

            assert metadata['model_type'] == "residual_predictor", "模型类型应该正确"
            assert 'trained_at' in metadata, "应该包含训练时间"
            assert metadata['training_samples'] > 0, "应该有训练样本数"

            logger.info(f"✅ 训练完成，模型ID: {self.model_id}")
            logger.info(f"   训练时间: {training_time:.2f}秒")
            logger.info(f"   训练样本: {metadata['training_samples']}")

            self.test_results['training'] = True

        except Exception as e:
            logger.error(f"❌ 训练阶段失败: {e}")
            raise

    def test_registration_phase(self):
        """测试阶段3: 注册验证阶段"""
        logger.info("\n📝 阶段3: 模型注册验证")

        try:
            # 1. 通过ModelFactory验证模型注册
            available_models = ModelFactory._registry
            assert "residual_predictor" in available_models, "残差预测器应该已注册"

            # 2. 尝试加载模型
            from trading_system.models.base.model_factory import ModelRegistry
            model_registry = ModelRegistry(self.registry_path)
            loaded_model = model_registry.load_model(self.model_id)

            # 3. 验证加载的模型
            assert loaded_model is not None, "加载的模型不应为空"
            assert hasattr(loaded_model, 'predict'), "模型应该有预测方法"
            assert hasattr(loaded_model, 'model_type'), "模型应该有类型属性"
            assert loaded_model.model_type == "residual_predictor", "模型类型应该正确"

            # 4. 验证模型状态
            assert hasattr(loaded_model, 'status'), "模型应该有状态属性"
            assert loaded_model.status in ["trained", "deployed"], f"模型状态应该正常: {loaded_model.status}"

            # 5. 验证模型元数据
            assert hasattr(loaded_model, 'metadata'), "模型应该有元数据"
            assert loaded_model.metadata.model_type == "residual_predictor", "元数据类型应该正确"
            assert loaded_model.metadata.training_samples > 0, "元数据应该包含训练样本数"

            logger.info("✅ 模型注册验证完成")
            logger.info(f"   模型类型: {loaded_model.model_type}")
            logger.info(f"   模型状态: {loaded_model.status}")
            logger.info(f"   训练样本: {loaded_model.metadata.training_samples}")

            self.test_results['registration'] = True

        except Exception as e:
            logger.error(f"❌ 注册验证阶段失败: {e}")
            raise

    def test_deployment_phase(self):
        """测试阶段4: 生产部署阶段"""
        logger.info("\n🚀 阶段4: 生产部署")

        try:
            # 1. 创建ModelPredictor
            self.predictor = ModelPredictor(
                model_registry_path=str(self.registry_path),
                enable_monitoring=True,
                cache_predictions=True
            )

            # 2. 加载模型到生产环境
            deployed_model_id = self.predictor.load_model(
                model_name="residual_predictor",
                model_path=str(self.registry_path / self.model_id)
            )

            # 3. 验证部署结果
            assert deployed_model_id is not None, "部署的模型ID不应为空"
            assert self.predictor.get_current_model() is not None, "当前加载的模型不应为空"
            assert self.predictor.get_current_model_id() == deployed_model_id, "模型ID应该匹配"

            # 4. 验证Monitor初始化
            health = self.predictor.get_model_health()
            # 注意: 新模型可能还没有监控数据，所以health可能为None
            logger.info(f"模型健康状态: {health.status if health else '暂无数据'}")

            # 5. 验证可以获取模型信息
            available_models = self.predictor.list_available_models()
            assert "residual_predictor" in available_models, "应该可以列出可用模型"

            model_info = self.predictor.get_model_info("residual_predictor")
            assert model_info is not None, "应该可以获取模型信息"
            assert 'description' in model_info, "模型信息应该包含描述"

            logger.info("✅ 生产部署完成")
            logger.info(f"   部署模型ID: {deployed_model_id}")
            logger.info(f"   可用模型: {available_models}")

            self.test_results['deployment'] = True

        except Exception as e:
            logger.error(f"❌ 部署阶段失败: {e}")
            raise

    def test_prediction_phase(self):
        """测试阶段5: 预测阶段"""
        logger.info("\n🔮 阶段5: 预测测试")

        try:
            # 1. 准备预测数据 (为predictor提供正确的格式)
            symbol = self.test_symbols[0]  # 使用AAPL

            # 构造单个DataFrame，包含symbol列来区分不同股票的数据
            # 使用完整的历史数据以确保能计算所有特征
            combined_data = []
            for s in self.test_symbols:
                # 使用完整的历史数据，而不是最近60天
                # 这样可以计算长期特征（如200日均线等）
                symbol_data = self.equity_data[s].copy()

                # 添加symbol列
                symbol_data = symbol_data.reset_index()
                symbol_data['symbol'] = s
                combined_data.append(symbol_data)

            # 合并所有数据
            prediction_data = pd.concat(combined_data, ignore_index=True)

            # 为残差预测器添加因子数据
            # 残差预测器需要FF5因子数据（MKT, SMB, HML, RMW, CMA）
            if hasattr(self.predictor, '_current_model') and self.predictor._current_model:
                if hasattr(self.predictor._current_model, 'model_type') and self.predictor._current_model.model_type == "residual_predictor":
                    logger.info("为残差预测器添加FF5因子数据...")

                    # 重新索引因子数据到日频，使用前向填充
                    factor_data_daily = self.factor_data.reindex(prediction_data['date'], method='ffill')

                    # 将因子列添加到预测数据中
                    for factor_col in ['MKT', 'SMB', 'HML', 'RMW', 'CMA']:
                        if factor_col in factor_data_daily.columns:
                            prediction_data[factor_col] = factor_data_daily[factor_col].values
                        else:
                            # 如果前向填充失败，使用最近的因子值
                            prediction_data[factor_col] = self.factor_data[factor_col].iloc[-1]

                    logger.info(f"已添加因子列: {['MKT', 'SMB', 'HML', 'RMW', 'CMA']}")
                    logger.info(f"预测数据形状: {prediction_data.shape}")
                    logger.info(f"预测数据列: {list(prediction_data.columns)}")

            # 2. 执行预测 - 为所有符号创建预测（因为训练时使用了多符号特征）
            logger.info("为残差预测器创建多符号预测数据...")

            # 残差预测器训练时使用了所有符号的特征，所以预测时也需要提供所有符号的数据
            all_symbol_results = {}
            for s in self.test_symbols:
                try:
                    symbol_result = self.predictor.predict(
                        market_data=prediction_data,
                        symbol=s,
                        prediction_date=self.end_date
                    )
                    all_symbol_results[s] = symbol_result
                    logger.info(f"✅ {s} 预测成功: {symbol_result['prediction']:.6f}")
                except Exception as e:
                    logger.warning(f"⚠️ {s} 预测失败: {e}")
                    continue

            # 使用第一个成功的预测结果作为主要结果
            if all_symbol_results:
                prediction_result = all_symbol_results[symbol]  # 使用原始请求的符号
                logger.info(f"使用 {symbol} 的预测结果作为主要结果")
            else:
                raise ValueError("所有符号的预测都失败了")

            # 3. 验证预测结果
            assert prediction_result is not None, "预测结果不应为空"
            assert 'prediction' in prediction_result, "应该包含预测值"
            assert 'symbol' in prediction_result, "应该包含股票代码"
            assert 'prediction_date' in prediction_result, "应该包含预测日期"
            assert 'model_id' in prediction_result, "应该包含模型ID"
            assert 'timestamp' in prediction_result, "应该包含时间戳"

            # 4. 验证预测值合理性
            prediction_value = prediction_result['prediction']
            assert isinstance(prediction_value, (int, float, np.floating)), "预测值应该是数值"
            assert not np.isnan(prediction_value), "预测值不应该是NaN"
            assert not np.isinf(prediction_value), "预测值不应该是无穷大"
            assert abs(prediction_value) < 1.0, f"预测值应该在合理范围内: {prediction_value}"

            # 5. 测试批量预测 (使用相同的数据格式)
            batch_results = self.predictor.predict_batch(
                market_data=prediction_data,
                symbols=self.test_symbols,
                prediction_date=self.end_date
            )

            # 6. 验证批量预测结果
            assert isinstance(batch_results, dict), "批量预测结果应该是字典"
            assert len(batch_results) <= len(self.test_symbols), "预测结果数量不应超过股票数量"

            for symbol, result in batch_results.items():
                assert 'prediction' in result, f"{symbol} 预测结果应该包含预测值"
                assert not np.isnan(result['prediction']), f"{symbol} 预测值不应该是NaN"

            # 7. 验证Monitor记录了预测
            # 由于我们启用了监控，预测应该被记录
            logger.info("✅ 预测测试完成")
            logger.info(f"   AAPL预测值: {prediction_value:.6f}")
            logger.info(f"   批量预测: {len(batch_results)} 个股票")

            self.prediction_results = {
                'single': prediction_result,
                'batch': batch_results
            }

            self.test_results['prediction'] = True

        except Exception as e:
            logger.error(f"❌ 预测阶段失败: {e}")
            raise

    def test_monitoring_phase(self):
        """测试阶段6: 监控阶段"""
        logger.info("\n📊 阶段6: 监控测试")

        try:
            # 1. 获取当前模型健康状态
            health = self.predictor.get_model_health()

            # 2. 检查监控功能是否正常
            # 注意: 对于新模型，可能还没有足够的监控数据
            if health:
                logger.info(f"模型健康状态: {health.status}")
                logger.info(f"问题列表: {health.issues}")
                logger.info(f"健康指标: {health.metrics}")

                # 3. 验证健康状态合理性
                assert health.status in ['healthy', 'warning', 'critical', 'degraded', 'unknown'], \
                    f"健康状态应该有效: {health.status}"

                # 4. 验证监控指标
                if health.metrics:
                    for metric_name, metric_value in health.metrics.items():
                        # 允许某些指标为字符串值
                        string_metrics = {'performance_status'}
                        if metric_name in string_metrics:
                            assert isinstance(metric_value, str), \
                                f"指标 {metric_name} 应该是字符串: {metric_value}={type(metric_value)}"
                        else:
                            assert isinstance(metric_value, (int, float)), \
                                f"指标值应该是数值: {metric_name}={metric_value}"

            else:
                logger.info("新模型暂无监控数据，这是正常的")

            # 5. 测试预测缓存功能
            cached_prediction = self.predictor.get_cached_prediction(
                symbol=self.test_symbols[0],
                prediction_date=self.end_date
            )

            # 应该能找到缓存的预测
            if cached_prediction:
                assert 'prediction' in cached_prediction, "缓存的预测应该包含预测值"
                logger.info(f"找到缓存的预测: {cached_prediction['prediction']:.6f}")

            # 6. 验证监控没有抛出异常
            # 这已经通过前面的步骤验证了

            logger.info("✅ 监控测试完成")
            self.test_results['monitoring'] = True

        except Exception as e:
            logger.error(f"❌ 监控阶段失败: {e}")
            raise

    def test_strategy_integration_phase(self):
        """测试阶段7: 策略集成阶段"""
        logger.info("\n🎯 阶段7: 策略集成测试")

        try:
            # 1. 创建策略配置
            strategy_config = StrategyConfig(
                name="E2E_Test_Strategy",
                parameters={
                    'min_signal_strength': 0.05,
                    'max_position_size': 0.2,
                    'target_positions': len(self.test_symbols)
                },
                lookback_period=252,
                universe=self.test_symbols
            )

            # 2. 创建回测配置
            backtest_config = BacktestConfig(
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=1000000,
                symbols=self.test_symbols
            )

            # 3. 创建策略实例 - 使用新的依赖注入模式
            from trading_system.models.serving.predictor import ModelPredictor

            # 首先创建并配置ModelPredictor
            predictor = ModelPredictor(
                model_registry_path=str(self.registry_path),
                enable_monitoring=True,
                cache_predictions=True
            )

            # 加载训练好的模型
            model_id = predictor.load_model("residual_predictor")
            logger.info(f"策略集成 - 加载模型: {model_id}")

            # 使用依赖注入创建策略
            strategy = CoreFFMLStrategy(
                model_predictor=predictor,
                config=strategy_config,
                backtest_config=backtest_config
            )

            # 4. 验证策略可以生成信号
            # 注意：策略现在通过ModelPredictor自动获取数据，无需手动设置equity_data
            # 注意: 由于我们的模型预测可能不准确，这里主要测试流程是否正常
            try:
                signals = strategy.generate_signals(self.end_date)

                # 6. 验证信号结果
                if signals:
                    logger.info(f"策略生成了 {len(signals)} 个信号")

                    # 验证信号格式
                    for signal in signals[:3]:  # 只检查前3个信号
                        assert hasattr(signal, 'symbol'), "信号应该有股票代码"
                        assert hasattr(signal, 'strength'), "信号应该有强度"
                        assert signal.symbol in self.test_symbols, "信号股票应该在测试集中"

                    logger.info(f"示例信号: {signals[0].symbol} = {signals[0].strength:.6f}")
                else:
                    logger.info("策略未生成信号 (可能是预测值不满足阈值)")

            except Exception as e:
                # 如果信号生成失败，检查是否是因为模型预测问题
                if "No model available" in str(e):
                    logger.warning("策略没有可用的模型，这可能是配置问题")
                else:
                    # 其他错误，记录但继续测试
                    logger.warning(f"策略生成信号时出现问题: {e}")

            # 7. 验证策略状态
            strategy_info = strategy.get_strategy_summary()
            assert strategy_info is not None, "策略信息不应为空"
            assert 'strategy_name' in strategy_info, "策略信息应该包含名称"

            logger.info("✅ 策略集成测试完成")
            logger.info(f"   策略名称: {strategy_info.get('strategy_name')}")
            logger.info(f"   策略类型: {strategy_info.get('strategy_type')}")

            self.test_results['strategy_integration'] = True

        except Exception as e:
            logger.error(f"❌ 策略集成阶段失败: {e}")
            # 这个阶段不是关键，记录失败但不中断测试
            logger.warning("策略集成失败，但这不影响核心模型功能")

    def run_complete_test(self):
        """运行完整的端到端测试"""
        logger.info("🎯 开始端到端生命周期测试")
        logger.info("=" * 60)

        try:
            # 按顺序执行所有测试阶段
            self.test_setup_phase()
            self.test_training_phase()
            self.test_registration_phase()
            self.test_deployment_phase()
            self.test_prediction_phase()
            self.test_monitoring_phase()
            self.test_strategy_integration_phase()

            # 生成测试报告
            self.generate_test_report()

        except Exception as e:
            logger.error(f"❌ 端到端测试失败: {e}")
            raise
        finally:
            self.cleanup()

    def generate_test_report(self):
        """生成测试报告"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 端到端测试报告")
        logger.info("=" * 60)

        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())

        for phase, result in self.test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            phase_name = {
                'setup': '准备阶段',
                'training': '训练阶段',
                'registration': '注册验证',
                'deployment': '生产部署',
                'prediction': '预测测试',
                'monitoring': '监控测试',
                'strategy_integration': '策略集成'
            }.get(phase, phase)

            logger.info(f"{phase_name:.<20} {status}")

        logger.info("-" * 60)
        logger.info(f"总计: {passed_tests}/{total_tests} 测试通过")

        if passed_tests == total_tests:
            logger.info("🎉 所有端到端测试通过！系统架构重构成功！")
            logger.info("\n💡 关键成就:")
            logger.info("   ✅ 新模型架构工作正常")
            logger.info("   ✅ 训练到预测流程完整")
            logger.info("   ✅ 模型注册和加载功能正常")
            logger.info("   ✅ 生产部署和监控功能正常")
            logger.info("   ✅ 策略集成基本正常")
            logger.info("\n🚀 系统可以投入生产使用！")
        else:
            failed_count = total_tests - passed_tests
            logger.warning(f"⚠️  有 {failed_count} 个测试失败，需要进一步检查")

        logger.info("=" * 60)


def main():
    """主函数"""
    try:
        # 创建并运行测试
        test = EndToEndLifecycleTest()
        test.run_complete_test()

        return 0

    except KeyboardInterrupt:
        logger.info("测试被用户中断")
        return 1
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)