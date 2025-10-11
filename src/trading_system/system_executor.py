import logging
import yaml
from datetime import datetime, timedelta
import pandas as pd
import argparse

# 导入所有需要的核心模块
from src.use_case.single_experiment import SystemOrchestrator
from .config.system import SystemConfig
from .orchestration.components.allocator import AllocationConfig, StrategyAllocation
from .orchestration.components.compliance import ComplianceRules
from .strategies.base_strategy import BaseStrategy
from .strategies.ml_strategy import MLStrategy
from .feature_engineering.pipeline import FeatureEngineeringPipeline
from .models.serving.predictor import ModelPredictor
from .utils.position_sizer import PositionSizer
from .data.yfinance_provider import YFinanceProvider
from .data.ff5_provider import FF5DataProvider

logger = logging.getLogger(__name__)

class SystemExecutor:
    """
    一个用于运行多策略系统回测的执行器。
    它负责读取配置、构建所有组件（策略、分配器等），
    并驱动 SystemOrchestrator 完成回测。
    """
    def __init__(self, config_path: str):
        """
        初始化执行器。
        Args:
            config_path: 指向 system_backtest_config.yaml 文件的路径。
        """
        logger.info(f"Loading system backtest configuration from: {config_path}")
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

    def run_backtest(self):
        """
        执行完整的回测流程。
        """
        logger.info("Starting multi-strategy backtest...")

        # 1. 创建数据提供方 (Providers)
        data_provider = self._create_provider('data_provider')
        factor_data_provider = self._create_provider('factor_data_provider')

        # 2. 构建策略列表
        strategies = self._build_strategies(data_provider, factor_data_provider)
        if not strategies:
            logger.error("No strategies could be built. Aborting backtest.")
            return

        # 3. 创建资金分配和合规性配置
        allocation_config = self._create_allocation_config()
        compliance_rules = self._create_compliance_rules()

        # 4. 初始化 SystemOrchestrator
        # 注意：这里的 system_config 只是为了获取 initial_capital，实际的分配和合规逻辑由新配置驱动
        system_config = SystemConfig(initial_capital=self.config['backtest_run']['initial_capital'])
        
        orchestrator = SystemOrchestrator(
            system_config=system_config,
            strategies=strategies,
            allocation_config=allocation_config,
            compliance_rules=compliance_rules
        )
        
        # 5. 运行回测时间循环
        if not orchestrator.initialize_system():
            logger.error("System Orchestrator failed to initialize. Aborting.")
            return

        start_date = datetime.strptime(self.config['backtest_run']['start_date'], "%Y-%m-%d")
        end_date = datetime.strptime(self.config['backtest_run']['end_date'], "%Y-%m-%d")
        date_range = pd.date_range(start=start_date, end=end_date, freq='B') # 'B' for business day

        logger.info(f"Running backtest from {start_date.date()} to {end_date.date()}...")
        for date in date_range:
            orchestrator.run_system(date=date)

        logger.info("Backtest loop finished.")

        # 6. 生成并打印最终报告
        self._generate_final_report(orchestrator)

    def _create_provider(self, provider_key: str):
        config = self.config.get(provider_key)
        if not config:
            return None
        
        provider_type = config.get('type')
        params = config.get('parameters', {})
        logger.info(f"Creating provider '{provider_key}' of type '{provider_type}'")

        if provider_type == "YFinanceProvider":
            return YFinanceProvider(**params)
        if provider_type == "FF5DataProvider":
            return FF5DataProvider(**params)
        
        raise ValueError(f"Unsupported provider type: {provider_type}")

    def _build_strategies(self, data_provider, factor_data_provider) -> list[BaseStrategy]:
        """根据配置构建并返回一个策略实例列表。"""
        strategy_list = []
        for strategy_config in self.config.get('strategies', []):
            name = strategy_config['name']
            logger.info(f"Building strategy: {name}")

            # a. 创建模型预测器 (CHANGED: Create predictor FIRST)
            model_params = strategy_config.get('parameters', {})
            model_predictor = ModelPredictor(
                model_id=model_params['model_id'],
                model_registry_path="./models/"
            )

            # b. 获取或构建特征工程管道
            fe_pipeline = None
            loaded_model = model_predictor.get_current_model()
            if loaded_model and getattr(loaded_model, 'is_trained', False) and hasattr(loaded_model, 'feature_pipeline'):
                # 如果模型已训练并包含特征管道，则直接使用它
                logger.info(f"Using feature pipeline loaded from pre-trained model for '{name}'.")
                fe_pipeline = loaded_model.feature_pipeline
            
            if fe_pipeline is None:
                # 否则，从配置中构建并拟合新的管道
                logger.info(f"Building and fitting a new feature pipeline for '{name}'.")
                fe_pipeline = FeatureEngineeringPipeline.from_config(strategy_config['feature_engineering'])
                
                # 回测开始前，需要用历史数据 "训练" 特征管道
                fit_end_date = datetime.strptime(self.config['backtest_run']['start_date'], "%Y-%m-%d") - timedelta(days=1)
                fit_start_date = fit_end_date - timedelta(days=3 * 365) # 使用过去3年的数据
                
                logger.info(f"Fitting feature pipeline for '{name}' with data up to {fit_end_date.date()}")
                
                fit_data = data_provider.get_data(start_date=fit_start_date, end_date=fit_end_date)
                
                pipeline_data = {'price_data': fit_data}
                if factor_data_provider:
                    pipeline_data['factor_data'] = factor_data_provider.get_data(start_date=fit_start_date, end_date=fit_end_date)

                fe_pipeline.fit(pipeline_data)

            # c. 获取策略的交易 universe
            # 优先从策略自己的参数中读取 universe
            strategy_universe = model_params.get('universe')
            if not strategy_universe:
                # 如果策略没有定义 universe，则回退到 data_provider 的全局 aum
                logger.warning(f"No universe defined for strategy '{name}'. "
                               f"Falling back to global symbols from data_provider config.")
                strategy_universe = self.config.get('data_provider', {}).get('parameters', {}).get('symbols', [])
            
            if not strategy_universe:
                logger.error(f"FATAL: No universe could be determined for strategy '{name}'. Please define it in the config.")
                continue # Skip this strategy if no universe is found

            # d. 创建仓位大小控制器 (Position Sizer)
            # 这里可以使用默认值，也可以从配置中读取更复杂的设置
            position_sizer = PositionSizer(volatility_target=0.15)
            
            # e. 实例化策略
            # 这里我们假设所有策略都使用通用的 MultiStockMLStrategy 类，它遵循新的架构
            strategy = MLStrategy(
                name=name,
                feature_pipeline=fe_pipeline,
                model_predictor=model_predictor,
                universe=strategy_universe,  # Pass the resolved universe
                data_provider=data_provider,
                factor_data_provider=factor_data_provider
            )
            strategy_list.append(strategy)
            
        return strategy_list

    def _create_allocation_config(self) -> AllocationConfig:
        alloc_conf = self.config['allocation']
        strategy_allocations = [
            StrategyAllocation(**sa) for sa in alloc_conf['strategy_allocations']
        ]
        return AllocationConfig(
            strategy_allocations=strategy_allocations,
            rebalance_threshold=alloc_conf.get('rebalance_threshold', 0.05),
            cash_buffer_weight=alloc_conf.get('cash_buffer_weight', 0.02)
        )

    def _create_compliance_rules(self) -> ComplianceRules:
        if 'compliance' not in self.config:
            logger.info("No compliance rules specified in config. Will be auto-generated from allocation.")
            return None # Orchestrator will auto-generate them
        
        comp_conf = self.config['compliance']
        # 注意: 这里的实现只映射了部分参数，可以按需扩展
        return ComplianceRules(
            max_single_position_weight=comp_conf.get('max_single_position_weight', 0.15)
        )

    def _generate_final_report(self, orchestrator: SystemOrchestrator):
        logger.info("\n" + "="*50 + "\n--- FINAL BACKTEST PERFORMANCE REPORT ---\n" + "="*50)
        
        reporter = orchestrator.performance_reporter
        history = orchestrator.execution_history

        if not history:
            logger.warning("No execution history was recorded. Cannot generate a report.")
            return

        final_portfolio = history[-1].portfolio_summary
        
        # 使用最后一个 portfolio 状态和整个回测期间的交易来生成报告
        all_trades = []
        for result in history:
            if result.is_successful:
                trades_dict_list = result.trades_summary.get('trade_list', [])
                # The trade_list is a list of dicts, not Trade objects. 
                # PerformanceReporter expects Trade objects. This needs a proper conversion.
                # For this example, we will generate a simpler summary.
        
        # 简化的报告逻辑
        initial_capital = orchestrator.config.initial_capital
        final_value = final_portfolio.get('total_value', initial_capital)
        total_return = (final_value - initial_capital) / initial_capital
        
        print(f"\nBacktest Period:")
        print(f"  Start Date: {self.config['backtest_run']['start_date']}")
        print(f"  End Date:   {self.config['backtest_run']['end_date']}")

        print(f"\nPortfolio Performance:")
        print(f"  Initial Capital: ${initial_capital:,.2f}")
        print(f"  Final Value:     ${final_value:,.2f}")
        print(f"  Total Return:    {total_return:.2%}")
        
        # TODO: A more detailed report would require processing the history
        # of portfolio values to calculate Sharpe, Drawdown, etc.
        # The PerformanceReporter can be enhanced to take a history of portfolios.

        print("\n" + "="*50)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s.%(funcName)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    parser = argparse.ArgumentParser(description="Run a multi-strategy system backtest.")
    parser.add_argument(
        '-c', '--config',
        type=str,
        default='configs/system_backtest_config.yaml',
        help='Path to the system backtest configuration file.'
    )
    args = parser.parse_args()

    setup_logging()
    
    executor = SystemExecutor(config_path=args.config)
    executor.run_backtest()


if __name__ == "__main__":
    main()
