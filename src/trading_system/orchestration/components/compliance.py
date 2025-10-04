"""
Compliance Monitor - IPS Compliance and Risk Monitoring

Monitors Investment Policy Statement (IPS) compliance, investment box
constraints, risk budgets, and generates compliance reports.

Extracted from SystemOrchestrator to follow Single Responsibility Principle.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from ...types.portfolio import Portfolio, Position
from ...types.enums import AssetClass
from ...data.stock_classifier import StockClassifier, InvestmentBox

logger = logging.getLogger(__name__)


class ComplianceStatus(Enum):
    """Compliance status levels."""
    COMPLIANT = "compliant"
    WARNING = "warning"
    VIOLATION = "violation"
    CRITICAL = "critical"


class ViolationType(Enum):
    """Types of compliance violations."""
    ALLOCATION_OUT_OF_BOUNDS = "allocation_out_of_bounds"
    CONCENTRATION_EXCEEDED = "concentration_exceeded"
    RISK_BUDGET_EXCEEDED = "risk_budget_exceeded"
    INVESTMENT_BOX_VIOLATION = "investment_box_violation"
    POSITION_SIZE_EXCEEDED = "position_size_exceeded"
    SECTOR_ALLOCATION_EXCEEDED = "sector_allocation_exceeded"


@dataclass
class ComplianceRule:
    """Individual compliance rule."""
    name: str
    violation_type: ViolationType
    description: str
    threshold_value: float
    current_value: float
    status: ComplianceStatus
    recommendation: str
    severity: str  # "low", "medium", "high", "critical"


@dataclass
class StrategyAllocationRule:
    """Allocation rule for a single strategy."""
    strategy_name: str
    min_weight: float
    max_weight: float


@dataclass
class ComplianceRules:
    """
    Complete set of compliance rules - supports flexible strategy allocations.

    Instead of hardcoded core/satellite rules, uses a list of strategy allocation rules
    that can be configured for any number of strategies.
    """
    # Strategy allocation rules - flexible for multiple strategies
    strategy_allocation_rules: List[StrategyAllocationRule] = None

    # Investment box exposure limits (e.g., sector limits, region limits)
    box_exposure_limits: Dict[str, Dict[str, float]] = None
    
    # Cash allocation rules
    cash_min_weight: float = 0.00
    cash_max_weight: float = 0.10

    # Position rules
    max_single_position_weight: float = 0.15
    max_sector_allocation: float = 0.25
    max_concentration_top5: float = 0.40
    max_concentration_top10: float = 0.60

    # Risk rules
    max_portfolio_volatility: float = 0.15  # 15% annual volatility
    max_var_95: float = 0.05  # 5% daily VaR
    max_drawdown: float = 0.15  # 15% max drawdown
    risk_budget_limit: float = 0.12  # Risk budget units

    # Investment box rules (asset class level)
    asset_class_rules: Dict[AssetClass, Tuple[float, float]] = field(default_factory=lambda: {
        AssetClass.EQUITY: (0.60, 0.95),
        AssetClass.BOND: (0.00, 0.30),
        AssetClass.ALTERNATIVE: (0.00, 0.15),
    })

    # NEW: Box deviation rules (e.g., sector, region)
    box_deviation_limits: List[Dict[str, Any]] = field(default_factory=list)

    # Liquidity rules
    min_liquidity_score: float = 0.7  # Minimum portfolio liquidity
    max_illiquid_allocation: float = 0.10  # Max allocation to illiquid assets
    
    def __post_init__(self):
        """Initialize default strategy allocation rules if not provided."""
        if self.strategy_allocation_rules is None:
            self.strategy_allocation_rules = []
    
    @classmethod
    def create_core_satellite(cls, 
                             core_min: float = 0.70, 
                             core_max: float = 0.80,
                             satellite_min: float = 0.20, 
                             satellite_max: float = 0.30,
                             core_name: str = "core",
                             satellite_name: str = "satellite") -> 'ComplianceRules':
        """
        Factory method to create traditional core-satellite compliance rules.
        Provides backward compatibility with the old hardcoded approach.
        """
        return cls(
            strategy_allocation_rules=[
                StrategyAllocationRule(
                    strategy_name=core_name,
                    min_weight=core_min,
                    max_weight=core_max
                ),
                StrategyAllocationRule(
                    strategy_name=satellite_name,
                    min_weight=satellite_min,
                    max_weight=satellite_max
                )
            ]
        )


@dataclass
class ComplianceReport:
    """Complete compliance report."""
    timestamp: datetime
    overall_status: ComplianceStatus
    total_violations: int
    violations: List[ComplianceRule]
    warnings: List[ComplianceRule]
    recommendations: List[str]
    portfolio_summary: Dict[str, Any]

    @property
    def is_compliant(self) -> bool:
        """Check if portfolio is fully compliant."""
        return self.overall_status == ComplianceStatus.COMPLIANT

    @property
    def has_critical_violations(self) -> bool:
        """Check if there are critical violations."""
        return any(rule.status == ComplianceStatus.CRITICAL for rule in self.violations)


class ComplianceMonitor:
    """
    Monitors and enforces IPS compliance.

    Responsibilities:
    - Check allocation compliance
    - Monitor concentration limits
    - Validate investment box constraints
    - Track risk budget compliance
    - Generate compliance reports
    - Provide remediation recommendations
    """

    def __init__(self, rules: ComplianceRules,
                 stock_classifier: Optional[StockClassifier] = None):
        """
        Initialize compliance monitor.

        Args:
            rules: Compliance rules to monitor
            stock_classifier: Optional classifier for box-based checks.
        """
        self.rules = rules
        self.stock_classifier = stock_classifier

        # Compliance history
        self.compliance_history: List[ComplianceReport] = []
        self.violation_history: List[Dict[str, Any]] = []

        # Statistics
        self.compliance_stats = {
            'total_checks': 0,
            'violations_detected': 0,
            'critical_violations': 0,
            'warnings_generated': 0
        }

        # Notification thresholds
        self.notification_thresholds = {
            'violation_count_warning': 3,
            'violation_count_critical': 5,
            'consecutive_violations': 3
        }

        logger.info("Initialized ComplianceMonitor")
        logger.info(f"Monitoring {len(self.rules.__dict__)} compliance rules")

    def check_compliance(self, portfolio: Portfolio,
                        benchmark_data: Optional[Any] = None) -> ComplianceReport:
        """
        Perform comprehensive compliance check.

        Args:
            portfolio: Current portfolio state
            benchmark_data: Optional benchmark data for comparison

        Returns:
            Complete compliance report
        """
        try:
            logger.info("Performing comprehensive compliance check")

            violations = []
            warnings = []

            # Check allocation compliance
            allocation_violations = self._check_allocation_compliance(portfolio)
            violations.extend(allocation_violations)

            # Check concentration compliance
            concentration_violations = self._check_concentration_compliance(portfolio)
            violations.extend(concentration_violations)

            # Check investment box compliance
            investment_violations = self._check_investment_box_compliance(portfolio)
            violations.extend(investment_violations)

            # Check risk budget compliance
            risk_violations = self._check_risk_budget_compliance(portfolio, benchmark_data)
            violations.extend(risk_violations)

            # Check position size compliance
            position_violations = self._check_position_size_compliance(portfolio)
            violations.extend(position_violations)

            # Check liquidity compliance
            liquidity_violations = self._check_liquidity_compliance(portfolio)
            violations.extend(liquidity_violations)

            # Separate warnings from violations
            warnings = [v for v in violations if v.status == ComplianceStatus.WARNING]
            violations = [v for v in violations if v.status in [ComplianceStatus.VIOLATION, ComplianceStatus.CRITICAL]]

            # Generate recommendations
            recommendations = self._generate_recommendations(violations, warnings, portfolio)

            # Determine overall status
            overall_status = self._determine_overall_status(violations, warnings)

            # Create portfolio summary
            portfolio_summary = self._create_portfolio_summary(portfolio)

            # Create compliance report
            report = ComplianceReport(
                timestamp=datetime.now(),
                overall_status=overall_status,
                total_violations=len(violations),
                violations=violations,
                warnings=warnings,
                recommendations=recommendations,
                portfolio_summary=portfolio_summary
            )

            # Update statistics
            self._update_statistics(report)

            # Store in history
            self.compliance_history.append(report)
            if len(self.compliance_history) > 100:  # Keep last 100 reports
                self.compliance_history = self.compliance_history[-100:]

            logger.info(f"Compliance check completed: {len(violations)} violations, {len(warnings)} warnings")
            return report

        except Exception as e:
            logger.error(f"Compliance check failed: {e}")
            # Return minimal report indicating failure
            return ComplianceReport(
                timestamp=datetime.now(),
                overall_status=ComplianceStatus.CRITICAL,
                total_violations=1,
                violations=[],
                warnings=[],
                recommendations=["Compliance system error - manual review required"],
                portfolio_summary={"error": str(e)}
            )

    def _check_allocation_compliance(self, portfolio: Portfolio) -> List[ComplianceRule]:
        """Check strategy allocation compliance."""
        violations = []

        if not portfolio.positions or portfolio.total_value == 0:
            return violations

        # Calculate current allocations (simplified - assumes position metadata includes strategy)
        strategy_values = {}
        sector_values = {}

        for position in portfolio.positions.values():
            position_value = position.quantity * position.current_price
            weight = position_value / portfolio.total_value

            # Strategy allocation (simplified)
            strategy = getattr(position, 'strategy_name', 'unknown')
            if 'core' in strategy.lower():
                strategy_values['core'] = strategy_values.get('core', 0) + weight
            elif 'satellite' in strategy.lower():
                strategy_values['satellite'] = strategy_values.get('satellite', 0) + weight

            # Sector allocation (simplified - would need sector mapping)
            sector = getattr(position, 'sector', 'unknown')
            sector_values[sector] = sector_values.get(sector, 0) + weight

        # Check strategy allocations based on configured rules
        for strategy_rule in self.rules.strategy_allocation_rules:
            strategy_weight = strategy_values.get(strategy_rule.strategy_name, 0)
            
            # Check minimum allocation
            if strategy_weight < strategy_rule.min_weight:
                violations.append(ComplianceRule(
                    name=f"{strategy_rule.strategy_name} Allocation Minimum",
                    violation_type=ViolationType.ALLOCATION_OUT_OF_BOUNDS,
                    description=f"{strategy_rule.strategy_name} allocation below minimum",
                    threshold_value=strategy_rule.min_weight,
                    current_value=strategy_weight,
                    status=ComplianceStatus.VIOLATION,
                    recommendation=f"Increase {strategy_rule.strategy_name} allocation to meet minimum requirement",
                    severity="high"
                ))
            
            # Check maximum allocation
            elif strategy_weight > strategy_rule.max_weight:
                violations.append(ComplianceRule(
                    name=f"{strategy_rule.strategy_name} Allocation Maximum",
                    violation_type=ViolationType.ALLOCATION_OUT_OF_BOUNDS,
                    description=f"{strategy_rule.strategy_name} allocation above maximum",
                    threshold_value=strategy_rule.max_weight,
                    current_value=strategy_weight,
                    status=ComplianceStatus.VIOLATION,
                    recommendation=f"Reduce {strategy_rule.strategy_name} allocation to meet maximum limit",
                    severity="high"
                ))

        # Check sector allocations
        for sector, sector_weight in sector_values.items():
            if sector_weight > self.rules.max_sector_allocation:
                violations.append(ComplianceRule(
                    name=f"Sector Allocation - {sector}",
                    violation_type=ViolationType.SECTOR_ALLOCATION_EXCEEDED,
                    description=f"Sector {sector} allocation exceeds limit",
                    threshold_value=self.rules.max_sector_allocation,
                    current_value=sector_weight,
                    status=ComplianceStatus.WARNING,
                    recommendation=f"Reduce {sector} exposure through rebalancing",
                    severity="medium"
                ))

        return violations

    def _check_concentration_compliance(self, portfolio: Portfolio) -> List[ComplianceRule]:
        """Check portfolio concentration limits."""
        violations = []

        if not portfolio.positions or portfolio.total_value == 0:
            return violations

        # Calculate position weights
        position_weights = []
        for position in portfolio.positions.values():
            weight = (position.quantity * position.current_price) / portfolio.total_value
            position_weights.append(weight)

        # Sort by weight (descending)
        position_weights.sort(reverse=True)

        # Check single position limit
        if position_weights and position_weights[0] > self.rules.max_single_position_weight:
            violations.append(ComplianceRule(
                name="Single Position Limit",
                violation_type=ViolationType.CONCENTRATION_EXCEEDED,
                description="Single position exceeds maximum weight",
                threshold_value=self.rules.max_single_position_weight,
                current_value=position_weights[0],
                status=ComplianceStatus.VIOLATION,
                recommendation="Reduce position size through partial rebalancing",
                severity="high"
            ))

        # Check top 5 concentration
        if len(position_weights) >= 5:
            top5_weight = sum(position_weights[:5])
            if top5_weight > self.rules.max_concentration_top5:
                violations.append(ComplianceRule(
                    name="Top 5 Concentration",
                    violation_type=ViolationType.CONCENTRATION_EXCEEDED,
                    description="Top 5 positions exceed concentration limit",
                    threshold_value=self.rules.max_concentration_top5,
                    current_value=top5_weight,
                    status=ComplianceStatus.WARNING,
                    recommendation="Determine concentration by adding smaller positions",
                    severity="medium"
                ))

        # Check top 10 concentration
        if len(position_weights) >= 10:
            top10_weight = sum(position_weights[:10])
            if top10_weight > self.rules.max_concentration_top10:
                violations.append(ComplianceRule(
                    name="Top 10 Concentration",
                    violation_type=ViolationType.CONCENTRATION_EXCEEDED,
                    description="Top 10 positions exceed concentration limit",
                    threshold_value=self.rules.max_concentration_top10,
                    current_value=top10_weight,
                    status=ComplianceStatus.WARNING,
                    recommendation="Determine by adding more positions to portfolio",
                    severity="low"
                ))

        return violations

    def _check_investment_box_compliance(self, portfolio: Portfolio) -> List[ComplianceRule]:
        """Check investment box compliance."""
        violations = []

        # This method is now significantly enhanced to use the StockClassifier
        if not self.stock_classifier:
            logger.debug("StockClassifier not provided, skipping detailed box compliance check.")
            return violations

        if not portfolio.positions or portfolio.total_value == 0:
            return violations

        # 1. Classify all stocks in the portfolio
        symbols = [p.symbol for p in portfolio.positions.values() if p.quantity > 0]
        try:
            # Note: This requires price_data, which is not available here.
            # This check will need to be refactored to either receive classified
            # positions or have access to the necessary data provider.
            # For now, we simulate classification based on position metadata if available.
            
            # Placeholder for aggregated weights
            aggregated_weights = {
                'sector': {},
                'region': {},
                'style': {},
                'size': {}
            }

            for position in portfolio.positions.values():
                if position.quantity <= 0:
                    continue
                
                weight = (position.quantity * position.current_price) / portfolio.total_value
                
                # In a real scenario, we'd use stock_classifier here.
                # We simulate by assuming positions have this metadata.
                cls_info = getattr(position, 'classification', {})
                if not cls_info:
                    # If you run this, it will likely use this path.
                    # The orchestrator needs to be updated to pass classified positions.
                    continue 

                for dim in aggregated_weights.keys():
                    if dim_value := cls_info.get(dim):
                        current_weight = aggregated_weights[dim].get(dim_value, 0)
                        aggregated_weights[dim][dim_value] = current_weight + weight

            # 2. Check deviation against configured limits
            # This part assumes a benchmark is either provided or implicitly defined (e.g., equal weight)
            for rule in self.rules.box_deviation_limits:
                dimension = rule.get('dimension')
                max_dev = rule.get('max_deviation')
                
                if not dimension or not max_dev or dimension not in aggregated_weights:
                    continue

                # Here we would compare against a benchmark's weights.
                # For simplicity, let's assume the check is against any single box
                # in a dimension exceeding a total allocation limit, which is a simpler rule.
                # Example rule: "max_sector_allocation: 0.25" is now handled here.
                
                for box_name, weight in aggregated_weights[dimension].items():
                     # This check is a simplification. A proper implementation
                     # would compare deviation from a benchmark's weight for that box.
                    if weight > max_dev:
                        violations.append(ComplianceRule(
                            name=f"Box Deviation - {dimension.capitalize()}: {box_name}",
                            violation_type=ViolationType.INVESTMENT_BOX_VIOLATION,
                            description=f"{dimension.capitalize()} '{box_name}' allocation of {weight:.1%} "
                                        f"exceeds the maximum deviation/limit of {max_dev:.1%}.",
                            threshold_value=max_dev,
                            current_value=weight,
                            status=ComplianceStatus.WARNING,
                            recommendation=f"Reduce exposure to {dimension.capitalize()} '{box_name}'.",
                            severity="medium"
                        ))

        except Exception as e:
            logger.error(f"Error during investment box compliance check: {e}", exc_info=True)
            # Add a violation to indicate the check failed
            violations.append(ComplianceRule(
                name="Box Compliance Check Failure",
                violation_type=ViolationType.INVESTMENT_BOX_VIOLATION,
                description=f"The box compliance check failed to execute: {e}",
                threshold_value=0, current_value=1, status=ComplianceStatus.CRITICAL,
                recommendation="Review system logs for compliance module.",
                severity="critical"
            ))
            
        return violations

    def _check_risk_budget_compliance(self, portfolio: Portfolio,
                                   benchmark_data: Optional[Any] = None) -> List[ComplianceRule]:
        """Check risk budget compliance."""
        violations = []

        if not hasattr(portfolio, 'volatility') or not hasattr(portfolio, 'max_drawdown'):
            return violations

        # Check volatility
        if hasattr(portfolio, 'volatility') and portfolio.volatility > self.rules.max_portfolio_volatility:
            violations.append(ComplianceRule(
                name="Portfolio Volatility",
                violation_type=ViolationType.RISK_BUDGET_EXCEEDED,
                description="Portfolio volatility exceeds risk budget",
                threshold_value=self.rules.max_portfolio_volatility,
                current_value=portfolio.volatility,
                status=ComplianceStatus.WARNING,
                recommendation="Consider reducing risk through diversification or hedging",
                severity="medium"
            ))

        # Check drawdown
        if hasattr(portfolio, 'max_drawdown') and abs(portfolio.max_drawdown) > self.rules.max_drawdown:
            violations.append(ComplianceRule(
                name="Maximum Drawdown",
                violation_type=ViolationType.RISK_BUDGET_EXCEEDED,
                description="Maximum drawdown exceeds limit",
                threshold_value=self.rules.max_drawdown,
                current_value=abs(portfolio.max_drawdown),
                status=ComplianceStatus.CRITICAL,
                recommendation="Immediate risk reduction required - consider stop-loss measures",
                severity="critical"
            ))

        return violations

    def _check_position_size_compliance(self, portfolio: Portfolio) -> List[ComplianceRule]:
        """Check individual position size compliance."""
        violations = []

        if not portfolio.positions or portfolio.total_value == 0:
            return violations

        for symbol, position in portfolio.positions.items():
            if position.quantity <= 0:
                continue

            position_weight = (position.quantity * position.current_price) / portfolio.total_value

            if position_weight > self.rules.max_single_position_weight:
                violations.append(ComplianceRule(
                    name=f"Position Size - {symbol}",
                    violation_type=ViolationType.POSITION_SIZE_EXCEEDED,
                    description=f"Position {symbol} exceeds maximum size",
                    threshold_value=self.rules.max_single_position_weight,
                    current_value=position_weight,
                    status=ComplianceStatus.WARNING,
                    recommendation=f"Reduce {symbol} position size",
                    severity="medium"
                ))

        return violations

    def _check_liquidity_compliance(self, portfolio: Portfolio) -> List[ComplianceRule]:
        """Check portfolio liquidity compliance."""
        violations = []

        # Simplified liquidity check - in practice would use actual liquidity metrics
        if not portfolio.positions:
            return violations

        # Count positions in liquid assets (simplified - assuming major ETFs are liquid)
        liquid_positions = 0
        total_positions = 0

        for symbol, position in portfolio.positions.items():
            if position.quantity > 0:
                total_positions += 1
                # Simplified liquidity check
                if symbol in ['SPY', 'QQQ', 'IWM', 'EFA', 'VTI', 'VOO']:  # Major liquid ETFs
                    liquid_positions += 1

        if total_positions > 0:
            liquidity_ratio = liquid_positions / total_positions
            if liquidity_ratio < self.rules.min_liquidity_score:
                violations.append(ComplianceRule(
                    name="Portfolio Liquidity",
                    violation_type=ViolationType.INVESTMENT_BOX_VIOLATION,
                    description="Portfolio liquidity below minimum threshold",
                    threshold_value=self.rules.min_liquidity_score,
                    current_value=liquidity_ratio,
                    status=ComplianceStatus.WARNING,
                    recommendation="Increase allocation to liquid assets",
                    severity="medium"
                ))

        return violations

    def _generate_recommendations(self, violations: List[ComplianceRule],
                                warnings: List[ComplianceRule],
                                portfolio: Portfolio) -> List[str]:
        """Generate remediation recommendations."""
        recommendations = []

        # Add recommendations from violations and warnings
        all_issues = violations + warnings
        for issue in all_issues:
            if issue.recommendation not in recommendations:
                recommendations.append(issue.recommendation)

        # Add general recommendations based on overall situation
        if len(violations) > self.notification_thresholds['violation_count_critical']:
            recommendations.append("CRITICAL: Multiple compliance violations - immediate remediation required")
        elif len(violations) > self.notification_thresholds['violation_count_warning']:
            recommendations.append("WARNING: Multiple compliance violations - review and address promptly")

        # Add rebalancing recommendation if needed
        if len(violations) > 0:
            recommendations.append("Consider portfolio rebalancing to address compliance issues")

        return recommendations

    def _determine_overall_status(self, violations: List[ComplianceRule],
                                warnings: List[ComplianceRule]) -> ComplianceStatus:
        """Determine overall compliance status."""
        if any(rule.status == ComplianceStatus.CRITICAL for rule in violations):
            return ComplianceStatus.CRITICAL
        elif len(violations) > 0:
            return ComplianceStatus.VIOLATION
        elif len(warnings) > self.notification_thresholds['violation_count_warning']:
            return ComplianceStatus.WARNING
        else:
            return ComplianceStatus.COMPLIANT

    def _create_portfolio_summary(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Create portfolio summary for compliance report."""
        return {
            'total_value': getattr(portfolio, 'total_value', 0),
            'position_count': len([p for p in portfolio.positions.values() if p.quantity > 0]),
            'cash_balance': getattr(portfolio, 'cash_balance', 0),
            'daily_return': getattr(portfolio, 'daily_return', 0),
            'total_return': getattr(portfolio, 'total_return', 0),
            'max_drawdown': getattr(portfolio, 'max_drawdown', 0),
            'volatility': getattr(portfolio, 'volatility', 0),
            'last_updated': datetime.now()
        }

    def _update_statistics(self, report: ComplianceReport) -> None:
        """Update compliance statistics."""
        self.compliance_stats['total_checks'] += 1
        self.compliance_stats['violations_detected'] += report.total_violations
        self.compliance_stats['critical_violations'] += len([v for v in report.violations if v.status == ComplianceStatus.CRITICAL])
        self.compliance_stats['warnings_generated'] += len(report.warnings)

        # Record violations in history
        for violation in report.violations:
            self.violation_history.append({
                'timestamp': report.timestamp,
                'violation_type': violation.violation_type.value,
                'severity': violation.severity,
                'description': violation.description
            })

        # Keep history manageable
        if len(self.violation_history) > 500:
            self.violation_history = self.violation_history[-500:]

    def get_compliance_summary(self, days: int = 30) -> Dict[str, Any]:
        """Get compliance summary for recent period."""
        cutoff_date = datetime.now() - timedelta(days=days)
        recent_reports = [r for r in self.compliance_history if r.timestamp >= cutoff_date]

        if not recent_reports:
            return {
                'period_days': days,
                'total_checks': 0,
                'compliance_rate': 0,
                'most_common_violations': [],
                'trend': 'insufficient_data'
            }

        total_checks = len(recent_reports)
        compliant_checks = len([r for r in recent_reports if r.is_compliant])
        compliance_rate = compliant_checks / total_checks

        # Count violation types
        violation_counts = {}
        for report in recent_reports:
            for violation in report.violations:
                violation_type = violation.violation_type.value
                violation_counts[violation_type] = violation_counts.get(violation_type, 0) + 1

        # Sort by frequency
        most_common_violations = sorted(violation_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # Determine trend
        if len(recent_reports) >= 10:
            recent_half = recent_reports[-len(recent_reports)//2:]
            earlier_half = recent_reports[:len(recent_reports)//2]

            recent_compliance = len([r for r in recent_half if r.is_compliant]) / len(recent_half)
            earlier_compliance = len([r for r in earlier_half if r.is_compliant]) / len(earlier_half)

            if recent_compliance > earlier_compliance + 0.1:
                trend = 'improving'
            elif recent_compliance < earlier_compliance - 0.1:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'

        return {
            'period_days': days,
            'total_checks': total_checks,
            'compliant_checks': compliant_checks,
            'compliance_rate': compliance_rate,
            'most_common_violations': most_common_violations,
            'trend': trend,
            'stats': self.compliance_stats.copy()
        }

    def validate_rules(self) -> Tuple[bool, List[str]]:
        """Validate compliance rules for consistency."""
        issues = []

        # Check strategy allocation bounds
        min_total = 0.0
        max_total = 0.0
        
        for strategy_rule in self.rules.strategy_allocation_rules:
            # Check individual strategy bounds
            if strategy_rule.min_weight >= strategy_rule.max_weight:
                issues.append(
                    f"Strategy '{strategy_rule.strategy_name}' allocation bounds are invalid: "
                    f"min={strategy_rule.min_weight:.1%} >= max={strategy_rule.max_weight:.1%}"
                )
            
            if strategy_rule.min_weight < 0 or strategy_rule.max_weight > 1:
                issues.append(
                    f"Strategy '{strategy_rule.strategy_name}' allocation bounds out of range [0, 1]"
                )
            
            min_total += strategy_rule.min_weight
            max_total += strategy_rule.max_weight

        # Check total allocation makes sense
        if min_total > 0.9:  # Allow for cash
            issues.append(
                f"Minimum allocation requirements ({min_total:.1%}) leave insufficient buffer for cash"
            )

        if max_total < 0.6:
            issues.append(
                f"Maximum allocation requirements ({max_total:.1%}) too restrictive"
            )

        # Check risk limits
        if self.rules.max_var_95 >= self.rules.max_drawdown:
            issues.append("VaR limit should be less than maximum drawdown limit")

        # Check concentration limits
        if self.rules.max_concentration_top5 <= self.rules.max_single_position_weight:
            issues.append("Top 5 concentration limit should be higher than single position limit")

        return len(issues) == 0, issues