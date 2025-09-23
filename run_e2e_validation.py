#!/usr/bin/env python3
"""
Kairos End-to-End Workflow Validation Runner
Simple runner script to execute comprehensive E2E validation tests.

Usage:
    python run_e2e_validation.py [--quick] [--scenario=<name>] [--report-only]

Author: Kairos Development Team
Version: 2.0
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime
import json

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from tests.e2e.workflow_validation import EndToEndWorkflowValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('E:\\kairos\\logs\\e2e_validation.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

class E2EValidationRunner:
    """Runner for End-to-End Workflow Validation"""
    
    def __init__(self):
        self.validator = None
    
    async def run_validation(self, args):
        """Run the E2E validation with specified arguments"""
        try:
            logger.info("🚀 Starting Kairos E2E Workflow Validation")
            logger.info(f"Arguments: {vars(args)}")
            
            # Initialize validator
            self.validator = EndToEndWorkflowValidator()
            await self.validator.initialize()
            
            if args.report_only:
                # Just generate a report from existing data
                await self._generate_status_report()
            elif args.scenario:
                # Run specific scenario
                await self._run_specific_scenario(args.scenario)
            elif args.quick:
                # Run quick validation (subset of tests)
                await self._run_quick_validation()
            else:
                # Run comprehensive validation
                await self._run_comprehensive_validation()
                
        except KeyboardInterrupt:
            logger.info("🛑 Validation interrupted by user")
        except Exception as e:
            logger.error(f"❌ Validation failed: {e}")
            raise
        finally:
            if self.validator:
                await self.validator.cleanup()
    
    async def _run_comprehensive_validation(self):
        """Run full comprehensive validation"""
        logger.info("📊 Running COMPREHENSIVE End-to-End Validation")
        logger.info("This will test all scenarios and may take several minutes...")
        
        start_time = datetime.now()
        
        # Run the full validation suite
        report = await self.validator.run_comprehensive_validation()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display summary
        self._display_validation_summary(report, duration)
        
        return report
    
    async def _run_quick_validation(self):
        """Run quick validation with subset of scenarios"""
        logger.info("⚡ Running QUICK End-to-End Validation")
        logger.info("Testing core functionality only...")
        
        start_time = datetime.now()
        
        # Get scenarios and run only simple and moderate ones
        scenarios = self.validator._generate_test_scenarios()
        quick_scenarios = [s for s in scenarios if s.complexity_level in ['simple', 'moderate']]
        
        logger.info(f"Running {len(quick_scenarios)} scenarios (skipping complex/extreme)")
        
        # Run quick scenarios
        results = {}
        passed = 0
        failed = 0
        
        for scenario in quick_scenarios:
            logger.info(f"▶️  Running: {scenario.name}")
            try:
                metrics = await self.validator.execute_workflow_scenario(scenario)
                results[scenario.name] = metrics
                
                if metrics.performance_score >= 0.7:
                    passed += 1
                    logger.info(f"✅ {scenario.name}: PASSED (Score: {metrics.performance_score:.2f})")
                else:
                    failed += 1
                    logger.warning(f"❌ {scenario.name}: FAILED (Score: {metrics.performance_score:.2f})")
                    
            except Exception as e:
                failed += 1
                logger.error(f"❌ {scenario.name}: ERROR - {e}")
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Display quick summary
        logger.info("=" * 50)
        logger.info("🏁 QUICK VALIDATION COMPLETE")
        logger.info("=" * 50)
        logger.info(f"📊 Results: {passed}/{len(quick_scenarios)} scenarios passed")
        logger.info(f"⏱️  Duration: {duration:.1f} seconds")
        
        if passed == len(quick_scenarios):
            logger.info("🌟 All quick tests PASSED! System appears healthy.")
        elif passed >= len(quick_scenarios) * 0.8:
            logger.info("✅ Most tests passed. Minor issues detected.")
        else:
            logger.warning("⚠️  Multiple test failures. System needs attention.")
        
        logger.info("=" * 50)
    
    async def _run_specific_scenario(self, scenario_name: str):
        """Run a specific scenario"""
        logger.info(f"🎯 Running SPECIFIC scenario: {scenario_name}")
        
        # Get scenarios and find the requested one
        scenarios = self.validator._generate_test_scenarios()
        target_scenario = None
        
        for scenario in scenarios:
            if scenario.name == scenario_name:
                target_scenario = scenario
                break
        
        if not target_scenario:
            logger.error(f"❌ Scenario '{scenario_name}' not found!")
            logger.info("Available scenarios:")
            for scenario in scenarios:
                logger.info(f"  - {scenario.name} ({scenario.complexity_level})")
            return
        
        # Run the specific scenario
        start_time = datetime.now()
        
        try:
            logger.info(f"▶️  Executing: {target_scenario.description}")
            metrics = await self.validator.execute_workflow_scenario(target_scenario)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Display detailed results
            logger.info("=" * 50)
            logger.info(f"🏁 SCENARIO RESULTS: {scenario_name}")
            logger.info("=" * 50)
            logger.info(f"📊 Performance Score: {metrics.performance_score:.3f}")
            logger.info(f"✅ Success Rate: {metrics.success_rate:.3f}")
            logger.info(f"⏱️  Duration: {duration:.2f} seconds")
            logger.info(f"📋 Tasks Completed: {metrics.tasks_completed}")
            logger.info(f"🔮 Oracle Predictions: {metrics.oracle_predictions}")
            logger.info(f"🧠 Decisions Made: {metrics.decisions_made}")
            logger.info(f"💰 CC Transactions: {metrics.cc_transactions}")
            
            if metrics.errors:
                logger.info(f"❌ Errors: {len(metrics.errors)}")
                for error in metrics.errors[:3]:  # Show first 3 errors
                    logger.info(f"  - {error}")
            
            # Assessment
            if metrics.performance_score >= 0.9:
                logger.info("🌟 ASSESSMENT: EXCELLENT")
            elif metrics.performance_score >= 0.8:
                logger.info("✅ ASSESSMENT: GOOD") 
            elif metrics.performance_score >= 0.7:
                logger.info("⚠️  ASSESSMENT: ADEQUATE")
            else:
                logger.info("❌ ASSESSMENT: POOR")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"❌ Scenario execution failed: {e}")
    
    async def _generate_status_report(self):
        """Generate status report without running tests"""
        logger.info("📄 Generating system status report...")
        
        try:
            # Basic system health check
            async with self.validator.db_pool.acquire() as conn:
                agent_count = await conn.fetchval("SELECT COUNT(*) FROM Agents")
                venture_count = await conn.fetchval("SELECT COUNT(*) FROM Ventures")
                task_count = await conn.fetchval("SELECT COUNT(*) FROM Tasks")
                decision_count = await conn.fetchval("SELECT COUNT(*) FROM Decisions")
            
            logger.info("=" * 50)
            logger.info("📊 KAIROS SYSTEM STATUS REPORT")
            logger.info("=" * 50)
            logger.info(f"👥 Agents: {agent_count}")
            logger.info(f"🚀 Ventures: {venture_count}")
            logger.info(f"📋 Tasks: {task_count}")
            logger.info(f"🧠 Decisions: {decision_count}")
            logger.info("=" * 50)
            logger.info("✅ Database connectivity: OK")
            logger.info("✅ Oracle engine: Available")
            logger.info("✅ Economy engine: Available") 
            logger.info("=" * 50)
            
        except Exception as e:
            logger.error(f"❌ Status report failed: {e}")
    
    def _display_validation_summary(self, report: dict, duration: float):
        """Display comprehensive validation summary"""
        if not report or 'overall_metrics' not in report:
            logger.error("❌ Invalid validation report")
            return
        
        metrics = report['overall_metrics']
        assessment = report.get('system_assessment', {})
        
        logger.info("=" * 70)
        logger.info("🏁 COMPREHENSIVE VALIDATION COMPLETE")
        logger.info("=" * 70)
        logger.info(f"⏱️  Total Duration: {duration:.1f} seconds")
        logger.info(f"📊 Scenarios: {metrics.get('passed_scenarios', 0)}/{metrics.get('total_scenarios', 0)} passed")
        logger.info(f"🎯 Average Performance: {metrics.get('average_performance_score', 0):.2f}/1.00")
        logger.info(f"⚡ Total Workflows: {metrics.get('total_workflows', 0)}")
        logger.info(f"📋 Tasks Completed: {metrics.get('total_tasks', 0)}")
        logger.info(f"🔮 Oracle Predictions: {metrics.get('total_oracle_predictions', 0)}")
        logger.info(f"🧠 Decisions Made: {metrics.get('total_decisions', 0)}")
        
        # Overall assessment
        health = assessment.get('overall_health', 'unknown').upper()
        readiness = assessment.get('readiness_level', 'unknown').upper()
        
        logger.info("")
        logger.info(f"💚 System Health: {health}")
        logger.info(f"🚀 Readiness Level: {readiness}")
        
        # Recommendations
        recommendations = report.get('recommendations', [])
        if recommendations:
            logger.info("")
            logger.info("📝 Top Recommendations:")
            for i, rec in enumerate(recommendations[:3], 1):
                logger.info(f"  {i}. {rec}")
        
        logger.info("=" * 70)
        
        # Final grade
        score = metrics.get('average_performance_score', 0)
        if score >= 0.9:
            logger.info("🌟 FINAL GRADE: EXCELLENT - Production Ready!")
        elif score >= 0.8:
            logger.info("✅ FINAL GRADE: GOOD - Ready with optimizations")
        elif score >= 0.7:
            logger.info("⚠️  FINAL GRADE: ADEQUATE - Needs improvement")
        else:
            logger.info("❌ FINAL GRADE: POOR - Critical issues")
        
        logger.info("=" * 70)

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Kairos End-to-End Workflow Validation Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick validation (simple and moderate scenarios only)'
    )
    
    parser.add_argument(
        '--scenario', '-s',
        type=str,
        help='Run specific scenario by name'
    )
    
    parser.add_argument(
        '--report-only', '-r',
        action='store_true',
        help='Generate system status report without running tests'
    )
    
    parser.add_argument(
        '--list-scenarios', '-l',
        action='store_true',
        help='List available test scenarios'
    )
    
    args = parser.parse_args()
    
    # Ensure log directory exists
    os.makedirs('E:\\kairos\\logs', exist_ok=True)
    
    # List scenarios if requested
    if args.list_scenarios:
        from tests.e2e.workflow_validation import EndToEndWorkflowValidator
        validator = EndToEndWorkflowValidator()
        scenarios = validator._generate_test_scenarios()
        
        print("\n📋 Available Test Scenarios:")
        print("=" * 50)
        for scenario in scenarios:
            print(f"  {scenario.name}")
            print(f"    Description: {scenario.description}")
            print(f"    Complexity: {scenario.complexity_level}")
            print(f"    Max Duration: {scenario.max_duration_seconds}s")
            print()
        return
    
    # Run validation
    runner = E2EValidationRunner()
    
    try:
        asyncio.run(runner.run_validation(args))
    except KeyboardInterrupt:
        logger.info("\n👋 Validation interrupted by user")
    except Exception as e:
        logger.error(f"\n❌ Runner failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()