#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import subprocess
import time
import os

class AnalysisLauncherNode(Node):
    def __init__(self):
        super().__init__('analysis_launcher_node')
        self.get_logger().info('Analysis Launcher Node Started')
        
        # Subscribers to collect results
        self.height_weight_sub = self.create_subscription(
            String, 'height_weight_results', self.height_weight_callback, 10)
        self.brain_weight_sub = self.create_subscription(
            String, 'brain_weight_results', self.brain_weight_callback, 10)
        self.boston_housing_sub = self.create_subscription(
            String, 'boston_housing_results', self.boston_housing_callback, 10)
        
        # Results storage
        self.results = {
            'height_weight': None,
            'brain_weight': None,
            'boston_housing': None
        }
        
        # Timer to check completion and generate report
        self.report_timer = self.create_timer(5.0, self.check_and_generate_report)
        self.report_generated = False

    def height_weight_callback(self, msg):
        self.results['height_weight'] = msg.data
        self.get_logger().info('Received Height-Weight results')

    def brain_weight_callback(self, msg):
        self.results['brain_weight'] = msg.data
        self.get_logger().info('Received Brain Weight results')

    def boston_housing_callback(self, msg):
        self.results['boston_housing'] = msg.data
        self.get_logger().info('Received Boston Housing results')

    def check_and_generate_report(self):
        if self.report_generated:
            return
            
        # Check if all results are received
        completed = sum(1 for result in self.results.values() if result is not None)
        
        if completed == 3:  # All analyses completed
            self.generate_final_report()
            self.report_generated = True
        else:
            self.get_logger().info(f'Analysis progress: {completed}/3 completed')

    def generate_final_report(self):
        try:
            self.get_logger().info('Generating Final Analysis Report...')
            
            # Create comprehensive report
            report = f"""
================================================================================
                    ROS2 LINEAR REGRESSION MULTI-DATASET ANALYSIS
                                    FINAL REPORT
================================================================================

ANALYSIS OVERVIEW:
This report presents the results of Linear Regression analysis performed on three
different datasets using ROS2 nodes. Each dataset was processed independently
by dedicated ROS2 nodes, demonstrating distributed computing capabilities.

DATASETS ANALYZED:
1. Human Height and Weight Dataset (10,000 records)
2. Human Brain Weight and Head Size Dataset (237 records)
3. Boston Housing Dataset (506 records, 14 features)

================================================================================

{self.results['height_weight']}

================================================================================

{self.results['brain_weight']}

================================================================================

{self.results['boston_housing']}

================================================================================

COMPARATIVE ANALYSIS:

Model Performance Comparison:
- The R² scores indicate how well each model explains the variance in the data
- Higher R² values (closer to 1.0) indicate better model performance
- MAE values show the average prediction error in the original units

Key Insights:
1. Height-Weight relationship shows strong linear correlation
2. Brain Weight-Head Size relationship demonstrates biological correlation
3. Boston Housing model handles multiple features with complex interactions

ROS2 Implementation Benefits:
- Modular analysis with separate nodes for each dataset
- Concurrent processing capabilities
- Message passing for result aggregation
- Scalable architecture for additional datasets

FILES GENERATED:
- height_weight_regression.png: Scatter plot with regression line
- brain_weight_regression.png: Head size vs brain weight visualization
- boston_feature_importance.png: Feature coefficient importance chart
- boston_actual_vs_predicted.png: Model prediction accuracy plot
- boston_residuals.png: Residual analysis for model diagnostics

================================================================================
                            ANALYSIS COMPLETED SUCCESSFULLY
================================================================================
            """
            
            # Save report to file
            results_dir = os.path.join(os.getcwd(), 'ros2_results')
            os.makedirs(results_dir, exist_ok=True)
            report_path = os.path.join(results_dir, 'final_analysis_report.txt')
            
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.get_logger().info(f'Final report saved to: {report_path}')
            self.get_logger().info('All analyses completed successfully!')
            
            # Print summary to console
            print("\n" + "="*80)
            print("ROS2 LINEAR REGRESSION ANALYSIS COMPLETE")
            print("="*80)
            print(f"Results and visualizations saved in: {results_dir}")
            print("Files generated:")
            print("  - final_analysis_report.txt")
            print("  - height_weight_regression.png")
            print("  - brain_weight_regression.png")
            print("  - boston_feature_importance.png")
            print("  - boston_actual_vs_predicted.png")
            print("  - boston_residuals.png")
            print("="*80)
            
        except Exception as e:
            self.get_logger().error(f'Error generating final report: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = AnalysisLauncherNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
