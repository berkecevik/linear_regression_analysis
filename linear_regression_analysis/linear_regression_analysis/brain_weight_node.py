#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os
from ament_index_python.packages import get_package_share_directory

class BrainWeightAnalysisNode(Node):
    def __init__(self):
        super().__init__('brain_weight_analysis_node')
        self.get_logger().info('Brain Weight Analysis Node Started')
        
        # Publisher for results
        self.result_publisher = self.create_publisher(String, 'brain_weight_results', 10)
        
        # Timer to run analysis
        self.timer = self.create_timer(2.0, self.run_analysis)
        self.analysis_done = False

    def run_analysis(self):
        if self.analysis_done:
            return
            
        try:
            self.get_logger().info('Starting Brain Weight-Head Size Linear Regression Analysis...')
            
            # Get package data directory
            package_share_directory = get_package_share_directory('linear_regression_analysis')
            data_path = os.path.join(package_share_directory, 'data', 'HumanBrain_WeightandHead_size.csv')
            
            # Load the dataset
            df = pd.read_csv(data_path)
            self.get_logger().info(f'Dataset loaded with {len(df)} records')
            
            # Display basic info
            self.get_logger().info('Dataset head:')
            self.get_logger().info(str(df.head()))
            
            # Separate features and labels
            X = df[['Head Size(cm^3)']]  # Feature (must be 2D)
            y = df['Brain Weight(grams)']  # Label
            
            # Split into train/test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluate
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results = f"""
Brain Weight-Head Size Linear Regression Results:
===============================================
Mean Absolute Error (MAE): {mae:.2f}
R-squared (R² Score): {r2:.4f}
Model Coefficient: {model.coef_[0]:.6f}
Model Intercept: {model.intercept_:.4f}
Training samples: {len(X_train)}
Test samples: {len(X_test)}
            """
            
            self.get_logger().info(results)
            
            # Publish results
            msg = String()
            msg.data = results
            self.result_publisher.publish(msg)
            
            # Save plot
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test, y_test, color='green', alpha=0.7, label='Actual', s=30)
            
            # Sort for smooth plotting
            X_test_sorted = X_test.sort_values('Head Size(cm^3)')
            y_pred_sorted = model.predict(X_test_sorted)

            #Plot
            plt.plot(X_test_sorted.values.flatten(), y_pred_sorted, color='red', linewidth=2, label='Predicted Line')

            
            plt.xlabel("Head Size (cm³)")
            plt.ylabel("Brain Weight (grams)")
            plt.title("ROS2 Linear Regression: Head Size vs Brain Weight")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot in results directory
            results_dir = os.path.join(os.getcwd(), 'ros2_results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, 'brain_weight_regression.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.get_logger().info(f'Plot saved to: {plot_path}')
            self.get_logger().info('Brain Weight Analysis Completed Successfully!')
            
            self.analysis_done = True
            
        except Exception as e:
            self.get_logger().error(f'Error in brain weight analysis: {str(e)}')
            self.analysis_done = True

def main(args=None):
    rclpy.init(args=args)
    node = BrainWeightAnalysisNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
