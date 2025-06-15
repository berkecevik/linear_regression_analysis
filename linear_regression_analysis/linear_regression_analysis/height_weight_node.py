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

class HeightWeightAnalysisNode(Node):
    def __init__(self):
        super().__init__('height_weight_analysis_node')
        self.get_logger().info('Height Weight Analysis Node Started')
        
        # Publisher for results
        self.result_publisher = self.create_publisher(String, 'height_weight_results', 10)
        
        # Timer to run analysis
        self.timer = self.create_timer(2.0, self.run_analysis)
        self.analysis_done = False

    def run_analysis(self):
        if self.analysis_done:
            return
            
        try:
            self.get_logger().info('Starting Height-Weight Linear Regression Analysis...')
            
            # Get package data directory
            package_share_directory = get_package_share_directory('linear_regression_analysis')
            data_path = os.path.join(package_share_directory, 'data', 'new_height_weight.csv')
            
            # Load the height-weight dataset
            df = pd.read_csv(data_path)
            self.get_logger().info(f'Dataset loaded with {len(df)} records')
            
            # Display first few rows
            self.get_logger().info('Dataset head:')
            self.get_logger().info(str(df.head()))
            
            # Show basic info
            self.get_logger().info('Dataset info:')
            buffer = df.info(buf=None)
            
            # 1. Separate features and labels
            X = df[['Height']]  # Features (2D)
            y = df['Weight']    # Label (1D)
            
            # 2. Split into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # 3. Create and train the model
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # 4. Predict on test set
            y_pred = model.predict(X_test)
            
            # 5. Evaluate performance
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results = f"""
Height-Weight Linear Regression Results:
========================================
Mean Absolute Error (MAE): {mae:.2f}
R-squared (R² Score): {r2:.4f}
Model Coefficient: {model.coef_[0]:.4f}
Model Intercept: {model.intercept_:.4f}
Training samples: {len(X_train)}
Test samples: {len(X_test)}
            """
            
            self.get_logger().info(results)
            
            # Publish results
            msg = String()
            msg.data = results
            self.result_publisher.publish(msg)
            
            # 6. Visualization
            plt.figure(figsize=(10, 6))
            plt.scatter(X_test.values.flatten(), y_test, color='blue', alpha=0.5, label='Actual', s=20)

            X_sorted = X_test.sort_values(by='Height')
            y_sorted_pred = model.predict(X_sorted)

            plt.plot(X_sorted.values.flatten(), y_sorted_pred, color='red', linewidth=2, label='Predicted Line')

            plt.xlabel("Height (cm)")
            plt.ylabel("Weight (kg)")
            plt.title("ROS2 Linear Regression: Height vs Weight")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save plot in results directory
            results_dir = os.path.join(os.getcwd(), 'ros2_results')
            os.makedirs(results_dir, exist_ok=True)
            plot_path = os.path.join(results_dir, 'height_weight_regression.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.get_logger().info(f'Plot saved to: {plot_path}')
            self.get_logger().info('Height-Weight Analysis Completed Successfully!')
            
            self.analysis_done = True
            
        except Exception as e:
            self.get_logger().error(f'Error in height-weight analysis: {str(e)}')
            self.analysis_done = True

def main(args=None):
    rclpy.init(args=args)
    node = HeightWeightAnalysisNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
