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
import numpy as np
from ament_index_python.packages import get_package_share_directory

class BostonHousingAnalysisNode(Node):
    def __init__(self):
        super().__init__('boston_housing_analysis_node')
        self.get_logger().info('Boston Housing Analysis Node Started')
        
        # Publisher for results
        self.result_publisher = self.create_publisher(String, 'boston_housing_results', 10)
        
        # Timer to run analysis
        self.timer = self.create_timer(2.0, self.run_analysis)
        self.analysis_done = False

    def run_analysis(self):
        if self.analysis_done:
            return
            
        try:
            self.get_logger().info('Starting Boston Housing Linear Regression Analysis...')
            
            # Get package data directory
            package_share_directory = get_package_share_directory('linear_regression_analysis')
            data_path = os.path.join(package_share_directory, 'data', 'boston_housing.csv')
            
            # Add column names (Boston dataset doesn't have headers)
            column_names = [
                'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
                'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV'
            ]
            
            # Load the dataset
            df = pd.read_csv(data_path, header=None, names=column_names)
            
            # Drop rows with missing target values
            df = df.dropna(subset=["MEDV"])
            
            self.get_logger().info(f'Dataset loaded with {len(df)} records')
            
            # Show info
            self.get_logger().info('Dataset head:')
            self.get_logger().info(str(df.head()))
            
            # Separate features and label
            X = df.drop('MEDV', axis=1)
            y = df['MEDV']
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Model training
            model = LinearRegression()
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Evaluation
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Get top 5 most important features
            feature_importance = pd.DataFrame({
                'feature': X.columns,
                'coefficient': model.coef_
            })
            feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
            top_features = feature_importance.nlargest(5, 'abs_coefficient')
            
            results = f"""
Boston Housing Linear Regression Results:
========================================
Mean Absolute Error (MAE): {mae:.2f}
R-squared (R² Score): {r2:.4f}
Model Intercept: {model.intercept_:.4f}
Training samples: {len(X_train)}
Test samples: {len(X_test)}

Top 5 Most Important Features:
{top_features[['feature', 'coefficient']].to_string(index=False)}
            """
            
            self.get_logger().info(results)
            
            # Publish results
            msg = String()
            msg.data = results
            self.result_publisher.publish(msg)
            
            # Create visualizations
            results_dir = os.path.join(os.getcwd(), 'ros2_results')
            os.makedirs(results_dir, exist_ok=True)
            
            # 1. Feature importance bar chart
            plt.figure(figsize=(12, 8))
            plt.barh(X.columns, model.coef_)
            plt.xlabel("Coefficient Value")
            plt.title("ROS2 Feature Importance in Boston Housing Prediction")
            plt.tight_layout()
            plot_path1 = os.path.join(results_dir, 'boston_feature_importance.png')
            plt.savefig(plot_path1, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Actual vs Predicted scatter plot
            plt.figure(figsize=(10, 8))
            plt.scatter(y_test, y_pred, alpha=0.6, color='blue', s=30)
            plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            plt.xlabel('Actual MEDV')
            plt.ylabel('Predicted MEDV')
            plt.title('ROS2 Boston Housing: Actual vs Predicted Values')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path2 = os.path.join(results_dir, 'boston_actual_vs_predicted.png')
            plt.savefig(plot_path2, dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Residuals plot
            residuals = y_test - y_pred
            plt.figure(figsize=(10, 6))
            plt.scatter(y_pred, residuals, alpha=0.6, color='green', s=30)
            plt.axhline(y=0, color='red', linestyle='--')
            plt.xlabel('Predicted MEDV')
            plt.ylabel('Residuals')
            plt.title('ROS2 Boston Housing: Residuals Plot')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plot_path3 = os.path.join(results_dir, 'boston_residuals.png')
            plt.savefig(plot_path3, dpi=300, bbox_inches='tight')
            plt.close()
            
            self.get_logger().info(f'Plots saved to: {results_dir}')
            self.get_logger().info('Boston Housing Analysis Completed Successfully!')
            
            self.analysis_done = True
            
        except Exception as e:
            self.get_logger().error(f'Error in Boston housing analysis: {str(e)}')
            self.analysis_done = True

def main(args=None):
    rclpy.init(args=args)
    node = BostonHousingAnalysisNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
