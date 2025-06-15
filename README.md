
# ROS2 Linear Regression Analysis 📈🤖

This ROS2 package performs linear regression analysis on three different datasets using Python, scikit-learn, and `rclpy`. It demonstrates modular data processing using ROS2 nodes, inter-node communication via topics, and visual output generation.

---

## Datasets Used

| Dataset                           | Feature(s)                     | Label                      | Records |
|----------------------------------|--------------------------------|----------------------------|---------|
| `new_height_weight.csv`          | Height (cm)                    | Weight (kg)                | 10,000  |
| `HumanBrain_WeightandHead_size.csv` | Head Size (cm³)             | Brain Weight (grams)       | 237     |
| `boston_housing.csv`             | All except `MEDV`              | `MEDV` (home price $1000s) | 506     |

---

## 📊 Outputs

All outputs will be saved under `ros2_results/`:

- `height_weight_regression.png`
- `brain_weight_regression.png`
- `boston_feature_importance.png`
- `boston_actual_vs_predicted.png`
- `boston_residuals.png`
- `final_analysis_report.txt`


## 👤 Author

Berke Çevik – [University of Europe for Applied Sciences]

---
