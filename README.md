# Airfare Price Prediction

## Goal
Predict airfare prices for various airlines using different machine learning models from linear regression to neural network. From using root mean squared error (RMSE) as a metric, the best model fit can be determined.

## Data
- **Airfare_Dataset.csv**: Raw data that consists flight prices for various airlines between the months of March and June of 2019. Go to the following Kaggle link where the data was retrieved for a detailed list of the features. Note, I only used the training dataset provided since it had the target column available for evaluation.
- **Dataset_Modified.csv**: Modified dataset after removing duplicate and null values along with performing feature engineering.

**Kaggle Link**: https://www.kaggle.com/code/vinayshaw/airfare-price-prediction/notebook

## Methods

### Exploratory Data Analysis (EDA)
#### Airfare Price Distribution
The airfare price distribution shows a positive skew.
![image](https://user-images.githubusercontent.com/70343375/229991176-a1241bbb-6b5f-4220-adeb-6de0a57ba7aa.png)
#### Correlation
Performing a heatmap indicates that Price, Total_Stops, and Duration are correlated. This means Total_Stops and Duration are important features to focus on for modeling. However, the other features have no correlation to Price.
![image](https://user-images.githubusercontent.com/70343375/229997340-d15d716b-9266-4ec4-9353-9556943beef6.png)
#### Total_Stops vs. Duration
After comparing Price with Total_Stops and Duration, Total_Stops does not provide many unique values compared to Duration. In this case, Duration will be focused for modeling.
![image](https://user-images.githubusercontent.com/70343375/229998206-87b7ae52-f72a-441e-9b87-b3746d3102bd.png)
![image](https://user-images.githubusercontent.com/70343375/229998472-f4f8409a-afd9-42b7-8598-0fa2ecac484c.png)

### Linear Regression
Preprocessing:
- Outliers outside a 99% confidence interval were removed (61 outliers total).
- Dataset split to 80% training (8320 instances) and 20% testing (2081 instances).

### Artificial Neural Network
Preprocessing:
- Outliers outside a 99% confidence interval were removed (61 outliers total).
- Dataset split to 80% training (8320 instances) and 20% testing (2081 instances).
- Standard features using StandardScalar().

Model Parameters using Sequential API in TensorFlow:
- Loss: Mean Squared Error (MSE)
- Optimizer: Adam
- Metric: RMSE
- Epoch: 100

## Analysis
Based on RMSE values, ANN is the better model but at the cost of interpretability.
| Model       | RMSE        |
| ----------- | ----------- |
| Linear Reg. | 4304        |
| ANN         | 3867        |

### Linear Regression
Due to the bias-variance tradeoff, the relationship between duration and price for linear regression can be easily visualized due to low bias. However, this is at the cost of high variance.
![image](https://user-images.githubusercontent.com/70343375/229999771-06bce621-670e-4a4c-a7ee-7efe59c08770.png)

### Artificial Neural Network
Based on the RMSE values, ANN is the more accurate model. However, due to the bias-variance tradeoff, the relationship between duration and price cannot be easily interpreted since ANN is more complex (high bias).
