# <a name="_ay9l4qpjpuei"></a>**Housing Price and Feature Prediction with Multi-Output Random Forest**
This repository contains Python scripts for training and making predictions with multi-output Random Forest models. The project focuses on predicting house features like beds and baths based on price and square footage, and conversely, predicting price based on features. Confidence intervals are included in the predictions to provide an estimate of prediction reliability.
## <a name="_x527vapsu0u0"></a>**Contents**
- **train\_multi\_output.py**: Script to preprocess the data, train multi-output Random Forest models, and save models for future predictions.
  - **Model 1**: Predicts Beds and Baths based on Price and Sqft.
  - **Model 2**: Predicts Price based on Beds, Baths, and Sqft.
- **predict\_price.py**: Loads a saved model and scaler to predict house prices from new data, with confidence intervals.
- **predict\_beds\_baths.py**: Uses a saved model and scaler to predict Beds and Baths for new houses based on price and square footage, with confidence intervals.
## <a name="_ihdmu6igdh9o"></a>**Prerequisites**
- Python 3.x
- Required packages: pandas, scikit-learn, joblib, numpy

Install dependencies with:

```bash pip install pandas scikit-learn joblib numpy ```
## <a name="_dwr5c78dlsk3"></a>**Data Format**
Input data should be in CSV format. Ensure:

- Columns for Price, Sqft, Beds, and Baths are available in the file.
- Non-numeric characters are removed from Price.
- Rows with missing values are handled by the script.

Example data format:

|**Price**|**Sqft**|**Beds**|**Baths**|**Features**|**Agent**|
| :-: | :-: | :-: | :-: | :-: | :-: |
|$300,000|2000|3|2|...|...|
## <a name="_2fyxh3uv8eun"></a>**Usage**
1. **Training Models**:
   1. Use train\_multi\_output.py to load and preprocess the data, train both models, and save them with scalers.
   1. Edit file\_path in the script to point to your CSV file.
   1. Models and scalers are saved as .pkl files for reuse.
1. Run: ```bash python train\_multi\_output.py ```
1. **Predicting Price**:
   1. Use predict\_price.py with new data containing Beds, Baths, and Sqft.
   1. The script loads the saved model and scaler, makes predictions, and calculates confidence intervals.
1. Run: ```bash python predict\_price.py ```
1. **Predicting Beds and Baths**:
   1. Use predict\_beds\_baths.py with new data containing Price and Sqft.
   1. This script loads the saved model and scaler, makes predictions, and includes confidence intervals.
1. Run: ```bash python predict\_beds\_baths.py ```
## **Outputs**
- **Predicted Values**: The scripts output predicted values for price, beds, and baths.
- **Confidence Intervals**: Each prediction includes a standard deviation-based confidence interval.
## **License**
This project is licensed under the MIT License. See LICENSE for more details.
## **Contact**
For questions or suggestions, please reach out via[ ](https://consultingjoe.com)[ConsultingJoe.com](https://consultingjoe.com).

