# Forecasting of COVID-19 cases in Peru
Implementation of a LSTM neural network to forecast the number of confirmed COVID-19 cases in Peru.

## Data & Pre-processing
Data was collected from the official [Peruvian government database of confirmed COVID-19 cases](https://www.datosabiertos.gob.pe/dataset/casos-positivos-por-covid-19-ministerio-de-salud-minsa) on January 07, 2023. Data pre-processing was applied to obtain the final dataset used in the modeling.

![Confirmed COVID-19 cases in Peru](https://raw.githubusercontent.com/leonardtd/Forecasting-of-COVID-19-cases-in-Peru/main/assets/Confirmed_cases.png "Confirmed COVID-19 cases in Peru")

### Note:
- Sklearn's [Robust Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html#sklearn.preprocessing.RobustScaler) was used as the time series contained outliers and the more common alternatives (`MinMaxScaler` and `StandardScaler`) proved to be inadequate for this problem.

## Modeling
A straightforward LSTM model was implemented using the [`pytorch`](https://pytorch.org) framework and [`pytorch lightning`](https://www.pytorchlightning.ai). The model was trained using MSE criterion and NAdam optimizer with a `learning_rate` of 0.001. Furthermore, `pytorch lightning` was used to fit the model with an Early Stopping configured to monitor the test loss.  A visualization of the train/test forecast is presented below.

![Training results](https://raw.githubusercontent.com/leonardtd/Forecasting-of-COVID-19-cases-in-Peru/main/assets/Test_results.png "Training results")

## Model usage
The model was implemented to predict the number of cases of the next day looking at the data from the previous 30 days. In order to forecast for longer periods of time, an **auto regressive** approach was utilized. In this way, the predictions of the model are concatenated to the batch and the 30-day window is shifted. With the approach described above, a forecast of the next 60 days was calculated.

![Forecast of COVID-19 cases in Peru for the next 60 days (auto regressive)](https://raw.githubusercontent.com/leonardtd/Forecasting-of-COVID-19-cases-in-Peru/main/assets/AR_predictions.png "Forecast of COVID-19 cases in Peru for the next 60 days")


## Next steps
- Perform hyper parameter optimization
- Automate data collection
- Deploy the model
