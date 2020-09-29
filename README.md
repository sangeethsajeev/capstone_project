# Predictive Maintenance using IoT Data

## Description
Using Machine Learning and Internet of Things to predict the pollution level at a time, given pollution level of last N timesteps.
We will simulate an IoT device using AWS Greengrass and collect streaming device data into IoT core. With pre configured rules, those reading will be pushed to IoT Analytics.
When we have sufficient data, then that data is pulled into AWS Sagemaker. Sagemaker will train a model and save artefacts to AWS S3.
Using lambda at edge (Greengrass) we will predict the pollution level for next hour given we have data for last N hours. 

## Dataset

Dataset is taken from the [Beijing PM2.5](https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data)
This dataset contains the hourly data for PM2.5 levels at US Embassy in Beijing between Jan 1st, 2010 to Dec 31st, 2014.

## Basic EDA steps 

1. Handling Missing Values - Drop as count of those is less than 5%. Also majority of missing is in Target fetures.

## Feature Engineering

1. Encoding Categorical value - Label Encoder
2. Data Normalization - Min-Max Scaler
3. Converting sequence data to timeseries data
4. Reshaping dataset - (Sample, Time-steps, Features)

## Model
As we will be forecasting the pollution level based on last N timsteps, LSTM Network seem fit for the problem. Working of LSTM Model is described in here [LSTM in Simple Words](https://medium.com/the-innovation/lstm-introduction-in-simple-words-fe544a45f1e7).Long Short Term Model is capable of retaning information like seasonal and periodical varaition in data along with the regression attributes. We have explored variations of the model before finalizing the current model.

Results on different variations in the model.

| Model     | Layers  | Neurons |  Activation | RMSE  |
| ----------|:-------:| -------:| -----------:| -----:|
| LSTM      | 1       | 50      |  tanh       |  32   |
| LSTM      | 1       | 50      |  relu       |  29   |
| LSTM      | 1       | 50      |  selu       |  27   |
| LSTM      | 3       | 50      |  selu       |  25   |
| BiLSTM    | 1       | 64      |  selu       |  27   |
| BiLSTM    | 3       | 64      |  relu       |  23   |
| BiLSTM    | 3       | 64      |  selu       |  20   |

Other parameters like epoch (20-100), optimizer(adam, sgd), learning rate(0.001 and 0.01) were also explored see notebook for details.

## Model Architecture

Placeholder for LSTM model image

## Model result comparision actual vs forecasted

Placeholder for results image

## Deployment Strategy

Placeholder for Architecture Image

The solutoion is deployed on AWS and we are leveraging AWS CFN to maintain infrastructure as code. 

### CloudFormation resources being deployed are grouped according to there purpose

1. Base Service - VPC, Subnet, Ruote Table, Security Group
2. Simulation Services -  Autoscaling + LaunchConfiguration + Greengrass Core
3. IoT Services -  Rule, Subscription, Topic
4. Machine Learning Services - Sagemaker Notebook and Training Jobs
5. Auxiallry Services -  S3, IAM, Certificates etc.

### Steps for Deployment

*TODO*

Use this command to install the requirements.txt file:

``` cat requirements.txt | xargs -n 1 pip install ```

