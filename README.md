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

1. Handling Missing Values - Drop as count of those is less than 5%. Also majority of missing is in Target features.

## Feature Engineering

1. Encoding Categorical value - Label Encoder
2. Data Normalization - Min-Max Scaler
3. Converting sequence data to timeseries data
4. Reshaping dataset - (Sample, Time-steps, Features)

## Model
As we will be forecasting the pollution level based on last N timsteps, LSTM Network seem fit for the problem. Working of LSTM Model is described over here [LSTM in Simple Words](https://medium.com/the-innovation/lstm-introduction-in-simple-words-fe544a45f1e7). Long Short Term Model is capable of retaning information like seasonal and periodical variation in data along with the regression attributes. We have explored variations of the model before finalizing the current model.

Results of different variations if the model we experimented with are below:

| Model     | Layers  | Neurons |  Activation | RMSE  |
| ----------|:-------:| -------:| -----------:| -----:|
| LSTM      | 1       | 50      |  tanh       |  32   |
| LSTM      | 1       | 50      |  relu       |  29   |
| LSTM      | 1       | 50      |  selu       |  27   |
| LSTM      | 3       | 50      |  selu       |  25   |
| BiLSTM    | 1       | 64      |  selu       |  27   |
| BiLSTM    | 3       | 64      |  relu       |  23   |
| BiLSTM    | 3       | 64      |  selu       |  20   |

Other parameters like epoch (20-100), optimizer(adam, sgd), learning rate(0.001 and 0.01) were also explored (see notebook for details).

## Model Architecture

![Model Architecture](https://github.com/sangeethsajeev/capstone_project/blob/dev/images/NeuralNetworkArch.jpg)

## Model result comparision actual vs forecasted

![Model Forecast Visuals](https://github.com/sangeethsajeev/capstone_project/blob/dev/images/PredictionResults.jpg)

## Deployment Strategy

![Deployment Architecture](https://github.com/sangeethsajeev/capstone_project/blob/dev/images/Deployment.jpg)

The solutoion is deployed on AWS and we are leveraging AWS CFN to maintain infrastructure as code. 

### CloudFormation resources being deployed are grouped according to their purpose

1. Base Service - VPC, Subnet, Route Table, Security Group
2. Simulation Services -  Autoscaling + LaunchConfiguration + Greengrass Core
3. IoT Services -  Rule, Subscription, Topic
4. Machine Learning Services - Sagemaker Notebook and Training Job
5. Auxillary Services -  S3, IAM, Certificates etc.

### Steps for Deployment

#### Prerequisite

1. Configure AWS credentials
2. Create an S3 bucket where artefacts(file, model, cfn) will be uploaded for deployment

#### Deploy

1. Clone the repo 
2. From root of the app directory deploy.sh while passing the arguemnts. -h or --help can be used to see all available arguemnts
3. Switch over to AWS Console and check cloudformation. Wait for cloudformation stack to come to status of CREATE_COMPLETE.
4. Head over to AWS Console for Greengrass, Deploy the <StackName>GreengrassGroup with Automatic Detection.
5. Switch to IoT Core console. Subscribe to the pollution/data topic. 
6. Update the publish topic to pollution/data/ingest/trigger and click Publish to topic.
7. Switch over to IoTAnalytic page. Run the dataset iotanalyticssqldataset.
8. Head over to SageMaker notebook and open notebook named SagemakerNotebookInstance. Follow instruction in notebook and run it.
9. Switch over Greengrass console and click on Group.
10. Select the group named <StackName>GreengrassGroup. Add a machine learning resource.
11. While prompted give name and select Use a model trained in AWS SageMaker. Select the training job you just trained prefixed with pollution-forecasting-lstm.
12. Give local path as /dest/.
13. Select Lambda affiliation, and pick the Lambda prefixed with <StackName>-InferenceLambda. Leave the Read-only access and click Save.
14. Expand Actions and click Deploy.
15. Subscribe to three topics pollution/data, pollution/data/infer, and pollution/data/model/accuracy.
16. Publish the default message to the topic pollution/data/infer/trigger.

### Cleaning up

1. Reset deployments on Greengrass group.
2. Delete the cloudformation stack.
3. Optinally clear your S3 bucket artefacts.



