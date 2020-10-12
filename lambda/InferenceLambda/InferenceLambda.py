import greengrasssdk
from pandas import read_csv
import json
from numpy import concatenate
from joblib import load
from keras.models import model_from_json
from pandas import DataFrame
from pandas import concat
import time

from model import LSTM_Model

lstm = LSTM_Model()

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')
data_dir = '/src/'
model_dir = '/dest'

def read_test_data():
    dataset = read_csv(data_dir + 'pollution.csv', header=0)
    test_dataset.dropna(inplace=True)
    return dataset

def publishToIotCloud(curr_row, predicted_y, actual_y):
    data = curr_row.to_dict(orient='records')[0]
    payload = json.dumps(data)
    client.publish(
        topic='pollution/data',
        payload=payload
    )
    output = {
        'predicted_pollution': float(predicted_y),
        'actual_pollution': float(actual_y),
        'date': data['date']
    }
    output_payload = json.dumps(output)
    client.publish(
        topic='pollution/data/infer',
        payload=output_payload
    )

def infer_data(test_X, test_y, model, test_data):
    for i in range(test_X.shape[0]-1):
        curr_test_X = test_X[i:i+1,:,:]
        curr_test_y = test_y[i:i+1]
        curr_y_hat = model.predict(curr_test_X)
        actual_y = test_data.iloc[i+1]['pollution']
        publishToIotCloud(test_data[i:i+1], curr_inv_yhat[0], actual_y)
        time.sleep(1)

def lambda_handler(event, context):
    test_data, test_X, test_y = lstm.test_data_process(read_test_data())
    lstm_model = lstm.load_model()
    infer_data(test_X, test_y, lstm_model, test_data)

    return