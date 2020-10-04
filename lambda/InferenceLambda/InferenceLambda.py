import greengrasssdk
from pandas import read_csv
import json
from numpy import concatenate
from joblib import load
from keras.models import model_from_json
from pandas import DataFrame
from pandas import concat
import time

# Creating a greengrass core sdk client
client = greengrasssdk.client('iot-data')
data_dir = '/src/'
model_dir = '/dest'

def read_models():
    scaler = load('{}/scaler.model'.format(model_dir ))
    json_file = open('{}/model.json'.format(model_dir), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights('{}/model.h5'.format(model_dir))
    loaded_model.compile(loss='mse', optimizer='adam')
    return scaler, loaded_model

def transform_to_supervised_series(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    #take the last n_in rows and create input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    #take the next n_out rows as forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # Merge all these rows together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows for NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def transform_input(values, scaler):
    values = values.astype('float32')
    scaled = scaler.transform(values)
    reframed = transform_to_supervised_series(scaled, 1, 1)
    reframed.drop(reframed.columns[[9,10,11,12,13,14,15]], axis=1, inplace=True)
    return reframed.values

def read_test_data():
    dataset = read_csv(data_dir + 'pollution.csv', header=0)
    n_test_hours = 4*365*24
    test_dataset = dataset[n_test_hours:]
    test_dataset.dropna(inplace=True)
    return test_dataset

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

def infer_data(test_X, test_y, model, test_data, scaler):
    for i in range(test_X.shape[0]-1):
        curr_test_X = test_X[i:i+1,:,:]
        curr_test_y = test_y[i:i+1]
        curr_y_hat = model.predict(curr_test_X)
        curr_test_X = curr_test_X.reshape((curr_test_X.shape[0], curr_test_X.shape[2]))
        curr_inv_yhat = concatenate((curr_y_hat, curr_test_X[:, 1:]), axis=1)
        curr_inv_yhat = scaler.inverse_transform(curr_inv_yhat)
        curr_inv_yhat = curr_inv_yhat[:,0]
        actual_y = test_data.iloc[i+1]['pollution']
        publishToIotCloud(test_data[i:i+1], curr_inv_yhat[0], actual_y)
        time.sleep(1)

def lambda_handler(event, context):
    test_data = read_test_data()
    indexed_test_Data = test_data.copy().set_index('date')
    scaler, lstm_model = read_models()
    transform_values = transform_input(indexed_test_Data.values, scaler)
    test_X, test_y = transform_values[:, :-1], transform_values[:, -1]
    test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    infer_data(test_X, test_y, lstm_model, test_data, scaler)

    return