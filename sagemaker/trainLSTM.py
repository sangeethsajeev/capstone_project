import argparse
import os
import pandas
from pandas import DataFrame
from pandas import concat
from pandas import read_csv
from datetime import datetime
import numpy
from numpy import concatenate
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from joblib import dump

import warnings
import json

warnings.simplefilter(action='ignore', category=FutureWarning)

if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=72)
    parser.add_argument('--n_train_hours', type=int, default=24*365*3)
    parser.add_argument('--n_validation_hours', type=int, default=24*365*1)

    # input data and model directories
    parser.add_argument('--model_dir', type=str)

    args, _ = parser.parse_known_args()
    
    train_dataset_dir = os.environ.get('SM_INPUT_DIR') + '/data/training/' 
    output_model_dir = os.environ.get('SM_MODEL_DIR')
    output_object_dir = os.environ.get('SM_OUTPUT_DATA_DIR')
    epochs = args.epochs
    batch_size = args.batch_size
    
    #Read the input dataset
    dataset = read_csv(train_dataset_dir + 'pollution.csv', header=0)
    
    #Drop non-feature columns
    for col in ['No', 'year', 'month', 'day', 'pm2.5_new']:
        df.drop(col, axis=1, inplace=True, errors='ignore')
    df.columns = ['Hour','PM25', 'Dew','Temperature', 'Pressure', 'WindDirection', 'WindSpeed', 'Snow', 'Rain']
    df.dropna(inplace=True)
    
    labelencoder = LabelEncoder()
    df['WindDirection'] = labelencoder.fit_transform(df['WindDirection'])

    min_max_scaler = MinMaxScaler()
    for col in df.columns:
        if col == 'PM25':
            continue
        else:
            df[col] = min_max_scaler.fit_transform(df[[col]])
    
    def transform_data_many_to_one(data, columns, time_steps=1):
        n_vars = data.shape[1]
        dataset = DataFrame(data)
        cols, names = list(), list()
        for i in range(time_steps, 0, -1):
            cols.append(dataset.shift(i))
            names += [('{}(t-{})'.format(columns[j], i)) for j in range(n_vars)]
        cols.append(dataset.shift(-0))
        names += [('{}(t)'.format(columns[j])) for j in range(n_vars)]
        new_df = concat(cols, axis=1)
        new_df.columns = names
        new_df.dropna(inplace=True)
    return new_df
    
    values = df.values
    values = values.astype('float32')
    n_hours = 4 #6
    n_features = 9
    transformed_df = transform_data_many_to_one(values, df.columns, n_hours)
    transformed_df.drop(['Hour(t)', 'Dew(t)','Temperature(t)', 'Pressure(t)', 'WindDirection(t)', 'WindSpeed(t)', 'Snow(t)', 'Rain(t)'], axis=1, inplace=True)
    transformed_df.reset_index(drop=True, inplace=True)

    values = transformed_df.values
    n_train_hours = 365*24*3 # 3 for 3 years
    train = values[:n_train_hours, :]
    validate = values[n_train_hours:n_train_hours+8740, :] #kept 1 year data for validation
    test = values[n_train_hours+8740:, :] # 1 year data for test
    n_attributes = n_hours * n_features
    train_X, train_y = train[:, :n_attributes], train[:, -1]
    test_X, test_y = test[:, :n_attributes], test[:, -1]
    validate_X, validate_y = validate[:, :n_attributes], validate[:, -1]
    train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
    validate_X = validate_X.reshape((validate_X.shape[0], n_hours, n_features))
    
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='selu', return_sequences=True), input_shape=(n_hours, n_features)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, activation='selu', return_sequences=True)))
    model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation='selu')))
    model.add(tf.keras.layers.Dense(1))
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    # fit network
    history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(validate_X, validate_y), verbose=0, shuffle=False)


    #Save the history
    with open(output_model_dir + '/history.json', 'w') as f:
        json.dump(history.history, f)
        
    #Save the Scaler
    dump(min_max_scaler, output_model_dir + '/scaler.model', protocol=2) 
    
    #Save the encoder
    dump(labelencoder, output_model_dir + '/encoder.model', protocol=2) 
    
    #Save the trained model
    model_json = model.to_json()
    with open(output_model_dir + "/model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(output_model_dir + "/model.h5")