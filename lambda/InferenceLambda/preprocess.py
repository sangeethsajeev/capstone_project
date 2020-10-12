import numpy
import pandas
from matplotlib import pyplot 
from datetime import datetime
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import pdpipe as pdp
from os import path
from joblib import load,dump

class PreProcessData(object):

  def __init__(self):
    ## These columns will be dropped from the dataframe
    self.drop_cols          = ['No', 'year', 'month', 'day']
    ## These columns will be processed further     
    self.cols               = ['hour','pm2.5','DEWP','TEMP','PRES','cbwd','Iws','Is','Ir']
    ##  These columns will be Label Encoded
    self.label_encode_cols  = ['cbwd']
    ## These columns will be excluded from MinMaxScaling
    self.scale_excCols      = ['pm2.5']
    ## This is the number of historic data we will consider before predicting the nth hour.
    self.n_hours            = 4
    ## The number of useful features in the dataset
    self.n_features         = 9
    ## Flag True for Testing Cases
    self.testFlag           = False

  ## Custom Function to load and dump previously used MinMaxScaling values
  def scaler(self,df):
    if path.exists('params/scaler.joblib'):
      scaler = load('params/scaler.joblib')
      print("MinMaxScaler from Previous Training Loaded...")
    else:
      scaler = MinMaxScaler()

    for i in self.cols:
      if i not in self.scale_excCols:
        df[i] = scaler.fit_transform(df[[i]])

    if not self.testFlag:
      print("Training Mode - Scaler Updated ...")
      dump(scaler,'params/scaler.joblib')
    else:
      print("Testing Mode - Scalers Not Updated ...")
      
    return df

  ## Custom function to load and dump previously used Encoding Schemes
  def encoder(self,df):
    if path.exists('params/encoder.joblib'):
      labelEncoder = load('params/encoder.joblib')
      print("LabelEncoder from Previous Training Loaded...")
    else:
      labelEncoder = LabelEncoder()

    for i in self.label_encode_cols:
      df[i] = labelEncoder.fit_transform(df[i])

    if not self.testFlag:
      print("Training Mode - Encoder Updated ...")
      dump(labelEncoder, 'params/encoder.joblib')
    else:
      print("Testing Mode - Encoders not Updated ...")

    return df

  ## Pipeline for Data Processing
  def data_pipeline(self,df):
    pipeline = pdp.ColDrop(self.drop_cols)
    pipeline+=pdp.DropNa()
    df = pipeline(df)

    df = self.encoder(df)
    df = self.scaler(df)

    return df

  def transform_data_many_to_one(self,data, columns, time_steps=1):
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

  def preprocess_df(self,df,testFlag):
    self.testFlag=testFlag
    df_ = self.data_pipeline(df)
    values = df_.values
    values = values.astype('float32')
    transformed_df = self.transform_data_many_to_one(values, df_.columns, self.n_hours)
    transformed_df.drop(['hour(t)', 'DEWP(t)','TEMP(t)','PRES(t)','cbwd(t)','Iws(t)','Is(t)',
      'Ir(t)'], axis=1, inplace=True)
    transformed_df.reset_index(drop=True, inplace=True)

    return transformed_df

