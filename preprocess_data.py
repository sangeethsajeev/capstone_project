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

class PreProcessData(object):

  def __init__(self):
    self.drop_cols     = ['No', 'year', 'month', 'day']
    self.cols          = ['hour','pm2.5','DEWP','TEMP','PRES','cbwd','Iws','Is','Ir']
    self.n_hours       = 4
    self.n_features    = 9

  def data_pipeline(self,df):
    pipeline = pdp.ColDrop(self.drop_cols)
    pipeline+=pdp.DropNa()
    pipeline+=pdp.Encode("cbwd")
    pipeline+=pdp.Scale('MinMaxScaler',exclude_columns=['pm2.5'])

    return pipeline(df)

  def transform_data_many_to_one(self,data, columns, time_steps=1):
    n_vars = data.shape[1]
    dataset = DataFrame(data)
    cols, names = list(), list()
    for i in range(time_steps, 0, -1):
      cols.append(dataset.shift(i))
      # names += [('{}(t-{})'.format(columns[j], i)) for j in range(n_vars)]
    cols.append(dataset.shift(-0))
    # names += [('{}(t)'.format(columns[j])) for j in range(n_vars)]
    new_df = concat(cols, axis=1)
    # new_df.columns = names
    new_df.dropna(inplace=True)
    return new_df

  def preprocess_df(self,df):
    df_ = self.data_pipeline(df)
    values = df_.values
    values = values.astype('float32')
    transformed_df = self.transform_data_many_to_one(values, df_.columns, self.n_hours)
    # transformed_df.drop(['hour(t)', 'DEWP(t)','TEMP(t)','PRES(t)','cbwd(t)','Iws(t)','Is(t)',
      # 'Ir(t)'], axis=1, inplace=True)
    transformed_df.reset_index(drop=True, inplace=True)

    return transformed_df

