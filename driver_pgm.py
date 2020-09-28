import pandas as pd

from train_model import TrainLSTM

trainLSTM = TrainLSTM()

df = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')

trainLSTM.fit_model(df)