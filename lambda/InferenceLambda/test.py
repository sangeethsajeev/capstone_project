import pandas as pd

from model import TrainLSTM

trainLSTM = TrainLSTM()

df = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')

trainLSTM.test_model(df)