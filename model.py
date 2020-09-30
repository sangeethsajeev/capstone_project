import tensorflow as tf
from os import path
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error

from preprocess import PreProcessData
preProcessData = PreProcessData()

class TrainLSTM(object):

	## Change the HyperParameters to the Model here.
	def __init__(self):
		self.activation_func = 'selu'
		self.optimizer		 = 'adam'
		self.loss			 = 'mse'
		self.n_train_years	 = 3			#Training on 3 Years of Data
		self.n_val_years	 = 1			#Validating on 1 Year of Data
		self.n_test_years	 = 1			#Testing on 1 Year of Data


	def create_model(self):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, 
			activation=self.activation_func, 
			return_sequences=True), input_shape=(preProcessData.n_hours, preProcessData.n_features)))
		model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, 
			activation=self.activation_func, 
			return_sequences=True)))
		model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, 
			activation=self.activation_func)))
		model.add(tf.keras.layers.Dense(1))
		model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

		return model

	def load_model(self, test_Flag=False):
		model = self.create_model()

		if path.exists('weights/lstm_model.h5') and test_Flag:
			model.load_weights('weights/lstm_model.h5')
			print("Training Weights Loaded Sucessfully..!")
		elif path.exists('weights/lstm_model.h5'):
			model.load_weights('weights/lstm_model.h5')
		else:
			print("Untrained Model")

		return model

	def fit_model(self,df):
		model 	= self.load_model()
		train_X,train_Y,validate_X,validate_Y = self.data_process(df)
		history = model.fit(train_X, train_Y, epochs=50, batch_size=72, 
			validation_data=(validate_X,validate_Y),
			verbose=0, shuffle=False)
		# print("Training Loss:	",history.history['loss'],
		# 	"Validation Loss:	",history.history['val_loss'],
		# 	"Accuracy:	",history.history['accuracy'])
		print("--------------------------------------------------------------------------------\n\n\n\n")
		print(model.summary())
		print("\n\n\n\n--------------------------------------------------------------------------------")
		
		model.save_weights('weights/lstm_model.h5')

		return "Fitting Complete"

	def get_hours(self):
		n_train_hours = 365*24*self.n_train_years
		n_valid_hours = 365*24*self.n_val_years
		n_test_hours = 365*24*self.n_test_years

		return n_train_hours,n_valid_hours,n_test_hours

	def test_model(self,df):
		model = self.load_model(test_Flag=True)
		test,test_X, test_Y = self.data_process(df,test_Flag=True)
		predictions = []
		for i in range(preProcessData.n_hours,test_X.shape[0]):
			if test_X.shape[0]<preProcessData.n_hours:
				print("Not enough samples to make a prediction")
				break
			x_input = test_X[i-preProcessData.n_hours:i]
			x_input = x_input.reshape((preProcessData.n_hours, preProcessData.n_hours, preProcessData.n_features))
			prediction = model.predict(x_input, verbose=0)
			predictions.append(prediction[0][0])
			if i%1000==0 and i>0:
				print("Predictions done for {} records".format(i))

		res_arr = np.array(predictions)
		res_arr = res_arr.reshape(len(res_arr), 1)
		rmse = sqrt(mean_squared_error(res_arr[:], test[preProcessData.n_hours:, 1:2]))
		print('RMSE Score: {}'.format(rmse))
		print("Prediction Complete")

		return res_arr

	def data_process(self,df,test_Flag=False):
		transformed_df = preProcessData.preprocess_df(df,test_Flag)
		values		   = transformed_df.values
		n_train_hours, n_valid_hours, n_test_hours = self.get_hours()
		train = values[:n_train_hours, :]
		validate = values[n_train_hours:n_train_hours+n_valid_hours, :] #kept 1 year data for validation
		test = values[n_train_hours+n_test_hours:, :] # 1 year data for test
		n_attributes = preProcessData.n_hours * preProcessData.n_features
		train_X, train_y = train[:, :n_attributes], train[:, -1]
		test_X, test_y = test[:, :n_attributes], test[:, -1]
		validate_X, validate_y = validate[:, :n_attributes], validate[:, -1]
		train_X = train_X.reshape((train_X.shape[0], preProcessData.n_hours, preProcessData.n_features))
		validate_X = validate_X.reshape((validate_X.shape[0], preProcessData.n_hours, preProcessData.n_features))

		if(test_Flag):
			return test,test_X,test_y
		else:
			return train_X,train_y,validate_X,validate_y




