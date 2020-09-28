import tensorflow as tf
from os import path


from preprocess_data import PreProcessData
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
		self.n_features		 = 9


	def create_model(self):
		model = tf.keras.Sequential()
		model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, 
			activation=activation_func, 
			return_sequences=True), input_shape=(n_hours, n_features)))
		model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, 
			activation=activation_func, 
			return_sequences=True)))
		model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, 
			activation=activation_func)))
		model.add(tf.keras.layers.Dense(1))
		model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

		return model

	def load_model(self):
		model=create_model()
		if path.exists('weights/lstm_model.h5'):
			model.load_weights('weights/lstm_model.h5')

		return model

	def fit_model(self,df):
		model 	= load_model()
		train_X,train_Y,validate_X,validate_Y = data_process(df)
		history = model.fit(train_X, train_Y, epochs=50, batch_size=72, 
			validation_data=(validate_X,validate_Y),
			verbose=0, shuffle=False)
		print("Training Loss:	",history.history['loss'],
			"Validation Loss:	",history.history['val_loss'],
			"Accuracy:	",history.history['accuracy'])
		print("--------------------------------------------------------------------------------")
		print(model.summary())
		print("--------------------------------------------------------------------------------")
		
		model.save_weights('weights/lstm_model.h5')

	def data_process(self,df):
		transformed_df = preProcessData().preprocess_df(df)
		values		   = transformed_df.values
		n_train_hours  = 365*24*n_train_years
		n_valid_hours  = 365*24*n_val_years
		n_test_hours   = 365*24*n_test_hours
		train = values[:n_train_hours, :]
		validate = values[n_train_hours:n_train_hours+n_valid_hours, :] #kept 1 year data for validation
		test = values[n_train_hours+n_test_hours:, :] # 1 year data for test
		n_attributes = n_hours * n_features
		train_X, train_y = train[:, :n_attributes], train[:, -1]
		test_X, test_y = test[:, :n_attributes], test[:, -1]
		validate_X, validate_y = validate[:, :n_attributes], validate[:, -1]
		train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
		validate_X = validate_X.reshape((validate_X.shape[0], n_hours, n_features))

		return train_X,train_y,validate_X,validate_y



