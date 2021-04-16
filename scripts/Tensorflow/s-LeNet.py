import tensorflow as tf
import time

checkpoint_path = "./model.ckpt"
device = "/cpu:0"
model = None

# Additional configs needed to solve the NotFoundError
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def loadData():
	mnist = tf.keras.datasets.mnist
	(X_train, y_train), (X_test, y_test) = mnist.load_data()
	X_train, X_test = X_train / 255.0, X_test / 255.0
	print(f"Shape of X_train: {X_train.shape}")
	print(f"Shape of y_train: {y_train.shape}")
	print(f"Shape of X_test: {X_test.shape}")
	print(f"Shape of y_test: {y_test.shape}")
	
	X_train = X_train[..., tf.newaxis].astype("float32")
	X_test = X_test[..., tf.newaxis].astype("float32")
	
	return (X_train, y_train), (X_test, y_test)


def createModel():
	global model
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(6, (5, 5), activation='sigmoid', use_bias=True, input_shape=(28, 28, 1)),
		tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=4, padding="valid"),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(10)
	])
	
	loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	model.compile(optimizer="SGD", loss=loss_fn, metrics=['accuracy'])


def trainModel(X_train, y_train):
	global model
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)
	model.fit(X_train, y_train, epochs=5, callbacks=[cp_callback])
	return checkpoint_path


def evaluate(X_test, y_test, verbose=False, device="/gpu:0"):
	global model
	deviceType = device.upper()[1:4]
	
	if model is None:
		createModel()
	
	model.load_weights(checkpoint_path)
	
	start = time.perf_counter()
	
	with tf.device(device):
		loss, acc = model.evaluate(X_test, y_test, verbose=0)
	
	end = time.perf_counter()
	
	elapsed_time = end - start
	elapsed_time *= 1000
	acc *= 100
	
	if verbose:
		print("[{}] (s-LeNet Inference) Elapsed Time = {} ms".format(deviceType, elapsed_time))
		print(f"Loss: {loss}")
		print(f"Accuracy: {acc} %")
	
	return elapsed_time, acc


def main():
	count = 100
	acc = 0
	averageTime = 0
	(X_train, y_train), (X_test, y_test) = loadData()
	# createModel()
	# checkpoint_path = trainModel(X_train, y_train)
	evaluate(X_test, y_test, verbose=False, device=device)
	
	for i in range(count):
		time, acc = evaluate(X_test, y_test, verbose=False, device=device)
		averageTime += time
	
	averageTime /= count
	print("[GPU] (s-LeNet Inference) Average Elapsed Time = {} ms".format(averageTime))
	print("[GPU] (s-LeNet Inference) Accuracy = {} %".format(acc))


if __name__ == "__main__":
	main()
