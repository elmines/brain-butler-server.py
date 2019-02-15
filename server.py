import json
import datetime
from collections import defaultdict
from collections import deque
#External libraries
import tensorflow as tf
import numpy as np
#Local libraries
import eeg
import rl
import erp

states = {
	"orientation": ["portrait", "landscape"],
	"brightness": ["full", "low"]
}
historyTimeout = 5000 #Number of milliseconds of events to record
minHistoryLength = 2 #Minimum number of events to have in the history

oddballWeight = 4
def conv(shape, weighted=False):
	"""
	:param shape: The shape of the inputs (should be [numChannels, samplesPerChannel])
	:param int classes: The number of different output classes
	"""
	inputs = tf.placeholder(tf.float32, (None,) + tuple(shape), name="inputs")
	classes = 2
	labels = tf.placeholder(tf.float32, (None, classes), name="labels")

	convolved = tf.layers.conv1d(inputs, filters=6, kernel_size=96, strides=1)
	pooled = tf.layers.max_pooling1d(convolved, pool_size=3, strides=2)
	logit = tf.squeeze( tf.layers.dense(tf.layers.flatten(pooled), classes) )

	if weighted:
		lossWeights = labels*(oddballWeight-1) + 1
		loss = tf.losses.sigmoid_cross_entropy(labels, logit, weights=lossWeights)
	else:
		loss = tf.losses.sigmoid_cross_entropy(labels, logit)


	optimizer = tf.train.AdamOptimizer()
	train_op = optimizer.minimize(loss)
	#TESTING
	#Graph
	predictions = tf.cast(tf.round(tf.nn.sigmoid(logit)), tf.int32, name="predictions")

	m = erp.ml.Model(inputs=inputs, labels=labels, loss=loss, train_op=train_op,
		predictions=predictions)

	return  m

class Server(object):
	"""
	A single-threaded server for receiving data packets
	"""

	def __init__(self):
		self._resetMembers()

	def receivePacket(self, packet):
		"""
		Description

		:param dict(str, obj) packet: The header, data, etc., packet

		:rtype: (int, str)
		:returns: An (id, action) tuple
		"""
		if packet["type"] == "header":
			self._resetMembers()
			self.headerProvided = True
			self.sess = tf.Session()

			self.model_a = conv(self._eegData.shape[1:])
			self.model_b = conv(self._fullData.shape[1:])

			self.sess.run(tf.global_variables_initializer())

			dateString = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M:%S")
			dataPath = "data_" + dateString + ".json"
			trainPath = "training_"+dateString + ".json"
			self.dataFile = _openOutfile(dataPath, packet["body"])
			self.trainFile = _openOutfile(trainPath, self._trainingHeader())
			return None
		if packet["type"] == "eof":
			self._resetMembers()
			return None
		if not self.headerProvided:
			print("Header has not been provided--rejecting packet")
			return None

		if packet["type"] == "data":
			return self._processData(packet)
		if packet["type"] == "event":
			return self._processEvent(packet)

		print(f'Received packet of unknown type {packet["type"]}--rejecting')
		return None

	def _contextVector(self, component, length, startTime, endTime):
		"""
		:param str component: What type of vector we're generating (\"brightness\", \"orientation\", etc.)
		:param int length: Size of time dimension
		:param int startTime: start time in milliseconds
		:param int endTime: end time in milliseconds
		"""
		duration = endTime - startTime
		history = self.history[component]

		width = len(states[component])
		if width < 3: width = 1
		vector = np.squeeze(np.zeros( [length, width] ))

		endIndex = length
		for (value, timestamp) in reversed(history):
			startIndex = int( (timestamp - startTime)/duration * length )
			if startIndex < 0: startIndex = 0
			stateIndex = states[component].index(value)
			if width == 1: vector[startIndex:endIndex] = stateIndex
			else:          vector[startIndex:endIndex, stateIndex] = 1
			endIndex = startIndex
		return vector

	def _processData(self, packet):
		if self.dataRecords > 0: self.dataFile.write(",")
		self.dataFile.write( json.dumps(packet["body"])[1:-1] )
		self.dataRecords += 1
		startTime = packet["body"][0]["timestamp"]
		endTime = packet["body"][-1]["timestamp"]

		eegOnly    = np.array([ reading["data"] for reading in packet["body"]] )
		eegOnly    = erp.tools.cleanSample(eegOnly, 80, 256)

		actInputs = self._appendContext(eegOnly, startTime, endTime)
		actLabels = self._deriveLabels(startTime, endTime)
		self._evalModels(actInputs, actLabels)

		self._loadData(actInputs, actLabels)
		if self._trainReady: self._trainModels()

	def _trainModels(self):
		self._train(self.model_a, self._eegData, self._labels)
		self._train(self.model_b, self._fullData, self._labels)
	def _train(self, model, inputs, labels):
		self.sess.run([model.train_op], {
			model.inputs: inputs, model.labels: labels
		})

	def _evalModels(self, fullSample, labels):
		assert type(labels) != list;
		eegData = fullSample[:, :-self._contextChannels]
		fullSample = np.expand_dims(fullSample, 0)
		eegData    = np.expand_dims(eegData, 0)
		labels = np.expand_dims(labels, 0)
		assert type(labels) != list;

		aPreds = self._eval(self.model_a, eegData, labels)
		bPreds = self._eval(self.model_b, fullSample, labels)

		#print("aPreds =", aPreds)
		#print("bPreds =", bPreds)
		#print("labels =", labels)
		labels = labels.tolist()
		aPreds = aPreds.tolist()
		bPreds = bPreds.tolist()

		trainRecord = {"labels": labels, "modelA": aPreds, "modelB": bPreds}

		#trainRecord = {"labels": labels.tolist(), "modelA": aPreds.tolist(),
			#"modelB": bPreds.tolist()
		#}
		if self.trainRecords > 0: self.trainFile.write(",")
		self.trainFile.write(json.dumps(trainRecord))
		self.trainRecords += 1
	def _eval(self, model, act_inputs, act_labels):
		[predictions] = self.sess.run([model.predictions],
			{model.inputs: act_inputs, model.labels: act_labels}
		)
		return predictions

	def _deriveLabels(self, startTime, endTime):
		if "orientation" in self.change:
			orientation = int( self.change["orientation"][-1] >= startTime )
		else: orientation = 0
		if "brightness" in self.change:
			brightness = int( self.change["brightness"][-1] >= startTime )
		else: brightness = 0
		actLabels = np.array([orientation, brightness])
		return actLabels

	def _appendContext(self, act_inputs, startTime, endTime):
		if act_inputs.ndim < 3:
			act_inputs = np.expand_dims(act_inputs, 0)
		orientVec = self._contextVector("orientation", 256, startTime, endTime)
		if orientVec.ndim < 2: orientVec = np.expand_dims(orientVec, -1)
		orientVec = np.expand_dims(orientVec, 0)
		brightVec = self._contextVector("brightness", 256, startTime, endTime)
		if brightVec.ndim < 2: brightVec = np.expand_dims(brightVec, -1)
		brightVec = np.expand_dims(brightVec, 0)

		act_inputs = np.concatenate([act_inputs, orientVec, brightVec], axis=-1)
		return np.squeeze(act_inputs)


	def _loadData(self, inputs, labels):
		self._fullData[self._writeIndex] = inputs
		self._labels[self._writeIndex] = labels
		self._writeIndex += 1
		if self._writeIndex >= self._dataLimit:
			self._trainReady = True
			self._writeIndex = 0



	def _processEvent(self, packet):
		if self.dataRecords > 0: self.dataFile.write(",")
		self.dataFile.write( json.dumps(packet["body"]) ) #Keep the curly braces
		self.dataRecords += 1

		packet = packet["body"]
		eventName = packet["eventName"]
		eventTimestamp = packet["timestamp"]
		eventValue = packet[eventName]
		eventTuple = (eventValue, eventTimestamp)
		if self._changed(eventName, eventTuple):
			self.change[eventName] = eventTuple
		self.history[eventName].append(eventTuple)
		self._flushOld(self.history[eventName])


	def _changed(self, eventName, eventTuple):
		#No previous state
		if eventName not in self.history: return False
		if not self.history[eventName]: return False
		#No change from last state
		if self.history[eventName][-1][0] == eventTuple[0]: return False

		#FIXME: Find a better way to track only decreasing brightness
		if eventName == "brightness":
			return eventTuple[0] == "low" #Only record darkness as a change

		return True

	def _flushOld(self, buffer):
		"""
		:param deque(tuple(str, int)) buffer: The buffer to flush
		"""
		if not buffer: return #Empty buffer
		while buffer[-1][1] - buffer[0][1] >= historyTimeout and len(buffer) > minHistoryLength:
			buffer.popleft()

	def _trainingHeader(self):
		header = {}
		header["classes"] = {
			"names": ["rotated", "darkened"],
			"disjoint": False
		}
		return header

	def _resetMembers(self):
		if hasattr(self, "dataFile") and self.dataFile:
			_closeOutfile(self.dataFile)
		self.dataFile = None
		if hasattr(self, "trainFile") and self.trainFile:
			_closeOutfile(self.trainFile)
		self.trainFile = None


		if hasattr(self, "sess") and self.sess:
			self.sess.close()
			tf.reset_default_graph()
		self.sess = None
		self.model_a = None
		self.model_b = None

		self.headerProvided = False
		self.dataRecords = 0
		self.trainRecords = 0

		self._trainReady = False
		self._contextChannels = 2
		self._dataLimit = 8
		self._writeIndex = 0

		self._fullData  = np.ndarray( (self._dataLimit, 256, 4+self._contextChannels) )
		self._eegData  = self._fullData[:, :, :-self._contextChannels]
		self._labels    = np.ndarray( (self._dataLimit, 2) )


		self.history = defaultdict(deque)
		"""
		dict(str) -> (str, int)

		Holds a deque of past values for each event
		"""

		self.change = {}
		"""
		dict(str) -> (str, int)

		Keys are event types (\"brightness\", \"orientation\", etc.)
		Values are the new state, and milliseconds since the epoch
		"""

def _openOutfile(filename, header):
	print(f"Opening outfile {filename}")
	f = open(filename, "w")
	f.write("{") #Begin root object
	f.write('\"header\":{}'.format(json.dumps(header)))
	f.write(', \"data\":[')
	return f
def _closeOutfile(dataFile):
	"""
	:param file dataFile: Pointer to file opened with _openOutfile
	"""
	dataFile.write("]") #End data
	dataFile.write("}") #End root object
	dataFile.close()
	print(f"Closed outfile {dataFile.name}")
