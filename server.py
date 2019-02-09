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
historyTimeout = 2000 #Number of milliseconds of events to record

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

			self.model_a = erp.ml.convolution( [256, 4], 2)
			contextChannels = sum(len(states[key]) for key in states)
			self.model_b = erp.ml.convolution( [256, 4 + contextChannels], 2)

			self.sess.run(tf.global_variables_initializer())

			self.currFile = Server._beginFile()
			self.currFile.write('\"header\":{}'.format( json.dumps(packet["body"]) ))
			self.currFile.write(', \"data\":[')
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

	def _contextVector(contextType, length, width, startTime, endTime):
		"""
		:param str contextType: What type of vector we're generating (\"brightness\", \"orientation\", etc.)
		:param int length: Size of time dimension
		:param int width: Number of features/classes
		:param int startTime: start time in milliseconds
		:param int endTime: end time in milliseconds
		"""
		width = endTime - startTime
		pass

	def _processData(self, packet):
		if self.dataRecords > 0: self.currFile.write(",")
		self.currFile.write( json.dumps(packet["body"])[1:-1] )
		self.dataRecords += 1

		timestamp = packet["body"][0]["timestamp"]

		if "orientation" in self.change:
			orientation = int( self.change["orientation"] >= timestamp )
		else: orientation = 0
		if "brightness" in self.change:
			brightness = int( self.change["brightness"] >= timestamp )
		else: brightness = 0

		eegOnly    = np.array([ reading["data"] for reading in packet["body"]] )

		act_inputs = np.expand_dims( erp.tools.cleanSample(eegOnly, 80, 256) , 0)
		act_labels = np.expand_dims( np.array([orientation, brightness]), 0 )

		sess = self.sess
		model = self.model_a

		(_, loss, predictions) = sess.run([model.train_op, model.loss, model.predictions],
			{model.inputs: act_inputs, model.labels: act_labels}
		)
		print("loss =", loss)
		print("predictions =", predictions)


	def _processEvent(self, packet):
		if self.dataRecords > 0: self.currFile.write(",")
		self.currFile.write( json.dumps(packet["body"]) ) #Keep the curly braces
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

		print("history =", self.history)
		print("change  =", self.change)


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
		while buffer[-1][1] - buffer[0][1] >= historyTimeout:
			buffer.popleft()

	def _resetMembers(self):
		if hasattr(self, "currFile") and self.currFile:
			Server._endFile(self.currFile)
		self.currFile = None

		if hasattr(self, "sess") and self.sess:
			self.sess.close()
			tf.reset_default_graph()
		self.sess = None
		self.model_a = None
		self.model_b = None

		self.headerProvided = False
		self.dataRecords = 0

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




	@staticmethod
	def _beginFile():
		currFile = Server._newOutfile()
		currFile.write("{") #Begin root object
		return currFile
	@staticmethod
	def _newOutfile():
		filename = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M:%S") + ".json"
		print(f"Opening outfile {filename}")
		return open(filename, "w")
	@staticmethod
	def _endFile(dataFile):
		dataFile.write("]") #End data
		dataFile.write("}") #End root object
		dataFile.close()
		print(f"Closed outfile {dataFile.name}")
