import json
import datetime
#External libraries
import tensorflow as tf
import numpy as np
#Local libraries
import eeg
import rl
import erp

class Server(object):
	"""
	A single-threaded server for receiving data packets
	"""

	def __init__(self):
		self._reset_members()

	def receivePacket(self, packet):
		"""
		Description

		:param dict(str, obj) packet: The header, data, etc., packet

		:rtype: (int, str)
		:returns: An (id, action) tuple
		"""
		if packet["type"] == "header":
			self._reset_members()
			self.header_provided = True
			self.sess = tf.Session()
			self.model = erp.ml.convolution( [256, 4], 2)
			self.sess.run(tf.global_variables_initializer())

			self.curr_file = Server._begin_file()
			self.curr_file.write('\"header\":{}'.format( json.dumps(packet["body"]) ))
			self.curr_file.write(', \"data\":[')
			return None
		if packet["type"] == "eof":
			self._reset_members()
			return None
		if not self.header_provided:
			print("Header has not been provided--rejecting packet")
			return None

		if packet["type"] == "data":
			return self._processData(packet)
		if packet["type"] == "event":
			return self._processEvent(packet)

		print(f'Received packet of unknown type {packet["type"]}--rejecting')
		return None

	def _processData(self, packet):
		if self.data_records > 0: self.curr_file.write(",")
		self.curr_file.write( json.dumps(packet["body"])[1:-1] )
		self.data_records += 1

		timestamp = packet["body"][0]["timestamp"]

		print("Got a data record with timestamp", timestamp)

		if "rotation" in self.last:
			rotation = int( self.last["rotation"] >= timestamp )
		else: rotation = 0
		if "darkening" in self.last:
			darkening = int( self.last["darkening"] >= timestamp )
		else: darkening = 0

		eegOnly    = np.array([ reading["data"] for reading in packet["body"]] )

		act_inputs = np.expand_dims( erp.tools.cleanSample(eegOnly, 80, 256) , 0)
		act_labels = np.expand_dims( np.array([rotation, darkening]), 0 )
		print("act_labels =",act_labels)

		sess = self.sess
		model = self.model

		(_, loss, predictions) = sess.run([model.train_op, model.loss, model.predictions],
			{model.inputs: act_inputs, model.labels: act_labels}
		)
		print("loss =", loss)
		print("predictions =", predictions)


		return None


	def _processEvent(self, packet):
		if self.data_records > 0: self.curr_file.write(",")
		self.curr_file.write( json.dumps(packet["body"]) ) #Keep the curly braces
		self.data_records += 1

		eventName = packet["body"]["eventName"]
		self.state[eventName] = packet["body"][eventName]
		self.last[eventName] = packet["body"]["timestamp"]

		print("Got an event record",packet["body"])

		return None

	#PRIVATE
	def _reset_members(self):
		if hasattr(self, "curr_file") and self.curr_file:
			Server._end_file(self.curr_file)
		self.curr_file = None

		if hasattr(self, "sess") and self.sess:
			self.sess.close()
			tf.reset_default_graph()
		self.sess = None
		self.model = None

		self.header_provided = False
		self.data_records = 0
		self.state = {}
		self.last = {}




	@staticmethod
	def _begin_file():
		curr_file = Server._new_outfile()
		curr_file.write("{") #Begin root object
		return curr_file
	@staticmethod
	def _new_outfile():
		filename = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M:%S") + ".json"
		print(f"Opening outfile {filename}")
		return open(filename, "w")
	@staticmethod
	def _end_file(data_file):
		data_file.write("]") #End data
		data_file.write("}") #End root object
		data_file.close()
		print(f"Closed outfile {data_file.name}")
