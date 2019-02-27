from data_file import DataFile

class Server(object):
	"""
	A single-threaded server for receiving data packets
	"""

	def __init__(self):
		if hasattr(self, "dataFile") and self.dataFile:
			_closeOutfile(self.dataFile)
		self.dataFile = None

	def receivePacket(self, packet):
		"""
		Description

		:param dict(str, obj) packet: The header, data, etc., packet

		:rtype: (int, str)
		:returns: An (id, action) tuple
		"""
		if packet["type"] == "header":
			self.dataFile = DataFile("data", packet, log=True)
			return None
		if packet["type"] == "eof":
			self.dataFile.close()
			self.dataFile = None
			return None

		if not self.dataFile:
			print("Header has not been provided--rejecting packet")
			return None

		if packet["type"] == "data":
			return self.dataFile.write(packet)
		if packet["type"] == "event":
			return self.dataFile.write(packet)

		print(f'Received packet of unknown type {packet["type"]}--rejecting')
		return None
