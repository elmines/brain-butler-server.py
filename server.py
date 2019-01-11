import json
import asyncio
import websockets
import netifaces
import datetime

lan_ip = netifaces.ifaddresses("wlo1")[netifaces.AF_INET][0]["addr"]
port = 8080
print("Serving from ws://{}:{}".format(lan_ip, port))

def new_outfile():
	filename = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M:%S") + ".json"
	print(f"Opening outfile {filename}")
	return open(filename, "w")

curr_file = None
header_provided = None
data_records = 0
def reset_globals():
	global curr_file
	global header_provided
	global data_records

	if curr_file: end_file(curr_file)
	curr_file = None
	header_provided = False
	data_records = 0
reset_globals()

def begin_file():
	global curr_file
	curr_file = new_outfile()
	curr_file.write("{")
	return curr_file
def end_file(data_file):
	data_file.write("]") #End data
	data_file.write("}") #End root object
	data_file.close()
	print(f"Closed outfile {data_file.name}")

async def print_message(websocket, path):
	global curr_file
	global header_provided
	global data_records

	async for message in websocket:
		json_obj = json.loads(message)
		if json_obj["type"] == "header":
			header_provided = True
			reset_globals()

			curr_file = begin_file()
			curr_file.write('\"header\":{}'.format( json.dumps(json_obj["body"]) ))
			curr_file.write(', \"data\":[')
			
		elif json_obj["type"] == "eof":
			reset_globals()
		else:
			if not header_provided:
				print("Header has not been provided--rejecting data record")
				continue

			if data_records > 0: curr_file.write(",")
			curr_file.write( json.dumps(json_obj["body"])[1:-1] )	
			data_records += 1
			

asyncio.get_event_loop().run_until_complete( websockets.serve(print_message, lan_ip, port) )
asyncio.get_event_loop().run_forever()
