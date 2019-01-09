import json
import asyncio
import websockets
import netifaces
import datetime

lan_ip = netifaces.ifaddresses("wlo1")[netifaces.AF_INET][0]["addr"]
port = 8080
print("Serving from ws://{}:{}".format(lan_ip, port))

def new_outfile():
	filename = datetime.datetime.now().strftime("%Y_%m_%d_%I:%M") + ".json"
	print(f"Opening outfile {filename}")
	return open(filename, "w")

curr_file = None
header_provided = None
data_records = None
def reset_globals():
	global curr_file
	global header_provided
	global data_records

	curr_file = new_outfile()
	curr_file.write("{")
	header_provided = False
	data_records = 0
reset_globals()


async def print_message(websocket, path):
	global header_provided
	global data_records

	async for message in websocket:
		json_obj = json.loads(message)
		if json_obj["type"] == "header":
			header_provided = True
			curr_file.write('\"header\":{}'.format( json.dumps(json_obj["body"]) ))

			curr_file.write(', \"data\":[')
			
		elif json_obj["type"] == "eof":
			curr_file.write("]") #End data
			curr_file.write("}") #End root object
			curr_file.close()
			print(f"Closed outfile {curr_file.name}")
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
