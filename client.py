# WS client example

import asyncio
import websockets

async def hello():
    async with websockets.connect(
            'ws://10.127.188.235:8080') as websocket:

        await websocket.send('{"type": "header", "body": {}}')

        greeting = await websocket.recv()
        print(f"< {greeting}")

asyncio.get_event_loop().run_until_complete(hello())
