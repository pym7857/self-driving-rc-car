from bluetooth import *

MAC = "34:F6:4B:A7:1E:D4"
port = 1

client_socket = BluetoothSocket(RFCOMM)
client_socket.connect((MAC, port))
try:
    while True:
        msg = input("Send : ")
        print(msg)
        client_socket.send(msg.encode())
except:
    print('err')

print("Finished")
client_socket.close()