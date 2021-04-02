from socket import *
import time

class Server(object):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.bind((self.ip, self.port))

    def __del__(self):
        self.sock.close()

    def listen(self):
        self.sock.listen()

    def client_accept(self):
        clientsocket, address = self.sock.accept()
        return clientsocket, address


    def StartListening(self):
        self.listen()
        print("Waiting for Connection")

        sock, addr = self.client_accept()
        print("Connection Finished")

        #while(1):
            #send recv
        while True:
            msg = sock.recv(1024)
            print("Receiving...", msg)
            time.sleep(1)
            sock.send(bytes(msg))
        #sock.shutdown(SHUT_WR)
        #sock.close()
        # close recv
    #close

if __name__ == "__main__":
    Server('localhost', 8888).StartListening()





