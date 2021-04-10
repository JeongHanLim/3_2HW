from socket import *
import os
import time
import errno
import sys
import select
from threading import Thread
#We do not need threading in client, as Client just need to send the string.
#Client has to get a threaded file, but library thread from server will do all the work.


class AsynchronousClient(object):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.connect((self.ip, self.port))
        self.BUFF_SIZE = 1024

    def __del__(self):
        self.sock.close()

    def send_file(self, ):
        while(True):
            data = input("here:")
            data = str(data)+".png"
            self.sock.send(bytes(data, encoding='utf8'))

    def recv_file(self):
            while True:
                received=''
                received = self.sock.recv(self.BUFF_SIZE).decode()
                if received == '' or received is None or received =='404 Not Found':
                    continue

                with open("recv/"+received, 'wb+') as f:
                    f.write(received.encode(encoding='utf-8'))
                print(received, "file receiving finished.")

    def StartSpeaking(self):
        finished = False
        Thread(target=self.send_file).start()
        Thread(target=self.recv_file).start()

def addr_split(addr):
    addrlist = []
    addrlist.append(addr.split('/'))
    ip, port = addrlist[2].split(':')[0], addrlist[2].split(':')[1]
    data = addrlist[3]
    return addrlist, data

if __name__ == "__main__":
    address = input("Write down your address http://")
    #ip, port = addr_split(address)
    #For a while
    #ip, port = "192.168.25.13", 10081
    ip, port = "192.168.0.16", 10080
    AsynchronousClient(ip, port).StartSpeaking()
