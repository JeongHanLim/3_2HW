from socket import *
import time
import os
import queue
import select
from threading import Thread
# Server needs to get the string data, and send the data to client back.
# when sending the data, server has to send data by threading, as file is big.
# When making thread, getting data from client is in while loop, so that threading should be done
# Within Function, which gives data to client.



class Server(object):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.bind((self.ip, self.port))
        self.q = queue.Queue()
        self.threads = []
        self.sock.listen()
        self.sock, self.addr = self.sock.accept()


    def __del__(self):
        self.sock.close()

    # Used in send_file
    def worker(self, item):
        self.sock.send(bytes(item, encoding='utf8'))


    def send_file(self):
        while(True):
            num_worker_threads=1
            item = self.q.get()

            if item == '' or item is None:
                continue

            if not os.path.exists("data/"+item):
                self.worker("404 Not Found")
                continue

            for i in range(num_worker_threads):
                t = Thread(target=self.worker, args=(item,))
                t.start()
                print("Thread starting...")
                self.threads.append(t)

            print(self.threads)

    def recv_link(self):
        while(True):
            msg = self.sock.recv(1024).decode()
            self.q.put(msg)

    def StartListening(self):
        Thread(target=self.recv_link).start()
        Thread(target=self.send_file).start()

if __name__ == "__main__":
    Server('localhost', 10080).StartListening()





