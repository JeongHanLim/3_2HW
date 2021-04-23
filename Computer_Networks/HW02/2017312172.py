import socket
import time
import os
import queue
import select
from threading import Thread
# Server needs to get the string data, and send the data to client back.
# when sending the data, server has to send data by threading, as file is big.
# When making thread, getting data from client is in while loop, so that threading should be done
# Within Function, which gives data to client.

#TODO: bind -> listen -> accept
#TODO : server bind permission denied.

class Server(object):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.reuse = True
        self.qsize = 10
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#        self.server_bind()
#        self.server_listen()


        self.q = queue.Queue()
        self.threads = []
        print("initialization Finished.")

    def __del__(self):
        self.socket.close()

    # Used in send_file

    def server_bind(self):
        if self.reuse==True:
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind((self.ip, self.port))

    def server_listen(self):
        self.socket.listen(self.qsize)

    def send_file(self, c, addr):
        while(True):
            item = self.q.get()
            if item == '' or item is None:
                continue
            if not os.path.exists(item):
                str_error = "404 Not found error"
                HTTP_RESPONSE = b'\r\n'.join([
                    b"HTTP/1.1 200 OK",
                    b"Connection: close",
                    b"Content-Type: text.html",
                    bytes("Content-Length: %s" % len(str_error), 'utf-8'),
                    b'', str_error.encode('utf-8')
                ])
                c.sendall(HTTP_RESPONSE)
                continue


            with open(item, 'rb') as f:
                data = f.read()
                exten = item.split('.')[1]
                img_ext_list = ["JPG","PNG", "GIF","WEBP","TIFF","PSD","RAW","BMP","jpg","png", "gif","webp","tiff","psd","raw","bmp" ]
                if exten in img_ext_list:
                    HTTP_RESPONSE = b'\r\n'.join([
                        b"HTTP/1.1 200 OK",
                        b"Connection: close",
                        b"Content-Type: image/jpg",
                        bytes("Content-Length: %s" % len(data), 'utf-8'),
                        b'', data
                    ])
                    c.sendall(HTTP_RESPONSE)
                else:
                    HTTP_RESPONSE = b'\r\n'.join([
                        b"HTTP/1.1 200 OK",
                        b"Connection: close",
                        b"Content-Type: text/html",
                        bytes("Content-Length: %s" % len(data), 'utf-8'),
                        b'', data
                    ])
                    c.sendall(HTTP_RESPONSE)
            return

    def recv_link(self, c, addr):
        while(True):

            msg = c.recv(1024).decode()
            if msg == '' or msg is None:
                continue
            processed = msg.split('/')[1].split(' ')[0]
            self.q.put(processed)
            return

    def acceptence(self, c, addr):

        self.recv_link(c, addr)
        self.send_file(c, addr)

    def StartListening(self):
        self.server_bind()
        self.server_listen()

        while True:

            c, addr = self.socket.accept()
            Thread(target=self.acceptence, args=(c, addr)).start()


        self.socket.close()


if __name__ == "__main__":
    #Server("192.168.25.13", 10081).StartListening()
    Server(ip="192.168.22.169", port=10080).StartListening()
