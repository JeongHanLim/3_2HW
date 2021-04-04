from socket import *
import time

class AsychronousClient(object):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.sock = socket(AF_INET, SOCK_STREAM)
        self.sock.connect((self.ip, self.port))

    def __del__(self):
        self.sock.close()

    def post_process(self, data):

        pass


    def send_data_thread(self, data):
        #TODO : Threading
        sentbyte = self.sock.send(bytes(data, encoding='utf8'))
        return sentbyte

    def received_data_thread(self):
        received = self.sock.recv(1024)
        return received


    def StartSpeaking(self):
        global received
        finished = False
        cnt = 3
        try:
            while not finished:
                data = str(cnt)
                sentbyte = self.send_data_thread(data)
                print("Speaking ", data, " byte ", sentbyte)
                #received = self.received_data_thread()
                #self.post_process(received)

                cnt+=1
        except:
            raise ConnectionError

if __name__ == "__main__":
   AsychronousClient("localhost", 8888).StartSpeaking()
