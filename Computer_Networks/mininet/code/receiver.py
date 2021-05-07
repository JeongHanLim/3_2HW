import sys
from logHandler import logHandler
import socket

# Client does not need bind, as client do not have to check.

class fileReceiver(object):

    def __init__(self, ip, port=10080):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.ip, self.port))
        self.BUFF_SIZE = 1024
        self.logProc = logHandler()

    def __del__(self):
        self.socket.close()

    def check_terminal(self, recv):
        if recv in ["''", None, "404 Not Found"]:
            return True
        return False

    # TODO: TERMINAL STATE IS NOT WELL DEFINED>
    def receive(self):
        done = False
        while not done:
            recv = self.socket.recv(self.BUFF_SIZE).decode()
            done = self.check_terminal(recv)
            with open(recv, 'wb+') as f:
                f.write(recv.encode(encoding='utf-8'))
              print(recv, "file received finished")


    def send_ACK(self):
        self.logProc.writeAck(1, "Something,...")

    def main(self):
        print('receiver program starts...')
        throughput = 0.0
        ########################
        #Write your Code here
        self.logProc.startLogging("testRecvLogFile.txt")
        self.receive()

        self.logProc.writePkt(0, "Use your log file Processor")
        self.logProc.writeAck(1, "Like this")
        self.logProc.writeEnd(throughput)

        #########################


if __name__=='__main__':
    ip = "115.145.179.117"
    fileReceiver(ip).main()
