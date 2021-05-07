import sys
from logHandler import logHandler
import socket
import queue

class filesender(object):
    def __init__(self, ip, port):
        self.ip = ip
        self.port = port
        self.reuse = True
        self.qsize = 10
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.q = queue.Queue()
        self.threads = []
        print("initilization Finished")

    def __del__(self):
        self.socket.close()

    def

def fileSender():
    print('sender program starts...')#remove this
    logProc = logHandler()
    
    throughput = 0.0
    avgRTT = 10.0
    ##########################
    
    #Write your Code here
    logProc.startLogging("testSendLogFile.txt")
    
    logProc.writePkt(0, "Use your log file Processor")
    logProc.writeAck(1, "Like this")
    logProc.writeEnd(throughput, avgRTT)
    ##########################


if __name__=='__main__':
    recvAddr = sys.argv[1]  #receiver IP address
    windowSize = int(sys.argv[2])   #window size
    srcFilename = sys.argv[3]   #source file name
    dstFilename = sys.argv[4]   #result file name

    fileSender()
