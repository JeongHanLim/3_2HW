import os
import time
from threading import Thread
import threading
from queue import Queue

def get_file_name():
    file_src = input("Input the file name: ")
    if file_src == "exit":
        return -1, 0, 0
    while not os.path.exists(file_src):
        file_src = input("Rewrite your Input the file name: ")
        if file_src == "exit":
            return -1, 0, 0

    dest_src = input("Input the new name: ")
    while os.path.exists(dest_src):
        dest_src= input("Same File name already exist. Reinput the new name: ")

    return 0, file_src, dest_src


def logging(q, f, start_t):
    #print(q.qsize())
    for i in range(q.qsize()):
        package = q.get()
        if package[0] == "r":
            f.write(str(round(package[3]-start_t, 2)) + " Start copying " + str(package[1])+ " to " + str(package[2])+"\n")
            #print(str(round(package[3]-start_t, 2)) + " Start copying " + str(package[1])+ " to " + str(package[2])+"\n")
        elif package[0] == "w":
            f.write(str(round(package[3]-start_t, 2)) +" "+str(package[2])+" is copied completely"+"\n")
            #print(str(round(package[3]-start_t, 2)) +" "+str(package[2])+" is copied completely"+"\n")


def copy_file( src, dst ,q):
    #DEBUGCODE=========
    read_t = time.time()
    package = ["r", src, dst, read_t]
    q.put(package, block = True)
    with open(src, "rb") as File:
        with open(dst, "wb+") as dstFile:
            i = 0
            while i < os.path.getsize(src):
                file_content = File.read(102)
                i += 102
                dstFile.write(file_content)
            # print(i)
    write_t = time.time()
    package = ["w", src, dst, write_t]
    q.put(package, block = True)

    return read_t, write_t



if __name__ == "__main__":
    start_t = time.time()
    f = open("log.txt", "w+")
    queue = Queue()
    index = 0
    while True:
        succ, src, dst = get_file_name()
        if succ == -1:
            break

        globals()['t{}'.format(index)] = Thread(target = copy_file, args = (src, dst, queue))
        exec('t%d.daemon = True' % (index))
        exec('t%d.start()'%(index))
        #print("number of thread-2", threading.active_count())
        index +=1

    print("Waiting for file copying...")
    for idx in range(index):
        exec('t%d.join()'%(idx))
    logging(queue, f, start_t)
    print("File transfer finished....")
    #print("number of thread-1", threading.active_count())

    f.close()
