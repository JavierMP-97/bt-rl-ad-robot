import socket
import time

class TcpServer():

    def __init__(self, host, port):
        self.HOST = host #'192.11'
        self.PORT = port #12345
        
    def open_connection(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)      
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        try:
            self.s.bind((self.HOST, self.PORT))
        except socket.error:
            print("Bind failed")
            return False

        self.s.listen(1)
        print("Server created")

        (self.conn, self.addr) = self.s.accept()
        print("Connected to {}".format(self.addr))

        return True
    
    def send_message(self, message):
        return self.conn.send(message)

    def receive_message(self, buffer):
        data = b''
        while len(data) < buffer:
            t = time.time()
            # doing it in batches is generally better than trying
            # to do it all in one go, so I believe.
            to_read = buffer - len(data)
            data += self.conn.recv(
                4096 if to_read > 4096 else to_read)
            #print("Pack {} took {} sec".format( len(data),time.time()-t))
        return data
    
    def close_connection(self):
        self.conn.close()
        #self.s.close()
    
class TcpClient():
    def __init__(self, host, port):
        self.HOST = host #'192.11'
        self.PORT = port #12345

    def open_connection(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        #self.s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        
        try:
            self.s.connect((self.HOST, self.PORT))
        except socket.error:
            print("Connect failed")
            return False

        
        self.addr = (self.HOST, self.PORT)

        print("Connected to {}".format(self.HOST))

        return True
    
    def send_message(self, message):
        self.s.send(message)

    def receive_message(self, buffer):
        
        data = b''
        while len(data) < buffer:
            # doing it in batches is generally better than trying
            # to do it all in one go, so I believe.
            to_read = buffer - len(data)
            data += self.s.recv(
                4096 if to_read > 4096 else to_read)
        return data

    def close_connection(self):
        self.s.close()
        #self.s.close()
    