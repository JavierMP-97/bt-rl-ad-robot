import serial
import time


class ArduinoTransmitter():

    def __init__(self):
        self.arduino = serial.Serial('/dev/ttyUSB0', 9600)
        #self.arduino = serial.Serial('/dev/ttyACM0', 9600)
        time.sleep(3)
    def open_connection(self):
        self.send_message(b'0')
        self.receive_data(1)
        print("Arduino connection opened\n")

    def send_message(self, message):
        self.arduino.write(message)

    def receive_data(self, buffer):
        data = b''
        while len(data) < buffer:
            # doing it in batches is generally better than trying
            # to do it all in one go, so I believe.
            to_read = buffer - len(data)
            data += self.arduino.read(
                4096 if to_read > 4096 else to_read)
            #print("Pack {} took {} sec".format( len(data),time.time()-t))
        return data

    def close_connection(self):
        self.arduino.close()