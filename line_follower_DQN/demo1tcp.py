import socket
import numpy as np
import pickle
import time

def decode_data(data):

    #print(data)

    LT = np.zeros(5)

    #print(len(data))

    for i in range(5):
        LT[i] = data[i*2] << 8

        LT[i] += data[i*2+1]

    #print(data)

    #print(md)
    print(LT[0], LT[1], LT[2], LT[3], LT[4])

    return LT[0], LT[1], LT[2], LT[3], LT[4]

if __name__ == '__main__':
    TCP_IP = ''
    TCP_PORT = 5000
    SERVER_BUFFER_SIZE = 10

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    s.settimeout(5)

    max_time = 0
    cont = 0

    with open( "calibration.pr", "rb" ) as f:
        infra_red_range = pickle.load(f)

    normalized_limit = np.zeros(5)

    for idx, irr in enumerate(infra_red_range):
        normalized_limit[idx] = (((irr[0] + irr[2]) / 2) - irr[2]) / (irr[1]-irr[2])

    while 1:

        i=0

        line_position = 2
        last_line_position = 1
        action = 7

        try:
            print("Waiting for connection...\n")
            conn, addr = s.accept()
        except:
            print("A connection couldn't be established. Trying again.\n")
        else:
            print("Connection established\n")

            while 1:
                start = time.time()
                try:
                    data = conn.recv(SERVER_BUFFER_SIZE)
                except:
                    print("No response (connection lost)\n")
                    conn.close()
                    print("Connection closed\n")
                    break
                tot_time = time.time() - start

                if max_time < tot_time:
                    max_time = tot_time
                if tot_time > 0.1:
                    cont += 1
                print(tot_time)
                print(max_time) 
                print(cont)

                decoded_data = decode_data(data)

                print(decoded_data)

                normalized_data = np.zeros(5)

                for idx, dat in enumerate(decoded_data):
                    normalized_data[idx] = (dat-infra_red_range[idx, 2])/(infra_red_range[idx, 1]-infra_red_range[idx, 2])

                print(normalized_data)

                print(normalized_limit)

                line_position = np.argmin(normalized_data)

                if normalized_data[line_position] > (normalized_limit[line_position]*1.5):
                    if last_line_position == 0:
                        action = 0
                    elif last_line_position == 1:
                        action = 3
                    elif last_line_position == 2:
                        action = 6

                else:
                    if line_position < 1:
                        last_line_position = 0
                    elif line_position > 3:
                        last_line_position = 2
                    else:
                        last_line_position = 1

                    action = line_position + 1
                #action = 2
                print(action)             
                conn.send(bytes(str(action),"ascii"))
    