import socket
import numpy as np
import pickle

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

    finished = False
    infra_red_range = np.zeros((5,3))

    while not(finished):

        try:
            print("Waiting for connection...\n")
            conn, addr = s.accept()
        except:
            print("A connection couldn't be established. Trying again.\n")
        else:
            print("Connection established\n")

            i=0

            infra_red_range = np.zeros((5,3))

            input("Press Enter after placing robot to start floor calibration...\n")

            while i<500:
                try:
                    data = conn.recv(SERVER_BUFFER_SIZE)
                except:
                    print("No response (connection lost)\n")
                    conn.close()
                    print("Connection closed\n")
                    break

                LT_values = decode_data(data)

                for idx, val in enumerate(LT_values):
                    
                    if i == 0:
                        infra_red_range[idx, 0] = val
                        infra_red_range[idx, 1] = val
                    else:
                        if val < infra_red_range[idx, 0]:
                            infra_red_range[idx, 0] = val
                        if val > infra_red_range[idx, 1]:
                            infra_red_range[idx, 1] = val

                i += 1

                action = 0

                if i == 500:
                    action = 7
         
                conn.send(bytes(str(action),"ascii"))

            if i < 500:
                print("Floor calibraiton failed\nRestarting...\n")
                continue

            print("Floor calibraiton finished\n")

            i = 0

            input("Press Enter after placing robot to start line calibration...\n")

            while i<500:
                try:
                    data = conn.recv(SERVER_BUFFER_SIZE)
                except:
                    print("No response (connection lost)\n")
                    conn.close()
                    print("Connection closed\n")
                    break

                LT_values = decode_data(data)

                for idx, val in enumerate(LT_values):
                    
                    if i == 0:
                        infra_red_range[idx, 2] = val
                    else:
                        if val < infra_red_range[idx, 2]:
                            infra_red_range[idx, 2] = val

                i += 1

                action = 0

                if i == 500:
                    action = 7

                conn.send(bytes(str(action),"ascii"))

            if i < 500:
                print("Line calibraiton failed\nRestarting...\n")
                continue

            print("Line calibraiton finished\n")

            finished = True

    print("Saving calibration info...\n")

    with open( "calibration.pr" , "wb" ) as f:
        pickle.dump( infra_red_range, f, protocol=pickle.HIGHEST_PROTOCOL )

    print("Calibration succesfully finished\n")

    print(infra_red_range)

    



            