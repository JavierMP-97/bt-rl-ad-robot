import socket



if __name__ == '__main__':
    TCP_IP = ''
    TCP_PORT = 5000
    BUFFER_SIZE = 2

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((TCP_IP, TCP_PORT))
    s.listen(1)
    s.settimeout(20)

    while 1:

        i=0
        f_giro=0
        ultimo_LT=1

        conn, addr = s.accept()
        last_data=b'\xc83'
        while 1:
            data = conn.recv(BUFFER_SIZE)
            ''' if data!= last_data:
                print(data)
                break
            '''

            if len(data)==0:
                break

            md = data[0]

            LT=data[1]-1

            LT_L = (LT & 4)>>2
            LT_M = (LT & 2)>>1
            LT_R = (LT & 1)

            print(data)

            print(md)
            print(LT_L,LT_M,LT_R)
            #print(LT_M)
            #print(LT_R)


            if f_giro==1:
                if(not LT_R):
                    conn.send(b'4')
                else:
                    f_giro=0
                    conn.send(b'2')

            elif f_giro==2:
                if(not LT_L):
                    conn.send(b'0')
                else:
                    f_giro=0
                    conn.send(b'2')
            
            elif f_giro==0:

                if md<15:
                    if i>250:
                        if LT_L:
                            f_giro=1
                        else:
                            f_giro=2
                        i=0
                    else:
                        i+=1
                    conn.send(b'5')
                else:
                    i=0
                    if LT_M:
                        ultimo_LT=1
                        conn.send(b'2')
                    elif LT_L:
                        ultimo_LT=0
                        conn.send(b'1')
                    elif LT_R:
                        ultimo_LT=2
                        conn.send(b'3')
                    else:
                        if ultimo_LT==0:
                            conn.send(b'0')
                        elif ultimo_LT==1:
                            conn.send(b'5')
                        elif ultimo_LT==2:
                            conn.send(b'4')


        conn.close()

    #r = bytes([request.data[0]+1])    

    