from flask import Flask
from flask import request

app = Flask(__name__)

i=0
f_giro=0
ultimo_LT=1

@app.route('/', methods = ['POST','GET'])
def hello_world():
    #r = bytes([request.data[0]+1])    
    global f_giro
    global i
    global ultimo_LT

    md = request.data[0]

    LT=request.data[1]-1

    LT_L = (LT & 4)>>2
    LT_M = (LT & 2)>>1
    LT_R = (LT & 1)
    
    
    #print(md)
    print(LT_L,LT_M,LT_R)
    #print(LT_M)
    #print(LT_R)
    

    if f_giro==1:
        if(not LT_R):
            return b'4'
        else:
            f_giro=0
            return b'2'

    elif f_giro==2:
        if(not LT_L):
            return b'0'
        else:
            f_giro=0
            return b'2'

    elif md<20:
        if i>50:
            if LT_L:
                f_giro=1
            else:
                f_giro=2
            i=0
            return b'0'
        i+=1
        return b'5'
    elif LT_M:
        ultimo_LT=1
        return b'2'
    elif LT_L:
        ultimo_LT=0
        return b'1'
    elif LT_R:
        ultimo_LT=2
        return b'3'
    else:
        if ultimo_LT==0:
            return b'0'
        elif ultimo_LT==1:
            return b'5'
        elif ultimo_LT==2:
            return b'4'

    return b'5'

if __name__ == '__main__':
    app.run(host="0.0.0.0")
