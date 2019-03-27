#min max to tic tac toi

import numpy as np
import math
import imgLib 
import sys
import baselib
import matplotlib
from rpyc.utils.server import ThreadedServer
import rpyc
import re 



class MyService(rpyc.Service):
    def on_connect(self, conn):
        # trecho que executa quando uma conexão e criada
        pass
    
    def on_disconnect(self, conn):
        # trecho que executa após conexão encerrada
        pass
    
    def exposed_get_size(self, imgG): # metodo publico
        return baselib.size(imgG)
    
    def exposed_get_Response(self, response, rmax, rmin): # metodo publico
        resp = ((response - rmin)/(rmax-rmin)) * 1000       
        return resp
    
#    def exposed_get_keyPoint(self,keyPointsList, w, h, img2): # metodo publico
#        for point in keyPointsList:
#		#cv.circle(img, (point[1],point[0]) , r, (0,0,255), -1)
#                i, j = point
#	
#                for u in range(-1,2):
#                        for v in range(-1,2):
#                                x = min(max(j + v, 0), w-1)
#                                y = min(max(i + u, 0), h-1)
#                                img2[y][x] = (255,0,0)
#        return img2


    exposed_last_year = 2018  # atributo publico
    
    def get_question(self):  # metodo privado
        return "Em que ano estamos?"

if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    t = ThreadedServer(MyService, port=18861, protocol_config = {'allow_all_attrs' : True})
    t.start()
