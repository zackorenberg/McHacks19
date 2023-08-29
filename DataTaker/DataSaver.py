
import os
import imageio
import time
import LocalVars



class DataSaver(object):
    def __init__(self):
        self.path = None
        pass


    def createSavePath(self,path):

        self.path = os.path.join(path,str(time.time())) + os.sep
        if LocalVars.RecordData:
            os.mkdir(self.path)
        return self.path

    def createGif(self):
        if not LocalVars.SAVE_GIF:
            return
        if self.path is None:
            return
        images = [ ]
        with imageio.get_writer(os.path.join(self.path,"simulation.gif"), mode='I') as writer:
            for f in os.listdir(self.path):
                if ".png" in f:
                    image = imageio.imread(f)
                    writer.append_data(image)

    def saveData(self,data):
        lines = []
        for dat in data:
            #print(dat)
            lines.append( ",".join([str(dat[0]),str(dat[1])]) + "\n" )
        #print(lines)

        if LocalVars.RecordData:
            with open(os.path.join(self.path,"temps.dat"), "w", newline='\n') as f:
                f.writelines(lines)




