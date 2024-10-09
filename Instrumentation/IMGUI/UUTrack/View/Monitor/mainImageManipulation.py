"""
    UUTrack.View.mainImageManipulation.py
    ========================================
    Manipulates the main image seen in the GUI with backgroundcorrections and variance views.

    .. sectionauthor:: Kevin Namink <k.w.namink@uu.nl>
"""

import numpy as np



class mainImageManip():
    """ Manipulates the main image seen in the GUI.
        Options: different bgcorrections and variance view
    """
    
    def __init__(self, size):
        self.messageList = []
        self.imsizex = 0
        self.imsizey = 0
        
        self.bgON = False
        self.bgmode = 0  
        self.bgmodeN = 5  # number of available bg modes
        self.bgimage = np.array([])
        self.bgarray = np.array([])
        self.bgarraylen = 20
        self.bgarrayiter = 0
        
        self.varON = False
        self.vararray = np.array([], dtype = np.float64)
        self.vararray_length = 20
        self.vararray_location = 0
        
        self.contrastON = False
        self.contrastarray_length = size
        self.contrastarray = np.array([], dtype = np.float64)
        self.contrastimage = np.array([], dtype = np.float64)
        
    
    def toggleBG(self, image):
        if self.bgON == True:
            self.bgON = False
            self.messageList.append('Background reduction deactivated')
            return self.bgON
        elif self.contrastON == True:
            self.messageList.append('Background cannot be activated while contrast is active')
            return self.varON
        else:
            if len(image>0):
                self.bgON = True
                self.initBG(image)
                self.messageList.append('Background reduction active')
                return self.bgON
            self.messageList.append("Background reduction couldn't activate: no image")
            return self.bgON
            
    def toggleVar(self, image):
        if self.varON == True:
            self.varON = False
            self.messageList.append('Variance view deactivated')
            return self.varON
        elif self.contrastON == True:
            self.messageList.append('Variance view cannot be activated while contrast is active')
            return self.varON
        else:
            if len(image>0):
                self.varON = True
                self.initVar(image)
                self.messageList.append('Variance view active')
                return self.varON
            self.messageList.append("Variance view couldn't activate: no image")
            return self.varON

    def toggleContrast(self, image):
        if self.contrastON == True:
            self.contrastON = False
            self.messageList.append('Contrast view deactivated')
            return self.contrastON
        elif self.varON == True:
            self.messageList.append('Contrast view cannot be activated while variance view is active')
            return self.contrastON
        elif self.bgON == True:
            self.messageList.append('Contrast view cannot be activated while background correction is active')
            return self.contrastON
        else:
            if len(image)>0:
                self.contrastON = True
                self.initContrast(image)
                self.messageList.append('Contrast view active')
                return self.contrastON
            self.messageList.append("Contrast view couldn't activate: no image")
            return self.contrastON
     
    def hasMessage(self):
        if len(self.messageList) > 0:
            return True
        else:
            return False
    
    def giveMessage(self):
        message = self.messageList[-1]
        self.messageList = self.messageList[:-1]
        return message
    
    def ROIchange(self, image):
        self.initBG(image)
        self.initVar(image)
        
    
    def initContrast(self, image):
        self.imsizex = len(image[:,0])
        self.imsizey = len(image[0,:])
        
        if not self.contrastON:
            return
        
        self.contrastimage = np.zeros_like(image)
        self.contrastvoltagearray = np.zeros(self.contrastarray_length)
        self.contrastarray = np.zeros((self.imsizex, self.imsizey, self.contrastarray_length))
        self.messageList.append('Initialized contrast view array')
        
    def initVar(self, image):
        self.imsizex = len(image[:,0])
        self.imsizey = len(image[0,:])
        
        if not self.varON:
            return
        
        for i in range(self.vararray_length):
            self.vararray = np.append(self.vararray, np.zeros((self.imsizex, self.imsizey)) )
        self.vararray = np.reshape(self.vararray, (self.vararray_length, self.imsizex, self.imsizey))
        self.messageList.append('Initialized variance view array')
        
    def nextBGmode(self, image):
        self.bgmode += 1
        if self.bgmode == self.bgmodeN:
            self.bgmode = 0
        self.initBG(image)
    
    def initBG(self, image):
        self.imsizex = len(image[:,0])
        self.imsizey = len(image[0,:])
        self.bgimage = image.astype(np.float64)
        
#        if not self.bgON:
#            return
        
        if self.bgmode == 0:
            self.bgimage = image.astype(np.float64)
            self.messageList.append("Set current image as BG.")
            
        if self.bgmode == 1:
            self.bgimage = image.astype(np.float64)
            self.messageList.append("Set every previous image as BG.")
            
        if self.bgmode == 2:
            self.bgarray = np.zeros([self.bgarraylen, self.imsizex, self.imsizey])
            for i in range(self.bgarraylen):
                self.bgarray[i] = image
            self.messageList.append("Set mean of 20 last frames as BG.")
            
        if self.bgmode == 3:
            self.bgarray = np.zeros([self.bgarraylen, self.imsizex, self.imsizey])
            for i in range(self.bgarraylen):
                self.bgarray[i] = image
            self.messageList.append("Showing mean of 20 frames")
            
        if self.bgmode == 4:
            self.bgarray = np.zeros([self.bgarraylen, self.imsizex, self.imsizey])
            for i in range(self.bgarraylen):
                self.bgarray[i] = image
            self.messageList.append("Set every %d-th frame before as BG." %(self.bgarraylen))
            
            
    def updateContrast(self, picture, voltagearray, clock):
        i = clock%self.contrastarray_length  # Next index
        
        self.contrastvoltagearray = voltagearray
        self.contrastarray[:,:,i] = picture.astype(np.float64)  # Update values
    
    
    def update(self, picture, binning):
        processed = picture.astype(np.float64)
        
        if self.contrastON == True:
            
            v_period_est = self.contrastarray_length/(np.argmax(np.abs(np.fft.rfft(self.contrastvoltagearray)[3:]))+3)  # Estimate period length
            
            v_max_list = [np.argmax(self.contrastvoltagearray[:int(v_period_est)])]
            while(v_max_list[-1]+v_period_est*1.25 < self.contrastarray_length):
                v_max_list.append(int(v_max_list[-1]+v_period_est*0.75)+np.argmax(self.contrastvoltagearray[int(v_max_list[-1]+v_period_est*0.75):int(v_max_list[-1]+v_period_est*1.25)]))
            v_period = np.mean(np.diff(v_max_list))  # Calculated period length
            
            
            cutoff = int(self.contrastarray_length%v_period)+1  # Calculate cutoff to have full periods
            
            voltedit = self.contrastvoltagearray[:-cutoff] - np.mean(self.contrastvoltagearray[:-cutoff])  # Set average voltage to 0 and apply cutoff to voltage
            self.contrastimage = np.matmul(self.contrastarray[:,:,:-cutoff], voltedit)  # Create contrast image
            
            processed = self.contrastimage  # Send new contrastimage
                
        
        if self.bgON == True:
            if self.bgmode == 0:
                processed = processed - self.bgimage 
            elif self.bgmode == 1:
                processed = processed - self.bgimage
                self.bgimage = picture.astype(np.float64)
            elif self.bgmode == 2:
                self.bgarrayiter += 1
                if self.bgarrayiter >= self.bgarraylen:
                    self.bgarrayiter = 0
                processed = processed - np.mean(self.bgarray, axis = 0)
                self.bgarray[self.bgarrayiter] = picture.astype(np.float64)
            elif self.bgmode == 3:
                self.bgarrayiter += 1
                if self.bgarrayiter >= self.bgarraylen:
                    self.bgarrayiter = 0
                self.bgarray[self.bgarrayiter] = processed
                processed = np.mean(self.bgarray, axis = 0)
            elif self.bgmode == 4:
                self.bgarrayiter += 1
                if self.bgarrayiter >= self.bgarraylen:
                    self.bgarrayiter = 0
                processed = processed - self.bgarray[self.bgarrayiter]
                self.bgarray[self.bgarrayiter] = picture.astype(np.float64)
            
        if self.varON == True:
            self.vararray_location += 1
            if self.vararray_location == self.vararray_length:
                self.vararray_location = 0
            self.vararray[self.vararray_location] = processed
            with np.errstate(divide='ignore'):
                processed = np.var(self.vararray, axis=0)/np.mean(self.vararray, axis=0)
        
        if binning == 2:
            processed = np.repeat(np.repeat(processed, 2, axis=1), 2, axis=0)
        elif binning == 4:
            processed = np.repeat(np.repeat(processed, 4, axis=1), 4, axis=0)
        
        return processed





if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    def gibpicca():
        xx, yy = np.meshgrid(np.arange(50), np.arange(50))
        r = np.random.random((50,50))
        f = 50+10*np.sin(xx/10) + r*np.exp(-((yy-25.5)**2+(xx-20.5)**2)/30)
        return f
    
    mim = mainImageManip()
    
    mim.toggleBG(gibpicca())
    while(mim.hasMessage()):
        print(mim.giveMessage())
    
    for i in range(2):
        mim.nextBGmode(gibpicca())
        while(mim.hasMessage()):
            print(mim.giveMessage())
    
    plt.imshow(mim.update(gibpicca()))
    while(mim.hasMessage()):
        print(mim.giveMessage())
    
    
    
        
        
    