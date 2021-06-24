import wx
import numpy as np
import pickle
from narsil.tracking.siamese.trainDatasets import oneSet

class mainWindow(wx.Frame):
    """ Main windows that helps loading data from one set of tracks 
    and annotate them
    """
    def __init__(self, parent, title):
        super(mainWindow, self).__init__(parent, title=title, size=(1000, 1000))
        self.InitUI()
        self.pathname = ''
        self.oneset = None
        self.idx1 = 0
        self.idx2 = 1
        self.blobState = "normal"
        self.links = {}
        self.leftBbox = -1
        self.rightBbox = -1


    def InitUI(self):

        # Menubar stuff
        menubar = wx.MenuBar()
        fileMenu = wx.Menu()
        fileItem = fileMenu.Append(wx.ID_EXIT, 'Quit', 'Quit Application')
        menubar.Append(fileMenu, '&File')
        self.SetMenuBar(menubar)
        self.Bind(wx.EVT_MENU, self.onQuit, fileItem)

        # Buttons to load, scroll, change, save, etc
        self.panel = wx.Panel(self)
        self.panel.SetBackgroundColour('#FFFFFF')
        hbox = wx.BoxSizer(wx.HORIZONTAL)
        btnLoad = wx.Button(self.panel, label='Load', size=(150, 30))
        hbox.Add(btnLoad, flag=wx.LEFT|wx.TOP, border = 5)
        btnNext = wx.Button(self.panel, label='Next', size=(150, 30) )
        hbox.Add(btnNext, flag = wx.LEFT|wx.TOP, border = 5)
        btnReloadCurrent = wx.Button(self.panel, label='Relaod Current Step', size=(150, 30))
        hbox.Add(btnReloadCurrent, flag= wx.LEFT| wx.TOP, border=5)
        btnClose = wx.Button(self.panel, label='Close', size=(150, 30))
        hbox.Add(btnClose, flag = wx.LEFT | wx.TOP, border = 5)
        btnSave = wx.Button(self.panel, label = 'Save', size=(150, 30))
        hbox.Add(btnSave, flag = wx.LEFT | wx.TOP, border = 5)
        btnPrintMatrix = wx.Button(self.panel, label = 'Print Matrix', size=(150, 30))
        hbox.Add(btnPrintMatrix, flag = wx.LEFT | wx.TOP, border = 5)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        btnShowBoxes = wx.Button(self.panel, label = 'Show Boxes', size=(150, 30))
        hbox1.Add(btnShowBoxes, flag = wx.LEFT | wx.TOP, border = 5)

        vbox_buttons = wx.BoxSizer(wx.VERTICAL)
        #Normal segmentation
        self.normal = wx.RadioButton(self.panel, label = 'Normal', pos=(20, 300), style=wx.RB_GROUP)
        self.normal.Bind(wx.EVT_RADIOBUTTON, self.setBlobState)
        self.badSeg = wx.RadioButton(self.panel, label = 'Bad Segmentation', pos=(20, 320))
        self.badSeg.Bind(wx.EVT_RADIOBUTTON, self.setBlobState)
        self.notCell = wx.RadioButton(self.panel, label = 'Not Cell', pos=(20, 340))
        self.notCell.Bind(wx.EVT_RADIOBUTTON, self.setBlobState)
        self.ignoreCell = wx.RadioButton(self.panel, label = 'Ignore Cell', pos = (20, 360))
        self.ignoreCell.Bind(wx.EVT_RADIOBUTTON, self.setBlobState)
        vbox_buttons.Add(self.normal)
        vbox_buttons.Add(self.badSeg)
        vbox_buttons.Add(self.notCell)
        vbox_buttons.Add(self.ignoreCell)

        #button bindings
        self.Bind(wx.EVT_BUTTON, self.load,id = btnLoad.GetId())
        self.Bind(wx.EVT_BUTTON, self.next,id = btnNext.GetId())
        self.Bind(wx.EVT_BUTTON, self.reload,id = btnReloadCurrent.GetId())
        self.Bind(wx.EVT_BUTTON, self.close, id = btnClose.GetId())
        self.Bind(wx.EVT_BUTTON, self.save, id = btnSave.GetId())
        self.Bind(wx.EVT_BUTTON, self.printMatrix, id = btnPrintMatrix.GetId())
        

        # show dummy images
        self.firstImage = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap("gui/1.jpeg", wx.BITMAP_TYPE_ANY))
        self.secondImage = wx.StaticBitmap(self.panel, wx.ID_ANY, wx.Bitmap("gui/2.png", wx.BITMAP_TYPE_ANY))
        self.firstImage.SetPosition((300, 50))
        self.secondImage.SetPosition((500, 50))
        self.firstImage.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.secondImage.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.currentLine = [0, 0, 0 , 0]

        # set panel sizer for the button hbox
        vbox = wx.BoxSizer(wx.VERTICAL)
        vbox.Add(hbox)
        vbox.Add(hbox1)
        vbox.Add(vbox_buttons)
        self.panel.SetSizer(vbox)

    def setBlobState(self, e):
        normal_set = self.normal.GetValue()
        badSeg_set = self.badSeg.GetValue()
        notCell_set = self.notCell.GetValue()
        ignoreCell_set = self.ignoreCell.GetValue()
        if normal_set == True:
            print("Normal cell")
            self.blobState = "normal"
        elif badSeg_set == True:
            print("Bad segmentation")
            self.blobState = "badSeg"
        elif notCell_set == True:
            print("Not cell set")
            self.blobState = "notCell"
        elif ignoreCell_set == True:
            print("Ignore cell set")
            self.blobState = "ignoreCell"



    def loadImages(self, idx1, idx2):
        self.img1 = wx.Image(self.oneset[idx1]['filename'], wx.BITMAP_TYPE_TIFF).ConvertToBitmap()
        self.img2 = wx.Image(self.oneset[idx2]['filename'], wx.BITMAP_TYPE_TIFF).ConvertToBitmap()
        self.firstImage.SetBitmap(wx.Bitmap(self.img1))
        self.secondImage.SetBitmap(wx.Bitmap(self.img2))
        self.dict_index = str(idx1) + "_" + str(idx2)
        self.links[self.dict_index] = np.zeros((len(self.oneset[idx1]['bbox']), len(self.oneset[idx2]['bbox'])))
        self.local_links_OF = np.array([], dtype = int)



    def findObjectNumber(self, idx, x, y):
        # This function returns the object number on the
        bbox_list = self.oneset[idx]['bbox']
        for i in range(len(bbox_list)):
            if y >= bbox_list[i][0] and y < bbox_list[i][2]:
                return (i, bbox_list[i][2] - bbox_list[i][0], bbox_list[i][3] - bbox_list[i][1], bbox_list[i][1], bbox_list[i][0])


    def onLeftDown(self, e):
        x, y = e.GetPosition()
        self.currentLine[0] = x + 300
        self.currentLine[1] = y + 50
        # find the bbox here
        bbox_number, height, width, draw_x, draw_y  = self.findObjectNumber(self.idx1, x, y)
        print("Bbox1 : ", bbox_number)
        self.leftBbox = bbox_number
        # draw the rectangle of the bbox
        self.dc2 = wx.ClientDC(self)
        if self.blobState == 'normal':
            self.dc2.SetPen(wx.Pen('#c56c00', 1, wx.SOLID))
        else:
            self.dc2.SetPen(wx.Pen('#004fc5', 1, wx.SOLID))
        self.dc2.DrawRectangle(draw_x + 700, draw_y + 50, width, height)

    
    def onLeftUp(self, e):
        x, y = e.GetPosition()
        self.currentLine[2] = x + 500
        self.currentLine[3] = y + 50
        self.drawCurrentLine()
        # find the bbox here
        bbox_number, height, width, draw_x, draw_y = self.findObjectNumber(self.idx2, x, y) 
        self.rightBbox = bbox_number
        print("Bbox2 : ", bbox_number)
        self.dc3 = wx.ClientDC(self)
        if self.blobState == 'normal':
            self.dc3.SetPen(wx.Pen('#c56c00', 1, wx.DOT))
        else:
            self.dc3.SetPen(wx.Pen('#004fc5', 1, wx.DOT))
        self.dc3.DrawRectangle(draw_x + 900, draw_y + 50, width, height)

        # After drawing the boxes, now set the links
        self.links[self.dict_index][self.leftBbox, self.rightBbox] = 1.0
        self.draw_links_OF = np.array([[self.idx1, self.leftBbox, self.rightBbox]])
        if self.local_links_OF.size == 0:
            self.local_links_OF = self.draw_links_OF
        else:
            self.local_links_OF = np.concatenate((self.local_links_OF, self.draw_links_OF))
        


    def printMatrix(self, e):
        print("----- Local links Matrix format ----- ")
        print(self.links[self.dict_index])
        print("----- Pipeline format: Local Links ------------")
        print(self.local_links_OF)
        print("----- Pipeline format: Globl links ------------")
        print(self.blobs_links_OF)




    def drawCurrentLine(self):
        self.dc = wx.ClientDC(self)
        self.dc.SetPen(wx.Pen('#4c4c4c', 1, wx.SOLID))
        self.dc.DrawLine(*self.currentLine)

    
    def save(self, e):
        savingFilename = self.pathname + 'links.npy'
        np.save(savingFilename, self.links)
        savingJustLinksFilename = self.pathname + 'justlinks'
        np.save(savingJustLinksFilename, self.blobs_links_OF )
        print(savingFilename, " SAVED !!!!")
        print(savingJustLinksFilename, " SAVED !!!!!")

    def load(self, e):
        # This function opens a directory dialog prompt to pick the directory containing the image sequence
        dirdialog = wx.DirDialog(self, message = "Open blob sequence directory", defaultPath='/home/batnet/Documents/trackingdata/blobsSingle/', style=wx.DD_DEFAULT_STYLE)
        if dirdialog.ShowModal() == wx.ID_CANCEL:
            return
        self.pathname = dirdialog.GetPath() + '/'
        print("Path Set:" , self.pathname)
        self.oneset = oneSet(self.pathname)
        #print("Oneset initialized:", self.oneset[0])
        print(self.oneset)
        self.idx1 = 0
        self.idx2 = 1
        self.loadImages(self.idx1, self.idx2)
        self.blobs_links_OF = np.array([], dtype = int)

    def next(self, e):
        # if not last time point increment indices
        if self.idx2 == len(self.oneset) - 1:
            wx.MessageBox('Sequence completed', 'Info', wx.OK | wx.ICON_STOP)
            if self.blobs_links_OF.size == 0:
                self.blobs_links_OF = self.local_links_OF
            else:
                self.blobs_links_OF = np.concatenate((self.blobs_links_OF, self.local_links_OF))
 
            return
        else:
            self.idx1 += 1
            self.idx2 +=1
            self.dc.Clear()
            if self.blobs_links_OF.size == 0:
                self.blobs_links_OF = self.local_links_OF
            else:
                self.blobs_links_OF = np.concatenate((self.blobs_links_OF, self.local_links_OF))
        self.loadImages(self.idx1, self.idx2)



    def reload(self, e):
        if self.pathname == None:
            return
        else:
            #self.idx1 = 0
            #self.idx2 = 1
            self.loadImages(self.idx1, self.idx2)
            self.dc.Clear()
            
    
    def close(self, e):
        self.Close()


    def onQuit(self, e):
        self.Close()


def run():
    app = wx.App()
    gui = mainWindow(None, title="Tracking train dataset creation")
    gui.Show()
    app.MainLoop()

#if __name__ == '__main__':
#    main()