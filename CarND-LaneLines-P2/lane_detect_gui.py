import tkinter as tk
from tkinter import ttk
import cv2
import PIL.Image, PIL.ImageTk
import time
import os
#from revision.tools_v2 import readFile
import argparse
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from frame_processing import processFrame as Det
import json
import math

Configs = {'Imageborder': {'color': 'gray10',
                         'thickness': 5},

           'Background': {'color': 'gray10'},

           'Misc': {'blueColor': '#4476a8',
                    'blueColorRGB': [68, 118, 168]}
          }
#global count_detect
#count_detect=0
def oddify(val):
    if math.floor(val) % 2 == 0:
        output = math.floor(val-1)
    else:
        output = math.floor(val)
    return output

def read_arg_filename():
    '''
    Read command line argument for video/image file to read.
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, help='Enter the file you would like to open')
    args = parser.parse_args()
    return args.filename

class readFile:
    def __init__(self, filepath):
        '''
        Class to import and store image and video variables.
        '''
        _, ext = os.path.splitext(filepath)
        if ext in ('.jpg', '.png', '.gif', '.jpeg'):
            self.image = cv2.imread(filepath)
            self.video = None
            self.is_image = True
            self.is_video = False  
        else:
            self.image = None
            self.video = cv2.VideoCapture(filepath)
            self.is_video = True  
            self.is_image = False

    def getFrame(self, width=None, height=None, play=True):
        '''
        Loop through image or video frames performing the function func.
        '''
        if self.is_video:
            try:
                ret, frame = self.video.read()
                if ret:
                    self.frame = frame
                else:
                    print('Looping...')
                    self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    _, self.frame = self.video.read()
            except:
                print("Video Ended.")
                cv2.destroyAllWindows()
                exit()

        elif self.is_image:
            self.frame = self.image
        self.setShape(width, height)
        self.frame = cv2.resize(self.frame, (self.width, self.height))
        return self.frame

    def setShape(self, width, height):
        if width != None and height != None:
            self.width = int(width)
            self.height = int(height)
        elif width != None: 
            (h, w) = self.frame.shape[:2]
            aspect = 1.0*w/h
            self.width = int(width)
            self.height = int(round(width/aspect))   
        elif height != None:
            (h, w) = self.frame.shape[:2]
            aspect = 1.0*w/h
            self.width = int(round(height*aspect))
            self.height = int(height)
        elif self.is_image:
            self.width = self.image.shape[1]
            self.height = self.image.shape[0]
        elif self.is_video:
            self.width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def setWidth(self, width):
        (h, w) = self.frame.shape[:2]
        aspect = 1.0*w/h
        new_h = lambda w: int(round(w/aspect))
        self.frame = cv2.resize(self.frame, (width, new_h(width)))

    def __del__(self):
        if self.is_video and self.video.isOpened():
            self.video.release()

def save_values(filename):
    with open(filename+".json", "w") as f:
        f.write(json.dumps(PARAM))
    print("saved values to {}.json!".format(filename))

def load_saved_values(filename):
    global PARAM
    PARAM = {}
    try:
        with open(filename+'.json', "r") as f:
            data = json.loads(f.read())
            for key, value in data.items():
                PARAM[key] = value
    except FileNotFoundError:
        print("(didn't find {}.json)".format(filename))

class livePlot:
    def __init__(self):
        #plt.ion()
        self.fig = plt.figure(figsize=(4.2, 3.2))
        self.fig.patch.set_facecolor((26.0/255,26.0/255,26.0/255))
        #self.fig.canvas.draw()

    def reload(self, x, y, xlim=800, ylim=1600):
        plt.clf()
        self.ax = self.fig.add_subplot(1,1,1)
        self.ax.plot(x, y, color=Configs['Misc']['blueColor'])
        self.ax.set_xlim([0, xlim])
        self.ax.set_facecolor((26.0/255,26.0/255,26.0/255))
        self.ax.set_ylim([0, ylim])
        self.ax.tick_params(axis='x', colors='white')
        self.ax.tick_params(axis='y', colors='#1a1a1a')
        for i in ['top', 'bottom', 'right', 'left']:
            self.ax.spines[i].set_color('#1a1a1a')
        #self.ax.tick_params(axis='y', colors='white')
        #self.ax.grid()
        #self.ax.set_title('Histogram Plot', color='white')
        #self.ax.set_ylabel('White pixel count', color='white')
        #self.ax.set_xlabel('Bin X location', color='white') 
        #self.fig.canvas.draw()

class App:
    def __init__(self, window, window_title, source='../../data/03-16-carla_vid.avi', default='/home/nickg/Pictures/revision_logos/withBackground.jpg'):
        # Necessary variables
        self.window = window
        self.window.title(window_title)
        self.source = readFile(source)
        self.panelx = (screenx-80)//(8*2) # Subtract length of side panel on homescreen
        self.panely = (screeny-80)//(6*2) # subtract length of top bar on homescreen
        self.imagex = self.panelx*2
        self.imagey = self.panely*4
        self.count = 0
        self.play_state = True

        # Setup Default image
        default_img = readFile(default)
        frame = cv2.cvtColor(default_img.getFrame(width=self.imagex, height=self.imagey), cv2.COLOR_BGR2RGB)
        photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))

        ##### Adding image panels #####
        self.add_image_panels(photo)
        
        ##### Adding Labels #####
        self.add_labels()

        ##### Adding Sliders #####
        self.add_slider_widgets()

        ##### Adding save/load widgets #####
        self.add_file_widgets()

        ##### Add my signature #####
        self.signature()

        self.delay = 10
        self.update()
        self.window.mainloop()

    def update(self):
        # Getting images from readFile object

        if self.play_state:
            frame = cv2.cvtColor(self.source.getFrame(width=self.imagex, height=self.imagey), cv2.COLOR_BGR2RGB)
            
            obj = Det(frame, PARAM)

            self.update_sliders()
            
            #color, perspective_opened, canny, threshed = obj.detect_lanes()
            
            BGR_binary, HSV_binary, HLS_binary, BGR_canny, HSV_canny, HLS_canny, combo_canny, img, combine, perspective = obj.detect_lanesV2()
            self.add_images([BGR_canny, HSV_canny, HLS_canny, combine, combo_canny, img])
            #self.frame = cv2.resize(self.frame, (self.width, self.height))
            '''
            hough = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(color))
            self.houghPanel.configure(image=hough)
            self.houghPanel.image = hough

            perspective = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(perspective_opened))
            self.perspectivePanel.configure(image=perspective)
            self.perspectivePanel.image = perspective
            
            original = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.originalPanel.configure(image=original)
            self.originalPanel.image = original
            
            canny = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(canny))
            self.cannyPanel.configure(image=canny)
            self.cannyPanel.image = canny
            
            color = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(threshed))
            self.colorPanel.configure(image=color)
            self.colorPanel.image = color
            
            '''
            # Histogram Plot DO NOT DELETE
            
            opened_height = perspective.shape[0]
            opened_width = perspective.shape[1]
            hist = np.count_nonzero(perspective[opened_height // 6:, :], axis=0)
            self.histPlot.reload(np.arange(opened_width), hist, xlim=self.imagex, ylim=self.imagey*2)
            self.histPanel.draw()
            
            '''
            # Perspective combobox and slider update
            if self.perspectPt_combobox.get() == "":
                self.perspectiveX_slider.set(PARAM['perspectPt_a'][0])
                self.perspectiveY_slider.set(PARAM['perspectPt_a'][1])
            else:
                self.perspectiveX_slider.set(float(PARAM[self.perspectPt_combobox.get()][0]))
                self.perspectiveY_slider.set(float(PARAM[self.perspectPt_combobox.get()][1]))
            '''
        if 'normal' == self.window.state():
            self.id = self.window.after(self.delay, self.update)

    def play(self):
        self.play_state=True
        print('play')

    def pause(self):
        self.play_state=False
        print('pause')

    def add_slider_widgets(self):
        ##### Setting up sliders #####
        common_configs = {'orient': tk.HORIZONTAL,
                          'width': 10,
                          'length': self.panelx,
                          'sliderlength': 10,
                          'bg': Configs['Misc']['blueColor'],
                          'troughcolor': Configs['Background']['color']}
        
        self.sliders = []
        for i in range(8):
            self.sliders.append([])
            for j in range(4):
                self.sliders[i].append(tk.Scale(self.window, common_configs, label='unused'))

        count = 0
        for item in ['BGR_high', 'BGR_low', 'HSV_high', 'HSV_low', 'HLS_high', 'HLS_low']:
            for i in range(3):
                self.add_slider(item, count, i, 0, 255, list_pos=i)
            count+=1

        self.add_slider('cannyThresh1', 7, 0, 0, 255)
        self.add_slider('cannyThresh2', 7, 1, 0, 255)
        
        # Perspective AOI sliders
        self.add_slider('AOIbasewidth', 7, 0, 1, self.imagex)
        self.add_slider('AOIhorizonHeight', 7, 1, 1, self.imagey)
        self.add_slider('AOIbaseOffset', 7, 2, 1, self.imagey)
        self.add_slider('AOIlinewidth', 7, 3, 1, self.imagex)

                # Perspective pts
        #self.perspectPt_combobox = ttk.Combobox(self.window, values=['', 'perspectPt_a', 'perspectPt_b', 'perspectPt_c', 'perspectPt_d'])
        #self.perspectPt_combobox.grid(row=10, column=4, pady=5, sticky=tk.W)
        
        #self.perspectiveX_slider = tk.Scale(self.window, common_configs, from_=0, to=self.imagex, label='X')
        #self.perspectiveX_slider.grid(row=10, column=5, sticky=tk.E, pady=5)
        #self.perspectiveX_slider.set(PARAM['AOIhorizonHeight'])

        #self.perspectiveY_slider = tk.Scale(self.window, common_configs, from_=0, to=self.imagey, label='Y')
        #self.perspectiveY_slider.grid(row=10, column=6, sticky=tk.W, pady=5)

    def add_file_widgets(self):
        self.input_filename_entry = tk.Entry(self.window)
        self.input_filename_entry.grid(row=1, column=6)

        self.output_filename_entry = tk.Entry(self.window)
        self.output_filename_entry.grid(row=2, column=6)

        self.load_button = tk.Button(self.window, command=self.load_config_file, text='Load')
        self.load_button.grid(row=1, column=7, sticky=tk.W)

        self.save_button = tk.Button(self.window, command=self.save_config_file, text='Save')
        self.save_button.grid(row=2, column=7, sticky=tk.W)

        self.play_button = tk.Button(self.window, command=self.play, text='Play')
        self.play_button.grid(row=0, column=6)

        self.pause_button = tk.Button(self.window, command=self.pause, text='Pause')
        self.pause_button.grid(row=0, column=7)

    def save_config_file(self):
        save_values(self.output_filename_entry.get())
        print('Saved configs to {}.json'.format(self.output_filename_entry.get()))

    def load_config_file(self):
        load_saved_values(self.input_filename_entry.get())
        print('loaded {}.json'.format(self.input_filename_entry.get()))

    def signature(self):
        self.my_signature = tk.Label(self.window, bg=Configs['Background']['color'], text='Developed by Nick Goberville, CEO Revision Autonomy LLC')
        self.my_signature.grid(row=0, column=6, columnspan=2, sticky=tk.N)

    def add_image_panels(self, photo):
        common_configs = {'highlightbackground': Configs['Imageborder']['color'],
                          'highlightthickness': Configs['Imageborder']['thickness'],
                          'width': self.imagex,
                          'height': self.imagey,
                          'bd': 0,
                          'image': photo}
        self.image_panels = []

        # ColorPanel
        self.imagePanel_1 = tk.Label(self.window, common_configs)
        self.imagePanel_1.grid(row=0, column=0, rowspan=4, columnspan=2)
        self.image_panels.append(self.imagePanel_1)      
        # CannyPanel
        self.imagePanel_2 = tk.Label(self.window, common_configs)
        self.imagePanel_2.grid(row=4, column=0, rowspan=4, columnspan=2)      
        self.image_panels.append(self.imagePanel_2)
        # HoughPanel
        self.imagePanel_3 = tk.Label(self.window, common_configs)
        self.imagePanel_3.grid(row=8, column=0, rowspan=4, columnspan=2)
        self.image_panels.append(self.imagePanel_3)
        # Original image panel
        self.imagePanel_4 = tk.Label(self.window, common_configs)
        self.imagePanel_4.grid(row=0, column=2, rowspan=4, columnspan=2)
        self.image_panels.append(self.imagePanel_4)
        # 2 extra images instead of large perspective
        self.imagePanel_5 = tk.Label(self.window, common_configs)
        self.imagePanel_5.grid(row=4, column=2, rowspan=4, columnspan=2)
        self.image_panels.append(self.imagePanel_5)

        self.imagePanel_6 = tk.Label(self.window, common_configs)
        self.imagePanel_6.grid(row=8, column=2, rowspan=4, columnspan=2)
        self.image_panels.append(self.imagePanel_6)        

        # Large PerspectivePanel
        #self.perspectivePanel = tk.Label(self.window, common_configs, 
        #                                width=self.imagex, height=self.imagey*2 + Configs['Imageborder']['thickness']*2)
        #self.perspectivePanel.grid(row=4, column=2, rowspan=8, columnspan=2, sticky=tk.N)
        # histogramPanel
        self.histPlot = livePlot()
        self.histPanel = FigureCanvasTkAgg(self.histPlot.fig, master=self.window)
        self.histPanel.get_tk_widget().grid(row=0, column=4, rowspan=4, columnspan=2, sticky='NSEW')
        self.histPanel.draw()

    def add_labels(self):
        self.colorLabel = tk.Label(self.window, text='Color Threshing', fg='white', bg='black', 
                                    highlightthickness=Configs['Imageborder']['thickness'], highlightbackground='black')
        self.colorLabel.grid(row=0, column=0, sticky=tk.N) 

    
    def update_sliders(self):
        perspect_list = ['perspectPt_a', 'perspectPt_b', 'perspectPt_c', 'perspectPt_d']
        label_list = {}
        for i in range(len(self.sliders)):
            for j in range(len(self.sliders[i])):
                label = self.sliders[i][j].config()['label'][-1]
                if label != 'unused':
                    if label in label_list.keys() and label not in perspect_list:
                        label_list[label].append([i, j])
                    else: 
                        label_list[label] = [[i, j]]

        for key in label_list.keys():
            for i in range(len(label_list[key])):
                row = label_list[key][i][0]
                col = label_list[key][i][1]
                PARAM[key][i] = int(self.sliders[row][col].get())

        '''
        for i in range(len(self.sliders)):
            for j in range(len(self.sliders[i])):
                label = self.sliders[i][j].config()['label'][-1]
                label_list.append(label)
                if label == "unused":
                    continue
                vals = []
                if label == last_label:
                    vals.append(int(self.sliders[i][j].get()))
                PARAM[label] = vals
        '''
        '''
        PARAM['laneThickness'] = int(self.laneThickness_slider.get())
        PARAM['cannyThresh1'] = self.cannyThresh1_slider.get()
        PARAM['cannyThresh2'] = self.cannyThresh2_slider.get()
        PARAM['AOIlinewidth'] = self.AOIlinewidth_slider.get()
        PARAM['adaptivemaxVal'] = int(self.adaptivemaxVal_slider.get())
        PARAM['adaptiveblockSize'] = int(oddify(self.adaptiveblockSize_slider.get()))
        PARAM['adaptiveC'] = self.adaptiveC_slider.get()
        PARAM['morphopenSize'] = self.morphopenSize_slider.get()
        PARAM['morphcloseSize'] = self.morphcloseSize_slider.get()
        PARAM['houghResolution'] = self.houghResolution_slider.get()
        PARAM['houghPiRes'] = self.houghPiRes_slider.get()
        PARAM['houghThresh'] = self.houghThresh_slider.get()
        PARAM['houghLength'] = self.houghLength_slider.get()
        PARAM['houghGap'] = self.houghGap_slider.get()
        PARAM['AOIbaseOffset'] = self.AOIbaseOffset_slider.get()
        PARAM['AOIhorizonHeight'] = self.AOIhorizonHeight_slider.get()
        '''
        '''
        if self.count==0:
            self.history = []

        
        # correctly update perspective slider
        if len(self.history) >= 1:
            if self.perspectPt_combobox.get() == self.history[-1]:
                perspX = int(self.perspectiveX_slider.get())
                perspY = int(self.perspectiveY_slider.get())
                newpt = [perspX, perspY]
                PARAM[self.perspectPt_combobox.get()] = newpt
                del self.history[0]
                self.history.append(self.perspectPt_combobox.get())
            else:
                del self.history[0]
                self.history.append(self.perspectPt_combobox.get())
        else:
            self.history.append(self.perspectPt_combobox.get())
        

        self.count+=1
        '''
    def add_images(self, images):
        count = 0
        for image in images:
            img = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(image))
            self.image_panels[count].configure(image=img)
            self.image_panels[count].image = img
            count+=1
    
    def add_slider(self, val, i, j, from_, to, list_pos=0):
        self.sliders[i][j].configure(from_=from_, to=to, label=val)
        self.sliders[i][j].grid(row=i+4, column=j+4, sticky=tk.N, pady=5)
        self.sliders[i][j].set(PARAM[val][list_pos])
    #def snapshot(self):
    #  # Get a frame from the video source
    #    frame = self.source.getFrame(width=300, height=10)
    #    cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

PARAM = {}
load_saved_values('parameters')
#load_saved_values('canny_params')

main_window = tk.Tk()
screenx = main_window.winfo_screenwidth()
screeny = main_window.winfo_screenheight()

main_window.geometry('{}x{}'.format(str(screenx), str(screeny)))
main_window.configure(background=Configs['Background']['color'])

menu=tk.Menu(main_window)
main_window.config(menu=menu)
fileobj = tk.Menu(menu)
fileobj.add_command(label="Save", command=save_values)

App(main_window, "Lane Detection Parameter Assistant", source=read_arg_filename())