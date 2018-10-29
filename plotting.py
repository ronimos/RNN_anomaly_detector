# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 21:24:04 2018

@author: Ron Simenhois
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D


class LivePlot(object):
    
    def __init__(self, data):
        self.data = data
        self.xdata = []
        self.ydata = []
        self.pdata = []
        self.edata = []
        self.time = 0
        
        self.fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, figsize=(12, 8))
        
        self.ax1 = ax1
        self.ax1.grid()
        self.ax1.set_title('True signal')
        self.ax1.set_xlim(0, 10)
        self.ax1.set_ylim(-2, 2)        
        self.line1, = self.ax1.plot([], [], color='green', lw=2)
        self.line1.set_data(self.xdata, self.ydata)
        
        self.ax2 = ax2
        self.ax2.grid()
        self.ax2.set_title('prediction')
        self.ax2.set_xlim(0, 10)
        self.ax2.set_ylim(-2, 2)
        self.line2, = self.ax2.plot([], [], color='black', lw=2)
        self.line2.set_data(self.xdata, self.pdata)
        
        self.ax3 = ax3
        self.ax3.grid()
        self.ax3.set_title('Square Error')
        self.ax3.set_xlim(0, 10)
        self.ax3.set_ylim(0, 1)
        self.line3, = self.ax3.plot([], [], color='red', lw=2)
        self.line3.set_data(self.xdata, self.edata)
    
    def init(self):
        
        self.time, _,_,_ = next(self.data)
        xmin, xmax = max(0, self.time-10), max(10, self.time)
        self.ax1.set_xlim(xmin, xmax)
        self.ax2.set_xlim(xmin, xmax)
        self.ax3.set_xlim(xmin, xmax)
        del self.xdata[:]
        del self.ydata[:]
        del self.pdata[:]
        del self.edata[:]
        
        return self.line1, self.line2, self.line3
    
    def draw_frame(self, i):
        
        self.time, y, p, e = next(self.data)
        self.xdata.append(self.time)
        self.ydata.append(y)
        self.pdata.append(p)
        self.edata.append(e)
        
        xmin, xmax = self.ax1.get_xlim()
        if self.time >= xmax:
            self.ax1.set_xlim(xmin+0.1, xmax+0.1)
            self.ax2.set_xlim(xmin+0.1, xmax+0.1)
            self.ax3.set_xlim(xmin+0.1, xmax+0.1)
        
        self.ax1.figure.canvas.draw()
        self.line1.set_data(self.xdata, self.ydata)
        self.ax2.figure.canvas.draw()
        self.line2.set_data(self.xdata, self.pdata)
        self.ax3.figure.canvas.draw()
        self.line3.set_data(self.xdata, self.edata)
        
        return self.line1, self.line2, self.line3
    
    def run(self):
        self.ani = FuncAnimation(self.fig, 
                                 self.draw_frame, 
                                 init_func=self.init, 
                                 interval=10, 
                                 blit=False)
        plt.tight_layout()
        plt.show()
        return self.ani
    
class Plot3D:
    def __init__(self, **kwargs):
        
        self.fig = plt.figure()
        
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.xlabel = kwargs.get('xlabel', '')
        self.ylabel = kwargs.get('ylabel', '')
        self.zlabel = kwargs.get('zlabel', '')        
        
    def plot(self, x, y, z, label=''):
        self.ax.plot(x,y,z, label=label)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.ax.set_zlabel(self.zlabel)

if __name__=='__main__':
    
    def data_gen(length=np.inf,t=0):
        
        while length > 0:
            t += 00.1
            length -= 1
            y = np.sin(2*np.pi*t)
            p = np.random.uniform(-0.5,0.5)
            e = y + p
            yield t, y, p, e
    
   
    data_generator = data_gen()
    
    ploter = LivePlot(data_generator)
    ani = ploter.run()
    
    
    
    
    p = Plot3D(**{'xlabel':'xlabel','ylabel':'ylabel','zlabel':'zlabel'})
    theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
    z = np.linspace(-2, 2, 100)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = r * np.cos(theta)
    p.plot(x+0.1,y+0.1,z+0.1, label='parametric curve')
