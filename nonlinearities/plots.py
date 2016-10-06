from __future__ import division

import glob
import os
import sys
import collections

from IPython.display import HTML

import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
mpl.rcParams['figure.figsize'] = 16,10
mpl.rcParams['lines.linewidth']=2

import tfutils
import cifar10.nonlinearities as nls

def code_style():
    return HTML("<style>code {background-color:#EEE !important; border:#CCC !important; padding:3px !important}</style>")



def load(root):

    runs = tfutils.PathDict()
    for dirname in glob.glob(os.path.join(root,'*')):
        runs[dirname.split('/')[1]] = tfutils.ScalarSummaries.join(
            os.path.join(dirname,"events.*"))

    cleanup = tfutils.PathDict()
    for path,value in runs.items():
        parts = path.split('/')
        if len(parts) != 4:
            continue

        nl,tv,_,_ = parts
        cleanup[tv,nl] = value

    return cleanup

class Plotter(object):
    def __init__(self,summaries):
        self.train = summaries.train
        self.valid = summaries.validate

    styles = collections.OrderedDict([
     ['linear','k'],
     ['tanh','cv'],
     ['ReLU','b^'],
     ['ReLU.down','bv'],
     ['ReLU.tangent','bs'],
     ['ReLU.shift','bo'],
     ['LReLU','b*'],
     ['ELU','gv'],
     ['ELU.up','g^'],
     ['softplus2','y^'],
     ['softplus2.down','yv'],
     ['bilinear','r*'],
     ['maxout','rs'],
     ['PReLU','ro']
    ])

    def plot(self, name, nls=None, xunits='epochs'):
        self._axes()

        if xunits == 'time':
            plt.xlim([0.25,16])
            plt.xlabel('Time [h]')
        elif xunits == 'epochs':
            plt.xlim([1,75])
            plt.xlabel('epochs')


        if nls is None:
            nls = self.train.keys()

        lines = self._single(self.train,nls,xunits,linestyle='-')

        plt.legend(lines,nls,loc='lower left')

        self._single(self.valid,nls,xunits,linestyle='--')

        plt.savefig(name + '.png')
        plt.close()

    @staticmethod
    def _axes():
        plt.figure()
        A = plt.axes()
        A.grid(True)

        plt.ylabel('Cross Entropy [digits]')
        plt.ylim([0.25,0.75])




    def _single(self,runs,select=None,xunits='epochs',**plotkwargs):
        steps = []

        #TODO: this is **BAD** duplicating constants, I just want to fix the plots.
        batches_per_report = 10
        batch_size = 128
        epoch_size = 50000

        if xunits == 'time':
            getx=lambda event:(event.time+10)/3600
            xticks = [1,2,4,6,8,12,16,20,24]
        elif xunits=='epochs':
            getx=lambda event:np.arange(1,len(event.time)+1)*batch_size*batches_per_report/epoch_size
            xticks = [1,3,10,20,30,40,50,75,100]
        else:
            raise ValueError('xunits must be in {"time","steps"}')

        if not select:
            select = runs.keys()

        lines = []
        for key in select:
            event = runs[key]
            lines.extend([getx(event),event.value])
            lines.append(self.styles[key])

        lines=plt.plot(*lines,markevery=200,markersize=10,markeredgecolor='k',**plotkwargs)

        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_xticks(xticks)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())


        return lines

if __name__ == "__main__":
    nls.plot()
    plt.savefig('nonlinearities.png')
    plt.close()

    scalar_summaries = load('logs')
    P = Plotter(scalar_summaries)
    
    P.plot('Glorot',['ReLU','softplus2','ELU.up','ReLU.shift','linear'],*sys.argv[1:])

    P.plot('Xu',["ReLU","PReLU","LReLU",'linear'],*sys.argv[1:])
    
    P.plot('Clevert',["ReLU","ELU","ReLU.down","softplus2.down","tanh",'linear'],*sys.argv[1:])
    
    P.plot('Piecewise',["ReLU",'bilinear','maxout','linear'],*sys.argv[1:])
