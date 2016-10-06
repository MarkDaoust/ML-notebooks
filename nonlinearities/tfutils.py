from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals



import collections
import glob
import os

import numpy as np

import tensorflow as tf

def xent(logits,labels,base = None):
    with tf.name_scope('xent'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits, labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        if base is None:
            return cross_entropy_mean
        else:
            return cross_entropy_mean/tf.log(float(base))

class Linear(object):
    scope_name = 'linear'

    def __init__(self,W,b):
        with tf.name_scope(self.scope_name) as scope:
            self.scope = scope
            self.W = W
            self.b = b
        
    def __call__(self,X):
        with tf.name_scope(self.scope):
            return tf.matmul(X,self.W)+self.b

class Saver(object):
    """ Simplified interface to `tf.train.Saver`

    Storing the session, path, and latest_filename in the Saver object"""

    def __init__(self,path,session=None,latest_filename=None,**kwargs):
        """**kwargs are transmitted directly to the underlying tf.train.Saver object
        """
        assert os.path.exists(path)
            
        self.path = path
        self.latest_filename = latest_filename
                    
        if session is None:
            session = tf.get_default_session()
            assert session is not None

        self.session = session
        
        self._saver = tf.train.Saver(**kwargs)
        
    def save(self,filename,global_step=None):
        self._saver.save(
            self.session,
            os.path.join(self.path,filename),
            global_step=global_step,
            latest_filename=self.latest_filename)
        
    def restore(self,filename=None,strict=False):
        if filename is None:
            filename = tf.train.latest_checkpoint(self.path,self.latest_filename)
        
        if filename is None:
            if strict:
                raise OSerror("No Restore Point Found")
        else:
            self._saver.restore(self.session,filename)


Event = collections.namedtuple('Event',['time','value'])


class PathDict(object):
    #TODO: this doesn't really follow any standard ABC

    def __init__(self,dct=None):
        if dct is None:
            dct = {}
        object.__setattr__(self,'_dict',dct)

    def items(self):
        kps = self._keypaths()
        for kp in kps:
            yield kp,self[kp]

    def keys(self):
        kps = self._keypaths()
        for kp in kps:
            yield kp

    def select(self,names):
        cls = type(self)
        result = cls()
        for name in names:
            result[name]=self[name]

        return result

    def __iter__(self):
        kps = self._keypaths()
        for kp in kps:
            yield kp

    def __dir__(self):
        return self._dict.keys()

    def __getattr__(self, item):
        return self._dict[item]

    def __setattr__(self,name,value):
        assert name not in dir(type(self))
        self._dict[name] = value

    def __getitem__(self, parts):
        if isinstance(parts, basestring):
            parts = parts.split('/')

        if not parts[-1]:
            parts = parts[:-1]

        result = self._dict[parts[0]]

        if len(parts) > 1:
            result = result[parts[1:]]

        return result

    def __setitem__(self,parts,value):
        if isinstance(parts,basestring):
            parts = parts.split('/')

        if not parts[-1]:
            parts = parts[:-1]

        if len(parts) == 1:
            self._dict[parts[0]] = value
            return

        if parts[0] in self._dict:
            next = self[parts[0]]
        else:
            next=type(self)()
            self[parts[0]] = next

        next[parts[1:]] = value

    def __str__(self):
        return '{cls}:\n    {keys}\n'.format(
            cls=type(self).__name__,
            keys='\n    '.join(self._keypaths()))

    def _keypaths(self):
        keys = []
        for key,val in self._dict.items():
            if isinstance(val, PathDict):
                subkeys = val._keypaths()
                keys.extend(['/'.join([key,subkey]) for subkey in subkeys])
            else:
                keys.append(key)

        return keys

    __repr__ = __str__


class ScalarSummaries(PathDict):
    @classmethod
    def load(cls,filename):
        """
        Args:
            filename: summary file to load
        """
        scalars = collections.defaultdict(list)
        for event in tf.train.summary_iterator(filename):
            for v in event.summary.value:
                scalars[v.tag].append(Event(event.wall_time,v.simple_value))

        scalars = {name:np.array(values) for [name,values] in scalars.items()}

        result = cls()
        for name,values in scalars.items():
            result[name] = Event(values[:,0],values[:,1])

        return result

    @classmethod
    def join(cls,filenames):
        """
        Joins summary files, after sorting by the timestamp in the file name.

        Resulting times are all made relative.

        Times are joined by setting the first time in the next file equal
        to the last time in the previous file
        """
        if isinstance(filenames,basestring):
            filenames = glob.glob(filenames)

        filenames = sorted(filenames,key=lambda name:int(os.path.split(name)[-1].split('.')[3]))

        summaries = [
            cls.load(filename)
            for filename in filenames]

        result = cls()
        for summ in summaries:
            for key,value in summ.items():
                try:
                    leaf = result[key]
                    endtime = leaf[-1].time[-1]
                    leaf.append(Event(
                        time=value.time-value.time[0]+endtime,
                        value=value.value))

                except KeyError:
                    result[key]=[
                        Event(
                            time=value.time-value.time[0],
                            value=value.value)]

        for key,values in result.items():
            value = np.concatenate(values,axis=1)
            result[key] = Event(time=value[0,:],value=value[1,:])

        return result