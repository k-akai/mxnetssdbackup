# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from __future__ import print_function
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter
import json
class Detector(object):
    """
    SSD detector which hold a detection network and wraps detection API

    Parameters:
    ----------
    symbol : mx.Symbol
        detection network Symbol
    model_prefix : str
        name prefix of trained model
    epoch : int
        load epoch of trained model
    data_shape : int
        input data resize shape
    mean_pixels : tuple of float
        (mean_r, mean_g, mean_b)
    batch_size : int
        run detection with batch size
    ctx : mx.ctx
        device to use, if None, use mx.cpu() as default context
    """
    def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels, \
                 batch_size=1, ctx=None):
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
        if symbol is None:
            symbol = load_symbol
        self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
        if not isinstance(data_shape, tuple):
            data_shape = (data_shape, data_shape)
        self.data_shape = data_shape
        self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape[0], data_shape[1]))])
        self.mod.set_params(args, auxs)
        self.mean_pixels = mean_pixels

    def detect(self, det_iter, show_timer=False):
        """
        detect all images in iterator

        Parameters:
        ----------
        det_iter : DetIter
            iterator for all testing images
        show_timer : Boolean
            whether to print out detection exec time

        Returns:
        ----------
        list of detection results
        """
        num_images = det_iter._size
        if not isinstance(det_iter, mx.io.PrefetchingIter):
            det_iter = mx.io.PrefetchingIter(det_iter)
        start = timer()
        detections = self.mod.predict(det_iter).asnumpy()
        time_elapsed = timer() - start
        if show_timer:
            print("Detection time for {} images: {:.4f} sec".format(
                num_images, time_elapsed))
        result = []
        for i in range(detections.shape[0]):
            det = detections[i, :, :]
            res = det[np.where(det[:, 0] >= 0)[0]]
            result.append(res)
        return result

    def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
        """
        wrapper for detecting multiple images

        Parameters:
        ----------
        im_list : list of str
            image path or list of image paths
        root_dir : str
            directory of input images, optional if image path already
            has full directory information
        extension : str
            image extension, eg. ".jpg", optional

        Returns:
        ----------
        list of detection results in format [det0, det1...], det is in
        format np.array([id, score, xmin, ymin, xmax, ymax]...)
        """
        test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
        test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                            is_train=False)
        return self.detect(test_iter, show_timer)

    def visualize_detection(self,imname, img, dets, classes=[], thresh=0.3):
        """
        visualize detections in one image

        Parameters:
        ----------
        img : numpy.array
            image, in bgr format
        dets : numpy.array
            ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
            each row is one object
        classes : tuple or list of str
            class names
        thresh : float
            score threshold
        """
        import matplotlib as mpl
        mpl.use('Agg')
        import matplotlib.pyplot as plt
        import random
        plt.imshow(img)
        height = img.shape[0]
        width = img.shape[1]
        colors = dict()
        for i in range(dets.shape[0]):
            cls_id = int(dets[i, 0])
            if cls_id >= 0:
                score = dets[i, 1]
                if score > thresh:
                    if cls_id not in colors:
                        colors[cls_id] = (random.random(), random.random(), random.random())
                    xmin = int(dets[i, 2] * width)
                    ymin = int(dets[i, 3] * height)
                    xmax = int(dets[i, 4] * width)
                    ymax = int(dets[i, 5] * height)
                    rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                                         ymax - ymin, fill=False,
                                         edgecolor=colors[cls_id],
                                         linewidth=3.5)
                    plt.gca().add_patch(rect)
                    class_name = str(cls_id)
                    if classes and len(classes) > cls_id:
                        class_name = classes[cls_id]
                    plt.gca().text(xmin, ymin - 2,
                                    '{:s} {:.3f}'.format(class_name, score),
                                    bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                                    fontsize=12, color='white')
        plt.show()

        name=imname.split(".")
        if len(name)==2:
          name=name[0]
        else:
          name=name[1]
        name=name.split("/")
        name=name[len(name)-1]
        print(imname)
        plt.savefig(os.path.join(outputdir,name))

    def writejson(self,outputdir,imname,dets,classes,thresh,width,height,viewclass):

      dictres={}
      for i in range(dets.shape[0]):
        cls_id = int(dets[i, 0])
        if cls_id >= 0:
          score = dets[i, 1]
          if score < thresh:
            continue
          xmin =int(dets[i,2]*width)
          ymin =int(dets[i,3]*height)
          xmax =int(dets[i,4]*width)
          ymax =int(dets[i,5]*height)
#          print(classes[cls_id])
#          print (score)
          cls=classes[cls_id]
          #print(str(dets[i,2])+":"+str(dets[i,3])+":"+str(dets[i,4])+":"+str(dets[i,5]))
#          print (str(xmin)+":"+str(ymin)+":"+str(xmax)+":"+str(ymax))

          res=(score,xmin,ymin,xmax,ymax)
          #if already add, append
          if cls in dictres:
             lis=dictres[cls]
             lis.append(res)
          else:
             lis=[]
             lis.append(res)
             dictres[cls]=lis

      #if viewclass is None, view all class except dammy
      vcls=""

      if viewclass == None:
        for i in classes:
          if i.find("dammy")==-1:
            vcls=vcls+","+i
      else:
        vcls=viewclass
#      print (vcls)
      views=vcls.split(",")
#      for i in views:
#        if i in dictres:
#          print ("dict:"+i+"=")
#          print (dictres[i])
#        else:
#          print ("dict:"+i+"=no data")

      #make format
      resj={}
      resj["conf_thresh"]=thresh
      resj["ground_truth"]="None"
      resj["num_thresh"]=0
      rlt={}
      resj["result"]=rlt
      for i in views:
        if i =="":
          continue
        rlt[i]=[]
        if i in dictres:
          datas=dictres[i]
          for j in datas:
            data=[j[1],j[2],j[3],j[4]]
            rlt[i].append({"bbox":data,"score":round(j[0],3)})
      #print (resj)
      name=imname.split(".")
      if len(name)==2:
        name=name[0]
      else:
        name=name[1]
      name=name.split("/")
      name=name[len(name)-1]
      #print(imname)
      f=open("out/"+name+"_result.json",'w')
      json.dump(resj,f)

    def detect_and_visualize(self,outputdir, make_image,im_list, root_dir=None, extension=None,
                             classes=[], thresh=0.6, show_timer=False, same=True,viewclass=None):
        """
        wrapper for im_detect and visualize_detection

        Parameters:
        ----------
        im_list : list of str or str
            image path or list of image paths
        root_dir : str or None
            directory of input images, optional if image path already
            has full directory information
        extension : str or None
            image extension, eg. ".jpg", optional

        Returns:
        ----------

        """
        #print("check")
        #print(make_image)
        import cv2
        dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
        if not isinstance(im_list, list):
            im_list = [im_list]
        assert len(dets) == len(im_list)
        height=0
        width=0
        for k, det in enumerate(dets):

            if k==0 or same ==False:
              img=cv2.imread(im_list[k])
              height = img.shape[0]
              width = img.shape[1]
            self.writejson(outputdir,im_list[k],det,classes,thresh,width,height,viewclass)
            if make_image==1:
              continue
            img = cv2.imread(im_list[k])
            img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
            self.visualize_detection(im_list[k],img, det, classes, thresh)
