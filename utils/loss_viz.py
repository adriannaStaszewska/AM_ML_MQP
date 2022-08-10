# -*- coding: utf-8 -*-
"""
@File    : d2_loss_visualization.py
@Time    : 11/6/20 11:24 AM
@Author  : Mingqiang Ning
@Email   : ningmq_cv@foxmail.com
@Modify Time        @Version    @Description
------------        --------    -----------
11/6/20 11:24 AM      1.0         None
# @Software: PyCharm
"""
import json
import re
from pylab import *
import matplotlib.pyplot as plt

fig = figure(figsize=(8,6), dpi=300)
y1 = fig.add_subplot(111)
y1.set_xlabel('Iterations')
y2 = y1.twinx()
y1.set_ylim(0,1.0)
parsed=[]
with open('/home/azstaszewska/Models/final_model_full_1/metrics.json') as f:
    # whole = f.read()
    # # #pattern = re.compile(r'json_stats: (\{.*\})')
    # pattern = re.compile(r'\"data_time\":.*')
    # # #r:ban zhaunyi
    # lis = pattern.findall(whole)
    try:
        #ke neng chu xian de yi chang kuai
        # parsed = [json.loads(j) for j in lis]
        for line in f:
            if line == "":
                continue
            parsed.append(json.loads(line))
        # print(parsed[0])

    except:
        #shang shu yi chang kuai de chu li fang fa
        print("json format is not correct")
        exit(1)

    _iter = [j['iteration'] for j in parsed]
    _loss = [ l.get('total_loss') for l in parsed]
    _loss_bbox = [ l.get('loss_box_reg') for l in parsed]
    _loss_cls = [ l.get('loss_cls') for l in parsed]
    try:
         _accuracy_cls = [ l.get('fast_rcnn/cls_accuracy') for l in parsed]
    except:
        _accuracy_cls = None
    _lr =  [l.get('lr') for l in parsed]
    try:
        _mask_loss = [ l.get('loss_mask') for l in parsed]
    except:
        _mask_loss = None
    _iter, _loss = zip(*sorted(zip(_iter, _loss)))

    y1.plot(_iter, _loss_bbox, color="green", linewidth=0.3,linestyle="-",label='loss_box_reg')
    y1.plot(_iter, _loss, color="blue", linewidth=0.4, linestyle="--",label='total_loss')
    y1.plot(_iter, _loss_cls, color="black", linewidth=0.3, linestyle="-",label='loss_cls')
    if _mask_loss is not None:
         y1.plot(_iter, _mask_loss, color="purple", linewidth=0.3, linestyle="-",label='mask_loss')
    #y2.set_ylim(0,0.001/0.8)
    #y2.plot(_iter, _lr, color="purple", linewidth=1.0, linestyle="-",label='lr')
    #y2.set_ylabel('lr')

    #????????
    #grid()
    #??
    y1.legend()
    savefig('/home/azstaszewska/fig.png')
    show()
    #fig2 = figure(figsize=(8,6), dpi=300)
    #fig2.plot(_iter, _accuracy_cls, color="red", linewidth=0.5, linestyle="-",label='cls_accuracy')
    #savefig('/home/azstaszewska/fig2_cls_acc.png')
