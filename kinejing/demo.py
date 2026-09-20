"""Deterministic synthetic integration demo. No neural weights or benchmark data."""
import json
from pathlib import Path
import cv2
import numpy as np
from .refinement import flow_knots,render


def demo(output):
    output=Path(output)
    if output.exists():raise FileExistsError('Choose a new demo output directory.')
    output.mkdir(parents=True)
    initial=np.full((240,320,3),(40,30,20),np.uint8)
    for x in range(20,320,30):cv2.line(initial,(x,40),(x,220),(70,65,50),1)
    for y in range(40,240,30):cv2.line(initial,(0,y),(320,y),(70,65,50),1)
    cv2.rectangle(initial,(85,100),(135,150),(60,190,235),-1)
    cv2.circle(initial,(220,130),25,(185,130,65),-1)
    streams=[]
    for sign in [0,-1,1]:
        donor=np.stack([cv2.warpAffine(initial,np.array([[1,0,sign*t*1.5],[0,1,t*.5]],np.float32),(320,240),borderMode=cv2.BORDER_REFLECT101) if sign else initial for t in range(17)])
        prediction=render(initial,flow_knots(initial,donor,.25)) if sign else donor
        streams.append(prediction)
    panels=[]
    for t in range(17):
        row=[]
        for name,stream in zip(['HEAD / PERSISTENCE','LEFT / MOTION','RIGHT / MOTION'],streams):
            frame=stream[t].copy();cv2.putText(frame,name,(12,24),cv2.FONT_HERSHEY_SIMPLEX,.5,(220,240,245),1);row.append(frame)
        panels.append(np.concatenate(row,axis=1))
    writer=cv2.VideoWriter(str(output/'demo.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),8,(960,240))
    if not writer.isOpened():raise RuntimeError('OpenCV video encoder unavailable')
    for frame in panels:writer.write(frame)
    writer.release()
    cv2.imwrite(str(output/'preview.png'),panels[8])
    manifest={'kind':'synthetic software integration demo','seed':42,'neural_weights_used':False,
              'official_score':None,'frames':17,'resolution':[960,240]}
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    return manifest
