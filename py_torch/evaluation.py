# https://medium.com/@timothycarlen/understanding-the-map-evaluation-metric-for-object-detection-a07fe6962cf3
# mAP for detection tasks!

"""
define different IOU threshold and if IOU > thres -> counts as pos.
classified => precision value -> average precision

SCORES ARE NEEDED FOR THIS (score == heat map peak value)

similarly for the classification performance, but for now 
we have only 1 class => makes no sense yet!
"""

