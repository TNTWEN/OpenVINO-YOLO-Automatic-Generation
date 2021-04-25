import os
import numpy as np
import argparse



def parse_model_cfg(path):
    if not path.endswith('.cfg'):
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists('cfg' + os.sep + path):
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')

    lines=[x for x in lines if x and not x.startswith("#")]
    lines=[x.rstrip().lstrip() for x in lines]
    mdefs = []
    for line in lines:
        if line.startswith('['):
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1]['batch_normalize'] = 0
        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape((-1, 2))
            elif (key in ['from', 'layers', 'mask']) or (key == 'size' and ',' in val):
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val

    # Check all fields are supported
    supported = ['type', 'batch_normalize', 'filters', 'size', 'stride', 'pad', 'activation', 'layers', 'groups',
                 'from', 'mask', 'anchors', 'classes', 'num', 'jitter', 'ignore_thresh', 'truth_thresh', 'random',
                 'stride_x', 'stride_y', 'weights_type', 'weights_normalization', 'scale_x_y', 'beta_nms', 'nms_kind',
                 'iou_loss', 'iou_normalizer', 'cls_normalizer', 'iou_thresh','group_id','resize']

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]
    assert not any(u), "Unsupported fields %s in %s." % (u, path)

    return mdefs


def generate(mdefs):
    yololayer=0
    conv_filter=[]

    for i,mdef in enumerate(mdefs):
        if mdef['type']=='convolutional':
            conv_filter.append(mdef['filters'])
        if mdef['type']=="yolo":
            yololayer+=1
            conv_filter.pop()
    return yololayer,conv_filter


    
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg",type=str,default='cfg/yolov4-tiny.cfg',help=('*.cfg path'))
    parser.add_argument("--threel",action='store_true')

    opt = parser.parse_args()
    path = opt.cfg
    mdefs=parse_model_cfg(path)
    mdefs.pop(0)
    yololayer,c=generate(mdefs)

    if opt.threel:
        modeltiny3l=["net = _conv2d_fixed_padding(inputs,%d,kernel_size=3,strides=2)"%c[0],\
    "net = _conv2d_fixed_padding(net, %d, kernel_size=3,strides=2)"%c[1],\
    "net,_ = _tiny_res_block(net,%d,%d,%d,%d,data_format)"%(c[2],c[3],c[4],c[5]),\
    "net,feat_1 = _tiny_res_block(net,%d,%d,%d,%d,data_format)"%(c[6],c[7],c[8],c[9]),\
    "net,feat = _tiny_res_block(net,%d,%d,%d,%d,data_format)"%(c[10],c[11],c[12],c[13]),\
    "net = _conv2d_fixed_padding(net,%d,kernel_size=3)"%c[14],\
    "net=_conv2d_fixed_padding(net,%d,kernel_size=1)"%c[15],\
    "route = net",\
    "net = _conv2d_fixed_padding(route,%d,kernel_size=3)"%c[16],\
    "detect_1 = _detection_layer(net, num_classes, _ANCHORS[6:9], img_size, data_format)",\
    "detect_1 = tf.identity(detect_1, name='detect_1')",\
    "net = _conv2d_fixed_padding(route,%d,kernel_size=1)"%c[17],\
    "upsample_size = feat.get_shape().as_list()\nnet = _upsample(net, upsample_size, data_format)",\
    "net = tf.concat([net,feat], axis=1 if data_format == 'NCHW' else 3)",\
    "net = _conv2d_fixed_padding(net,%d,kernel_size=3)"%c[18],\
    "route = net",\
    "detect_2 = _detection_layer(net, num_classes, _ANCHORS[3:6], img_size, data_format)",\
    "detect_2 = tf.identity(detect_2, name='detect_2')",
    "net = _conv2d_fixed_padding(route,%d,kernel_size=1)"%c[19],\
    "upsample_size = feat_1.get_shape().as_list()\nnet = _upsample(net, upsample_size, data_format)",\
    "net = tf.concat([net,feat_1], axis=1 if data_format == 'NCHW' else 3)",\
    "net = _conv2d_fixed_padding(net,%d,kernel_size=3)"%c[20],\
    "detect_3 = _detection_layer( net, num_classes, _ANCHORS[0:3], img_size, data_format)",\
    "detect_3 = tf.identity(detect_3, name='detect_3')",\
    "detections = tf.concat([detect_1, detect_2,detect_3], axis=1)\ndetections = tf.identity(detections, name='detections')"]
        for x in modeltiny3l:
            print(x)
    else:
        modeltiny=["net = _conv2d_fixed_padding(inputs,%d,kernel_size=3,strides=2)"%c[0],\
    "net = _conv2d_fixed_padding(net, %d, kernel_size=3,strides=2)"%c[1],\
    "net,_ = _tiny_res_block(net,%d,%d,%d,%d,data_format)"%(c[2],c[3],c[4],c[5]),\
    "net,_ = _tiny_res_block(net,%d,%d,%d,%d,data_format)"%(c[6],c[7],c[8],c[9]),\
    "net,feat = _tiny_res_block(net,%d,%d,%d,%d,data_format)"%(c[10],c[11],c[12],c[13]),\
    "net = _conv2d_fixed_padding(net,%d,kernel_size=3)"%c[14],\
    "feat2=net",\
    "net=_conv2d_fixed_padding(feat2,%d,kernel_size=1)"%c[15],\
    "route = net",\
    "net = _conv2d_fixed_padding(route,%d,kernel_size=3)"%c[16],\
    "detect_1 = _detection_layer(net, num_classes, _ANCHORSTINY[3:6], img_size, data_format)",\
    "detect_1 = tf.identity(detect_1, name='detect_1')",\
    "net = _conv2d_fixed_padding(route,%d,kernel_size=1)"%c[17],\
    "upsample_size = feat.get_shape().as_list()\nnet = _upsample(net, upsample_size, data_format)",\
    "net = tf.concat([net,feat], axis=1 if data_format == 'NCHW' else 3)",\
    "net = _conv2d_fixed_padding(net,%d,kernel_size=3)"%c[18],\
    "detect_2 = _detection_layer(net, num_classes, _ANCHORSTINY[1:4], img_size, data_format)",
    "detect_2 = tf.identity(detect_2, name='detect_2')",
    "detections = tf.concat([detect_1, detect_2], axis=1)\ndetections = tf.identity(detections, name='detections')"]
        for x in modeltiny:
            print(x)
