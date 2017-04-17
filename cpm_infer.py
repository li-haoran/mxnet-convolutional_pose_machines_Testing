#import cv2
from PIL import Image
import numpy as np
import mxnet as mx
from cv2 import resize as imresize
from cv2 import INTER_LINEAR
from cpm_sym import get_sym
from datetime import datetime
from visual import heat_plot
# rect is [x,y,w,h], img is [h,w,3], offset is [x,y] remind the axis for me 
#imresize the size is[w,h]
INPUT_SIZE=368
NSTAGE=6
def preprocess_image(img):
    '''
    if opencv bgr  should change to rgb
    else
    PIL image is ok

    a=a/256-0.5
    '''
    mean = np.array([0.5, 0.5,0.5])
    img = np.array(img, dtype=np.float32)
    img/=256.0
    reshaped_mean = mean.reshape(1, 1, 3)
    img = img - reshaped_mean
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    img = np.expand_dims(img, axis=0)
    return img

def get_center_map(rect,sigma):
    [X,Y]=np.meshgrid(range(rect[2]),range(rect[3]))
    X=X-rect[2]/2.0
    Y=Y-rect[3]/2.0
    D2=(X**2+Y**2)/(2.0*sigma*sigma)
    return np.expand_dims(np.expand_dims(np.exp(-D2),0),0)

def get_roi(img,scale,ori_roi):
    '''
    here img is [h,w,3]
    
    '''
    shape=[int(s*scale) for s in img.shape]
    roi=np.ceil(ori_roi*scale)
    temp=imresize(img,(shape[1],shape[0]))
    #temp=temp_im[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2],:]
    a=np.ones(shape=(INPUT_SIZE,INPUT_SIZE,3))*128
    offset_h=max(int(roi[1]+roi[3]/2-INPUT_SIZE/2.0),0)
    offset_w=max(int(roi[0]+roi[2]/2-INPUT_SIZE/2.0),0)
    h=min(INPUT_SIZE/2.0,roi[1]+roi[3]/2)+min(INPUT_SIZE/2.0,shape[0]-(roi[1]+roi[3]/2))
    w=min(INPUT_SIZE/2.0,roi[0]+roi[2]/2)+min(INPUT_SIZE/2.0,shape[1]-(roi[0]+roi[2]/2))

    a_offest_h=max(int(INPUT_SIZE/2.0-(roi[1]+roi[3]/2)),0)
    a_offest_w=max(int(INPUT_SIZE/2.0-(roi[0]+roi[2]/2)),0)

    a[a_offest_h:a_offest_h+h,a_offest_w:a_offest_w+w,:]=temp[offset_h:offset_h+h,offset_w:offset_w+w,:]
    return a ,(-a_offest_h+offset_h,-a_offest_w+offset_w,h,w)

def load_params(save_params):
    save_dict = mx.nd.load(save_params)
    arg_params = {}
    aux_params = {}
    for k, value in save_dict.items():
        arg_type, name = k.split(':', 1)
        if arg_type == 'arg':
            arg_params[name] = value
        elif arg_type == 'aux':
            aux_params[name] = value
        else:
            raise ValueError("Invalid param file " + fname)
    return arg_params,aux_params

def multi_scale_infer(img_path,rect):
    starting=rect[3]*1.2*0.8
    ending=rect[3]*1.2*3.0
    ms=np.arange(np.log2(INPUT_SIZE/ending),np.log2(INPUT_SIZE/starting),1.0/4.0)
    multi_scale=2**ms
    
    ####get sym and module
    ctx=mx.cpu()
    sym=get_sym()
    mod=mx.mod.Module(sym,data_names=('image','center_map',),label_names=(),context=ctx)
    mod.bind(data_shapes=[('image',(1,3,368,368)),('center_map',(1,1,368,368))],label_shapes=[],for_training=False)
    arg_params,aux_params=load_params('cpm_infer/mpii.params')
    mod.init_params(initializer=None, arg_params=arg_params, aux_params=aux_params,allow_missing=False, force_init=True)

    img = Image.open(img_path)
    img = np.array(img, dtype=np.float32)
    center_map=get_center_map((0,0,368,368),21)
    output=[]
    offset=[]
    stamp =  datetime.now().strftime('%H_%M_%S')
    print stamp
    for scale in multi_scale:
        
        im,off=get_roi(img,scale,rect)
        image=preprocess_image(im)
        mod.forward(mx.io.DataBatch([mx.nd.array(image),mx.nd.array(center_map)],[]),is_train=False)
        out=[np.squeeze(ot.asnumpy()) for ot in mod.get_outputs()]
        output.append(out)
        offset.append(off)
        stamp =  datetime.now().strftime('%H_%M_%S')
        print stamp
    final=np.zeros((img.shape[0],img.shape[1],15,NSTAGE))
    for k,scale in enumerate(multi_scale):
        op=output[k]
        os=offset[k]
        tmp=np.zeros((img.shape[0]*scale,img.shape[1]*scale,15))
        tmp_offset_h=max(os[0],0)
        tmp_offset_w=max(os[1],0)
        h=os[2]
        w=os[3]
        op_offset_h=max(-os[0],0)
        op_offset_w=max(-os[1],0)
        for i in range(NSTAGE):
            opi=op[i]
            opi = np.swapaxes(opi, 0, 1)
            opi = np.swapaxes(opi, 1, 2)
            tm=imresize(opi,(368,368))
            tmp[tmp_offset_h:tmp_offset_h+h,tmp_offset_w:tmp_offset_w+w,:]=tm[op_offset_h:op_offset_h+h,op_offset_w:op_offset_w+w,:]
            final[:,:,:,i]+=imresize(tmp,(img.shape[1],img.shape[0]))
    final/=multi_scale.shape[0]

    heat_plot(img.astype(np.int8),final[:,:,:,5],15)


if __name__ =='__main__':
    img_path='cpm_infer/test.jpg'
    rect=(750,150,440,800)
    multi_scale_infer(img_path,np.array(rect))




















