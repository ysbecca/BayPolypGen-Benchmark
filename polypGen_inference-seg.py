#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 14:36:02 2021

@author: endocv2021@generalizationChallenge
"""

import network

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import skimage
from skimage import io
from  tifffile import imsave
import skimage.transform
from collections import OrderedDict
import matplotlib.pyplot as plt

def create_predFolder(root, model_desc, test_data=None, lr=None, sharpen=False, epoch=0):
    #path = f"{root}predictions/images_C6_pred/{model_desc}/"
    #if test_data:
    folder_path = f"{root}/predictions/images_{test_data}/"
    if not os.path.exists(folder_path):
      os.mkdir(folder_path)
    path = f"{root}/predictions/images_{test_data}/{model_desc}/"

    if sharpen:
        path += f"{epoch}s_lr{lr}/"

    if not os.path.exists(path):
      os.mkdir(path)
        
    return path

def detect_imgs(infolder, ext='.tif'):
    import os

    items = os.listdir(infolder)

    flist = []
    for names in items:
        if names.endswith(ext) or names.endswith(ext.upper()):
            flist.append(os.path.join(infolder, names))

    return np.sort(flist)


def get_argparser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet', 'pspNet', 'segNet', 'FCN8', 'resnet-Unet', 'axial', 'unet'], help='model name')
    parser.add_argument("--root", type=str, default="",
                        help='absolute path to EndoCV2021')

    parser.add_argument("--model_desc", type=str, default='test',
                        help='model description for loading moments')
    parser.add_argument("--moment_count", type=int, default='2',
                        help="total number of moments from posterior")

    parser.add_argument("--is_sharpen", type=bool, default=False,
                        help="sharpened?")
    parser.add_argument("--epoch", type=int,
                        help="sharpening epoch for sharpened")
    parser.add_argument("--lr", type=float,
                        help="sharpening lr")

    parser.add_argument("--backbone", type=str, default='resnet50',
                        choices=['vgg19',  'resnet34' , 'resnet50',
                                 'resnet101', 'densenet121', 'none'], help='model name')

    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    
    parser.add_argument("--crop_size", type=int, default=512)
    
    # "Sent as an argument so see the script file instead!!!"
    parser.add_argument("--ckpt", type=str, default='./checkpoints/best_deeplabv3plus_resnet50_voc_os16_kvasir.pth',
                        help="checkpoint file")

    parser.add_argument("--gpu_id", type=str, default='3',
                        help="GPU ID")
    
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")

    parser.add_argument("--test_set", type=str, default="C6_pred",
                        help="options: C6_pred, EndoCV_DATA3, EndoCV_DATA4")
    return parser

    

def mymodel():
    '''
    Returns
    -------
    model : TYPE
        DESCRIPTION.
    device : TYPE
        DESCRIPTION.
    '''
    opts = get_argparser().parse_args() 

    print("Model description: ", opts.model_desc)
    print("Moment count:      ", opts.moment_count)
    print("Dataset:           ", opts.test_set)
    print("="*30)

    # ---> explicit classs number
    opts.num_classes = 2
    
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)


    # Set up model
    if opts.model == 'pspNet':
        from network.network import PSPNet
        #,FCN8,SegNet
        model = PSPNet(num_classes=opts.num_classes, pretrained=True)

    elif opts.model == 'FCN8':
        from network.network import FCN8
        model = FCN8(num_classes=opts.num_classes)

    elif opts.model =='resnet-Unet':
        from backboned_unet import Unet
        # net = Unet(backbone_name='densenet121', classes=2)
        # model = Unet(backbone_name=opts.backbone, pretrained=True, classes=opts.num_classes, decoder_filters=(512, 256, 128, 64, 32))
        
        # for VGG
        model = Unet(backbone_name=opts.backbone, pretrained=True, classes=opts.num_classes)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('number of trainable parameters:', pytorch_total_params)   
    
    elif opts.model =='axial':
        from network.axialNet import axial50s
        model = axial50s(pretrained=True)
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        print('number of trainable parameters:', pytorch_total_params)
        
    elif opts.model =='unet':
        from network.unet import UNet
        model = UNet(n_channels=3, n_classes=1, bilinear=True)


    else:
        model_map = {
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
        }
    
    
        model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        
    return model, device
    # checkpoint = torch.load(opts.ckpt, map_location=device)
    # # model.load_state_dict(checkpoint["model_state"])
    # state_dict =checkpoint['model_state']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()

    # for k, v in state_dict.items():
    #     if 'module' not in k:
    #         k = 'module.'+k
    #     else:
    #         k = k.replace('features.module.', 'module.features.')
    #     new_state_dict[k]=v


def load_moment(moment_id, model, device):

    if opts.is_sharpen:
        # 0_2s_lr0.01.pt
        ckpt_name = f"{moment_id}_{opts.epoch}s_lr{opts.lr}"
    else:
        ckpt_name = f"{moment_id}"

    print(f"[INFO] checkpoint: {ckpt_name}")
    checkpoint = torch.load(f"{opts.root}moments/{opts.model_desc}/{ckpt_name}.pt", map_location=device)
    state_dict = checkpoint['model_state']

    try:
        model.load_state_dict(state_dict)

    except:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
             if 'module' not in k:
                 k = 'module.'+k
             else:
                 k = k.replace('features.module.', 'module.features.')
             new_state_dict[k]=v

        model.load_state_dict(new_state_dict)
            
    model.eval()

    return model

        
if __name__ == '__main__':
    '''
     You are not allowed to print the images or visualizing the test data according to the rule. 
     We expect all the users to abide by this rule and help us have a fair challenge "EndoCV2021-Generalizability challenge"
     
     FAQs:
         1) Most of my predictions do not have polyp.
            --> This can be the case as this is a generalisation challenge. The dataset is very different and can produce such results. In general, not all samples 
            have polyp.
        2) What format should I save the predictions.
            --> you can save it in the tif or jpg format. 
        3) Can I visualize the results or copy them in to see?
            --> No, you are not allowed to do this. This is against challenge rules!!!
        4) Can I use my own test code?
            --> Yes, but please make sure that you follow the rules. Any visulization or copy of test data is against the challenge rules. We make sure that the 
            competition is fair and results are replicative.
    '''
    # weird thing where if I don't call it here it can't find it later
    torch.cuda.is_available()
    torch.cuda.device_count()

    model, device = mymodel()

    opts = get_argparser().parse_args() 
#    if opts.model != 'resnet-Unet':
#        dirN = 'test_best_endocv2021'+opts.model
#    else:
#        dirN = 'test_best_endocv2021'+opts.model+'_'+opts.backbone
    # set image folder here! ./frosty/segmentation
    saveDir = create_predFolder(opts.root, opts.model_desc, opts.test_set,
        sharpen=opts.is_sharpen,
        epoch=opts.epoch,
        lr=opts.lr,
    )
    
    # ----> three test folders [https://github.com/sharibox/EndoCV2021-polyp_det_seg_gen/wiki/EndoCV2021-Leaderboard-guide]
    
#    please uncomment for challenge paper -->
    # subDirs = ['EndoCV_DATA1', 'EndoCV_DATA2', 'EndoCV_DATA3', 'EndoCV_DATA4']
    
#    last directory is for the data paper --> these are single and sequence datasets respectively, please change the names accordingly

    all_epistemics = []

    #subDirs = ['EndoCV_DATA4', 'EndoCV_DATAPaperC6']
    print(opts.test_set)
    subDirs = [opts.test_set]
    for j in range(0, len(subDirs)):
        
        # ---> Folder for test data location!!! (Warning!!! do not copy/visulise!!!)
        #imgfolder='/well/rittscher/users/sharib/deepLabv3_plus_pytorch/datasets/endocv2021-test-noCopyAllowed-v3/' + subDirs[j]

        if not opts.root:
            imgfolder = '/resstore/b0211/Users/scpecs/' 
        else:
            imgfolder = opts.root
        if opts.test_set == "C6_pred":
          imgfolder += f"datasets/EndoCV2021/data_C6/images_C6/"
        else:
          imgfolder += f"datasets/endocv2021-test-noCopyAllowed-v3_confidential/" + subDirs[j]

        # # set folder to save your checkpoints here!
        #saveDir = os.path.join(directoryName , subDirs[j]+'_pred')

        imgfiles = detect_imgs(imgfolder, ext='.jpg')
    
        from torchvision import transforms
        data_transforms = transforms.Compose([
            # transforms.RandomResizedCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
        #start = torch.cuda.Event(enable_timing=True)
        #end = torch.cuda.Event(enable_timing=True)
        # file = open(saveDir + '/'+"timeElaspsed" + subDirs[j] +'.txt', mode='w')
        # timeappend = []

        for imagePath in imgfiles[:]:
            """plt.imshow(img1[:,:,(2,1,0)])
            Grab the name of the file. 
            """
            filename = (imagePath.split('/')[-1]).split('.jpg')[0]
            print('filename is printing :: =====>>', filename)
            
            img1 = Image.open(imagePath).convert('RGB').resize((512,512), resample=0)
            image = data_transforms(img1)
            # perform inference here:
            images = image.to(device, dtype=torch.float32)
            #            
            size = skimage.io.imread(imagePath).shape
            #start.record()

            # bayesian pass thru all moments
            m_preds = []

            for m in [0, 1, 2, 3]:
                model = load_moment(m, model, device)

                outputs = model(images.unsqueeze(0))
                pred = outputs.detach().max(dim=1)[1].cpu().numpy()[0]*255
                pred = pred.astype(np.uint8)
                m_preds.append(pred)

            # end.record()
            # torch.cuda.synchronize()
            # print(start.elapsed_time(end))
            # timeappend.append(start.elapsed_time(end))

            # [MOMENTS, PRED_w, PRED_h]
            m_preds = np.array(m_preds)
            # get epistemic uncertainties.... and average for single value 
            # accumulate epistemic uncertainties
            #temp = (m_preds - np.broadcast_to(np.mean(m_preds, axis=0), (opts.moment_count, *m_preds.shape)))**2
            #epis_ = np.sqrt(np.sum(temp, axis=0)) / opts.moment_count
            #epis_ = epis_.astype(np.double)

            # take mean
            #epi = epis_.max()
            epi = np.var(m_preds.astype(np.float32), axis=0)
            # temp = (m_preds - np.broadcast_to(np.mean(m_preds, axis=0), (opts.moment_count, *m_preds.shape)))**2
            # epis_ = np.sqrt(np.sum(temp, axis=0)) / opts.moment_count
            # epis_ = epis_.astype(np.double)

            # take mean
            all_epistemics.append(epi)

            # final averaged prediction seg map
            # [PRED_w, PRED_h]
            m_preds = np.mean(m_preds, axis=0)

            img_mask = skimage.transform.resize(m_preds, (size[0], size[1]), anti_aliasing=True)

            pil_image = Image.fromarray(img_mask.astype(np.uint8))
            pil_image.save(saveDir +'/'+ filename +'_mask.jpg')

            # imsave(saveDir +'/'+ filename +'_mask.jpg', img_mask.astype(np.uint8))

    all_epistemics = np.array(all_epistemics)

    if opts.test_set == "C6_pred":
      np.save(f"{saveDir}/epis_images_C6.npy", all_epistemics)
    else:
      np.save(f"{saveDir}/epis_{subDirs[j]}.npy", all_epistemics)

    # file.write('%s -----> %s \n' % 
       # ('average_t', np.mean(timeappend)))

