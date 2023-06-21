#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 14:12:16 2021

@author: endocv challenges

Disclaimer: Most codes are imported from the previous EndoCV challenges!!!

Requires:
!pip install --upgrade scikit-learn
!pip install numba==0.49.1
!pip install hausdorff
"""

# TODO: Add distance metrics for evaluation!!!

import numpy as np

def rescale_im_stack(imfiles, size=(256,256)):
    from skimage.transform import resize
    from skimage.io import imread
    X = []
    for imfile in imfiles:
        im = imread(imfile)
        im = resize(im, size).astype(np.float32)
        X.append(im[None,:])
        
    X = np.concatenate(X, axis=0)
    
    return X
def panel_imgs(imgs, grid):
    
    N,m,n,c = imgs.shape
    
    new_im = np.zeros((grid[0]*m, grid[1]*n, c))
    for ii in range(grid[0]):
        for jj in range(grid[1]):
            im = imgs[ii*grid[1]+jj]
            new_im[ii*m:(ii+1)*m, jj*n:(jj+1)*n] = im.copy()
            
    return new_im

def locate_folders(rootfolder):
    import os 
    folders = []
    for root, dirs, files in os.walk(rootfolder):
        for f in files:
            if 'c' in f and '.tif' in f:
                if root not in folders:
                    folders.append(root)
    return np.hstack(folders)

def roi_area(mask):
    
    from skimage.measure import label, regionprops
    
    areas = []
    if mask.max()>0:
        reg = regionprops(label(mask))
        
        for re in reg:
            areas.append(re.area)
            
        areas = np.hstack(areas)     
        return np.mean(areas)
    else:
        return np.sum(mask)
    

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="Semantic segmentation of EndoCV2021 challenge", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--GT_maskDIR", type=str, default="/media/sharib/development/EndoCV2021-test_analysis/codes-seg/EndoCV2021_groundTruth-v1/segmentation/EndoCV_DATA3_GT", help="ground truth mask image (5 channel tif image only)")
    parser.add_argument("--Eval_maskDIR", type=str, default="/media/sharib/development/EndoCV2021-test_analysis/codes-seg/EndoCV2021/segmentation/EndoCV_DATA3_pred", help="provide folder for testType1 dataset under that name")
    parser.add_argument("--jsonFileName", type=str, default="metric_seg_score.json", help="all evaluation scores used for grading")

    parser.add_argument("--root", type=str, default="",
                        help='absolute path to EndoCV2021')

    parser.add_argument("--model_desc", type=str, default='test',
                        help='model description for loading moments')

    # Peter: the results will be in {args.root}/EndoCV2021/{args.model_desc}/segmentation/...
    # in wandb.init dict set "name": args.model_desc

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    import glob
    import os
    from misc import EndoCV_misc 
    import cv2
    from metrics_seg import get_confusion_matrix_elements, jac_score, dice_score, F2, precision, recall, get_confusion_matrix_torch
    import time
    
    # ---> requires: !pip install hausdorff (first install !pip install numba==0.49.1)
    # from hausdorff import hausdorff_distance

    classTypes=['polyp']
    args = get_args()
    
    # can be multiple test sets: 1 -- 5
    # ground truth folder
    GT_folder = args.GT_maskDIR
    GT_files = glob.glob(os.path.join(GT_folder,'*.jpg'))
    
    # evaluation/predicted folder
    participantsFolder = args.Eval_maskDIR
    
    # save folder
#    savefolder = 'semantic_results'
#    os.makedirs(savefolder, exist_ok=True)
    
    fnames = []
    fpath = participantsFolder
    
#    os.makedirs(savefolder, exist_ok = True)

    pred_mask_files = glob.glob(os.path.join(fpath, '*.jpg'))
    fnames.append(pred_mask_files)
        
    print('running endocv segmentation...')
    print(pred_mask_files)
    if len(pred_mask_files) > 0:
        gt_mask_files = np.hstack([os.path.join(GT_folder, (os.path.split(f)[-1].split('.')[0])+'.jpg') for f in pred_mask_files])
        
        jac_scores = []
        dice_scores = []
        f2_scores = []
        PPV_scores = []
        Rec_scores = []
        acc_scores = []
        Hfd_score = []
        start = time.time()
        for jj in range(len(pred_mask_files))[:]:
            print(gt_mask_files[jj])
            gt_mask = (cv2.imread(gt_mask_files[jj]) > 0).astype(np.uint8)[:,:,0]
            pred_mask = (cv2.imread(pred_mask_files[jj]) > 0).astype(np.uint8)
            
            # make same size as GT
            pred_mask = cv2.resize(pred_mask, (gt_mask.shape[1], gt_mask.shape[0]), interpolation = cv2.INTER_AREA)[:,:,0]
            
            
            # computation
            # %%timeit
            tn, fp, fn, tp = get_confusion_matrix_elements(gt_mask.flatten().tolist(), pred_mask.flatten().tolist())
            
            # %%timeit
            # A = (get_confusion_matrix_torch(gt_mask.flatten().tolist(), pred_mask.flatten().tolist())).numpy()
            # # A =  A.numpy()

            # tn = A[0][0]
            # fp = A[0][1]
            # fn = A[1][0]
            # tp = A[1][1]
            
            overall_acc = (tp+tn)/(tp+tn+fp+fn)
            # %%timeit
            # Hf = hausdorff_distance(gt_mask, pred_mask, distance='euclidean')
            
            jac_set = np.hstack([jac_score(gt_mask,pred_mask)])
            dice_set = np.hstack([dice_score(gt_mask,pred_mask)])
            f2_set = np.hstack([F2(gt_mask,pred_mask)])
            PPV_set = np.hstack([precision(gt_mask,pred_mask)])
            Rec_set = np.hstack([recall(gt_mask,pred_mask)])
            acc = np.hstack([overall_acc])
            
            jac_scores.append(jac_set)
            dice_scores.append(dice_set)
            f2_scores.append(f2_set)
            PPV_scores.append(PPV_set)
            Rec_scores.append(Rec_set)
            acc_scores.append(acc)
            # Hfd_score.append(Hf)
            
        
        jac_scores = np.vstack(jac_scores)
        dice_scores = np.vstack(dice_scores)
        f2_scores = np.vstack(f2_scores)
        PPV_scores = np.vstack(PPV_scores)
        Rec_scores = np.vstack(Rec_scores)
        acc_scores = np.vstack(acc_scores)
        
        # if np.sum(Hfd_score) == 0.0:
        #     hfmean = 0
        #     hfstd = 0
        # else:
        #     hfmean = np.mean(Hfd_score/np.max(Hfd_score)+0.0000001)
        #     hfstd = np.std(Hfd_score/np.max(Hfd_score))
        
        print('----')
        print ('jac: ', jac_scores.mean(axis=0)), '+', jac_scores.mean(axis=0).mean()
        print('dice: ', dice_scores.mean(axis=0)), '+', dice_scores.mean(axis=0).mean()
        print('F2: ', f2_scores.mean(axis=0)), '+', f2_scores.mean(axis=0).mean()
        print('PPV: ', PPV_scores.mean(axis=0)), '+', PPV_scores.mean(axis=0).mean()
        print('Rec: ', Rec_scores.mean(axis=0)), '+', Rec_scores.mean(axis=0).mean()
        print('Acc: ', acc_scores.mean(axis=0)), '+', acc_scores.mean(axis=0).mean()
        # Normalise
        # print('Hdf: ', hfmean), '+', hfmean
        print('++++')
          

        # all_scores = np.vstack([jac_scores.mean(axis=0),
        #                         dice_scores.mean(axis=0),
        #                         f2_scores.mean(axis=0),
        #                         PPV_scores.mean(axis=0),
        #                         Rec_scores.mean(axis=0),
        #                         acc_scores.mean(axis=0),
        #                         hfmean])
        all_scores = np.vstack([jac_scores.mean(axis=0),
                        dice_scores.mean(axis=0),
                        f2_scores.mean(axis=0),
                        PPV_scores.mean(axis=0),
                        Rec_scores.mean(axis=0),
                        acc_scores.mean(axis=0)])
    
        # all_scores = np.hstack([np.hstack(['jac',
        #                                    'dice',
        #                                    'F2',
        #                                    'PPV',
        #                                    'Rec', 'Acc', 'Hfd'])[:,None], all_scores])
        
        all_scores = np.hstack([np.hstack(['jac',
                                   'dice',
                                   'F2',
                                   'PPV',
                                   'Rec', 'Acc'])[:,None], all_scores])
        
        end = time.time()
        print('Elapsed time is...{}'.format(end - start))
        # final scores are wrapped in json file
        # my_dictionary = {"EndoCV2021":{
        #             "dice":{
        #             "value":  ( dice_scores.mean(axis=0)[0]) 
        #             },
        #             "jaccard":{
        #             "value": (jac_scores.mean(axis=0)[0])
        #             },
        #             "typeIIerror":{
        #             "value": (f2_scores.mean(axis=0)[0])
        #             },
        #             "PPV":{
        #             "value": (PPV_scores.mean(axis=0)[0]),
        #             },
        #             "recall":{
        #             "value": (Rec_scores.mean(axis=0)[0]),
        #             }, 
        #             "OverallAcc":{
        #             "value": (np.mean(acc_scores)),
        #             },
        #             "hausdorff_distance":{
        #             "value": (hfmean),
        #             },
        #             "dice_std":{
        #             "value": (np.std(dice_scores)),
        #             },
        #             "jc_std":{
        #             "value": (np.std(jac_scores)),
        #             },
        #             "f2_std":{
        #             "value": (np.std(f2_scores)),
        #             },
        #             "ppv_std":{
        #             "value": (np.std(PPV_scores)),
        #             },
        #             "r_std":{
        #             "value": (np.std(Rec_scores)),
        #             },                   
        #             "acc_std":{
        #             "value": (np.std(acc_scores)),
        #             },   
        #             "hdf_std":{
        #             "value": (hfstd),
        #             }, 
        #         }
        # } 
        my_dictionary = {"EndoCV2021":{
                    "dice":{
                    "value":  ( dice_scores.mean(axis=0)[0]) 
                    },
                    "jaccard":{
                    "value": (jac_scores.mean(axis=0)[0])
                    },
                    "typeIIerror":{
                    "value": (f2_scores.mean(axis=0)[0])
                    },
                    "PPV":{
                    "value": (PPV_scores.mean(axis=0)[0]),
                    },
                    "recall":{
                    "value": (Rec_scores.mean(axis=0)[0]),
                    }, 
                    "OverallAcc":{
                    "value": (np.mean(acc_scores)),
                    },
                    "dice_std":{
                    "value": (np.std(dice_scores)),
                    },
                    "jc_std":{
                    "value": (np.std(jac_scores)),
                    },
                    "f2_std":{
                    "value": (np.std(f2_scores)),
                    },
                    "ppv_std":{
                    "value": (np.std(PPV_scores)),
                    },
                    "r_std":{
                    "value": (np.std(Rec_scores)),
                    },                   
                    "acc_std":{
                    "value": (np.std(acc_scores)),
                    }, 
                }
        }   
                
        
        # write to json      
        jsonFileName=args.jsonFileName
        EndoCV_misc.write2json(jsonFileName, my_dictionary)
        
    


