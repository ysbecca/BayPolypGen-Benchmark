from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

import torch.nn.functional as F
from torch.utils import data
import wandb

from datasets import VOCSegmentation_polypGen2021 as polyGenSeg

from utils import ext_transforms as et
from metrics import StreamSegMetrics

from pytorch_pcgrad.pcgrad import PCGrad

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize
from collections import OrderedDict


import random
import string
from nltk.corpus import words

def plotInference( imgs, depth):
    f =  plt.figure()
    f.add_subplot(1,2, 1)
    plt.imshow(resize((imgs), (256, 256)))
    f.add_subplot(1,2, 2)
    depth_masked = resize(depth, (256, 256))
    plt.imshow(depth_masked)
    # plt.show(block=True)
    return f


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dev_run", type=bool, default=False,
                        help='does not save checkpoints or log metrics when in dev mode')
    parser.add_argument("--model_desc", type=str, default='',
                        help="description of baseline Bayesian model")
    # debiasing options
    parser.add_argument("--sharpen", type=str, default=False,
                        help="use posterior sharpening method during training")
    parser.add_argument("--kappa", type=float, default=2.0,
                    help="weighting scalar")

    parser.add_argument("--loss_type", type=str, default="pcgrad",
                    choices=['pcgrad', 'sharpen'],
                    help="multi-task loss")

    parser.add_argument("--max_epochs", type=int, default=20,
                    help="max epochs to sharpen for")

    parser.add_argument("--lr", type=float, default=0.1,
                    help="lr")
    parser.add_argument("--total_epochs", type=int, default=10, 
                        help="epochs to run")
    parser.add_argument("--moment_count", type=int, default=2, 
                        help="model moments to use")

    # Dataset Options
    parser.add_argument("--root", type=str, default="",
                        help='absolute path to EndoCV2021')

    parser.add_argument("--data_root", type=str, default='EndoCV2021/trainData_polypGen/',
                        help="path to Dataset")
    parser.add_argument("--dataType", type=str, default='polypGen',
                        help="path to Dataset")
    parser.add_argument("--dataset", type=str, default='polypGen',
                        choices=['polypGen'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=2,
                        help="num classes (default: None)")
    parser.add_argument("--download", action='store_true', default=False,
                        help="download datasets")
    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet50',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet', 'pspNet', 'segNet', 'FCN8', 'resnet-Unet', 'axial', 'unet'], help='model name')

    # if unet backbone
    parser.add_argument("--backbone", type=str, default='resnet50',
                        choices=['vgg19',  'resnet34' , 'resnet50',
                                 'resnet101', 'densenet121', 'none'], help='model name')


    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--test_only", action='store_true', default=False)
    
    parser.add_argument("--save_val_results", action='store_true', default=True,
                        help="save segmentation results to \"./results_polypGen\"")
    
    parser.add_argument("--lr_policy", type=str, default='poly', choices=['poly', 'step'],
                        help="learning rate scheduler policy")
    parser.add_argument("--step_size", type=int, default=10000)
    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--val_batch_size", type=int, default=16,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=512)
    
    parser.add_argument("--ckpt", default=None, type=str,
                        help="restore from checkpoint")
    parser.add_argument("--continue_training", action='store_true', default=False)

    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    parser.add_argument("--random_seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--print_interval", type=int, default=10,
                        help="print interval of loss (default: 10)")
    parser.add_argument("--val_interval", type=int, default=100,
                        help="epoch interval for eval (default: 100)")

    parser.add_argument("--log_masks_wandb", default=False,
                        help="log --vis_num_samples samples during training on wandb")

    # Visdom options
    parser.add_argument("--enable_vis", action='store_true', default=False,
                        help="use visdom for visualization")
    parser.add_argument("--vis_port", type=str, default='13570',
                        help='port for visdom')
    parser.add_argument("--vis_env", type=str, default='main',
                        help='env for visdom')
    parser.add_argument("--vis_num_samples", type=int, default=8,
                        help='number of samples for visualization (default: 8)')
    return parser

def get_dataset(opts):
    """ Dataset And Augmentation
    """
    if opts.dataset == 'polypGen':
        train_transform = et.ExtCompose([
             et.ExtRandomScale((0.5, 2.0)),
            et.ExtRandomCrop(size=(opts.crop_size, opts.crop_size), pad_if_needed=True),
            et.ExtRandomHorizontalFlip(),
            et.ExtToTensor(),
            et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])
        if opts.crop_val:
            val_transform = et.ExtCompose([
                et.ExtResize(opts.crop_size),
                et.ExtCenterCrop(opts.crop_size),
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
        else:
            val_transform = et.ExtCompose([
                et.ExtToTensor(),
                et.ExtNormalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
            ])
            

    train_dst = polyGenSeg(
        root=f"{opts.root}datasets/{opts.data_root}",
        image_set='train_polypGen',
        download=opts.download,
        indices=True,
        transform=train_transform,
    )

    val_dst = polyGenSeg(
        root=f"{opts.root}datasets/{opts.data_root}",
        image_set='val_polypGen',
        download=False,
        indices=True,
        transform=val_transform
    )
        
    return train_dst, val_dst


def main():
    torch.cuda.is_available()
    torch.cuda.device_count()

    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 2 # foreground + background

    project_name = "sharpen"
    model_desc = opts.model_desc + "_sharpen"
    if not opts.dev_run:
        wandb.init(
            project=project_name,
            name=model_desc,
            config={
                "name": opts.model_desc,
                "learning_rate": opts.lr,
                "max_epochs": opts.max_epochs,
            }
        )

        print(f"Running new experiment under {project_name} named: {wandb.run.name}")
        # if os.path.exists(f"moments/{model_desc}"):
        #     print("[ERROR] {model_desc} already exists. Aborting.")
        #     exit()
        # utils.mkdir(f"moments/{model_desc}")

    # Setup visualization
    # vis = Visualizer(port=opts.vis_port,
                     # env=opts.vis_env) if opts.enable_vis else None
    # if vis is not None:  # display options
        # vis.vis_table("Options", vars(opts))

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup random seed
    torch.manual_seed(opts.random_seed)
    np.random.seed(opts.random_seed)
    random.seed(opts.random_seed)

    # Setup dataloader
    if opts.dataset=='voc' and not opts.crop_val:
        opts.val_batch_size = 1
    
    train_dst, val_dst = get_dataset(opts)
    
    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=False, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=False, num_workers=2)
    print("Dataset: %s, Train set: %d, Val set: %d" %
          (opts.dataset, len(train_dst), len(val_dst)))

    # for tempering
    train_size = len(train_dst)
    num_batches = train_size / opts.batch_size + 1

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

    models = [model for m in range(opts.moment_count)]
    print("Loaded models", len(models))

    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    if (opts.model != 'pspNet') and (opts.model != 'segNet') and (opts.model != 'FCN8'):
        utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)


    def get_optimizer():
               # Set up optimizer
        if opts.model in ['pspNet', 'segNet', 'FCN8']:
            optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay) 

        elif opts.model == 'resnet-Unet':
            optimizer = torch.optim.Adam(params=model.parameters(), lr=opts.lr) 

        elif opts.model == 'unet':
            optimizer = torch.optim.RMSprop(params=model.parameters(), lr=opts.lr, weight_decay=1e-8, momentum=0.9)
            
        elif opts.model == 'axial':
            optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
        else:
            optimizer = torch.optim.SGD(params=[
                {'params': model.backbone.parameters(), 'lr': 0.1*opts.lr},
                {'params': model.classifier.parameters(), 'lr': opts.lr},
            ], lr=opts.lr, momentum=0.8, weight_decay=opts.weight_decay)  
        return optimizer

    def weighting_function(epis):
        return torch.pow((1.0 + epis), opts.kappa)

    optims = [get_optimizer() for m in range(opts.moment_count)]
    print("Loaded optims", len(optims))

    if opts.loss_type == "pcgrad":
        optims = [PCGrad(o) for o in optims]

    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    # if opts.lr_policy=='poly':
    #     scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    # elif opts.lr_policy=='step':
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    scheduler = None

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    # if opts.loss_type == 'focal_loss':
    #     criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    # elif opts.loss_type == 'cross_entropy':
    criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')

    if not opts.dev_run:
        utils.mkdir('checkpoints')

    def save_ckpt(path):
        """ save current model
        """
        if not opts.dev_run:
            torch.save({
                "cur_itrs": cur_itrs,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_score": best_score,
            }, path)

        print(f"[{not opts.dev_run}] Model saved as {path}")    

    def save_moment(model, moment_id, epoch=-1):
        """ save moment checkpoint
        """
        path = f"{opts.root}/moments/{model_desc}/{moment_id}"
        if epoch > -1:
            path += f"_{epoch}s"

        path += ".pt"

        if not opts.dev_run:
            torch.save({
                "model_state": model.state_dict(),
            }, path)
        print(f"[{not opts.dev_run}] Model MOMENT {moment_id} saved")

    def standard_loss(outputs, labels, criterion, weights=[], device=None):
        if len(weights):
            loss = F.cross_entropy(outputs, labels, reduction="none")
            adj_w = torch.tensor(weights).unsqueeze(dim=1).unsqueeze(dim=1).to(device)

            loss *= adj_w
            return loss.sum() / len(labels)
        else:
            return F.cross_entropy(outputs, labels)

    def weighting_function(epistemics):
        return torch.pow((1.0 + torch.tensor(epistemics)), opts.kappa)


    def load_moment(moment_id, model, device):

        checkpoint = torch.load(f"{opts.root}/moments/{opts.model_desc}/{moment_id}.pt", map_location=device)
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

    def predict_full_posterior(models, loader, size, compute_acc=False):

        m_logits = []

        if opts.dev_run:
            true_targets = np.zeros((opts.batch_size, 512, 512))
        else:
            true_targets = np.zeros((size, 512, 512))

        for batch in enumerate(loader):
            _, (_, targets, idxes) = batch
            true_targets[idxes] = targets

            if opts.dev_run:
                break

        print(true_targets.shape)
        for model_idx, m in enumerate(models):
            m_preds = []
            for batch in enumerate(loader):
                batch_idx, (images, labels, idxes) = batch

                outputs = model(images)
                preds_batch = outputs.detach().max(dim=1)[1].cpu().numpy()*255
                preds_batch = preds_batch.astype(np.uint8)
                m_preds.append(preds_batch)

                if opts.dev_run:
                    break

            m_logits.append(np.concatenate(m_preds))

        # [N_MOMENTS, N_SAMPLES, 512, 512]
        m_logits = np.array(m_logits)
        print("m_logits.shape", m_logits.shape)

        # [N_SAMPLES, 512, 512]
        m_preds = np.mean(m_logits, axis=0)
        print("m_preds.shape", m_preds.shape)

        temp = (m_logits - np.broadcast_to(m_preds, (opts.moment_count, *m_preds.shape)))**2
        epis_ = np.sqrt(np.sum(temp, axis=0)) / opts.moment_count
        epis_ = epis_.astype(np.double)

        # [N_SAMPLES, 512, 512]
        print("epis_.shape before collapse", epis_.shape)
        # take max or mean?
        epis = epis_.mean(axis=(1, 2))
        print("epis.shape", epis.shape)

        if compute_acc:
            metrics.reset()
            # compute useful metrics on validation set... 
            metrics.update(true_targets, m_preds)

            score = metrics.get_results()
            return score
        else:
            return m_preds, epis

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0

    # TODO rewrite for Bay version
    # if opts.ckpt is not None and os.path.isfile(opts.ckpt):
    #     checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint["model_state"])
    #     model = nn.DataParallel(model)
    #     model.to(device)
    #     if opts.continue_training:
    #         optimizer.load_state_dict(checkpoint["optimizer_state"])
    #         scheduler.load_state_dict(checkpoint["scheduler_state"])
    #         cur_itrs = checkpoint["cur_itrs"]
    #         best_score = checkpoint['best_score']
    #         print("Training state restored from %s" % opts.ckpt)
    #     print("Model restored from %s" % opts.ckpt)
    #     del checkpoint  # free memory
    # else:
    #     print("[!] Retrain")
    #     model = nn.DataParallel(model)
    #     model.to(device)

    #==========   Train Loop   ==========#
    vis_sample_id = np.random.randint(0, len(val_loader), opts.vis_num_samples,
                                      np.int32) if (opts.enable_vis or opts.log_masks_wandb) else None 
    denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  
    interval_loss = 0

    # assumes unshuffled set! only once on train
    mean_preds, epis = predict_full_posterior(models, train_loader, len(train_dst))
    weights = weighting_function(epis)

    train_loader = data.DataLoader(
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    
    for e in range(opts.max_epochs):
        print("Epoch", e)
        for moment_id, m in enumerate(models):
            model = load_moment(moment_id, model.to(device), device)

            model.train()
            for batch in enumerate(train_loader):
                batch_idx, (images, labels, idxes) = batch
                mean_preds_batch = mean_preds[idxes]
                weights_batch = weights[idxes]

                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
     
                optims[model_idx].zero_grad()
                outputs = m(images)
        
                # =============== LOSS ======================
                std_loss = standard_loss(outputs, labels, criterion, weights, device)
                sharpen_loss = standard_loss(outputs, mean_preds_batch, weights, device)

                if opts.loss_type == "pcgrad":
                    optims[model_idex].pc_backward([sharpen_loss, std_loss])
                elif opts.loss_type == "sharpen":
                    sharpen_loss.backward()
                else:
                    std_loss.backward()
    
                optims[model_idex].step()

                # log batch loss...
                if not opts.dev_run:
                    wandb.log({"std loss": std_loss, "sharpen loss": sharpen_loss})
                else:
                    # single batch per moment per epoch on dev mode
                    break

            # save sharpened moment without overwriting.
            save_moment(model, model_idx, epoch=e)

    score = predict_full_posterior(models, val_loader, len(val_dst), compute_acc=True)
    if opts.dev_run:
        print(score)
    else:
        wandb.log(score)
    


if __name__ == '__main__':
    main()
