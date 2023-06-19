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

import torch
import torch.nn as nn
from utils.visualizer import Visualizer

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from skimage.transform import resize

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

    # debiasing options
    parser.add_argument("--epiupwt", type=str, default=False,
                        help="use EpiUpWt de-biasing method during training")
    parser.add_argument("--sharpen", type=str, default=False,
                        help="use posterior sharpening method during training")
    parser.add_argument("--kappa", type=float, default=2.0,
                    help="weighting scalar")
    parser.add_argument("--cycle_length", type=int, default=150,
                    help="default cycle length")
    parser.add_argument("--cycles", type=int, default=1,
                    help="number of total inference cycles")
    parser.add_argument("--models_per_cycle", type=int, default=5,
                    help="number of posterior samples per cycle")
    # TODO replace with true training set size
    parser.add_argument("--temperature", type=float, default=1./1159,
                    help="posterior cooling temperature")
    parser.add_argument("--alpha", type=float, default=0.3,
                    help="1: SGLD; <1: HMC")

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
    
    parser.add_argument("--total_itrs", type=int, default=30e3, # not used for bay numbers
                        help="epoch number (default: 30k)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
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

    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")

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
            
    if opts.epiupwt:
        epi_dims = (opts.models_per_cycle, 2, 512, 512)
    else:
        epi_dims = None 

    train_dst = polyGenSeg(
        root=f"{opts.root}datasets/{opts.data_root}",
        image_set='train_polypGen',
        download=opts.download,
        transform=train_transform,
        epi_dims=epi_dims,
    )

    val_dst = polyGenSeg(
        root=f"{opts.root}datasets/{opts.data_root}",
        image_set='val_polypGen',
        download=False,
        transform=val_transform
    )
        
    return train_dst, val_dst

def bay_inference(opts, model, loader, device, metrics, mt_count=0, ret_samples_ids=None, model_desc=None):
    """ Do Bayesian inference from posterior samples and return metrics, preds and epis. """
    metrics.reset()
    ret_samples = []
    directoryName = 'results_%s_%s'%(opts.dataType, opts.model)
    # if opts.save_val_results:
    #     if not os.path.exists(directoryName):
    #         os.mkdir(directoryName)
    #     denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
    #                                std=[0.229, 0.224, 0.225])
    #     img_id = 0

    with torch.no_grad():

        def get_targets_from_dataloader(dataloader):
            targets = []
            for _, target in dataloader:
                targets.append(target)
            targets_tensor = torch.tensor(targets)
            return targets_tensor

        targets = get_targets_from_dataloader(loader).cpu().numpy()


        m_logits = {i: [] for i in range(mt_count)}
        for m in range(mt_count):
            moment_path = f"moments/{model_desc}_{moment_id}.pt"
            moment_ckpt = torch.load(moment_path, map_location=torch.device('cpu'))
           
            model.load_state_dict(checkpoint["model_state"])
            model = model.to(device)

            for i, (images, labels) in enumerate(loader):
                
                images = images.to(device, dtype=torch.float32)
                outputs = model(images)
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                print("preds.shape", preds.shape)
                exit()

                m_logits[m].append(preds)
                
        # [N_MOMENTS, N_SAMPLES, N_CLASSES]
        m_logits = torch.stack([torch.cat(m_logits[i], dim=0) for i in range(mt_count)])
        # [SAMPLES, N_CLASSES]
        m_preds = m_logits.mean(dim=0)

        # TODO check the shape here; need another mean across dims probably!
        print("m_shape.shape", m_shape.shape)

        # accumulate epistemic uncertainties
        temp = (m_logits - m_preds.expand((mt_count, *m_preds.shape)))**2
        epis_ = torch.sqrt(torch.sum(temp, axis=0)) / mt_count
        epis_ = torch.Tensor([e[int(i)] for e, i in zip(epis_, targets)]).double().to(device)
        epis = epis_ * 10.

    metrics.update(targets, m_preds)
    score = metrics.get_results()

    return score

def validate(opts, model, loader, device, metrics, ret_samples_ids=None):
    """Do validation and return specified samples"""
    metrics.reset()
    ret_samples = []
    directoryName = 'results_%s_%s'%(opts.dataType, opts.model)
    if opts.save_val_results:
        if not os.path.exists(directoryName):
            os.mkdir(directoryName)
        denorm = utils.Denormalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
        img_id = 0

    with torch.no_grad():
        for i, (images, labels) in tqdm(enumerate(loader)):
            
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()
            
            metrics.update(targets, preds)
            if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                ret_samples.append(
                    (images[0].detach().cpu().numpy(), targets[0], preds[0]))

            if opts.save_val_results:
                for i in range(len(images)):
                    image = images[i].detach().cpu().numpy()
                    target = targets[i]
                    pred = preds[i]

                    image = (denorm(image) * 255).transpose(1, 2, 0).astype(np.uint8)
                    target =(target).astype(np.uint8)
                    pred = (pred).astype(np.uint8)

                    Image.fromarray(image).save(directoryName + '/%d_image.png' % img_id)
                    Image.fromarray(target).save(directoryName +'/%d_target.png' % img_id)
                    Image.fromarray(pred).save(directoryName+'/%d_pred.png' % img_id)

                    fig = plt.figure()
                    plt.imshow(image)
                    plt.axis('off')
                    plt.imshow(pred, alpha=0.7)
                    ax = plt.gca()
                    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                    plt.savefig(directoryName+'/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                    plt.close()
                    img_id += 1

        score = metrics.get_results()
    return score, ret_samples


def main():
    torch.cuda.is_available()
    torch.cuda.device_count()

    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 2 # foreground + background

    model_desc = "test"
    if not opts.dev_run:        
        project_name = "baybaseline"
        if opts.epiupwt:
            project_name = "epiupwt"
        elif opts.sharpen:
            project_name = "sharpen"

        wandb.init(
            project=project_name,
            config={
                "learning_rate": opts.lr,
                "cycle_length": opts.cycle_length,
                "cycles": opts.cycles,
                "alpha": opts.alpha,
                "kappa": opts.kappa,
                "models_per_cycle": opts.models_per_cycle,
            }
        )

        model_desc = wandb.run.name
        print(f"Running new experiment under {project_name} named: {wandb.run.name}")
        if os.path.exists(f"moments/{model_desc}"):
            print("[ERROR] {model_desc} already exists. Aborting.")
            exit()
        utils.mkdir(f"moments/{model_desc}")

    # Setup visualization
    vis = Visualizer(port=opts.vis_port,
                     env=opts.vis_env) if opts.enable_vis else None
    if vis is not None:  # display options
        vis.vis_table("Options", vars(opts))

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
        train_dst, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = data.DataLoader(
        val_dst, batch_size=opts.val_batch_size, shuffle=True, num_workers=2)
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
        
    model = model.to(device)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    if (opts.model != 'pspNet') and (opts.model != 'segNet') and (opts.model != 'FCN8'):
        utils.set_bn_momentum(model.backbone, momentum=0.01)
    
    # Set up metrics
    metrics = StreamSegMetrics(opts.num_classes)

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
        ], lr=opts.lr, momentum=1.0 - opts.alpha, weight_decay=opts.weight_decay)  

    #optimizer = torch.optim.SGD(params=model.parameters(), lr=opts.lr, momentum=0.9, weight_decay=opts.weight_decay)
    #torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)
    # if opts.lr_policy=='poly':
    #     scheduler = utils.PolyLR(optimizer, opts.total_itrs, power=0.9)
    # elif opts.lr_policy=='step':
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.step_size, gamma=0.1)
    scheduler = None

    # Set up criterion
    #criterion = utils.get_loss(opts.loss_type)
    if opts.loss_type == 'focal_loss':
        criterion = utils.FocalLoss(ignore_index=255, size_average=True)
    elif opts.loss_type == 'cross_entropy':
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

    def save_moment(model_desc, model, moment_id):
        """ save moment checkpoint
        """
        path = f"{opts.root}/moments/{model_desc}/{moment_id}.pt"

        if not opts.dev_run:
            torch.save({
                "model_state": model.state_dict(),
            }, path)
        print(f"[{not opts.dev_run}] Model MOMENT {moment_id} saved")

    # bayesian csg-mcmc functions
    def noise_loss(lr):
        noise_loss = 0.0
        noise_std = (2 / lr * opts.alpha)**0.5
        for var in model.parameters():
            means = torch.zeros(var.size()).to(device)
            noise_loss += torch.sum(var * torch.normal(means, std=noise_std).to(device))

        return noise_loss

    def adjust_learning_rate(model, batch_idx, optim, current_epoch):
        rcounter = (current_epoch) * num_batches + batch_idx

        cos_inner = np.pi * (rcounter % (opts.total_itrs // opts.cycles))
        cos_inner /= opts.total_itrs // opts.cycles
        cos_out = np.cos(cos_inner) + 1
        lr = 0.5 * cos_out * opts.lr

        if opts.alpha < 1.0:
            # decay u-net backbone at same rate, factor 1/10 
            optim.param_groups[0]['lr'] = 0.1*lr
            optim.param_groups[1]['lr'] = lr

        return lr

    def update_params(model, lr, current_epoch):
        """ manual update to params for HMC only not LD """
        for p in model.parameters():
            if not hasattr(p,'buf'):
                p.buf = torch.zeros(p.size()).to(device)
            d_p = p.grad.data
            d_p.add_(p.data, alpha=opts.weight_decay)

            buf_new = (1 - opts.alpha) * p.buf - lr * d_p
            if (current_epoch % opts.cycle_length) + 1 > (opts.cycle_length - opts.models_per_cycle):
                eps = torch.randn(p.size()).to(device)
                buf_new += (2.0 * opts.lr * opts.alpha * opts.temperature / train_size)**.5 * eps
            p.data.add_(buf_new)
            p.buf = buf_new

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

    # Restore
    best_score = 0.0
    cur_itrs = 0
    cur_epochs = 0
    moment_count = 0

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
    while True: 

        # =====  Train  =====
        model.train()
        for batch in enumerate(train_loader):
            if opts.epiupwt or opts.sharpen:
                batch_idx, (images, labels, idxes) = batch
            else:
                batch_idx, (images, labels) = batch
    
            cur_itrs += 1

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
     
            optimizer.zero_grad()
            outputs = model(images)
 
           # ======= Epistemic uncertainties ==========================================
            epistemics = []
            weights = np.ones((16))

            if opts.epiupwt and (cur_epochs % opts.cycle_length + 1 == opts.cycle_length):
                # which index in cycle is this moment
                moment_id = moment_count % opts.models_per_cycle

                # save in correct indices
                # [moment_id, idxes, 2, 512, 512]
                train_dst.p_hats[moment_id][idxes.cpu()] = outputs.detach().cpu().numpy()

            # if not first cycle, use epis for dynamic upweighting
            if opts.epiupwt and (cur_epochs > opts.cycle_length):
                # p_bars = y pred, p_hats = p_theta_i(y | x)
                p_hats = train_dst.p_hats[:, idxes.cpu()]
                p_bars = p_hats.mean(axis=0)

                temp = (p_hats - np.broadcast_to(p_bars, (opts.models_per_cycle, *p_bars.shape)))**2
                epistemics = np.sqrt(np.sum(temp, axis=0)) / opts.models_per_cycle
                # [models per cycle, batch, 2, 512, 512]
                epistemics = epistemics.astype(np.double)
                condensed_epis = np.mean(np.max(epistemics, axis=1), axis=(1, 2))
                weights = weighting_function(condensed_epis)

            # =============== LOSS ======================
            loss = standard_loss(outputs, labels, criterion, weights, device)
            lr = adjust_learning_rate(model, batch_idx, optimizer, cur_epochs)
            wandb.log({"lr": lr, "train_loss": loss})

            if opts.alpha == 1.0:
                if (cur_epochs % opts.cycle_length) + 1 > (opts.cycle_length - opts.models_per_cycle):
                    loss_noise = noise_loss(lr) * (opts.temperature / train_size)**.5
                    loss += loss_noise
                loss.backward()
            else:  # alpha < 1.0 is HMC
                loss.backward()
                update_params(model, lr, cur_epochs)

            optimizer.step()

            np_loss = loss.detach().cpu().numpy()
            interval_loss += np_loss
            if vis is not None:
                vis.vis_scalar('Loss', cur_itrs, np_loss)

            if (cur_itrs) % 10 == 0:
                interval_loss = interval_loss/10
                print("Epoch %d, Itrs %d/%d, Loss=%f" %
                      (cur_epochs, cur_itrs, opts.total_itrs, interval_loss))
                interval_loss = 0.0  

            if (cur_itrs) % opts.val_interval == 0:
                # save_ckpt('checkpoints_polypGen/latest_%s_%s_os%d_%s_%s.pth' %
                #           (opts.model, opts.dataset, opts.output_stride, opts.dataType, opts.backbone))
                print("validation...")
                model.eval()
                val_score, ret_samples = validate(
                    opts=opts,
                    model=model,
                    loader=val_loader,
                    device=device,
                    metrics=metrics,
                    ret_samples_ids=vis_sample_id,
                )

                print(metrics.to_str(val_score))
                if val_score['Mean IoU'] > best_score:  # save best model
                    best_score = val_score['Mean IoU']
                    # save_ckpt('checkpoints_polypGen/best_%s_%s_os%d_%s_%s.pth' %
                    #           (opts.model, opts.dataset,opts.output_stride, opts.dataType, opts.backbone))
                wandb.log({"val_mean_iou": val_score['Mean IoU']})
                # if vis is not None:  # visualize validation score and samples
                #     vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                #     vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                #     vis.vis_table("[Val] Class IoU", val_score['Class IoU'])

                wandb.log({"val_acc": val_score['Overall Acc'], "val_class_iou": val_score['Class IoU']})

                if opts.log_masks_wandb:
                    samples = []
                    for k, (img, target, lbl) in enumerate(ret_samples):
                        img = (denorm(img) * 255).astype(np.uint8)
                        target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                        lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                        concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                        # vis.vis_image('Sample %d' % k, concat_img)
                        samples.append(concat_img)
                    wandb.log({"samples": np.array(samples)})

            model.train()
            if scheduler:
                scheduler.step()  

            if opts.dev_run: # single itr per epoch only on dev run
                break

        # within sampling phase
        if ((cur_epochs % opts.cycle_length) + 1) > (opts.cycle_length - opts.models_per_cycle):
            save_moment(model_desc, model, moment_count)
            moment_count += 1

        cur_epochs += 1

        if cur_epochs > (opts.cycle_length * opts.cycles):
            break


    # to save time let's just rely on the validation performance
    # bay_inference(
    #     opts,
    #     model,
    #     loader,
    #     device,
    #     metrics,
    #     mt_count=moment_count,
    #     ret_samples_ids=None,
    #     model_desc=model_desc
    # )
        
if __name__ == '__main__':
    main()
