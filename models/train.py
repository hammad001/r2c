"""
Training script. Should be pretty adaptable to whatever.
"""
import argparse
import os
import shutil

from tensorboardX import SummaryWriter
import torch.nn.functional as F
import multiprocessing
import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.training.learning_rate_schedulers import LearningRateScheduler
from allennlp.training.optimizers import Optimizer
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d
from tqdm import tqdm

from dataloaders.vcr import VCR, VCRLoader
from utils.pytorch_misc import time_batch, save_checkpoint, clip_grad_norm, \
    restore_checkpoint, print_para, restore_best_checkpoint

import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', level=logging.DEBUG)

# This is needed to make the imports work
from allennlp.models import Model
import models

#################################
#################################
######## Data loading stuff
#################################
#################################

parser = argparse.ArgumentParser(description='train')
parser.add_argument(
    '-params',
    dest='params',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-folder',
    dest='folder',
    help='folder location',
    type=str,
)
parser.add_argument(
    '-tensorboard_log',
    dest='tensorboard_log',
    help='tensorboard log location',
    type=str,
)
parser.add_argument(
    '-no_tqdm',
    dest='no_tqdm',
    action='store_true',
)

args = parser.parse_args()

writer = SummaryWriter(args.tensorboard_log)

params = Params.from_file(args.params)
train, val = VCR.splits(embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                        only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True))
NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")

def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            td[k] = {k2: v.cuda(non_blocking=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
                non_blocking=True)
    return td

num_workers = 64 # (4 * NUM_GPUS if NUM_CPUS >= 32 else 2*NUM_GPUS)-1
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': 96 // NUM_GPUS, 'num_gpus':NUM_GPUS, 'num_workers':num_workers}

# Train Loaders
train_loader = VCRLoader.from_dataset(train, **loader_params)

# Val Loaders
val_loader = VCRLoader.from_dataset(val, **loader_params)

ARGS_RESET_EVERY = 100
print("Loading {} ".format(params['model'].get('type', 'WTF?')), flush=True)

model_qa = Model.from_params(vocab=train.vocab, params=params['model'])
model_ra = Model.from_params(vocab=train.vocab, params=params['model_ra'])

def make_backbone_req_grad_false(model):
    for submodule in model.detector.backbone.modules():
        if isinstance(submodule, BatchNorm2d):
            submodule.track_running_stats = False
        for p in submodule.parameters():
            p.requires_grad = False

make_backbone_req_grad_false(model_qa)
make_backbone_req_grad_false(model_ra)

model_qa = DataParallel(model_qa).cuda() if NUM_GPUS > 1 else model_qa.cuda()
model_ra = DataParallel(model_ra).cuda() if NUM_GPUS > 1 else model_ra.cuda()

model_params_qa = [x for x in model_qa.named_parameters() if x[1].requires_grad]
model_params_ra = [x for x in model_ra.named_parameters() if x[1].requires_grad]

optimizer = Optimizer.from_params(model_params_qa + model_params_ra,
                                  params['trainer']['optimizer'])

lr_scheduler_params = params['trainer'].pop("learning_rate_scheduler", None)
scheduler = LearningRateScheduler.from_params(optimizer, lr_scheduler_params) if lr_scheduler_params else None

if os.path.exists(args.folder):
    print("Found folder! restoring", flush=True)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model_qa, 'qa', optimizer, serialization_dir=args.folder,
                                                           learning_rate_scheduler=scheduler)
    start_epoch, val_metric_per_epoch = restore_checkpoint(model_ra, 'ra', optimizer, serialization_dir=args.folder,
                                                           learning_rate_scheduler=scheduler)
   
else:
    print("Making directories")
    os.makedirs(args.folder, exist_ok=True)
    start_epoch, val_metric_per_epoch = 0, []
    shutil.copy2(args.params, args.folder)


def log_tensorboard(mode, it, qa_loss, ra_loss, qar_loss, qa_acc, ra_acc, qar_acc):
    writer.add_scalar('{}.qa_loss'.format(mode), qa_loss, it)
    writer.add_scalar('{}.ra_loss'.format(mode), ra_loss, it)
    writer.add_scalar('{}.qar_loss'.format(mode), qar_loss, it)
    writer.add_scalar('{}.qa_acc'.format(mode), qa_acc, it)
    writer.add_scalar('{}.ra_acc'.format(mode), ra_acc, it)
    writer.add_scalar('{}.qar_acc'.format(mode), qar_acc, it)


def cal_accuracy(preds, labels):
    try:
        preds = np.argmax(preds, axis=1)
    except:
        import ipdb
        ipdb.set_trace()
    matches = (preds == labels) 
    return matches, np.mean(matches)


def cal_net_accuracy(qa_preds, qa_label, ra_preds, ra_label):
    qa_matches, qa_acc = cal_accuracy(qa_preds, qa_label)
    ra_matches, ra_acc = cal_accuracy(ra_preds, ra_label)
    qar_matches = qa_matches * ra_matches

    return qa_acc, ra_acc, np.mean(qar_matches)

# criterion_ra = torch.nn.CrossEntropyLoss().cuda()
param_shapes = print_para(model_qa)
num_batches = 0
tot_epoch_batch = len(train_loader)
for epoch_num in range(start_epoch, params['trainer']['num_epochs'] + start_epoch):
    train_results = []
    norms = []

    model_qa.train()
    model_ra.train()

    for b, (time_per_batch, batch) in enumerate(time_batch(train_loader if args.no_tqdm else tqdm(train_loader), reset_every=ARGS_RESET_EVERY)):
        
        batch = _to_gpu(batch)
        batch_sz = batch['label'].shape[0]

        optimizer.zero_grad()
        
        output_dict_qa = model_qa(batch)
        logits = output_dict_qa['label_probs']
        output_dict_ra = model_ra(batch) 
        
        all_ra_loss = output_dict_ra['all_ra_loss']
        all_ra_loss = torch.t(torch.reshape(all_ra_loss, (batch_sz, -1)))
        all_reg_loss = output_dict_ra['all_ra_reg_loss']

        exp_loss = torch.zeros(batch_sz).cuda()
        logits_sum = torch.zeros(batch_sz).cuda()

        for _ in range(64):
            samp_ind = (torch.multinomial(logits, 1)).squeeze().cuda()
            exp_loss += all_ra_loss[samp_ind, range(batch_sz)] * logits[range(batch_sz), samp_ind]
            logits_sum += logits[range(batch_sz), samp_ind]

        exp_loss = exp_loss/logits_sum

        loss_qa = output_dict_qa['loss'].mean() + output_dict_qa['cnn_regularization_loss'].mean()
        loss_ra = exp_loss.mean() + all_reg_loss.mean()

        loss = loss_qa + loss_ra

        loss.backward()

        num_batches += 1
        if scheduler:
            scheduler.step_batch(num_batches)

        norms.append(
            clip_grad_norm(list(model_qa.named_parameters()) + list(model_ra.named_parameters()),
                max_norm=params['trainer']['grad_norm'], clip=True, verbose=False)
        )
        optimizer.step()

        qa_label = batch['label']
        ra_label = batch['label_ra']

        _, best_pred_quest_ind = torch.max(logits, 1)
        pred_probs_all = output_dict_ra['all_ra_label_probs'].reshape(batch_sz, 4, 4)
        out_ra_probs = pred_probs_all[range(batch_sz), best_pred_quest_ind, :].detach().cpu().numpy()

        qa_accuracy, ra_accuracy, qar_accuracy = cal_net_accuracy(output_dict_qa['label_probs'].detach().cpu().numpy(),
                                                           qa_label.detach().cpu().numpy(),
                                                           out_ra_probs,
                                                           ra_label.detach().cpu().numpy())
        
        log_tensorboard('train', epoch_num * tot_epoch_batch + b, loss_qa.detach().cpu().item(), loss_ra.detach().cpu().item(), 
                         loss.detach().cpu().item(), qa_accuracy, ra_accuracy, qar_accuracy)
        
        train_results.append(pd.Series({'loss_qa': loss_qa.detach().cpu().item(),
                                        'loss_ra': loss_ra.detach().cpu().item(),
                                        'net_loss': loss.detach().cpu().item(),
                                        'accuracy_qa': qa_accuracy,
                                        'accuracy_ra': ra_accuracy,
                                        'net_accuracy': qar_accuracy,
                                        'sec_per_batch': time_per_batch,
                                        'hr_per_epoch': len(train_loader) * time_per_batch / 3600,
                                        }))
        if b % ARGS_RESET_EVERY == 0 and b > 0:
            norms_df = pd.DataFrame(pd.DataFrame(norms[-ARGS_RESET_EVERY:]).mean(), columns=['norm']).join(
                param_shapes[['shape', 'size']]).sort_values('norm', ascending=False)

            print("e{:2d}b{:5d}/{:5d}. norms: \n{}\nsumm:\n{}\n~~~~~~~~~~~~~~~~~~\n".format(
                epoch_num, b, len(train_loader),
                norms_df.to_string(formatters={'norm': '{:.2f}'.format}),
                pd.DataFrame(train_results[-ARGS_RESET_EVERY:]).mean(),
            ), flush=True)

    print("---\nTRAIN EPOCH {:2d}:\n{}\n----".format(epoch_num, pd.DataFrame(train_results).mean()))
    val_probs_qa = []
    val_probs_ra = []
    val_labels_qa = []
    val_labels_ra = []
    val_loss_sum_qa = 0.0
    val_loss_sum_ra = 0.0
    val_loss_sum_qar = 0.0
    
    model_qa.eval()
    model_ra.eval()
    


    for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
        with torch.no_grad():
            batch = _to_gpu(batch)
    
            output_dict_qa = model_qa(batch)
            
            logits = output_dict_qa['label_probs']
            output_dict_ra = model_ra(batch)
            
            all_ra_loss = output_dict_ra['all_ra_loss']
            batch_sz = batch['label'].shape[0]

            all_ra_loss = output_dict_ra['all_ra_loss']
            all_ra_loss = torch.t(torch.reshape(all_ra_loss, (batch_sz, -1)))
    
            exp_loss = torch.zeros(batch_sz).cuda()
            logits_sum = torch.zeros(batch_sz).cuda()
    
            for _ in range(64):
                samp_ind = (torch.multinomial(logits, 1)).squeeze().cuda()
                exp_loss += all_ra_loss[samp_ind, range(batch_sz)] * logits[range(batch_sz), samp_ind]
                logits_sum += logits[range(batch_sz), samp_ind]
    
            exp_loss = exp_loss/logits_sum
            
            loss_ra = exp_loss.mean().item() * batch['label_ra'].shape[0] 
            val_loss_sum_ra += loss_ra

            loss_qa = output_dict_qa['loss'].mean().item() * batch['label'].shape[0] 
            val_loss_sum_qa += loss_qa

            val_loss_sum_qar += loss_qa + loss_ra
            
            qa_label = batch['label']
            ra_label = batch['label_ra']

            _, best_pred_quest_ind = torch.max(logits, 1)
            pred_probs_all = output_dict_ra['all_ra_label_probs'].reshape(batch_sz, 4, 4)
            out_ra_probs = pred_probs_all[range(batch_sz), best_pred_quest_ind, :].detach().cpu().numpy()

            val_probs_qa.append(output_dict_qa['label_probs'].detach().cpu().numpy())
            val_probs_ra.append(out_ra_probs)
            val_labels_qa.append(qa_label.detach().cpu().numpy())
            val_labels_ra.append(ra_label.detach().cpu().numpy())

    val_labels_qa = np.concatenate(val_labels_qa, 0)
    val_labels_ra = np.concatenate(val_labels_ra, 0)
    val_probs_qa = np.concatenate(val_probs_qa, 0)
    val_probs_ra = np.concatenate(val_probs_ra, 0)
    
    qa_accuracy, ra_accuracy, qar_accuracy = cal_net_accuracy(val_probs_qa, val_labels_qa,
                                                           val_probs_ra, val_labels_ra)

    val_loss_avg_qa = val_loss_sum_qa / val_labels_qa.shape[0]
    val_loss_avg_ra = val_loss_sum_ra / val_labels_ra.shape[0]
    val_loss_avg_qar = val_loss_sum_qar / val_labels_qa.shape[0]

    log_tensorboard('val', epoch_num * tot_epoch_batch, val_loss_avg_qa, val_loss_avg_ra, 
                         val_loss_avg_qar, qa_accuracy, ra_accuracy, qar_accuracy)


    val_metric_per_epoch.append(qar_accuracy)
    if scheduler:
        scheduler.step(val_metric_per_epoch[-1], epoch_num)

    print("Val epoch {} has qa acc {:.3f} and qa loss {:.3f}".format(epoch_num, qa_accuracy, val_loss_avg_qa), flush=True)
    print("Val epoch {} has ra acc {:.3f} and ra loss {:.3f}".format(epoch_num, ra_accuracy, val_loss_avg_ra), flush=True)
    print("Val epoch {} has qar acc {:.3f} and qar loss {:.3f}".format(epoch_num, qar_accuracy, val_loss_avg_qar), flush=True)
    
    if int(np.argmax(val_metric_per_epoch)) < (len(val_metric_per_epoch) - 1 - params['trainer']['patience']):
        print("Stopping at epoch {:2d}".format(epoch_num))
        break
    save_checkpoint(model_qa, 'qa',  optimizer, args.folder, epoch_num, val_metric_per_epoch,
                    is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))
    save_checkpoint(model_ra, 'ra', optimizer, args.folder, epoch_num, val_metric_per_epoch,
                    is_best=int(np.argmax(val_metric_per_epoch)) == (len(val_metric_per_epoch) - 1))

print("STOPPING. now running the best model on the validation set", flush=True)
# Load best
restore_best_checkpoint(model_qa, 'qa', args.folder)
restore_best_checkpoint(model_ra, 'ra', args.folder)
    
model_qa.eval()
model_ra.eval()
 
val_probs_qa = []
val_probs_ra = []
val_labels_qa = []
val_labels_ra = []


for b, (time_per_batch, batch) in enumerate(time_batch(val_loader)):
    with torch.no_grad():
        batch = _to_gpu(batch)
        batch_sz = batch['label'].shape[0]
        
        output_dict_qa = model_qa(batch)

        logits = output_dict_qa['label_probs']
        output_dict_ra = model_ra(batch)

        val_probs_qa.append(output_dict_qa['label_probs'].detach().cpu().numpy())
        
        _, best_pred_quest_ind = torch.max(logits, 1)
        pred_probs_all = output_dict_ra['all_ra_label_probs'].reshape(batch_sz, 4, 4)
        out_ra_probs = pred_probs_all[range(batch_sz), best_pred_quest_ind, :].detach().cpu().numpy()

        qa_label = batch['label']
        ra_label = batch['label_ra']
        
        val_labels_qa.append(qa_label.detach().cpu().numpy())
        val_labels_ra.append(ra_label.detach().cpu().numpy())
        val_probs_ra.append(out_ra_probs)

val_labels_qa = np.concatenate(val_labels_qa, 0)
val_labels_ra = np.concatenate(val_labels_ra, 0)
val_probs_qa = np.concatenate(val_probs_qa, 0)
val_probs_ra = np.concatenate(val_probs_ra, 0)
    
qa_accuracy, ra_accuracy, qar_accuracy = cal_net_accuracy(val_probs_qa, val_labels_qa,
                                                           val_probs_ra, val_labels_ra)

print("Final qa val accuracy is {:.3f}".format(qa_accuracy))
print("Final ra val accuracy is {:.3f}".format(ra_accuracy))
print("Final qar val accuracy is {:.3f}".format(qar_accuracy))
np.save(os.path.join(args.folder, f'valpreds_qa.npy'), val_probs_qa)
np.save(os.path.join(args.folder, f'valpreds_ra.npy'), val_probs_ra)
