"""
Evaluation script for the leaderboard
"""
import argparse
import logging
import multiprocessing

import numpy as np
import pandas as pd
import torch
from allennlp.common.params import Params
from allennlp.nn.util import device_mapping
from torch.nn import DataParallel
from torch.nn.modules import BatchNorm2d

from dataloaders.vcr import VCR, VCRLoader
from utils.pytorch_misc import time_batch

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
    default='multiatt/default.json',
    help='Params location',
    type=str,
)
parser.add_argument(
    '-answer_ckpt',
    dest='answer_ckpt',
    default='/vcr2/vcr/gumbel-softmax-2/saves/qa.best.th',
    help='Answer checkpoint',
    type=str,
)
parser.add_argument(
    '-rationale_ckpt',
    dest='rationale_ckpt',
    default='/vcr2/vcr/gumbel-softmax-2/saves/ra.best.th',
    help='Rationale checkpoint',
    type=str,
)
parser.add_argument(
    '-output',
    dest='output',
    default='submission.csv',
    help='Output CSV file to save the predictions to',
    type=str,
)

args = parser.parse_args()
params = Params.from_file(args.params)

NUM_GPUS = torch.cuda.device_count()
NUM_CPUS = multiprocessing.cpu_count()
if NUM_GPUS == 0:
    raise ValueError("you need gpus!")


def _to_gpu(td):
    if NUM_GPUS > 1:
        return td
    for k in td:
        if k != 'metadata':
            td[k] = {k2: v.cuda(async=True) for k2, v in td[k].items()} if isinstance(td[k], dict) else td[k].cuda(
            async=True)
    return td


num_workers = (4 * NUM_GPUS if NUM_CPUS == 32 else 2 * NUM_GPUS) - 1
print(f"Using {num_workers} workers out of {NUM_CPUS} possible", flush=True)
loader_params = {'batch_size': 96 // NUM_GPUS, 'num_gpus': NUM_GPUS, 'num_workers': num_workers}

test = VCR.eval_splits(embs_to_load=params['dataset_reader'].get('embs', 'bert_da'),
                        only_use_relevant_dets=params['dataset_reader'].get('only_use_relevant_dets', True))

test_loader = VCRLoader.from_dataset(test, **loader_params)

model_qa = Model.from_params(vocab=test.vocab, params=params['model'])
model_ra = Model.from_params(vocab=test.vocab, params=params['model_ra'])

def make_backbone_req_grad_false(model):
    for submodule in model.detector.backbone.modules():
        if isinstance(submodule, BatchNorm2d):
            submodule.track_running_stats = False
        
make_backbone_req_grad_false(model_qa)
make_backbone_req_grad_false(model_ra)

model_state_qa = torch.load(getattr(args, f'answer_ckpt'), map_location=device_mapping(-1))
model_qa.load_state_dict(model_state_qa)

model_state_ra = torch.load(getattr(args, f'rationale_ckpt'), map_location=device_mapping(-1))
model_ra.load_state_dict(model_state_ra)

model_qa = DataParallel(model_qa).cuda() if NUM_GPUS > 1 else model_qa.cuda()
model_ra = DataParallel(model_ra).cuda() if NUM_GPUS > 1 else model_ra.cuda()

model_qa.eval()
model_ra.eval()

val_probs_qa = []
val_probs_ra = []
test_ids = []

for b, (time_per_batch, batch) in enumerate(time_batch(test_loader)):
    with torch.no_grad():
        batch = _to_gpu(batch)
        
        output_dict_qa = model_qa(batch)

        logits = output_dict_qa['label_logits']
        output_dict_ra = model_ra(logits, 1, batch)

        val_probs_qa.append(output_dict_qa['label_probs'].detach().cpu().numpy())
        val_probs_ra.append(output_dict_ra['label_probs'].detach().cpu().numpy())
        #val_probs_ra.append(F.softmax(out_logits_ra, dim=-1).detach().cpu().numpy())
        
        test_ids += [m['annot_id'] for m in batch['metadata']]

        if (b > 0) and (b % 10 == 0):
            print("Completed {}/{} batches in {:.3f}s".format(b, len(test_loader), time_per_batch * 10), flush=True)

val_probs_qa = np.concatenate(val_probs_qa, 0)
val_probs_ra = np.concatenate(val_probs_ra, 0)

################################################################################
# This is the part you'll care about if you want to submit to the leaderboard!
################################################################################

# Double check the IDs are in the same order for everything

probs_grp = np.stack([val_probs_qa, val_probs_ra], 1)
# essentially probs_grp is a [num_ex, 2, 4] array of probabilities. The 2 'groups' are
# [answer, rationale_conditioned_on_a_predicted]
# We will flatten this to a CSV file so it's easy to submit.
group_names = ['answer'] + [f'rationale_conditioned_on_a_predicted']
probs_df = pd.DataFrame(data=probs_grp.reshape((-1, 8)),
                        columns=[f'{group_name}_{i}' for group_name in group_names for i in range(4)])
probs_df['annot_id'] = test_ids 
probs_df = probs_df.set_index('annot_id', drop=True)
probs_df.to_csv(args.output)
