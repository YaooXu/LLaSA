#!/usr/bin/env python
# -*- coding: utf-8 -*-
# encoding=utf8
import os
import copy
import numpy as np

import utils.tool
from utils.configure import Configure
from tqdm import tqdm


class EvaluateTool(object):
    """
    The meta evaluator
    """
    def __init__(self, meta_args):
        self.meta_args = meta_args

    def evaluate(self, preds, golds, section):
        meta_args = self.meta_args
        summary_held_in = {}
        summary_held_out = {}
        wait_for_eval = {}

        for pred, gold in zip(preds, golds):
            if gold['arg_path'] not in wait_for_eval.keys():
                wait_for_eval[gold['arg_path']] = {'preds': [], "golds":[]}
            wait_for_eval[gold['arg_path']]['preds'].append(pred)
            wait_for_eval[gold['arg_path']]['golds'].append(gold)

        lst = [(arg_path, preds_golds) for arg_path, preds_golds in wait_for_eval.items()]
        print([arg_path for arg_path, preds_golds in lst])
        held_out_tasks = ['finqa', 'wikitabletext', 'sqa'] 
        for arg_path, preds_golds in tqdm(lst):
            print("Evaluating {}...".format(arg_path))
            args = Configure.refresh_args_by_file_cfg(os.path.join(meta_args.dir.configure, arg_path), meta_args)
            evaluator = utils.tool.get_evaluator(args.evaluate.tool)(args)
            summary_tmp = evaluator.evaluate(preds_golds['preds'], preds_golds['golds'], section)
            print(summary_tmp)
            for key, metric in summary_tmp.items():  # TODO
                if any(held_out_task in arg_path for held_out_task in held_out_tasks):
                    summary_held_out[os.path.join(arg_path, key)] = metric
                else:
                    summary_held_in[os.path.join(arg_path, key)] = metric
            # summary[os.path.join(arg_path, args.train.stop)] = summary_tmp[args.train.stop]

        to_mean = ['acc', 'sacrebleu', 'all', 'all_ex', 'all_micro', 'exact_match', 'all_acc']
        summary_held_in['avr'] = float(np.mean([float(v) for k, v in summary_held_in.items() if os.path.basename(k) in to_mean]))
        summary_held_out['held_out_avr'] = float(np.mean([float(v) for k, v in summary_held_out.items() if os.path.basename(k) in to_mean]))

        summary = {**summary_held_in, **summary_held_out}
        return summary

