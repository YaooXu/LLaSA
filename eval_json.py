import argparse
import importlib
import os
import json
from utils.configure import Configure

def eval_loose_json(args, data=None, record_results=False):
    cfgargs = Configure.Get('')
    evaluator = importlib.import_module("metrics.meta_tuning.evaluator").EvaluateTool(cfgargs)

    if not data:
        with open(args.json_file, "rb") as f:
            data = json.load(f)
        
    if '<|end_of_text|>' in data[0]['prediction']:
        for item in data:
            item['prediction'] = item['prediction'].split('<|end_of_text|>')[0].strip('"')
    if '<|endoftext|>' in data[0]['prediction']:
        for item in data:
            item['prediction'] = item['prediction'].split('<|endoftext|>')[0].strip('"')

    preds = [item['prediction'].split('<s>')[-1].split('\n\n')[0].strip() for item in data]
    labs = data
    summary = evaluator.evaluate(preds, labs, "test")
    print(summary)
    
    if record_results:
        output_path = args.json_file.replace('.json', '_summary.json')
        with open(output_path, "w") as f:
            json.dump(summary, f, indent=4)
    
    return summary

def main(args):
    # use import lib to import EvaluateTool from metrics.{args.dataset_name}.evaluator
    output_path= f"{args.run_name}"
    predictions_path = os.path.join(output_path,"predictions_predict.json")
    config_path = f"{args.run_name}.cfg"
    args = Configure.Get(config_path)
    evaluator = importlib.import_module("metrics.meta_tuning.evaluator").EvaluateTool(args)


    with open(predictions_path, "rb") as f:
        data = json.load(f)
        
    if '<|end_of_text|>' in data[0]['prediction']:
        for item in data:
            item['prediction'] = item['prediction'].split('<|end_of_text|>')[0].strip('"')
    
    preds = [item['prediction'].split('<s>')[-1].split('\n\n')[0].strip() for item in data]
    labs = data
    summary = evaluator.evaluate(preds, labs, "test")
    print(summary)
    with open(os.path.join(output_path, "summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

if __name__=="__main__":
    # args: name of the data, name of the run, 
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="run1")
    parser.add_argument("--json_file", type=str, default=None)
    parser.add_argument("--dir", type=str, default=None)
    args = parser.parse_args()
    if args.json_file:
        eval_loose_json(args, record_results=True)
    elif args.dir:
        subfolders = [f for f in os.listdir(args.dir) if os.path.isdir(os.path.join(args.dir, f)) and f.startswith('test')]
        for f in reversed(subfolders):
            print(f)
            args.json_file = os.path.join(args.dir, f'{f}/predictions.json')
            eval_loose_json(args, record_results=True) 
    else:
        main(args)