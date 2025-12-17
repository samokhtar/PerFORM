# Import libraries
import numpy as np
import argparse
import torch
import json
import time
import os
#print(f"Installed Torch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# print(device)

#Load all functions
from building_sdf.data_prep import XYZ_SDF_Perf_HgtMaps_Dataset
from building_sdf.sampling import NumpyArrayEncoder
from building_sdf.learning import img_cond
from building_sdf.learning.network_loss import loss_fn_perf
from building_sdf.learning.network_training import fit_perf
from building_sdf.learning.network_utils import *
from building_sdf.learning.network_modules import AutoDecoderWrapper


def main():

    # Parse the model directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model directory location")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--include_val", type=int, default=0, help="Include a validation set during training")
    parser.add_argument("--ckpt", type=int, help="Epoch number or None")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed")
    args = parser.parse_args()

    # Get arguments
    model_dir = args.model_dir
    include_val = False if(args.include_val == 0) else True
    
    # # Define the seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model parameters
    parameters_dir = model_dir + '/' + 'parameters'
    perf_param, loss_param, rep_param, dataset_train_param, dataset_val_param = loadModelParam(model_dir, param_ar = ["perf_param", "loss_param", "rep_param", "dataset_train_param", "dataset_val_param"])
    rep_param['seed'] = args.seed
    with open(parameters_dir + '/' + 'rep_param.json', 'w') as f:
        json.dump(rep_param, f, indent=2, cls=NumpyArrayEncoder)

    # Define the dataloaders
    dataset_params = [dataset_train_param, dataset_val_param] if(include_val) else [dataset_train_param]
    dataset_names = ['train', 'val'] if(include_val) else ['train']
    data_loaders, equiv_ids_mlp = [], []
    for i in range(len(dataset_params)):
        dataset_param = dataset_params[i]
        # Replace parameter   
        to_rep_n = ['batch_size','num_workers']
        to_rep_val = [args.batch_size,args.num_workers]
        for j in range(len(to_rep_n)):
            dataset_param[to_rep_n[j]]=to_rep_val[j]
        with open(parameters_dir + '/' + 'dataset_'+dataset_names[i]+'_param' + '.json', 'w') as f:
            json.dump(dataset_param, f, indent=2, cls=NumpyArrayEncoder)
        point_sampling_params={'grd_pt_set':dataset_param['grd_pt_set'],
                       'grd_pt_hgts':dataset_param['grd_pt_hgts']}        
        xyz_sdf_perf_dataset = XYZ_SDF_Perf_HgtMaps_Dataset(directory=dataset_param['dataset_directory'],
                                            dataset_type=dataset_param['dataset_type'],
                                            subsets=dataset_param['dataset_subsets'],
                                            geo_types=dataset_param['dataset_geotypes'],
                                            point_sampling=dataset_param['point_sampling'],
                                            point_sampling_params=point_sampling_params,        
                                            perf_metric = dataset_param['perf_metric'],
                                            split_type = dataset_param['model_type'],
                                            max_instances = dataset_param['max_num_instances'],
                                            batch_size = dataset_param['batch_size'],
                                            only_zero_orientation = dataset_param['only_zero_orientation'],
                                            inc_localMap = dataset_param['inc_localMap'])
        #print('The length of the '+dataset_names[i]+' dataset is ' + str(xyz_sdf_perf_dataset.__getlenCurDataset__())+".\n", flush=True)
        persistent_workers = False if(dataset_param['num_workers'] == 0) else True
        data_loader_ml = torch.utils.data.DataLoader(xyz_sdf_perf_dataset, batch_size=dataset_param['batch_size'], drop_last=False, pin_memory=True, num_workers=dataset_param['num_workers'], persistent_workers=persistent_workers, shuffle = True)
        data_loaders.append(data_loader_ml)
        
#     # Test out different num_workers
#     # Current 8/18, 16/16
#     num_workers_sr =[1,2,4,8,16,19,1,2,4,8,16,19,1,2,4,8,16,19,1,2,4,8,16,19,1,2,4,8,16,19,1,2,4,8,16,19,1,2,4,8,16,19]
#     batch_size_ar = [8,8,8,8,8,8,16,16,16,16,16,16,32,32,32,32,32,32,64,64,64,64,64,64,128,128,128,128,128,128,256,256,256,256,256,256,512,512,512,512,512,512]
    
#     start_time = time.time()
# # for i in range(1):
# #     data_loader_ml_iter = iter(data_loader_ml)
# #     print(f"Time: {time.time()-start_time}\n")
# #     for batch in data_loader_ml_iter:
# #         model_input, sdf, perf = batch
    
#     for i in range(len(num_workers_sr)):
#         start_time = time.time()
#         data_loader_ml = torch.utils.data.DataLoader(xyz_sdf_perf_dataset, batch_size=batch_size_ar[i], drop_last=False, pin_memory=True, num_workers=num_workers_sr[i], persistent_workers=True, shuffle = True)
#         data_loader_ml_iter = iter(data_loader_ml)
#         for batch in data_loader_ml_iter:
#              model_input, sdf, perf = batch
#         print('batch size ' + str(batch_size_ar[i]) + ', num_workers ' + str(num_workers_sr[i]) + ': ' + str(time.time()-start_time)+ "\n", flush=True)
    
    
    
    
   
     # Define model
    perf_model = img_cond.hgtMapPerformance(perf_param).to(device)
    #print(perf_model, flush=True)
    
    # Run the training
    fit_perf(
            model = perf_model,
            model_dir = model_dir,
            train_dataloader = data_loaders[0],
            loss_fn = loss_fn_perf,
            summary_fn = trainSummary,
            plotting_function = None,
            fromCheckPt = args.ckpt,
            val_dataloader = data_loaders[1] if(include_val) else None,
          )

if __name__ == "__main__":
    main()
