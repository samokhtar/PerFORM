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

#print(device)

#Load all functions
from building_sdf.data_prep import XYZ_SDF_Perf_Dataset
from building_sdf.sampling import NumpyArrayEncoder
from building_sdf.learning import global_cond
from building_sdf.learning.network_loss import loss_fn_perf
from building_sdf.learning.network_training import fit_perf
from building_sdf.learning.network_utils import *

def main():

    # Parse the model directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model directory location")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--num_points", type=int, default=0, help="Number of points per batch")
    parser.add_argument("--include_val", type=int, default=0, help="Include a validation set during training")
    parser.add_argument("--include_ctxt", type=int, default=1, help="Include a context latent vector")
    parser.add_argument("--ckpt", type=int, help="Epoch number or None")
    parser.add_argument("--geo_model_dir", type=str, default='', help="Trained sdf model directory location")
    parser.add_argument("--geo_load_type", type=str, default='', help="Load type: final, current or epoch")
    parser.add_argument("--geo_model_type", type=str, default='', help="Model type: train, test or val")
    parser.add_argument("--geo_epoch", type=int, help="Epoch number to load")
    parser.add_argument("--geo_include_val", type=int, help="The geometry training data includes validation")
    parser.add_argument("--geo_include_rot", type=int, help="The geometry training data includes all rotations")
    parser.add_argument("--geo_override_latent", type=int, help="The number of latents in the geometry model")
    parser.add_argument("--rot_model_dir", type=str, help="Trained rotation encoder model directory location")
    parser.add_argument("--rot_load_type", type=str, help="Load type: final, current or epoch")
    parser.add_argument("--rot_epoch", type=int, help="Epoch number to load")
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
    ctxtmod_param, perfdec_param, rep_param, dataset_train_param, dataset_val_param = loadModelParam(model_dir, param_ar = ["ctxtmod_param", "perfdec_param", "rep_param", "dataset_train_param", "dataset_val_param"])
    ctxtmod_param = ctxtmod_param if(args.include_ctxt == 1) else None
    perfdec_param['ctxt_feature_dim'] = perfdec_param['ctxt_feature_dim'] if(args.include_ctxt == 1) else 0
    rep_param['seed'] = args.seed
    with open(parameters_dir + '/' + 'rep_param.json', 'w') as f:
        json.dump(rep_param, f, indent=2, cls=NumpyArrayEncoder)
    with open(parameters_dir + '/' + 'perfdec_param.json', 'w') as f:
        json.dump(perfdec_param, f, indent=2, cls=NumpyArrayEncoder)
    # Load geometry latents
    geo_model_dir = args.geo_model_dir if(args.geo_model_dir != '') else dataset_train_param['geo_model_dir']
    geo_model_type = args.geo_model_type if(args.geo_model_type != '') else dataset_train_param['geo_model_type']
    geo_load_type = args.geo_load_type if(args.geo_load_type != '') else dataset_train_param['geo_load_type']
    geo_epoch = args.geo_epoch if(args.geo_epoch is not None) else dataset_train_param['geo_epoch']
    geo_include_val = bool(args.geo_include_val) if(args.geo_include_val is not None) else dataset_train_param['geo_include_val']
    geo_include_rot = bool(args.geo_include_rot) if(args.geo_include_rot is not None) else dataset_train_param['geo_include_rot']
    geo_override_latent = args.geo_override_latent if(args.geo_override_latent is not None) else dataset_train_param['geo_override_latent']

    if(os.path.exists(geo_model_dir)):
        geo_latents, geo_dataset_ids_cur = global_cond.loadLatents(geo_model_dir, model_category='LatentNeuralField', model_type = geo_model_type, load_type = geo_load_type, epoch = geo_epoch, device = device.type, include_val=geo_include_val, include_rot=geo_include_rot, return_ids=True, override_latent=geo_override_latent)

    # Define the dataloaders
    dataset_params = [dataset_train_param, dataset_val_param] if(include_val) else [dataset_train_param]
    dataset_names = ['train', 'val'] if(include_val) else ['train']
    data_loaders, equiv_ids_mlp = [], []
    for i in range(len(dataset_params)):
        dataset_param = dataset_params[i]
        # Replace parameter   
        to_rep_n = ['batch_size','num_workers','num_points','geo_model_dir','geo_model_type',
                    'geo_load_type','geo_epoch','geo_include_val','geo_include_rot','geo_override_latent',
                    'rot_model_dir','rot_load_type','rot_epoch','geo_dataset_ids']
        to_rep_val = [args.batch_size,args.num_workers,args.num_points if(args.num_points != 0) else dataset_param['num_points'],
                      geo_model_dir,geo_model_type,geo_load_type,geo_epoch,geo_include_val,geo_include_rot,geo_override_latent,
                      args.rot_model_dir if(args.rot_model_dir is not None) else dataset_param['rot_model_dir'],
                     args.rot_load_type if(args.rot_load_type is not None) else dataset_param['rot_load_type'],
                     args.rot_epoch if(args.rot_epoch is not None) else dataset_param['rot_epoch'],geo_dataset_ids_cur]
        for j in range(len(to_rep_n)):
            dataset_param[to_rep_n[j]]=to_rep_val[j]
        with open(parameters_dir + '/' + 'dataset_'+dataset_names[i]+'_param' + '.json', 'w') as f:
            json.dump(dataset_param, f, indent=2, cls=NumpyArrayEncoder)
        point_sampling_params={'srf_pt_set':dataset_param['srf_pt_set'],
                       'grd_pt_set':dataset_param['grd_pt_set'],
                       'grd_pt_hgts':dataset_param['grd_pt_hgts'],
                       'exclude_grd':dataset_param['exclude_grd'],
                       'exclude_srf':dataset_param['exclude_srf'],
                       'grd_per_loop':dataset_param['grd_per_loop'],
                       'stype_per_loop':dataset_param['stype_per_loop'],
                       'sampling_types':dataset_param['sampling_types'],
                       'sampling_distr':dataset_param['sampling_distr']}        
        xyz_sdf_perf_dataset = XYZ_SDF_Perf_Dataset(directory=dataset_param['dataset_directory'],
                                            dataset_type=dataset_param['dataset_type'],
                                            subsets=dataset_param['dataset_subsets'],
                                            geo_types=dataset_param['dataset_geotypes'],
                                            point_sampling=dataset_param['point_sampling'],
                                            point_sampling_params=point_sampling_params,        
                                            distr_posneg=dataset_param['distr_posneg'],
                                            num_points=dataset_param['num_points'],
                                            s_range=dataset_param['s_range'],
                                            perf_metric = dataset_param['perf_metric'],
                                            split_type = dataset_param['model_type'],
                                            max_instances = dataset_param['max_num_instances'],
                                            geo_dataset_avail = geo_dataset_ids_cur,
                                            batch_size = dataset_param['batch_size'],
                                            only_zero_orientation = dataset_param['only_zero_orientation'],
                                            include_sdf_inpt = dataset_param['include_sdf_inpt'] if('include_sdf_inpt' in list(dataset_param.keys())) else False)

        persistent_workers = False if(dataset_param['num_workers'] == 0) else True
        data_loader_ml = torch.utils.data.DataLoader(xyz_sdf_perf_dataset, batch_size=dataset_param['batch_size'], drop_last=False, pin_memory=True, num_workers=dataset_param['num_workers'], persistent_workers=persistent_workers, shuffle = True)
        data_loaders.append(data_loader_ml)
        
    # Iterate over the dataloader
    start_time = time.time()
    batch = next(iter(data_loaders[0]))
    print('Time for next batch: ' + str(time.time()-start_time)+ "\n", flush=True)
    model_input, sdf, perf = batch
    print(f"Name: {model_input['item_n']}", flush=True)
    print(f"PERF shape: {perf.shape}" + "\n", flush=True)

    # Load the rotation model
    if(((dataset_train_param['rot_model_dir'] is not None) and (dataset_train_param['rot_model_dir'] != 'None')) and os.path.exists(dataset_train_param['rot_model_dir'])):
        rotEnc_model = global_cond.loadModel(dataset_train_param['rot_model_dir'], model_category='LatentRotationEncoder', model_type = 'train', load_type = dataset_train_param['rot_load_type'], epoch = dataset_train_param['rot_epoch'], device = device.type)
        rotEnc_model.eval()
    else:
        rotEnc_model = None

    
    # Define model
    perf_model = global_cond.LatentNeuralPerformanceField(ctxtmod_param, perfdec_param, geo_latents, list(geo_dataset_ids_cur), rotEnc_model).to(device)
    
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
