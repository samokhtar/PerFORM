# Import libraries
import numpy as np
import argparse
import torch
import json
from json import JSONEncoder
import time
print(f"Installed Torch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

print(device)

#Load all functions
from building_sdf.data_prep import XYZ_SDF_Dataset, XYZ_SDF_Dataset_Rotations
from building_sdf.learning import hybrid_cond
from building_sdf.learning.network_loss import loss_fn
from building_sdf.learning.network_training import fit_hybrid
from building_sdf.learning.network_utils import trainSummary, loadModelParam
from building_sdf.learning.network_modules import AutoDecoderWrapper

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

def main():

    # Parse the model directory
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="", help="Model directory location")
    parser.add_argument("--ckpt", type=int, help="Epoch number or None")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--num_points", type=int, default=0, help="Number of points per batch")
    parser.add_argument("--include_val", type=int, default=0, help="Include validation dataset in training")
    parser.add_argument("--include_rot", type=int, default=0, help="Include rotations in dataset")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed")
    args = parser.parse_args()

    # Get arguments
    model_dir = args.model_dir
    include_val = bool(args.include_val)
    include_rot = bool(args.include_rot)
    print('include_val: ' +str(include_val) + ', include_rot: ' + str(include_rot))
    
    # # Define the seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load model parameters
    parameters_dir = model_dir + '/' + 'parameters'
    grid_param, field_param, loss_param, rep_param, dataset_param = loadModelParam(model_dir, param_ar = ['grid_param', 'field_param', 'loss_param', 'rep_param', 'dataset_train_param'])
    print('grid_param',grid_param)
    
    # Replace batch_size parameter   
    num_points = dataset_param['num_points'] if(args.num_points == 0) else args.num_points
    dataset_param['batch_size'] = args.batch_size
    dataset_param['num_points'] = num_points if(args.num_points != 0) else dataset_param['num_points']
    grid_param['seed'] = args.seed

    # Define dataset
    dataset_class = XYZ_SDF_Dataset_Rotations if(include_rot) else XYZ_SDF_Dataset
    xyz_sdf_dataset = dataset_class(directory=dataset_param['dataset_directory'],
                                            dataset_type=dataset_param['dataset_type'],
                                            subsets=dataset_param['dataset_subsets'],
                                            geo_types=dataset_param['dataset_geotypes'],
                                            sampling_types=dataset_param['sampling_types'],
                                            sampling_distr=dataset_param['sampling_distr'],
                                            distr=dataset_param['distr'],
                                            num_points=num_points,
                                            s_range = dataset_param['s_range'],
                                            split_type = 'train' if(not include_val) else ['train','val'],
                                            max_instances = dataset_param['max_num_instances'],
                                            batch_size = args.batch_size,
                                            multiplier = dataset_param['multiplier'] if('multiplier' in list(dataset_param.keys())) else 1)
    
    print('The length of the training dataset is ' + str(xyz_sdf_dataset.__getlenCurDataset__())+".\n")
    
    # Replace changed parameters for the record
    dataset_param['max_num_instances'] = int(xyz_sdf_dataset.__getlenCurDataset__())
    dataset_param['total_instances'] = int(xyz_sdf_dataset.__getTotalInstances__())
    dataset_param['dataset_ids'] = xyz_sdf_dataset.__getUniqueNames__()

    val_add = 'val' if(include_val) else ''
    rot_add = 'rot' if(include_rot) else ''
    d_name = 'dataset_train'+val_add+rot_add+'_param'
    print('d_name',d_name)
    #print('dataset_param',dataset_param)
    with open(parameters_dir + '/' + d_name + '.json', 'w') as f:
        json.dump(dataset_param, f, indent=2, cls=NumpyArrayEncoder)
    with open(parameters_dir + '/' + 'grid_param.json', 'w') as f:
        json.dump(grid_param, f, indent=2, cls=NumpyArrayEncoder)
        
    # Define the dataloader
    start_time = time.time()
    persistent_workers = False if(args.num_workers == 0) else True
    data_loader_ml = torch.utils.data.DataLoader(xyz_sdf_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, num_workers=args.num_workers, persistent_workers=persistent_workers, shuffle=True, prefetch_factor=1)
    print('Time for dataloader instance: ' + str(time.time()-start_time) + "\n")

    # Define model
    print('Dataset size: ' + str(dataset_param['max_num_instances']) + ' and number of latents: ' + str(xyz_sdf_dataset.__getlenCurDataset__()))
    model_grid = hybrid_cond.LatenttoHybridGroundLatentGrid_ConvDecoder(grid_param=grid_param).to(device) if(grid_param['grid_type'] == 'ground') else hybrid_cond.LatenttoHybridVoxelLatentGrid_ConvDecoder(grid_param=grid_param).to(device)
    print(model_grid)
    grid_shape = (1, grid_param['feature_dim'], grid_param['grid_size'], grid_param['grid_size']) if(grid_param['grid_type'] == 'ground') else (1, grid_param['feature_dim'], grid_param['grid_size'], grid_param['grid_size'], grid_param['grid_size'])
    model_grid_ad = AutoDecoderWrapper(dataset_param['max_num_instances'], model_grid, param_name='grid.latent', in_wgt=grid_param['latent_init']).to(device)
    print(model_grid_ad)
    model_field = hybrid_cond.LatentField(grid_param=grid_param, field_param=field_param).to(device)
    print(model_field)

    print('Time for model instance: ' + str(time.time()-start_time)+ "\n")

    # Run the training
    fit_hybrid(
            model_grid = model_grid_ad,
            model_field = model_field,
            model_dir = args.model_dir,
            train_dataloader = data_loader_ml,
            loss_fn = loss_fn,
            summary_fn = trainSummary,
            plotting_function = None,
            fromCheckPt = args.ckpt,
            dataset_override = None if((not include_val) and (not include_rot)) else dataset_param
          )

if __name__ == "__main__":
    main()





