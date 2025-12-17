# Import libraries
import numpy as np
import argparse
import torch
import torch.nn as nn
import json
from json import JSONEncoder
import time
import os
import copy
print(f"Installed Torch version: {torch.__version__}")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

print(device)

#Load all functions
from building_sdf.data_prep import XYZ_SDF_Dataset, XYZ_SDF_Dataset_Rotations, XYZ_SDF_ArbitraryD_Dataset
from building_sdf.learning import hybrid_cond
from building_sdf.learning.network_loss import loss_fn_latent
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
    parser.add_argument("--load_type", type=str, default="final", help="Load type: final, current or epoch")
    parser.add_argument("--epoch", type=int, help="Epoch number for trained model")
    parser.add_argument("--ckpt", type=int, help="Epoch number or None")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--num_points", type=int, default=0, help="Number of points per batch")
    parser.add_argument("--include_val", type=int, default=0, help="Include validation dataset in training")
    parser.add_argument("--include_rot", type=int, default=0, help="Include rotations in dataset")
    parser.add_argument("--seed", type=int, default=0, help="Torch seed")
    parser.add_argument("--otherdataset", type=str, default="", help="Other dataset")
    parser.add_argument("--epochN", type=int, default=0, help="Number of epochs to override")
    parser.add_argument("--latentlr", type=int, default=0, help="Latent learning rate initial override")
    args = parser.parse_args()

    # Get arguments
    model_dir = args.model_dir.replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF')
    include_val = bool(args.include_val)
    include_rot = bool(args.include_rot)
    print('include_val: ' +str(include_val) + ', include_rot: ' + str(include_rot))
    
    # # Define the seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    #np.random.seed((args.seed+1)*(int(1000 * time.time()) % 2**32))

    # Load model parameters
    parameters_dir = model_dir + '/' + 'parameters'
    test_model_dir = model_dir + '/' + 'test' if(args.otherdataset == "") else model_dir + '/' + args.otherdataset
    print('test_model_dir',test_model_dir)

    t_params = 'dataset_param_' + args.otherdataset if(args.otherdataset != "") else 'dataset_test_param' 
    grid_param, field_param, loss_param, rep_param, dataset_param = loadModelParam(model_dir, param_ar = ['grid_param', 'field_param', 'loss_param', 'rep_param', t_params])
    print('grid_param',grid_param)
    
    # Replace batch_size parameter   
    num_points = dataset_param['num_points'] if(args.num_points == 0) else args.num_points
    dataset_param['batch_size'] = args.batch_size
    dataset_param['num_points'] = num_points if(args.num_points != 0) else dataset_param['num_points']
    rep_param['total_epochs_test'] = rep_param['total_epochs_test'] if(args.epochN == 0) else args.epochN
    loss_param['lr_sch_initial_latent'] = loss_param['lr_sch_initial_latent'] if(args.latentlr == 0) else 1/np.power(10,args.latentlr)
    # grid_param['seed'] = args.seed

    # Define dataset
    dataset_class = XYZ_SDF_ArbitraryD_Dataset if(args.otherdataset != "") else XYZ_SDF_Dataset_Rotations if(include_rot) else XYZ_SDF_Dataset
    print('nor_real',args.otherdataset != "")
    #print('dataset_param',dataset_param)
    if(args.otherdataset == ""):
        xyz_sdf_dataset = dataset_class(directory=dataset_param['dataset_directory'].replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF'),
                                                dataset_type=dataset_param['dataset_type'],
                                                subsets=dataset_param['dataset_subsets'],
                                                geo_types=dataset_param['dataset_geotypes'],
                                                sampling_types=dataset_param['sampling_types'],
                                                sampling_distr=dataset_param['sampling_distr'],
                                                distr=dataset_param['distr'],
                                                num_points=num_points,
                                                s_range = 'N1_1',
                                                split_type = 'test',
                                                max_instances = dataset_param['max_num_instances'],
                                                batch_size = args.batch_size,
                                                multiplier = dataset_param['multiplier'] if('multiplier' in list(dataset_param.keys())) else 1)
    else:
        xyz_sdf_dataset = dataset_class(directory=dataset_param['dataset_directory'].replace('/home/gridsan/smokhtar/building_urban_sdf_project/BuildingUrbanSDF','/home/gridsan/smokhtar/3d_geometry_learning_shared/physics_surrogate/BuildingUrbanSDF'),
                                                dataset_type=dataset_param['dataset_type'],
                                                subsets=dataset_param['dataset_subsets'],
                                                category=dataset_param['category'],
                                                sampling_types=dataset_param['sampling_types'],
                                                sampling_distr=dataset_param['sampling_distr'],
                                                distr=dataset_param['distr'],
                                                num_points=num_points,
                                                s_range = 'N1_1',
                                                max_instances = dataset_param['max_num_instances'],
                                                batch_size = args.batch_size,
                                                apply_rot = False)

    print('The length of the training dataset is ' + str(xyz_sdf_dataset.__getlenCurDataset__())+".\n")
    
    # Replace changed parameters for the record
    dataset_param['max_num_instances'] = int(xyz_sdf_dataset.__getlenCurDataset__())
    dataset_param['total_instances'] = int(xyz_sdf_dataset.__getTotalInstances__())
    dataset_param['dataset_ids'] = xyz_sdf_dataset.__getUniqueNames__()
    dataset_param['s_range'] = 'N1_1'
    print('max_num_instances',xyz_sdf_dataset.__getlenCurDataset__())
    print('total_instances',xyz_sdf_dataset.__getTotalInstances__())
    print('dataset_ids',xyz_sdf_dataset.__getUniqueNames__())

    #print('dataset_param',dataset_param)
    with open(parameters_dir + '/' + t_params + '.json', 'w') as f:
        json.dump(dataset_param, f, indent=2, cls=NumpyArrayEncoder)
    # with open(parameters_dir + '/' + 'grid_param.json', 'w') as f:
    #     json.dump(grid_param, f, indent=2, cls=NumpyArrayEncoder)
    with open(parameters_dir + '/' + 'rep_param'+'.json', 'w') as f:
        json.dump(rep_param, f, indent=2, cls=NumpyArrayEncoder)
    with open(parameters_dir + '/' + 'loss_param'+'.json', 'w') as f:
        json.dump(loss_param, f, indent=2, cls=NumpyArrayEncoder)
        
    # Define the dataloader
    start_time = time.time()
    persistent_workers = False if(args.num_workers == 0) else True
    data_loader_ml = torch.utils.data.DataLoader(xyz_sdf_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, num_workers=args.num_workers, persistent_workers=persistent_workers, shuffle=True)
    print('Time for dataloader instance: ' + str(time.time()-start_time) + "\n")
    
    
    # Load trained model
    if(os.path.exists(model_dir)):
        model_grid, model_field = hybrid_cond.loadModels(model_dir, model_type = 'train', load_type = args.load_type, epoch = args.epoch, device = device.type, include_val=include_val, include_rot=include_rot)
        model_grid.eval()
        model_field.eval()
    else:
        print('The model directory does not exist.')

    # Create a duplicate copy of the model
    try:
        model_grid_test = copy.deepcopy(model_grid)
    except:
        if(isinstance(model_grid, nn.DataParallel)):
            print(model_grid.module)
            print(model_grid)
            weights = {}
            for module in model_grid.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        weights[hook.name] = getattr(self, hook.name)
                        delattr(module, hook.name)
            model_grid_test = copy.deepcopy(model_grid)
            for module in model_test.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        hook(module, None)
            for name, value in weights.items():
                setattr(model_grid_test, name, value)
        else:
            weights = {}
            for module in model_grid.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        weights[hook.name] = getattr(self, hook.name)
                        delattr(module, hook.name)
            model_grid_test = copy.deepcopy(model_grid)
            for module in model_grid_test.modules():
                for _, hook in module._forward_pre_hooks.items():
                    if isinstance(hook, WeightNorm):
                        hook(module, None)
            for name, value in weights.items():
                setattr(model_grid_test, name, value)

    # Create a latent embedding with the number of test geometries
    max_instances = xyz_sdf_dataset.max_instances
    print('max_num_instances', max_instances)
    latents = nn.Embedding(num_embeddings = max_instances, embedding_dim = grid_param['feature_dim'])
    latents.weight.data.normal_(0, grid_param['latent_init'])
    try:
        m_s = model_grid.latents.weight.shape
        model_grid_test.latents = latents
        model_grid_test = model_grid_test.to(device)
        model_field = model_field.to(device)
    except:
        model_grid_test.module.latents = latents
        model_grid_test = model_grid_test.module.to(device)
        model_field = model_field.module.to(device)

    del model_grid
    print(model_grid_test)
    
    print('Time for model instance: ' + str(time.time()-start_time)+ "\n")

    # Run the training
    fit_hybrid(
            model_grid = model_grid_test,
            model_field = model_field,
            model_dir = test_model_dir,
            train_dataloader = data_loader_ml,
            loss_fn = loss_fn_latent,
            summary_fn = trainSummary,
            plotting_function = None,
            fromCheckPt = args.ckpt,
            optLatentOnly = True,
          )

if __name__ == "__main__":
    main()









