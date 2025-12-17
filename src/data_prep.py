import torch
import numpy as np
import pandas as pd
import os
import trimesh
import time
import json
from json import JSONEncoder
import cv2
from zipfile import ZipFile,BadZipFile

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

    
# Dataset preparation
class XYZ_SDF(torch.utils.data.Dataset):

    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                subsets=['0000_0499'],
                geo_types=['building'],
                sampling_types=['rejection','surface','uniform','zeroSurface'],
                sampling_distr=[1,1,0.1,1],
                distr = [0.4,0.4,0.2],
                num_points = 50000,
                s_range = '0_1', #or N1_1
                ) -> None:
    
        self.dataset_type = dataset_type
        self.subsets = subsets
        self.sampling_types = sampling_types
        self.sampling_distr = sampling_distr
        self.distr = np.array(distr)
        self.num_points = num_points
        self.s_range = s_range
        
        st_base = time.time()
        self._np_cache = {}

        # Define geometry and sampling directories
        self.sampling_dir = directory + '/Samples'
        self.geometry_dir = directory + '/Geometry'
        self.parameters_dir = directory + '/Parameters'
        
        # Define combined sampling name
        if(len(self.sampling_types)!= 1):
            self.sampling_name = '_'.join([self.sampling_types[i][0:3] + '_' + str(int(self.sampling_distr[i]*100)) for i in range(len(self.sampling_types))])
            sampling_exists = all([os.path.exists(self.sampling_dir + '/' + self.dataset_type + '/' + s + '/combined/' + self.sampling_name) for s in self.subsets])
        else:
            self.sampling_name = self.sampling_types[0]
            sampling_exists = all([os.path.exists(self.sampling_dir + '/' + self.dataset_type + '/' + s + '/' + self.sampling_name) for s in self.subsets])
            
        if(not sampling_exists):
            print('The sampling ' + self.sampling_name + ' does not exist for all subsets.')
            
        # Load dataset stats
        data_stats_df = pd.read_csv(directory + '/Parameters/' + dataset_type + '_datasetStats.csv')
        sub_df = data_stats_df[(data_stats_df['b_type'].isin(geo_types)) & (data_stats_df['subset'].isin(subsets))]
        #print(sub_df.columns)
        
        # Define dataset subset - accounting for existing geometries and corresponding sampling
        g_exists = np.array(sub_df['geometry'] == 1)
        if(len(self.sampling_types)!= 1):
            c_exists = np.array(sub_df[self.sampling_name] == 1) if(self.sampling_name in sub_df.columns) else np.repeat(0,len(sub_df))
            self.sub_df_exist = sub_df[(g_exists.astype(int) + c_exists.astype(int)) == 2][['b_type','idx']]
        else:
            s_exists = (np.sum(np.array([(sub_df[s]==1) for s in sampling_types]).T.astype(int), axis=1)==len(sampling_types))
            self.sub_df_exist = sub_df[(g_exists.astype(int) + s_exists.astype(int)) == 2][['b_type','idx']]
            
        #print(self.sub_df_exist)

        # Get the maximum size of the train/val/test splits datasets
        #print('Total dataset size: ' + str(len(self.sub_df_exist)))
        self.split_len = {}
        self.split_ids = {}
        train_test_splits = json.loads(open(self.parameters_dir + '/' + 'traintestParam_5000.json').read())
        for k in train_test_splits.keys():
            subKey = np.array(train_test_splits[k])
            inc_idx = self.sub_df_exist[self.sub_df_exist['idx'].isin(subKey)]
            self.split_len[k] = len(inc_idx)
            self.split_ids[k] = np.array(inc_idx['idx'].values)
            #print(k + ': ' + str(len(inc_idx)))
        #print("\n")
        
        #print('st_base',time.time()-st_base)
        self.num_pos = int(self.num_points * self.distr[0])
        self.num_neg = int(self.num_points * self.distr[1])
        self.num_zero = self.num_points - self.num_pos - self.num_neg
        self.crop_min = 0.1464
        self.crop_max = 0.8536
        
        
        
        super().__init__()
        
    # def _cached_npy_load(self, path, allow_pickle=False):
    #     """Loads a file from cache if available, otherwise loads from disk."""
    #     if path not in self._np_cache:
    #         self._np_cache[path] = np.load(path, allow_pickle=allow_pickle)
    #     return self._np_cache[path]
    
    def __len__(self):
        return len(self.sub_df_exist)
    
    def __splitlen__(self):
        return self.split_len
    
    def __splitids__(self):
        return self.split_ids

    def __getitem__(self, idx, geo_type):
        
        # Define random seed as time
        np.random.seed(int(1000 * time.time()) % 2**32)
        #np.random.seed(0)
        
        # Define the cur_directories
        idx_s = int(idx/500)
        set_fromIDX = "{:04d}".format(idx_s*500) + '_' + "{:04d}".format(((idx_s+1)*500)-1)
        g_dir = self.geometry_dir + '/' + self.dataset_type + '/' + set_fromIDX
        s_dir = self.sampling_dir + '/' + self.dataset_type + '/' + set_fromIDX
        
        # Check if geometry, all sampling types and combined sampling exist
        geoPar_exists = (len(self.sub_df_exist[(self.sub_df_exist['b_type']==geo_type) & (self.sub_df_exist['idx']==idx)])>0)

        st_load = time.time()
        if(geoPar_exists):
            g_name = geo_type + '_' + str(idx) if(self.dataset_type == '01_Buildings') else str(idx) + '_' + geo_type + '_combined'
            
            # Read the point combined sampling files from the folder
            if(len(self.sampling_types)!= 1):
                try:
                    d_sdf_pos_cat = np.load(s_dir + '/' + 'combined' + '/' + self.sampling_name + '/' + g_name + '_pos.npy') #, mmap_mode='r'
                    #d_sdf_pos_cat = self._cached_npy_load(s_dir + '/' + 'combined' + '/' + self.sampling_name + '/' + g_name + '_pos.npy')
                    #mask = np.all((d_sdf_pos_cat[:, :3] >= self.crop_min) & (d_sdf_pos_cat[:, :3] <= self.crop_max), axis=1)
                    # x_in_bounds = (d_sdf_pos_cat[:, 0] >= self.crop_min) & (d_sdf_pos_cat[:, 0] <= self.crop_max)
                    # z_in_bounds = (d_sdf_pos_cat[:, 2] >= self.crop_min) & (d_sdf_pos_cat[:, 2] <= self.crop_max)
                    # mask = x_in_bounds & z_in_bounds
                    # d_sdf_pos_cat = d_sdf_pos_cat[mask]
                    d_sdf_pos_cat = d_sdf_pos_cat[np.random.randint(d_sdf_pos_cat.shape[0], size=self.num_pos)]#.astype(float)
                    #print('d_sdf_pos_cat',d_sdf_pos_cat.shape)
                    d_sdf_neg_cat = np.load(s_dir + '/' + 'combined' + '/' + self.sampling_name + '/' + g_name + '_neg.npy') #, mmap_mode='r'
                    #d_sdf_neg_cat = self._cached_npy_load(s_dir + '/' + 'combined' + '/' + self.sampling_name + '/' + g_name + '_neg.npy')
                    d_sdf_neg_cat = d_sdf_neg_cat[np.random.randint(d_sdf_neg_cat.shape[0], size=self.num_neg)]#.astype(float)
                    #print('d_sdf_neg_cat',d_sdf_neg_cat.shape)
                    d_sdf_zero_cat = np.load(s_dir + '/' + 'combined' + '/' + self.sampling_name + '/' + g_name + '_zero.npy') #, mmap_mode='r'
                    #d_sdf_zero_cat = self._cached_npy_load(s_dir + '/' + 'combined' + '/' + self.sampling_name + '/' + g_name + '_zero.npy')
                    d_sdf_zero_cat = d_sdf_zero_cat[np.random.randint(d_sdf_zero_cat.shape[0], size=self.num_zero)]#.astype(float)
                    #print('d_sdf_zero_cat',d_sdf_zero_cat.shape)
                except:
                    d_sdf_pos_cat = None
                    d_sdf_neg_cat = None
                    d_sdf_zero_cat = None
            else:
                d_sdf_pos_cat = None
                d_sdf_neg_cat = None
                d_sdf_zero_cat = None
                #npz_data = self._cached_npy_load(s_dir + '/' + self.sampling_name + '/' + g_name + '_' + self.sampling_name + '.npz', allow_pickle=True)
                if(self.sampling_name != 'zeroSurface'):
                    d_sdf_pos_cat = np.load(s_dir + '/' + self.sampling_name + '/' + g_name + '_' + self.sampling_name + '.npz', allow_pickle=True)['pos']
                    #d_sdf_pos_cat = npz_data['pos']
                    #print('d_sdf_pos_cat',d_sdf_pos_cat.shape)
                    #d_sdf_neg_cat = npz_data['neg']
                    d_sdf_neg_cat = np.load(s_dir + '/' + self.sampling_name + '/' + g_name + '_' + self.sampling_name + '.npz', allow_pickle=True)['neg']
                    #print('d_sdf_neg_cat',d_sdf_neg_cat.shape)
                else:
                    #d_sdf_zero_cat = npz_data['alls']
                    d_sdf_zero_cat = np.load(s_dir + '/' + self.sampling_name + '/' + g_name + '_' + self.sampling_name + '.npz', allow_pickle=True)['alls']
                    #print('d_sdf_zero_cat',d_sdf_zero_cat.shape)
            #print('st_load',time.time()-st_load)
                    
            # Combine in array
            if((d_sdf_pos_cat is not None) and (d_sdf_neg_cat is not None)):
                subset_ar = [d_sdf_pos_cat,d_sdf_neg_cat,d_sdf_zero_cat]
                z_n = True if(d_sdf_zero_cat is not None) else False
                del d_sdf_pos_cat, d_sdf_neg_cat, d_sdf_zero_cat

                #torch.from_numpy(.float()

                st_zero = time.time()
                if(z_n):
                    # Create equivalent arrays with zeros for SDF at surface, and zeros for normals outside surface
                    zero_np = np.array([0])
                    sdf_subset_ar_zero = np.concatenate((subset_ar[2][:,0:3].T,zero_np.repeat(len(subset_ar[2])).reshape(len(subset_ar[2]),1).T)).T
                    nor_subset_ar_posneg = [np.concatenate((subset_ar[i][:,0:3].T,zero_np.repeat(len(subset_ar[i])*3).reshape(len(subset_ar[i]),3).T)).T for i in range(2)]

                    # Concatenate positive and negative instances
                    sdf_subset = np.concatenate([subset_ar[0], subset_ar[1], sdf_subset_ar_zero])
                    nor_subset = np.concatenate([nor_subset_ar_posneg[0], nor_subset_ar_posneg[1], subset_ar[2]])
                    del subset_ar, nor_subset_ar_posneg, sdf_subset_ar_zero

                    # Shuffle the order of positive/negative/zero points
                    rand_pts_idx = np.random.permutation(len(sdf_subset))
                    sdf_subset = torch.from_numpy(sdf_subset[rand_pts_idx]).float()
                    nor_subset = torch.from_numpy(nor_subset[rand_pts_idx]).float()
                    
                    # sdf_subset = torch.from_numpy(sdf_subset).float()
                    # nor_subset = torch.from_numpy(nor_subset).float()

                    # Separate coordinates and sdf values
                    xyz = sdf_subset[:,:3]
                    sdf = sdf_subset[:,3:4]
                    nor = nor_subset[:,3:6]
                else:
                    # Concatenate the positive and negative sets
                    sdf_subset = np.concatenate(subset_ar)
                    del subset_ar

                    # Shuffle the order of positive/negative points
                    rand_pts_idx = np.random.permutation(len(sdf_subset))
                    sdf_subset = torch.from_numpy(sdf_subset[rand_pts_idx]).float()

                    # Separate coordinates and sdf values
                    xyz = sdf_subset[:,:3]
                    sdf = sdf_subset[:,3:4]
                    nor = None
                #print('st_zero',time.time()-st_zero)
            else:
                xyz = None
                sdf = None
                nor = None
                
                
        else:
            print('The geometry ' + geo_type + '_' + str(idx) + ' does not exist in the dataset or does not have all sampling types.')
            xyz = None
            sdf = None
            nor = None

        # return model_input dictionary and output
        return xyz, sdf, nor

class XYZ_SDF_Dataset(torch.utils.data.Dataset):
    """
    A subclass that considers the train/val/test splits and maximum number of instances.

    """
    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                subsets=['0000_0499'],
                geo_types=['building'],
                sampling_types=['rejection','surface','uniform','zeroSurface'],
                sampling_distr=[1,1,0.1,1],
                distr = [0.4,0.4,0.2],
                num_points = 50000,
                s_range = '0_1', # or N1_1
                split_type='train', # or val or test
                max_instances = -1, #-1 to use full dataset
                batch_size = 32,
                apply_rot=False,
                geo_dataset_avail = -1,
                multiplier=1,
                ) -> None:

        # Generate dataset
        self._data = XYZ_SDF(directory=directory,
                          dataset_type=dataset_type,
                          subsets=subsets,
                          geo_types=geo_types,
                          sampling_types=sampling_types,
                          sampling_distr=sampling_distr,
                          distr=distr,
                          num_points=num_points,
                          s_range=s_range)
        self.batch_size = batch_size
        self.apply_rot = apply_rot
        self.s_range = s_range
        
        # Load the indices for the train/val/test sets ONLY
        if(isinstance(split_type, list)):
            ids_in = []
            for sp_t in split_type:
                ids_in.append(list(np.array(json.loads(open(self._data.parameters_dir + '/' + 'traintestParam_5000.json').read())[sp_t])))
            self.ids_inc = np.array([x for row in ids_in for x in row])
        else:
            self.ids_inc = np.array(json.loads(open(self._data.parameters_dir + '/' + 'traintestParam_5000.json').read())[split_type])
        self._dataset = self._data.sub_df_exist
        self._dataset['g_name'] = self._dataset['b_type'] + '_' + self._dataset['idx'].astype(str)
        self._dataset = self._dataset[self._dataset['idx'].isin(self.ids_inc)]
        self._dataset = self._dataset[self._dataset['g_name'].isin(geo_dataset_avail)] if(geo_dataset_avail != -1) else self._dataset
        self.set_names = self._dataset['g_name'].values
        self.max_instances =  np.min([max_instances, len(self._dataset)])  if (max_instances != -1) else len(self._dataset)
        self.multiplier = multiplier

    def __len__(self):
        return int(np.ceil(self.max_instances/self.batch_size))*self.batch_size
    
    def __getlenCurDataset__(self):
        return self.max_instances
    
    def __getTotalInstances__(self):
        return len(self._dataset)
    
    def __getUniqueNames__(self):
        return self.set_names[0:self.max_instances]

    def __getUnrotatedItem__(self, idx, returnNone = False, formatforModel=False):
        idx = idx%self.max_instances
        g_name = self.set_names[idx]
        real_idx = int(g_name.split('_')[1])
        geo_type = g_name.split('_')[0]
        outputs = self._data.__getitem__(real_idx, geo_type)
        
        if((None in outputs) and (not returnNone)):
            # If any of the outputs is None - find another random sample to replace it with - to avoid dataloader errors
            while(None in outputs):
                idx = np.random.randint(0, self.max_instances)
                g_name = self.set_names[idx]
                real_idx = int(g_name.split('_')[1])
                geo_type = g_name.split('_')[0]
                outputs = self._data.__getitem__(real_idx, geo_type)
        
        if(not(None in outputs)):
            xyz, sdf, nor = outputs
            xyz = (xyz*2)-1 if(self.s_range == 'N1_1') else xyz
            model_input = {'xyz': xyz*self.multiplier,
                            'idx': torch.tensor([idx]),
                            'rot': torch.tensor([0]),
                            'geo_code': torch.tensor([0,int(g_name.split('_')[1])]) if(g_name.split('_')[0]=='building') else torch.tensor([1,int(g_name.split('_')[1])])}
            model_input = {'xyz':model_input['xyz'].unsqueeze(0)*self.multiplier, 
                       'idx':model_input['idx'].unsqueeze(0), 
                       'rot':model_input['rot'].unsqueeze(0), 
                       'geo_code': model_input['geo_code'].unsqueeze(0)} if(formatforModel and (xyz is not None)) else model_input

            return model_input, sdf, nor, g_name 
        else:
            return None, None, None, None 
    
    def __getRotatedItem__(self, idx, override_rot=None, returnNone = False, formatforModel=False):
        idx = idx%self.max_instances
        g_name = self.set_names[idx]
        real_idx = int(g_name.split('_')[1])
        geo_type = g_name.split('_')[0]
        outputs = self._data.__getitem__(real_idx, geo_type)
        
        if((None in outputs) and (not returnNone)):
            # If any of the outputs is None - find another random sample to replace it with - to avoid dataloader errors
            while(None in outputs):
                idx = np.random.randint(0, self.max_instances)
                g_name = self.set_names[idx]
                real_idx = int(g_name.split('_')[1])
                geo_type = g_name.split('_')[0]
                outputs = self._data.__getitem__(real_idx, geo_type)
        
        if(not(None in outputs)):
            xyz, sdf, nor = outputs
            rot_val = np.random.randint(0,360) if(override_rot is None) else override_rot
            
            center = torch.tensor([0.5, 0.5, 0.5], device=xyz.device, dtype=xyz.dtype)
            xyz_centered = xyz - center                      
            R = torch_rotation_matrix_y(rot_val, device=xyz.device, dtype=xyz.dtype)
            xyz_rot = torch.matmul(xyz_centered, R.T) + center  
            
            # xyz_rot = np.concatenate([(xyz.detach().numpy() + [-0.5,-0.5,-0.5]),np.ones((xyz.detach().numpy().shape[0],1))],1)
            # xyz_rot = np.dot(rotation_matrix_y(rot_val),xyz_rot.T).T[:,:3].round(6) + [0.5,0.5,0.5]
            
            # print("Rotation:", rot_val)
            # print("Before:", xyz[0])
            # print("After:", xyz_rot[0])
            
            xyz_rot = (xyz_rot*2)-1 if(self.s_range == 'N1_1') else xyz_rot

            model_input = {'xyz': xyz_rot*self.multiplier,  #'xyz': torch.from_numpy(xyz_rot).float()*self.multiplier,'xyz': torch.from_numpy(xyz_rot).float()*self.multiplier, # 
                           'idx': torch.tensor([idx]), 
                           'rot': torch.tensor([rot_val]), 
                           'geo_code': torch.tensor([0,int(g_name.split('_')[1])]) if(g_name.split('_')[0]=='building') else torch.tensor([1,int(g_name.split('_')[1])])}
            model_input = {'xyz':model_input['xyz'].unsqueeze(0)*self.multiplier, 
                       'idx':model_input['idx'].unsqueeze(0), 
                       'rot':model_input['rot'].unsqueeze(0), 
                       'geo_code': model_input['geo_code'].unsqueeze(0)} if(formatforModel and (xyz is not None)) else model_input

            return model_input, sdf, nor, g_name    
        else:
            return None, None, None, None
    
    def __getitem__(self, idx, override_rot=None, formatforModel=False, returnNone=False):
        idx = idx%self.max_instances
        model_input, sdf, nor, g_name = self.__getUnrotatedItem__(idx, returnNone = returnNone) if(not self.apply_rot) else self.__getRotatedItem__(idx, override_rot, returnNone = returnNone)
        
        # yield model_input dictionary and output
        return model_input, sdf, nor, g_name   
    
#     def __getIncNoneItem__(self, idx, override_rot=None, formatforModel = False, returnNone=False):
#         idx = idx%self.max_instances
#         model_input, sdf, nor, g_name = self.__getUnrotatedItem__(idx, formatforModel = formatforModel, returnNone=returnNone) if(not self.apply_rot) else self.__getRotatedItem__(idx, override_rot, returnNone = returnNone, formatforModel = formatforModel)

#         # yield model_input dictionary and output
#         return model_input, sdf, nor, g_name    
    
    # def __iter__(self):
    #     # Iterate through full dataset
    #     for i in range(int(np.ceil(self.max_instances/self.batch_size))*self.batch_size):
    #         outputs = self.__getitem__(i)
    #         yield outputs

    
    
    
    
class XYZ_SDF_Dataset_Rotations(torch.utils.data.Dataset):
    """
    A subclass that considers the train/val/test splits and maximum number of instances.

    """
    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                subsets=['0000_0499'],
                geo_types=['building'],
                sampling_types=['rejection','surface','uniform','zeroSurface'],
                sampling_distr=[1,1,0.1,1],
                distr = [0.4,0.4,0.2],
                num_points = 50000,
                s_range = '0_1', # or N1_1
                split_type='train', # or val or test
                max_instances = -1, #-1 to use full dataset
                batch_size = 32,
                rotations=np.arange(8)*45,
                geo_dataset_avail = -1,
                multiplier=1,
                ) -> None:

        # Generate dataset
        self._data = XYZ_SDF(directory=directory,
                          dataset_type=dataset_type,
                          subsets=subsets,
                          geo_types=geo_types,
                          sampling_types=sampling_types,
                          sampling_distr=sampling_distr,
                          distr=distr,
                          num_points=num_points,
                          s_range=s_range)
        self.batch_size = batch_size
        self.s_range = s_range
        
        # Load the indices for the train/val/test sets ONLY
        if(isinstance(split_type, list)):
            ids_in = []
            for sp_t in split_type:
                ids_in.append(list(np.array(json.loads(open(self._data.parameters_dir + '/' + 'traintestParam_5000.json').read())[sp_t])))
            self.ids_inc = np.array([x for row in ids_in for x in row])
        else:
            self.ids_inc = np.array(json.loads(open(self._data.parameters_dir + '/' + 'traintestParam_5000.json').read())[split_type])
        self._dataset = self._data.sub_df_exist
        self._dataset['g_name'] = self._dataset['b_type'] + '_' + self._dataset['idx'].astype(str)
        self._dataset = self._dataset[self._dataset['idx'].isin(self.ids_inc)]
        self._dataset = self._dataset[self._dataset['g_name'].isin(geo_dataset_avail)] if(geo_dataset_avail != -1) else self._dataset
        self.set_names = self._dataset['g_name'].values
        rot_list = np.repeat(rotations,len(self.set_names)).reshape(len(rotations),len(self.set_names)).T.flatten().astype(str)
        names_list = np.repeat(self.set_names,len(rotations))
        self.aug_set_names = np.array([names_list[i] + '_' + rot_list[i] for i in range(len(names_list))])
        self.max_instances =  np.min([max_instances, len(self.aug_set_names)])  if (max_instances != -1) else len(self.aug_set_names)
        self.multiplier = multiplier

    def __len__(self):
        return int(np.ceil(self.max_instances/self.batch_size))*self.batch_size
    
    def __getlenCurDataset__(self):
        return self.max_instances
    
    def __getTotalInstances__(self):
        return len(self._dataset)
    
    def __getUniqueNames__(self):
        return self.aug_set_names[0:self.max_instances]
    
    def __getitem__(self, idx, override_rot=None, returnNone=False):
        idx = idx%self.max_instances
        full_name = self.aug_set_names[idx]
        rot = int(full_name.split('_')[2])
        g_name = '_'.join(full_name.split('_')[0:2])
        real_idx = int(g_name.split('_')[1])
        geo_type = g_name.split('_')[0]
        outputs = self._data.__getitem__(real_idx, geo_type)
        
        if((None in outputs) and (not returnNone)):
            # If any of the outputs is None - find another random sample to replace it with - to avoid dataloader errors
            while(None in outputs):
                idx = np.random.randint(0, self.max_instances)
                full_name = self.aug_set_names[idx]
                rot = int(full_name.split('_')[2])
                g_name = '_'.join(full_name.split('_')[0:2])
                real_idx = int(g_name.split('_')[1])
                geo_type = g_name.split('_')[0]
                outputs = self._data.__getitem__(real_idx, geo_type)
        
        if(not(None in outputs)):
            xyz, sdf, nor = outputs
            rot_val = rot if(override_rot is None) else override_rot
            
            center = torch.tensor([0.5, 0.5, 0.5], device=xyz.device, dtype=xyz.dtype)
            xyz_centered = xyz - center                      
            R = torch_rotation_matrix_y(rot_val, device=xyz.device, dtype=xyz.dtype)
            xyz_rot = torch.matmul(xyz_centered, R.T) + center  
            
            # xyz_rot = np.concatenate([(xyz.detach().numpy() + [-0.5,-0.5,-0.5]),np.ones((xyz.detach().numpy().shape[0],1))],1)
            # xyz_rot = np.dot(rotation_matrix_y(rot_val),xyz_rot.T).T[:,:3].round(6) + [0.5,0.5,0.5]
            
            # print("Rotation:", rot_val)
            # print("Before:", xyz[0])
            # print("After:", xyz_rot[0])
            
            xyz_rot = (xyz_rot*2)-1 if(self.s_range == 'N1_1') else xyz_rot
            
            model_input = {'xyz': xyz_rot*self.multiplier,#'xyz':torch.from_numpy(xyz_rot).float()*self.multiplier,#
                           'idx': torch.tensor([idx]), 
                           'rot': torch.tensor([rot_val]), 
                           'geo_code': torch.tensor([0,int(g_name.split('_')[1])]) if(g_name.split('_')[0]=='building') else torch.tensor([1,int(g_name.split('_')[1])])}
            
            if torch.isnan(xyz).any() or torch.isinf(xyz).any():
                print(f"[BAD xyz] NaN or Inf detected in {g_name}, rot_val {rot_val}")
            if xyz.min() < -1e3 or xyz.max() > 1e3:
                print(f"[BAD xyz] Extreme values detected in {g_name}, rot_val {rot_val}, min={xyz.min()}, max={xyz.max()}")

            return model_input, sdf, nor, full_name    
        else:
            return None, None, None, None
        
    

def combineSampTypes(geo_directory, samp_directory, samp_types, samp_prop = [1,1,1,1], b_type = 'building'):
    
    # Create a unique name for the combined sampling
    samp_name = '_'.join([samp_types[i][0:3] + '_' + str(int(samp_prop[i]*100)) for i in range(len(samp_types))])
    print('Sampling type name: ' + samp_name)
    
    # Create sampling folder for the combined proportions
    if(not os.path.exists(samp_directory + '/' + 'combined')):
        os.makedirs(samp_directory + '/' + 'combined')
    if(not os.path.exists(samp_directory + '/' + 'combined' + '/' + samp_name)):
        os.makedirs(samp_directory + '/' + 'combined' + '/' + samp_name)
    samp_directory_combined = samp_directory + '/' + 'combined' + '/' + samp_name
    
    # Get all mesh files in geometry directory
    all_mesh_files = os.listdir(geo_directory)

    for i in range(len(all_mesh_files)):   

        bldg_file_name = all_mesh_files[i]
        
        if((b_type in bldg_file_name) and not(all([os.path.exists(o) for o in [samp_directory_combined + '/' + bldg_file_name.split('.')[0] + '_' + p + '.npy' for p in ['pos','neg','zero']]]))):
            building_mesh = geo_directory + '/' + bldg_file_name
            samp_dirs = np.array([samp_directory + '/' + smTy + '/' + bldg_file_name.split('.')[0] + '_' + smTy + '.npz' for smTy in samp_types])
            #print('samp_dirs',samp_dirs)
            try:
                b_id = int(bldg_file_name.split('.')[0].split('_')[0])
            except:
                b_id = int(bldg_file_name.split('.')[0].split('_')[1])
            #print('i',str(b_id))
            if(all([os.path.exists(fSmp) for fSmp in samp_dirs]) and os.path.exists(building_mesh)):
                print('All files exist for geometry ' + str(b_id) + '.')   
                try:
                    # Get all samples
                    b_mesh = trimesh.load(building_mesh)
                    #print(b_mesh)
                    # Concatenate all sampling types  
                    max_prSp_pos = [np.load(sD)['pos'].shape[0] if (sD.split('/')[-2]!='zeroSurface') else 0 for sD in samp_dirs]
                    #print(max_prSp_pos)
                    max_prSp_neg = [np.load(sD)['neg'].shape[0] if (sD.split('/')[-2]!='zeroSurface') else 0 for sD in samp_dirs]
                    #print(max_prSp_neg)
                    max_prSp_zero = [np.load(sD)['alls'].shape[0] if (sD.split('/')[-2]=='zeroSurface') else 0 for sD in samp_dirs]
                    #print(max_prSp_zero)
                    d_sdf_pos_cat = np.concatenate([np.load(samp_dirs[m])['pos'][np.random.randint(low=0,high=max_prSp_pos[m],size=int(max_prSp_pos[m]*samp_prop[m]))] for m in range(len(samp_dirs)) if (samp_dirs[m].split('/')[-2]!='zeroSurface')])
                    #print('pos_shape',d_sdf_pos_cat.shape)
                    d_sdf_neg_cat = np.concatenate([np.load(samp_dirs[m])['neg'][np.random.randint(low=0,high=max_prSp_neg[m],size=int(max_prSp_neg[m]*samp_prop[m]))] for m in range(len(samp_dirs)) if (samp_dirs[m].split('/')[-2]!='zeroSurface')])
                    #print('neg_shape',d_sdf_neg_cat.shape)
                    d_sdf_zero_cat = np.concatenate([np.load(samp_dirs[m])['alls'][np.random.randint(low=0,high=max_prSp_zero[m],size=int(max_prSp_zero[m]*samp_prop[m]))] for m in range(len(samp_dirs)) if (samp_dirs[m].split('/')[-2]=='zeroSurface')]) if ('zeroSurface' in samp_types) else None
                    #print('zero_shape',d_sdf_zero_cat.shape)
                    # Save the file 
                    np.save(samp_directory_combined + '/' + bldg_file_name.split('.')[0] + '_pos', np.array(d_sdf_pos_cat))
                    np.save(samp_directory_combined + '/' + bldg_file_name.split('.')[0] + '_neg', np.array(d_sdf_neg_cat))
                    np.save(samp_directory_combined + '/' + bldg_file_name.split('.')[0] + '_zero', np.array(d_sdf_zero_cat))
                    print('The combined sampling files have been saved for geometry ' + str(b_id) + '.')
                except:
                    print('There was an error loading one of the sdf files for geometry ' + str(b_id) + '.')
            else:
                print('There was an error loading one of the sdf files or the geometry file for geometry ' + str(b_id) + '.')

                
# Dataset preparation
class XYZ_SDF_Perf(torch.utils.data.Dataset):

    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                subsets=['0000_0499'],
                geo_types=['building'],
                point_sampling='scattered', #'sliced' or 'both'
                point_sampling_params={'sampling_types':['rejection','surface','uniform'],'sampling_distr':[1,1,1],'srf_pt_set':'20','grd_pt_set':'512','grd_pt_hgts':'all','exclude_grd':False,'exclude_srf':False,'grd_per_loop':2},
                distr_posneg = [0.9,0.1],
                num_points = 50000,
                s_range = '0_1', #or N1_1
                bbox_size = 500,
                buffer = 10,
                return_fact=False,
                add_ext='') -> None:
        
        self.directory = directory
        self.dataset_type = dataset_type
        self.subsets = subsets
        self.pt_samp = point_sampling
        self.pt_samp_params = point_sampling_params
        self.distr_posneg = np.array(distr_posneg) if(distr_posneg is not None) else None
        self.num_points = num_points
        self.s_range = s_range
        self.return_fact = return_fact

        # Define geometry and sampling directories
        self.sampling_dir = self.directory + '/Samples'
        self.geometry_dir = self.directory + '/Geometry'
        self.parameters_dir = self.directory + '/Parameters'
        self.performance_dir = self.directory + '/Performance'
        
        # Load dataset stats
        data_stats_df = pd.read_csv(directory + '/Parameters/' + dataset_type + '_datasetStats.csv')
        self.sub_df = data_stats_df[(data_stats_df['b_type'].isin(geo_types)) & (data_stats_df['subset'].isin(subsets))]
        
        # Define bounding box
        self.buffered_bbox = np.array(json.loads(open(self.parameters_dir + '/' + 'bboxParam_' + str(bbox_size)  + '_' + str(buffer) +'.json').read())['bbox'])
        
        # Check if geometry exists
        g_exists = np.array(self.sub_df['geometry'] == 1)

        # Define dataset subset - accounting for existing geometries and corresponding sampling
        if(self.pt_samp == 'sliced'):
            s_exists = np.array(self.sub_df['performance'] == 1)
            self.sub_df_exist = self.sub_df[(s_exists.astype(int) + g_exists.astype(int)) == 2][['geometry','performance','idx','b_type']]
        elif(self.pt_samp == 'scattered'):
            s_exists = (np.sum([np.array(self.sub_df[s] == 1) for s in self.pt_samp_params['sampling_types']],0)==len(self.pt_samp_params['sampling_types'])).astype(int)
            cols_to_keep = [['geometry','idx','b_type'],self.pt_samp_params['sampling_types']]
            cols_to_keep = [x for xs in cols_to_keep for x in xs]
            self.sub_df_exist = self.sub_df[(s_exists.astype(int) + g_exists.astype(int)) == 2][cols_to_keep]
        else:
            s_exists_1 = np.array(self.sub_df['performance'] == 1)
            s_exists_2 = (np.sum([np.array(self.sub_df[s] == 1) for s in self.pt_samp_params['sampling_types']],0)==len(self.pt_samp_params['sampling_types'])).astype(int)
            cols_to_keep = [['geometry','performance','idx','b_type'],self.pt_samp_params['sampling_types']]
            cols_to_keep = [x for xs in cols_to_keep for x in xs]
            self.sub_df_exist = self.sub_df[(s_exists_1.astype(int) + s_exists_2.astype(int) + g_exists.astype(int)) == 3][cols_to_keep]

        # Get the maximum size of the train/val/test splits datasets
        #print('Total dataset size: ' + str(len(self.sub_df_exist)))
        self.split_len = {}
        self.split_ids = {}
        train_test_splits = json.loads(open(self.parameters_dir + '/' + 'traintestParam_5000.json').read())
        for k in train_test_splits.keys():
            subKey = np.array(train_test_splits[k])
            inc_idx = self.sub_df_exist[self.sub_df_exist['idx'].isin(subKey)]
            self.split_len[k] = len(inc_idx)
            self.split_ids[k] = np.array(inc_idx['idx'].values)
            #print(k + ': ' + str(len(inc_idx)))
        #print("\n")
        
        if((self.pt_samp == 'sliced') or (self.pt_samp == 'both')):
            # Get performance point types
            srf = ['srf_' + str(self.pt_samp_params['srf_pt_set'])] if(('srf_pt_set' in self.pt_samp_params.keys()) and (self.pt_samp_params['srf_pt_set'] is not None)) else [] 
            fld_vals = [15,30,45,60,75,90,105,120,135,150,300,600,1200,2400,4800] if(self.pt_samp_params['grd_pt_hgts'] == 'all') else self.pt_samp_params['grd_pt_hgts']
            gd = ['XYgrid_' + str(self.pt_samp_params['grd_pt_set']) + '_30_'+ str(x) for x in fld_vals] if(self.pt_samp_params['grd_pt_set'] is not None) else []
            all_point_sets = [x  for sublist in [srf, gd] for x in sublist]
            self.point_sets = all_point_sets if(('exclude_grd' not in self.pt_samp_params.keys()) or ((not self.pt_samp_params['exclude_grd']) and (not self.pt_samp_params['exclude_srf']))) else srf if(self.pt_samp_params['exclude_grd']) else gd
            #print('Active point sets: ' + str(self.point_sets))
            or_res = 512*512
            ar_n = np.arange(or_res)
            sqrt_n = int(np.sqrt(or_res))
            idx_all = np.array(np.split(ar_n, sqrt_n))
            st_ix = int(sqrt_n/4)
            ed_ix = int(sqrt_n/4)+int(sqrt_n/2)
            self.idx_subset = idx_all[st_ix:ed_ix].T[st_ix:ed_ix].T.flatten()

        # Get performance results status updates
        stats_ext = '' if(self.pt_samp == 'sliced') else '_Sampling' if(self.pt_samp == 'scattered') else '_Combined'
        cfd_stats = pd.read_csv(self.parameters_dir + '/' + self.dataset_type + '_CFDdatasetStats'+stats_ext+add_ext+'.csv')
        self.cfd_df_sub =  cfd_stats[(cfd_stats['b_type'].isin(geo_types)) & (cfd_stats['subset'].isin(subsets))]
        svf_stats = pd.read_csv(self.parameters_dir + '/' + self.dataset_type + '_SVFdatasetStats'+stats_ext+'.csv')
        self.svf_df_sub = svf_stats[(svf_stats['b_type'].isin(geo_types)) & (svf_stats['subset'].isin(subsets))]

        super().__init__()
    
    def __len__(self):
        return len(self.sub_df_exist)
    
    def __splitlen__(self):
        return self.split_len
    
    def __splitids__(self):
        return self.split_ids

    def __getitem__(self, idx, geo_type, perf_metric, orien_val = None, return_fact = False):
        
        # Define random seed as time
        np.random.seed(int(1000 * time.time()) % 2**32)
        
        # Define the cur_directories
        idx_s = int(idx/500)
        self.set_fromIDX = "{:04d}".format(idx_s*500) + '_' + "{:04d}".format(((idx_s+1)*500)-1)
        self.g_dir = self.geometry_dir + '/' + self.dataset_type + '/' + self.set_fromIDX
        self.s_dir = self.sampling_dir + '/' + self.dataset_type + '/' + self.set_fromIDX
        self.orien_val = orien_val
        self.perf_metric = perf_metric
        
        # Check if geometry, all sampling types and combined sampling exist
        s_cols = ['geometry', 'performance'] if(self.pt_samp=='sliced') else [x for xs in [['geometry'], self.pt_samp_params['sampling_types']] for x in xs] if(self.pt_samp == 'scattered') else [x for xs in [['geometry','performance'], self.pt_samp_params['sampling_types']] for x in xs]
        geoPar_exists = (np.sum(self.sub_df_exist[(self.sub_df_exist['b_type']==geo_type) & (self.sub_df_exist['idx']==idx)][s_cols].values)==len(s_cols))
        #print(self.sub_df_exist[(self.sub_df_exist['b_type']==geo_type) & (self.sub_df_exist['idx']==idx)])

        # Check if performance files exist for the selected performance metric
        stats_file = self.cfd_df_sub if(perf_metric in ['U','P']) else self.svf_df_sub
        if(self.pt_samp=='sliced'):
            perf_cols = [perf_metric + '_' + nT + '_R' + str(orien_val) for nT in ['srf','fld']] if(perf_metric == 'U') else [perf_metric + '_' + 'srf' + '_R' + str(orien_val)] if(perf_metric == 'P') else ['srf_' + str(self.pt_samp_params['srf_pt_set']), 'XYgrid_' + str(self.pt_samp_params['grd_pt_set']) + '_30']
        elif(self.pt_samp == 'scattered'):
            perf_cols = [perf_metric + '_' + nT[0:3] + '_' + str(orien_val) for nT in self.pt_samp_params['sampling_types']] if(perf_metric in ['U','P']) else [perf_metric + '_' + nT[0:3] for nT in self.pt_samp_params['sampling_types']]
        else:
            perf_cols_1 = [perf_metric + '_' + nT + '_R' + str(orien_val) for nT in ['srf','fld']] if(perf_metric == 'U') else [perf_metric + '_' + 'srf' + '_R' + str(orien_val)] if(perf_metric == 'P') else ['srf_' + str(self.pt_samp_params['srf_pt_set']), 'XYgrid_' + str(self.pt_samp_params['grd_pt_set']) + '_30']
            perf_cols_2 = [perf_metric + '_' + nT[0:3] + '_' + str(orien_val) for nT in self.pt_samp_params['sampling_types']] if(perf_metric in ['U','P']) else [perf_metric + '_' + nT[0:3] for nT in self.pt_samp_params['sampling_types']]
            perf_cols = [x for xs in [perf_cols_1,perf_cols_2] for x in xs]
        p_exists = np.sum(stats_file[(stats_file['b_type']==geo_type) & (stats_file['idx']==idx)][perf_cols].sum().values)>0       
        
        # If the geometry of the building shape exists
        if(geoPar_exists and p_exists):
            self.g_name = geo_type + '_' + str(idx) if(self.dataset_type == '01_Buildings') else str(idx) + '_' + geo_type + '_combined'
            #print(self.g_name + '_' + str(self.orien_val) + ' ')
            self.g_name_perf = geo_type + '_' + str(idx) if(self.dataset_type == '01_Buildings') else geo_type + '_' + str(idx) + '_urb'
            
            # Get the performance array for all point sets
            d_pts_all, d_ptsrot_all, d_sdf_all, d_perf_all = [], [], [], []
            
            if(self.pt_samp=='sliced' or self.pt_samp=='both'):
                # Sub select a point set randomly
                gd_sub = np.array([p for p in self.point_sets if('XYgrid' in p)])
                np.random.shuffle(gd_sub)
                gd_sub = gd_sub[0:self.pt_samp_params['grd_per_loop']]
                #gd_sub = list(np.array(gd_sub)[np.random.randint(1, len(gd_sub), self.pt_samp_params['grd_per_loop'])])
                srf_sub = [p for p in self.point_sets if('srf' in p)]
                sub_point_sets = [x  for sublist in [srf_sub, gd_sub] for x in sublist]
                # Looping through point sets
                for s in sub_point_sets:
                    d_pts, d_ptsrot, d_sdf, d_perf = getPerfArrayfromPSet(self, s, return_fact=return_fact)
                    if(d_pts is not None):
                        d_pts_all.append(d_pts)
                        d_ptsrot_all.append(d_ptsrot)
                        d_sdf_all.append(d_sdf)
                        d_perf_all.append(d_perf)       
            if(self.pt_samp=='scattered' or self.pt_samp=='both'):
                # Subselecting a point sub-set
                samp_sub = self.pt_samp_params['sampling_types'].copy()
                np.random.shuffle(samp_sub)
                sampling_types_subset = samp_sub[0:self.pt_samp_params['stype_per_loop']] if('stype_per_loop' in list(self.pt_samp_params.keys())) else samp_sub   
                # Looping through point sets
                for s in sampling_types_subset:
                    d_pts, d_ptsrot, d_sdf, d_perf = getPerfArrayfromSampSet(self, s, return_fact=return_fact)
                    if(d_pts is not None):
                        d_pts_all.append(d_pts)
                        d_ptsrot_all.append(d_ptsrot)
                        d_sdf_all.append(d_sdf)
                        d_perf_all.append(d_perf)   
            
            # Concatenate all:
            if(len(d_pts_all)>0):
                # Combine each performance in one np array
                d_pts_all, d_ptsrot_all, d_sdf_all, d_perf_all = [np.concatenate(d) for d in [d_pts_all, d_ptsrot_all, d_sdf_all, d_perf_all]]
                d_all_combined = np.concatenate([d_ptsrot_all, d_pts_all, d_sdf_all, d_perf_all],1)
                d_types = ['XYZ','XYZ_orig','SDF','Perf']
                d_type_shapes = np.array([d.shape[1] for d in [d_ptsrot_all, d_pts_all, d_sdf_all, d_perf_all]])
    
                # Remove the points outside the bounding box
                min_XYZ = -1 if(self.s_range == 'N_1') else 0
                max_XYZ = 1
                fil_bboxInc = ((d_ptsrot_all[:,0]>=min_XYZ)&(d_ptsrot_all[:,1]>=min_XYZ)&(d_ptsrot_all[:,2]>=min_XYZ)&(d_ptsrot_all[:,0]<=max_XYZ)&(d_ptsrot_all[:,1]<=max_XYZ)&(d_ptsrot_all[:,2]<=max_XYZ))
                d_all_combined = d_all_combined[fil_bboxInc]
                
                # Exclude points with positive sdf that have zero or near-zero perf value
                fil_err = (d_all_combined[:,6]>0)&(np.absolute(d_all_combined[:,7])<0.0000001)
                d_all_combined = d_all_combined[~fil_err]
    
                # Get positive/negative SDF filter
                if(self.distr_posneg is not None):
                    fil_posSDF = (d_all_combined[:,6]>0)
                    #print('Size: ' + str(d_all_combined.shape[0]) + ', filtered: ' + str(np.sum(fil_posSDF)))
                    set_ar = [d_all_combined[fil_posSDF],d_all_combined[fil_posSDF==False]]
                    set_size = [int(self.num_points*self.distr_posneg[i]) for i in range(len(set_ar))]
                    set_size[len(set_ar)-1] = self.num_points - (np.sum(set_size[0:-1]))
                    if((set_ar[0].shape[0]>0)and(set_ar[1].shape[0]>0)):
                        f_subset = np.concatenate([set_ar[i][np.random.randint(low=0,high=set_ar[i].shape[0]-1, size=set_size[i])].astype(float) for i in range(len(set_ar))],0)
                    else:
                        f_subset = d_all_combined[np.random.randint(d_all_combined.shape[0], size=int(self.num_points))]
                else:
                    f_subset = d_all_combined[np.random.randint(d_all_combined.shape[0], size=int(self.num_points))]
    
                # Shuffle the order of points
                rand_pts_idx = np.random.permutation(f_subset.shape[0])
                f_subset = torch.from_numpy(f_subset[rand_pts_idx]).float()
    
                # Separate coordinates, sdf and performance values
                xyz = f_subset[:,0:3] if(self.s_range == '0_1') else ((f_subset[:,0:3]*2)-1)
                xyz_org = f_subset[:,3:6] if(self.s_range == '0_1') else ((f_subset[:,3:6]*2)-1)
                sdf = f_subset[:,6:7]
                perf = f_subset[:,7:]
            else:
                #print('The geometry ' + geo_type + '_' + str(idx) + ' or its sampling/performance do not exist in the dataset: total number of points is zero!')
                xyz, xyz_org, sdf, perf = None, None, None, None

        else:
            #print('The geometry ' + geo_type + '_' + str(idx) + ' or its sampling/performance do not exist in the dataset: ' + 'geoPar_exists: ' + str(geoPar_exists) + ', p_exists: ' + str(p_exists))
            xyz, xyz_org, sdf, perf = None, None, None, None

        # return xyz, xyz_org, sdf, perf
        return xyz, xyz_org, sdf, perf
    
    def __getSelPtSet__(self, idx, geo_type, perf_metric, s_set, orien_val, fil_bBox = True, return_fact = False, apply_sdf_filter=True):
        
        # Define random seed as time
        np.random.seed(int(1000 * time.time()) % 2**32)
        
        # Define the cur_directories
        idx_s = int(idx/500)
        self.set_fromIDX = "{:04d}".format(idx_s*500) + '_' + "{:04d}".format(((idx_s+1)*500)-1)
        self.g_dir = self.geometry_dir + '/' + self.dataset_type + '/' + self.set_fromIDX
        self.s_dir = self.sampling_dir + '/' + self.dataset_type + '/' + self.set_fromIDX
        self.orien_val = orien_val
        self.perf_metric = perf_metric
        
        # Check if geometry, all sampling types and combined sampling exist
        s_cols = ['geometry', 'performance']
        geoPar_exists = (np.sum(self.sub_df_exist[(self.sub_df_exist['b_type']==geo_type) & (self.sub_df_exist['idx']==idx)][s_cols].values)==len(s_cols))
        #print(self.sub_df_exist[(self.sub_df_exist['b_type']==geo_type) & (self.sub_df_exist['idx']==idx)])
        
        # Check if performance files exist for the selected performance metric
        if(('_XZgrid_' in s_set) or ('_YZgrid_' in s_set) or ('_XZgrid45_' in s_set) or ('_YZgrid45_' in s_set)):
            p_exists = True
        else:
            stats_file = self.cfd_df_sub if(perf_metric in ['U','P']) else self.svf_df_sub
            perf_cols = [perf_metric + '_' + nT + '_R' + str(orien_val) for nT in ['srf','fld']] if(perf_metric == 'U') else [perf_metric + '_' + 'srf' + '_R' + str(orien_val)] if(perf_metric == 'P') else ['srf_' + str(self.pt_samp_params['srf_pt_set']), 'XYgrid_' + str(self.pt_samp_params['grd_pt_set']) + '_30']
            p_exists = np.sum(stats_file[(stats_file['b_type']==geo_type) & (stats_file['idx']==idx)][perf_cols].sum().values)>0     
        
        # If the geometry of the building shape exists
        if(geoPar_exists and p_exists):
            self.g_name = geo_type + '_' + str(idx) if(self.dataset_type == '01_Buildings') else str(idx) + '_' + geo_type + '_combined'
            #print(self.g_name + '_' + str(self.orien_val) + ' ')
            self.g_name_perf = geo_type + '_' + str(idx) if(self.dataset_type == '01_Buildings') else geo_type + '_' + str(idx) + '_urb'
            
            # Get the performance array for selected point set
            d_pts, d_ptsrot, d_sdf, d_perf = getPerfArrayfromPSet(self, s_set, return_fact=return_fact)

            # Concatenate all:
            if(d_pts is not None):
                d_all = np.concatenate([d_ptsrot, d_pts, d_sdf, d_perf],1)
                #print('d_all',d_all.shape)
                # print('d_all_combined', d_all_combined.shape)
                d_types = ['XYZ','XYZ_orig','SDF','Perf']
                d_type_shapes = np.array([d.shape[1] for d in [d_ptsrot, d_pts, d_sdf, d_perf]])
    
                # Remove the points outside the bounding box
                if(fil_bBox):
                    fil_bboxInc = ((d_ptsrot[:,0]>=0)&(d_ptsrot[:,1]>=0)&(d_ptsrot[:,2]>=0)&(d_ptsrot[:,0]<=1)&(d_ptsrot[:,1]<=1)&(d_ptsrot[:,2]<=1))
                    d_all = d_all[fil_bboxInc]
                
                # Exclude points with positive sdf that have zero or near-zero perf value
                if(apply_sdf_filter):
                    fil_err = (d_all[:,6]>0)&(np.absolute(d_all[:,7])<0.0000001)
                    d_all = d_all[~fil_err]

                # Separate coordinates, sdf and performance values
                d_all = torch.from_numpy(d_all).float()
                xyz = d_all[:,0:3] if(self.s_range == '0_1') else ((d_all[:,0:3]*2)-1)
                xyz_org = d_all[:,3:6] if(self.s_range == '0_1') else ((d_all[:,3:6]*2)-1)
                sdf = d_all[:,6:7]
                perf = d_all[:,7:]
            else:
                #print('The geometry ' + geo_type + '_' + str(idx) + ' or its sampling/performance do not exist in the dataset: total number of points is zero!')
                xyz, xyz_org, sdf, perf = None, None, None, None

        else:
            #print('The geometry ' + geo_type + '_' + str(idx) + ' or its sampling/performance do not exist in the dataset: ' + 'geoPar_exists: ' + str(geoPar_exists) + ', p_exists: ' + str(p_exists))
            xyz, xyz_org, sdf, perf = None, None, None, None

        # return xyz, xyz_org, sdf, perf
        return xyz, xyz_org, sdf, perf

    def __getSelPtSetHgtMap__(self, idx, geo_type, perf_metric, s_set, orien_val, return_fact = False, return_filter=False, matchhgtmaps=True):
        
        # Define random seed as time
        np.random.seed(int(1000 * time.time()) % 2**32)
        
        # Define the cur_directories
        idx_s = int(idx/500)
        self.set_fromIDX = "{:04d}".format(idx_s*500) + '_' + "{:04d}".format(((idx_s+1)*500)-1)
        self.g_dir = self.geometry_dir + '/' + self.dataset_type + '/' + self.set_fromIDX
        self.s_dir = self.sampling_dir + '/' + self.dataset_type + '/' + self.set_fromIDX
        self.orien_val = orien_val
        self.perf_metric = perf_metric
        
        # Check if geometry, all sampling types and combined sampling exist
        s_cols = ['geometry', 'performance']
        geoPar_exists = (np.sum(self.sub_df_exist[(self.sub_df_exist['b_type']==geo_type) & (self.sub_df_exist['idx']==idx)][s_cols].values)==len(s_cols))
        #print(self.sub_df_exist[(self.sub_df_exist['b_type']==geo_type) & (self.sub_df_exist['idx']==idx)])
        
        # Check if performance files exist for the selected performance metric
        stats_file = self.cfd_df_sub if(perf_metric in ['U','P']) else self.svf_df_sub
        perf_cols = [perf_metric + '_' + nT + '_R' + str(orien_val) for nT in ['fld']] if(perf_metric == 'U') else ['XYgrid_' + str(self.pt_samp_params['grd_pt_set']) + '_30']
        p_exists = np.sum(stats_file[(stats_file['b_type']==geo_type) & (stats_file['idx']==idx)][perf_cols].sum().values)>0       
        
        # If the geometry of the building shape exists
        if(geoPar_exists and p_exists):
            self.g_name = geo_type + '_' + str(idx) if(self.dataset_type == '01_Buildings') else str(idx) + '_' + geo_type + '_combined'
            #print(self.g_name + '_' + str(self.orien_val) + ' ')
            self.g_name_perf = geo_type + '_' + str(idx) if(self.dataset_type == '01_Buildings') else geo_type + '_' + str(idx) + '_urb'
            
            # Get the performance array for selected point set
            d_pts, d_ptsrot, d_sdf, d_perf, hgtmap = getPerfArrayfromPSet(self, s_set, return_fact=return_fact, return_hgtMap=True, matchhgtmaps=matchhgtmaps)

            # Concatenate all:
            if(d_pts is not None):
                d_all = np.concatenate([d_ptsrot, d_pts, d_sdf, d_perf, hgtmap],1)
                #print('d_all',d_all.shape)
                # print('d_all_combined', d_all_combined.shape)
                d_types = ['XYZ','XYZ_orig','SDF','Perf','HgtM']
                d_type_shapes = np.array([d.shape[1] for d in [d_ptsrot, d_pts, d_sdf, d_perf, hgtmap]])
                
                # Get filter for bounding box
                if(return_filter):
                    fil_bboxInc = ((d_ptsrot[:,0]>=0)&(d_ptsrot[:,1]>=0)&(d_ptsrot[:,2]>=0)&(d_ptsrot[:,0]<=1)&(d_ptsrot[:,1]<=1)&(d_ptsrot[:,2]<=1))

                # Separate coordinates, sdf and performance values
                d_all = torch.from_numpy(d_all).float()
                xyz = d_all[:,0:3] if(self.s_range == '0_1') else ((d_all[:,0:3]*2)-1)
                xyz_org = d_all[:,3:6] if(self.s_range == '0_1') else ((d_all[:,3:6]*2)-1)
                sdf = d_all[:,6:7]
                perf = d_all[:,7:-2]
                hgtmap = d_all[:,-2:]
            else:
                #print('The geometry ' + geo_type + '_' + str(idx) + ' or its sampling/performance do not exist in the dataset: total number of points is zero!')
                if(return_filter):
                    xyz, xyz_org, sdf, perf, hgtmap, fil_bboxInc = None, None, None, None, None, None
                else:
                    xyz, xyz_org, sdf, perf, hgtmap = None, None, None, None, None

        else:
            #print('The geometry ' + geo_type + '_' + str(idx) + ' or its sampling/performance do not exist in the dataset: ' + 'geoPar_exists: ' + str(geoPar_exists) + ', p_exists: ' + str(p_exists))
            if(return_filter):
                xyz, xyz_org, sdf, perf, hgtmap, fil_bboxInc = None, None, None, None, None, None
            else:
                xyz, xyz_org, sdf, perf, hgtmap = None, None, None, None, None

        # return xyz, xyz_org, sdf, perf
        if(return_filter):
            return xyz, xyz_org, sdf, perf, hgtmap, fil_bboxInc
        else:
            return xyz, xyz_org, sdf, perf, hgtmap


    
class XYZ_SDF_Perf_Dataset(torch.utils.data.Dataset):
    """
    A subclass that considers the train/val/test splits and maximum number of instances.

    """
    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                subsets=['0000_0499'],
                geo_types=['building'],
                point_sampling='scattered', #'sliced'
                point_sampling_params={'sampling_types':['rejection','surface','uniform'],'sampling_distr':[1,1,1],'srf_pt_set':'20','grd_pt_set':'512','grd_pt_hgts':'all','exclude_grd':False,'exclude_srf':False,'grd_per_loop':2},
                distr_posneg = [0.9,0.1],
                num_points = 50000,
                s_range = '0_1', #or N1_1
                perf_metric = 'U',
                split_type='train', # or val or test
                max_instances = -1, #-1 to use full dataset
                geo_dataset_avail = -1, #-1 to use full dataset and ignore trained geometry datasets
                batch_size = 32,
                num_workers = 8,
                return_fact = False,
                only_zero_orientation = False,
                include_sdf_inpt=False,
                add_ext='') -> None:

        # Generate dataset
        self._data = XYZ_SDF_Perf(directory=directory,
                          dataset_type=dataset_type,
                          subsets=subsets,
                          geo_types=geo_types,
                          point_sampling=point_sampling,
                          point_sampling_params=point_sampling_params,
                          distr_posneg = distr_posneg,
                          num_points=num_points,
                          s_range=s_range,
                          return_fact=return_fact,
                          add_ext=add_ext)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.perf_metric = perf_metric
        self.return_fact = return_fact
        self.include_sdf_inpt = include_sdf_inpt

        # Check if performance files exist for the selected performance metric
        perf_data = self._data.cfd_df_sub if(perf_metric in ['U','P']) else self._data.svf_df_sub
        if(point_sampling=='sliced'):
            srf_pt_set = point_sampling_params['srf_pt_set']
            grd_pt_set = point_sampling_params['grd_pt_set']
            perf_cols = [perf_metric + '_' + 'srf', perf_metric + '_' + 'fld'] if(perf_metric == 'U') else [perf_metric + '_' + 'srf'] if(perf_metric == 'P') else ['srf_' + str(srf_pt_set), 'XYgrid_' + str(grd_pt_set) + '_30']
        else:
            sampling_types = point_sampling_params['sampling_types']
            perf_cols = [perf_metric + '_' + nT[0:3] for nT in sampling_types] 
        perf_data['p_part'] = perf_data[perf_cols].sum(axis=1)  
        perf_data = perf_data[perf_data['p_part']>0]
        if(point_sampling=='sliced'):
            perf_cols = [perf_metric + '_' + 'srf', perf_metric + '_' + 'fld', perf_metric + '_' + 'allR', perf_metric + '_' + 'totR'] if(perf_metric == 'U') else [perf_metric + '_' + 'srf', perf_metric + '_' + 'allR', perf_metric + '_' + 'totR'] if(perf_metric == 'P') else ['srf_' + str(srf_pt_set), 'XYgrid_' + str(grd_pt_set) + '_30']
        else:
            perf_cols = [x for xs in [perf_cols, [perf_metric + '_' + 'allR', perf_metric + '_' + 'totR']] for x in xs] if(perf_metric in ['U','P']) else perf_cols
        perf_cols.append('idx')
        perf_cols.append('b_type')
        perf_data = perf_data[perf_cols]
        
        # Load the indices for the train/val/test sets ONLY
        ids_inc = np.array(json.loads(open(self._data.parameters_dir + '/' + 'traintestParam_5000.json').read())[split_type])
        
        # Filter the indices based on the type of dataset
        self.ids_exist_merged = pd.merge(self._data.sub_df_exist[self._data.sub_df_exist['idx'].isin(ids_inc)], perf_data[perf_data['idx'].isin(ids_inc)], on=['b_type', 'idx'], how='inner')
        
        # Define the maximum instances based on the filtered set (of available geometry and performance)
        self.ids_exist_merged['g_name'] = (self.ids_exist_merged['b_type'] + '_' + self.ids_exist_merged['idx'].astype(str)).values
        if((geo_dataset_avail != -1)and(len(geo_dataset_avail[0].split('_'))==2)):
            self.ids_exist_merged = self.ids_exist_merged[self.ids_exist_merged['g_name'].isin(geo_dataset_avail)] if(geo_dataset_avail != -1) else self.ids_exist_merged

        self.max_geo_instances = len(self.ids_exist_merged)
        self.max_instances_withrot = self.ids_exist_merged[perf_metric + '_totR'].sum() if(perf_metric in ['U','P']) else self.max_geo_instances*8 if((perf_metric == 'SVF') and (not only_zero_orientation)) else self.max_geo_instances
        self.uniq_names = self.ids_exist_merged['g_name'].values
       # print('self.uniq_names',self.uniq_names)
        self.uniq_geo_names = self.ids_exist_merged['g_name'].values
        if(self.max_instances_withrot > 0):
            if(perf_metric in ['U','P']):
                orientations = [list(map(int, o.split('[')[1].split(']')[0].split(','))) if(o.split('[')[1].split(']')[0].split(',') != ['']) else [] for o in self.ids_exist_merged[perf_metric + '_allR']]
                g_names_rep = [list(np.repeat(self.ids_exist_merged['g_name'].iloc[i], len(orientations[i]))) for i in range(len(self.ids_exist_merged))]
                g_names_rep = [item for row in g_names_rep for item in row]
                orientations = [item for row in orientations for item in row]
            if(perf_metric == 'SVF'):
                g_names_rep = np.repeat(self.ids_exist_merged['g_name'].values, 8)
                orientations = np.repeat(np.arange(8)*45, len(self.ids_exist_merged['g_name'].values)).reshape(8,len(self.ids_exist_merged['g_name'].values)).T.flatten()
            self.uniq_names = [g_names_rep[i] + '_' + str(orientations[i]) for i in range(len(g_names_rep))]
            self.uniq_geo_names = g_names_rep
        else:
            print('There are no elements in the dataset.')

        if((geo_dataset_avail != -1)and(len(geo_dataset_avail[0].split('_'))==3)):
            if(len(self.uniq_names[0].split('_'))==2):
                self.uniq_names = [n + '_0' for n in self.uniq_names]
            self.uniq_names = np.array(self.uniq_names)[np.isin(self.uniq_names, geo_dataset_avail)]# self.uniq_names[self.uniq_names.isin(geo_dataset_avail)]
            self.uniq_geo_names = np.unique(['_'.join(s.split('_')[0:2]) for s in self.uniq_geo_names])
        
        if(only_zero_orientation):
            if(len(self.uniq_names[0].split('_'))==3):
                self.uniq_names = [n for n in self.uniq_names if(int(n.split('_')[2])==0)]
                self.uniq_geo_names = np.unique(['_'.join(s.split('_')[0:2]) for s in self.uniq_geo_names])
        self.max_total_instances = np.minimum(len(self.uniq_names), max_instances) if(max_instances != -1) else len(self.uniq_names)
            
    def __len__(self):
        return int(np.ceil(self.max_total_instances/self.batch_size))*self.batch_size
    
    def __getlenbyGeo__(self):
        return self.max_geo_instances

    def __getlenbyGeoRot__(self):
        return self.max_instances_withrot
    
    def __getlenCurDataset__(self):
        return self.max_total_instances
    
    def __getUniqueNames__(self):
        return self.uniq_names
    
    def __getUniqueGeoNames__(self):
        return np.unique(self.uniq_geo_names[0:self.max_total_instances])
        
    def __getitemIncNone__(self, idx, include_sdf = True, formatforModel = True, return_fact = False): 
        idx = idx % self.max_total_instances
        g_name = self.uniq_names[idx]
        outputs = self._data.__getitem__(idx=int(g_name.split('_')[1]), geo_type=g_name.split('_')[0], perf_metric=self.perf_metric, orien_val=int(g_name.split('_')[2])) # if(self.perf_metric in ['U','P']) else None
        xyz, xyz_org, sdf, perf = outputs
        model_input = {'xyz': xyz,
                       'idx': torch.tensor([idx]),
                       'rot': torch.tensor([int(g_name.split('_')[2])]), # if(self.perf_metric in ['U','P']) else torch.tensor([0]),
                       'item_n': g_name,
                       'geo_n': '_'.join(g_name.split('_')[0:2]),
                       'geo_code': torch.tensor([0,int(g_name.split('_')[1])]) if(g_name.split('_')[0]=='building') else torch.tensor([1,int(g_name.split('_')[1])])}
        model_input = {'xyz':model_input['xyz'].unsqueeze(0), 
                       'idx':model_input['idx'].unsqueeze(0), 
                       'rot':model_input['rot'].unsqueeze(0), 
                       'item_n': [model_input['item_n']], 
                       'geo_n': [model_input['geo_n']],
                       'geo_code': model_input['geo_code'].unsqueeze(0)} if(formatforModel and (xyz is not None)) else model_input
        if(self.include_sdf_inpt):
            model_input['sdf'] = sdf.unsqueeze(0) if(formatforModel and (xyz is not None)) else sdf
            
        if(include_sdf):
            return model_input, sdf, perf
        else:
            return model_input, perf
        
    def __getitembySet__(self, idx, p_set, fil_bBox = False, include_sdf = True, formatforModel = True, return_fact = False):    #self, idx, geo_type, perf_metric, p_set, rot, fil_bBox = True, return_fact = False
        idx = idx % self.max_total_instances
        g_name = self.uniq_names[idx]
        outputs = self._data.__getSelPtSet__(idx=int(g_name.split('_')[1]), geo_type=g_name.split('_')[0], perf_metric=self.perf_metric, orien_val=int(g_name.split('_')[2]), s_set = p_set, fil_bBox=fil_bBox, return_fact=return_fact)  #if(self.perf_metric in ['U','P']) else None
        xyz, xyz_org, sdf, perf = outputs
        model_input = {'xyz': xyz,
                       'idx': torch.tensor([idx]),
                       'rot': torch.tensor([int(g_name.split('_')[2])]), # if(self.perf_metric in ['U','P']) else torch.tensor([0]),
                       'item_n': g_name,
                       'geo_n': '_'.join(g_name.split('_')[0:2]),
                       'geo_code': torch.tensor([0,int(g_name.split('_')[1])]) if(g_name.split('_')[0]=='building') else torch.tensor([1,int(g_name.split('_')[1])])}
        model_input = {'xyz':model_input['xyz'].unsqueeze(0), 
                       'idx':model_input['idx'].unsqueeze(0), 
                       'rot':model_input['rot'].unsqueeze(0), 
                       'item_n': [model_input['item_n']], 
                       'geo_n': [model_input['geo_n']],
                       'geo_code': model_input['geo_code'].unsqueeze(0)} if(formatforModel and (xyz is not None)) else model_input
        if(self.include_sdf_inpt):
            model_input['sdf'] = sdf.unsqueeze(0) if(formatforModel and (xyz is not None)) else sdf
            
        if(include_sdf):
            return model_input, sdf, perf
        else:
            return model_input, perf
        
    def __getitem__(self, idx, include_sdf = True, return_fact = False):  
        
        idx = idx % self.max_total_instances
        g_name = self.uniq_names[idx]
       # rt = int(g_name.split('_')[2]) if(self.perf_metric in ['U','P']) else None
        #print(str(idx) + ', ' + g_name + ', ' + 'real_idx: ' + g_name.split('_')[1] + ', geo_type: ' + g_name.split('_')[0] + ', perf_metric: ' + self.perf_metric + ', r: ' + str(rt))
        
        outputs = self._data.__getitem__(idx=int(g_name.split('_')[1]), geo_type=g_name.split('_')[0], perf_metric=self.perf_metric, orien_val=int(g_name.split('_')[2]), return_fact=return_fact)  #if(self.perf_metric in ['U','P']) else None
        #print('outputs',outputs)
        
        # If any of the outputs is None - find another random sample to replace it with - to avoid dataloader errors
        while(None in outputs):
            idx = np.random.randint(0, self.max_total_instances)
            g_name = self.uniq_names[idx]
            outputs = self._data.__getitem__(idx=int(g_name.split('_')[1]), geo_type=g_name.split('_')[0], perf_metric=self.perf_metric, orien_val=int(g_name.split('_')[2]), return_fact=return_fact)  #if(self.perf_metric in ['U','P']) else None
        
        xyz, xyz_org, sdf, perf = outputs
        model_input = {'xyz': xyz,
                       'idx': torch.tensor([idx]),
                       'rot': torch.tensor([int(g_name.split('_')[2])]),# if(self.perf_metric in ['U','P']) else torch.tensor([0]),
                       'item_n': g_name,
                       'geo_n': '_'.join(g_name.split('_')[0:2]),
                       'geo_code': torch.tensor([0,int(g_name.split('_')[1])]) if(g_name.split('_')[0]=='building') else torch.tensor([1,int(g_name.split('_')[1])])}
        if(self.include_sdf_inpt):
            model_input['sdf'] = sdf
            
        if(include_sdf):
            return model_input, sdf, perf
        else:
            return model_input, perf

        
class XYZ_SDF_Perf_HgtMaps_Dataset(torch.utils.data.Dataset):
    """
    A subclass that considers the train/val/test splits and maximum number of instances.

    """
    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                subsets=['0000_0499'],
                geo_types=['building'],
                point_sampling='sliced', 
                point_sampling_params={'grd_pt_set':'512','grd_pt_hgts':'all'},
                perf_metric = 'U',
                split_type='train', # or val or test
                max_instances = -1, #-1 to use full dataset
                geo_dataset_avail = -1, #-1 to use full dataset and ignore trained geometry datasets
                batch_size = 32,
                num_workers = 8,
                return_fact = False,
                only_zero_orientation = False,
                include_sdf_inpt=False,
                inc_localMap = True) -> None:

        # Generate dataset
        self._data = XYZ_SDF_Perf(directory=directory,
                          dataset_type=dataset_type,
                          subsets=subsets,
                          geo_types=geo_types,
                          point_sampling=point_sampling,
                          point_sampling_params=point_sampling_params,
                          return_fact=return_fact)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.perf_metric = perf_metric
        self.return_fact = return_fact
        self.include_sdf_inpt = include_sdf_inpt
        self.inc_localMap = inc_localMap

        # Check if performance files exist for the selected performance metric
        perf_data = self._data.cfd_df_sub if(perf_metric in ['U']) else self._data.svf_df_sub
        #print('perf_data',perf_data)
        grd_pt_set = point_sampling_params['grd_pt_set']
        self.grd_pt_set = grd_pt_set
        perf_cols = [perf_metric + '_' + 'fld'] if(perf_metric == 'U') else ['XYgrid_' + str(grd_pt_set) + '_30']
        #print('perf_cols',perf_cols)
        perf_data['p_part'] = perf_data[perf_cols].sum(axis=1)  
        perf_data = perf_data[perf_data['p_part']>0]
        perf_cols = [perf_metric + '_' + 'fld', perf_metric + '_' + 'allR', perf_metric + '_' + 'totR'] if(perf_metric == 'U') else ['XYgrid_' + str(grd_pt_set) + '_30']
        perf_cols.append('idx')
        perf_cols.append('b_type')
        #print('perf_cols',perf_cols)
        perf_data = perf_data[perf_cols]
        #print('perf_data',perf_data)
        fld_vals = [15,30,45,60,75,90,105,120,135,150,300,600,1200,2400,4800] if(point_sampling_params['grd_pt_hgts'] == 'all') else point_sampling_params['grd_pt_hgts']
        
        # Load the indices for the train/val/test sets ONLY
        ids_inc = np.array(json.loads(open(self._data.parameters_dir + '/' + 'traintestParam_5000.json').read())[split_type])
        
        # Filter the indices based on the type of dataset
        self.ids_exist_merged = pd.merge(self._data.sub_df_exist[self._data.sub_df_exist['idx'].isin(ids_inc)], perf_data[perf_data['idx'].isin(ids_inc)], on=['b_type', 'idx'], how='inner')
        
        # Define the maximum instances based on the filtered set (of available geometry and performance)
        self.ids_exist_merged['g_name'] = (self.ids_exist_merged['b_type'] + '_' + self.ids_exist_merged['idx'].astype(str)).values
        if((geo_dataset_avail != -1)and(len(geo_dataset_avail[0].split('_'))==2)):
            self.ids_exist_merged = self.ids_exist_merged[self.ids_exist_merged['g_name'].isin(geo_dataset_avail)] if(geo_dataset_avail != -1) else self.ids_exist_merged

        self.max_geo_instances = len(self.ids_exist_merged)
        self.max_instances_withrot = self.ids_exist_merged[perf_metric + '_totR'].sum() if(perf_metric in ['U','P']) else self.max_geo_instances*8 if((perf_metric == 'SVF') and (not only_zero_orientation)) else self.max_geo_instances
        self.uniq_names = self.ids_exist_merged['g_name'].values
       # print('self.uniq_names',self.uniq_names)
        self.uniq_geo_names = self.ids_exist_merged['g_name'].values
        if(self.max_instances_withrot > 0):
            if(perf_metric in ['U','P']):
                orientations = [list(map(int, o.split('[')[1].split(']')[0].split(','))) if(o.split('[')[1].split(']')[0].split(',') != ['']) else [] for o in self.ids_exist_merged[perf_metric + '_allR']]
                g_names_rep = [list(np.repeat(self.ids_exist_merged['g_name'].iloc[i], len(orientations[i]))) for i in range(len(self.ids_exist_merged))]
                g_names_rep = [item for row in g_names_rep for item in row]
                orientations = [item for row in orientations for item in row]
            if(perf_metric == 'SVF'):
                g_names_rep = np.repeat(self.ids_exist_merged['g_name'].values, 8)
                orientations = np.repeat(np.arange(8)*45, len(self.ids_exist_merged['g_name'].values)).reshape(8,len(self.ids_exist_merged['g_name'].values)).T.flatten()
            self.uniq_names = [g_names_rep[i] + '_' + str(orientations[i]) for i in range(len(g_names_rep))]
            self.uniq_geo_names = g_names_rep
        else:
            print('There are no elements in the dataset.')

        if((geo_dataset_avail != -1)and(len(geo_dataset_avail[0].split('_'))==3)):
            if(len(self.uniq_names[0].split('_'))==2):
                self.uniq_names = [n + '_0' for n in self.uniq_names]
            self.uniq_names = np.array(self.uniq_names)[np.isin(self.uniq_names, geo_dataset_avail)]# self.uniq_names[self.uniq_names.isin(geo_dataset_avail)]
            self.uniq_geo_names = np.unique(['_'.join(s.split('_')[0:2]) for s in self.uniq_geo_names])
        
        if(only_zero_orientation):
            if(len(self.uniq_names[0].split('_'))==3):
                self.uniq_names = [n for n in self.uniq_names if(int(n.split('_')[2])==0)]
                self.uniq_geo_names = np.unique(['_'.join(s.split('_')[0:2]) for s in self.uniq_geo_names])

        # Include heights as separate data points
        grd_pt_hgts = [15,30,45,60,75,90,105,120,135,150,300,600,1200,2400,4800] if(point_sampling_params['grd_pt_hgts'] == 'all') else point_sampling_params['grd_pt_hgts']
        rp_n = np.repeat(self.uniq_names,len(grd_pt_hgts))
        rp_rot = np.repeat(grd_pt_hgts,len(self.uniq_names)).reshape(len(grd_pt_hgts),len(self.uniq_names)).T.flatten()
        self.uniq_names = [rp_n[i] + '_' + str(rp_rot[i]) for i in range(len(rp_n))]

        #self.uniq_names = []self.uniq_names fld_vals
        self.max_total_instances = np.minimum(len(self.uniq_names), max_instances) if(max_instances != -1) else len(self.uniq_names)
            
    def __len__(self):
        return int(np.ceil(self.max_total_instances/self.batch_size))*self.batch_size
    
    def __getlenbyGeo__(self):
        return self.max_geo_instances

    def __getlenbyGeoRot__(self):
        return self.max_instances_withrot
    
    def __getlenCurDataset__(self):
        return self.max_total_instances
    
    def __getUniqueNames__(self):
        return self.uniq_names
    
    def __getUniqueGeoNames__(self):
        return np.unique(self.uniq_geo_names[0:self.max_total_instances])
        
    def __getitemIncNone__(self, idx, include_sdf = True, formatforModel = True, return_fact = False): 
        idx = idx % self.max_total_instances
        un_name = self.uniq_names[idx]
        g_name = '_'.join(un_name.split('_')[0:3])
        hgt = un_name.split('_')[-1]    
        outputs = self._data.__getSelPtSetHgtMap__(idx=int(g_name.split('_')[1]), geo_type=g_name.split('_')[0], perf_metric=self.perf_metric, s_set='XYgrid_'+str(self.grd_pt_set)+'_30_'+hgt, orien_val=int(g_name.split('_')[2]), return_fact=return_fact) # if(self.perf_metric in ['U','P']) else None
        xyz, xyz_org, sdf, perf, hgtmap = outputs
        model_input = {'xyz': xyz,
                       'hgtmap':hgtmap if(self.inc_localMap) else hgtmap[:,:1],
                       'idx': torch.tensor([idx]),
                       'rot': torch.tensor([int(g_name.split('_')[2])]), # if(self.perf_metric in ['U','P']) else torch.tensor([0]),
                       'item_n': g_name,
                       'geo_n': '_'.join(g_name.split('_')[0:2]),
                       'geo_code': torch.tensor([0,int(g_name.split('_')[1])]) if(g_name.split('_')[0]=='building') else torch.tensor([1,int(g_name.split('_')[1])]),
                       'hgt':torch.tensor([int(hgt)])}
        model_input = {'xyz':model_input['xyz'].unsqueeze(0), 
                       'hgtmap':model_input['hgtmap'].unsqueeze(0),
                       'idx':model_input['idx'].unsqueeze(0), 
                       'rot':model_input['rot'].unsqueeze(0), 
                       'item_n': [model_input['item_n']], 
                       'geo_n': [model_input['geo_n']],
                       'geo_code': model_input['geo_code'].unsqueeze(0),
                       'hgt':model_input['hgt'].unsqueeze(0)} if(formatforModel and (xyz is not None)) else model_input
        if(self.include_sdf_inpt):
            model_input['sdf'] = sdf.unsqueeze(0) if(formatforModel and (xyz is not None)) else sdf
            
        if(include_sdf):
            return model_input, sdf, perf
        else:
            return model_input, perf
        
    def __getitem__(self, idx, include_sdf = True, return_fact = False):  
       # __getitem__(self, idx, geo_type, perf_metric, s_set, orien_val, fil_bBox = True, return_fact = False)
        
        idx = idx % self.max_total_instances
        un_name = self.uniq_names[idx]
        g_name = '_'.join(un_name.split('_')[0:3])
        hgt = un_name.split('_')[-1]            
       # rt = int(g_name.split('_')[2]) if(self.perf_metric in ['U','P']) else None
        #print(str(idx) + ', ' + g_name + ', ' + 'real_idx: ' + g_name.split('_')[1] + ', geo_type: ' + g_name.split('_')[0] + ', perf_metric: ' + self.perf_metric + ', r: ' + str(rt))
        
        outputs = self._data.__getSelPtSetHgtMap__(idx=int(g_name.split('_')[1]), geo_type=g_name.split('_')[0], perf_metric=self.perf_metric, s_set='XYgrid_'+str(self.grd_pt_set)+'_30_'+hgt, orien_val=int(g_name.split('_')[2]), return_fact=return_fact)  #if(self.perf_metric in ['U','P']) else None
        #print('outputs',outputs)
        
        # If any of the outputs is None - find another random sample to replace it with - to avoid dataloader errors
        while(None in outputs):
            idx = np.random.randint(0, self.max_total_instances)
            un_name = self.uniq_names[idx]
            g_name = '_'.join(un_name.split('_')[0:3])
            hgt = un_name.split('_')[-1]  
            outputs = self._data.__getSelPtSetHgtMap__(idx=int(g_name.split('_')[1]), geo_type=g_name.split('_')[0], perf_metric=self.perf_metric, s_set='XYgrid_'+str(self.grd_pt_set)+'_30_'+hgt, orien_val=int(g_name.split('_')[2]), return_fact=return_fact)  #if(self.perf_metric in ['U','P']) else None
        
        xyz, xyz_org, sdf, perf, hgtmap = outputs
        model_input = {'xyz': xyz,
                       'hgtmap':hgtmap if(self.inc_localMap) else hgtmap[:,:1],
                       'idx': torch.tensor([idx]),
                       'rot': torch.tensor([int(g_name.split('_')[2])]),# if(self.perf_metric in ['U','P']) else torch.tensor([0]),
                       'item_n': g_name,
                       'geo_n': '_'.join(g_name.split('_')[0:2]),
                       'geo_code': torch.tensor([0,int(g_name.split('_')[1])]) if(g_name.split('_')[0]=='building') else torch.tensor([1,int(g_name.split('_')[1])]),
                        'hgt':torch.tensor([int(hgt)])}
        if(self.include_sdf_inpt):
            model_input['sdf'] = sdf
            
        if(include_sdf):
            return model_input, sdf, perf
        else:
            return model_input, perf
        
        
def getPerfArrayfromSampSet(self, s, return_fact=False):
    
    pref_bytype = 'wSp' if(self.perf_metric=='U') else 'pCf'

    # Read the sdf point samples for performance types
    pt_file = self.s_dir + '/' + s + '/' + self.g_name + '_' + s + '.npz'
    if(self.perf_metric in ['U','P']):
        perf_file = self.performance_dir +'/'+ self.perf_metric + '/' + self.perf_metric + '_' + s[0:3] + '/'+ self.dataset_type + '/' + self.set_fromIDX + '/' + self.g_name_perf + '_' + pref_bytype + '_' + s + '_pos_' + str(self.orien_val) + '.npz'
    if(self.perf_metric == 'SVF'):
        perf_file = self.performance_dir + '/' + self.perf_metric + '/' + self.dataset_type + '/' + self.set_fromIDX + '/' + self.g_name_perf + '/' + self.g_name_perf + '_' + s[0:3] + '_pos_svf.npy'
    # print('pt_file',pt_file)

    if(os.path.exists(pt_file) and os.path.exists(perf_file)):
        d_sdf_pos = np.load(pt_file, allow_pickle=True)['pos']
        sort_d_sdf_byXYZ_pos = np.lexsort((d_sdf_pos[:,1],d_sdf_pos[:,2],d_sdf_pos[:,0]))
        d_sdf_pos = d_sdf_pos[sort_d_sdf_byXYZ_pos]
        d_sdf_neg = np.load(pt_file, allow_pickle=True)['neg']
        sort_d_sdf_byXYZ_neg = np.lexsort((d_sdf_neg[:,1],d_sdf_neg[:,2],d_sdf_neg[:,0]))
        d_sdf_neg = d_sdf_neg[sort_d_sdf_byXYZ_neg]
        # print('d_sdf_neg',d_sdf_neg.shape)

        # Rotate points based on orientation
        d_pts_pos = d_sdf_pos[:,[0,1,2]]
        d_pts_neg = d_sdf_neg[:,[0,1,2]]

        # Read the cfd point results
        d_perf_ptsN = 0

        if(self.perf_metric in ['U','P']):
            # Read the cfd results
            d_perf, d_perf_ptsN, d_pts_pos = readCFDPerfFile(perf_file, self.perf_metric, self.return_fact, self.buffered_bbox, return_points=True)
            d_perf_neg = np.zeros((d_sdf_neg.shape[0],d_perf.shape[-1]), dtype=np.float64) if(d_perf is not None) else None

        # Read the SVF results
        if(self.perf_metric == 'SVF'):
            d_perf, d_perf_ptsN = readSVFPerfFile(perf_file, sort_d_sdf_byXYZ_pos)
            d_perf_neg = np.zeros((d_sdf_neg.shape[0],d_perf.shape[-1]), dtype=np.float64) if(d_perf is not None) else None

        if(d_perf is not None):
            # print('d_pts_pos',d_pts_pos.shape)
            # print('d_pts_neg',d_pts_neg.shape)
            if((self.orien_val is not None)):# and (self.perf_metric != 'SVF')):
                d_pts_rot_pos = np.concatenate([(d_pts_pos + [-0.5,-0.5,-0.5]),np.ones((d_pts_pos.shape[0],1))],1)
                d_pts_rot_pos = np.dot(rotation_matrix_y(self.orien_val),d_pts_rot_pos.T).T[:,:3].round(6) + [0.5,0.5,0.5]
                d_pts_rot_neg = np.concatenate([(d_pts_neg + [-0.5,-0.5,-0.5]),np.ones((d_pts_neg.shape[0],1))],1)
                d_pts_rot_neg = np.dot(rotation_matrix_y(self.orien_val),d_pts_rot_neg.T).T[:,:3].round(6) + [0.5,0.5,0.5]
            else:
                d_pts_rot_pos = d_pts_pos
                d_pts_rot_neg = d_pts_neg

            # Return values
            d_perf_ptsN = d_perf_ptsN
            #print('d_perf_ptsN',d_perf_ptsN)
            if((d_perf_ptsN == d_sdf_pos.shape[0]) and (d_perf_ptsN != 0) and (d_perf is not None)):
                # Combine pos+neg
                d_pts = np.concatenate([d_pts_pos,d_pts_neg],0)
                d_pts_rot = np.concatenate([d_pts_rot_pos,d_pts_rot_neg],0)
                d_sdf = np.concatenate([d_sdf_pos,d_sdf_neg],0)
                d_perf_comb = np.concatenate([d_perf,d_perf_neg],0)
                return d_pts, d_pts_rot, d_sdf[:,3:], d_perf_comb 
            else:
                return None, None, None, None
        else:
            return None, None, None, None
    else:
        #print('Sampling file does not exists for: ' + self.g_name)
        return None, None, None, None
    
def getPerfArrayfromPSet(self, p, return_fact=False, return_hgtMap = False, matchhgtmaps=True):
    
    pref_bytype = 'wSp' if(self.perf_metric=='U') else 'pCf'
    cfd_add_temp = 'srf' if(p.startswith('srf')) else 'fld' if(p.startswith('XYgrid')) else 'vert'
    upd_idx = False

    # Read the sdf point samples for performance types
    pt_file = self.s_dir + '/' + 'performance' + '/' + self.g_name +'/' + self.g_name + '_' + p + '.npz' 
    #print(pt_file)
    if(self.perf_metric in ['U','P']):
        perf_file = self.performance_dir +'/'+ self.perf_metric + '/' + self.perf_metric + '_' + cfd_add_temp + '/'+ self.dataset_type + '/' + self.set_fromIDX + '/' + self.g_name_perf + '_' + pref_bytype + '_'+ p + '_' + str(self.orien_val) + '.npz' 
        if((not os.path.exists(perf_file)) and ('XYgrid_256_' in p)):
            perf_file = self.performance_dir +'/'+ self.perf_metric + '/' + self.perf_metric + '_' + cfd_add_temp + '/'+ self.dataset_type + '/' + self.set_fromIDX + '/' + self.g_name_perf + '_' + pref_bytype + '_'+ p.replace('_256_','_512_') + '_' + str(self.orien_val) + '.npz' 
            upd_idx = True
    if(self.perf_metric == 'SVF'):
        perf_file = self.performance_dir + '/' + self.perf_metric + '/' + self.dataset_type + '/' + self.set_fromIDX + '/' + self.g_name_perf + '/' + self.g_name_perf + '_' + p + '_svf.npy'
    #print(perf_file)

    if(return_hgtMap):
        p_512 = p.replace('_256_','_512_') if('_256_' in p) else p
        hgt_file = self.performance_dir + '/' + 'HgtMaps' + '/' + self.dataset_type + '/' + self.set_fromIDX + '/' + self.g_name_perf + '/' + self.g_name_perf + '_' + str(self.orien_val) + '_' + p_512
        hgt_bool = (np.array([os.path.exists(hgt_file+'_'+m+'map.npy') for m in ['glb','loc']])).all()

    allFExist = (os.path.exists(pt_file) and (os.path.exists(perf_file) or os.path.exists(perf_file.replace('/Performance/','/Performance/CFD_postP/'))) and hgt_bool) if(return_hgtMap) else (os.path.exists(pt_file) and (os.path.exists(perf_file) or (os.path.exists(perf_file.replace('/Performance/','/Performance/CFD_postP/')))))
    # print('allFExist',allFExist)
    # print('os.path.exists(pt_file)',os.path.exists(pt_file))
    # print('os.path.exists(perf_file)',os.path.exists(perf_file))
    if(allFExist):
        d_sdf = np.load(pt_file, allow_pickle=True)['allv']
        sort_d_sdf_byXYZ = np.lexsort((d_sdf[:,1],d_sdf[:,2],d_sdf[:,0]))
        d_sdf = d_sdf[sort_d_sdf_byXYZ]

        # Rotate points based on orientation
        d_pts = d_sdf[:,[0,1,2]]
        if((self.orien_val is not None)):# and (self.perf_metric != 'SVF')):
            d_pts_rot = np.concatenate([(d_pts + [-0.5,-0.5,-0.5]),np.ones((d_pts.shape[0],1))],1)
            d_pts_rot = np.dot(rotation_matrix_y(self.orien_val),d_pts_rot.T).T[:,:3].round(6) + [0.5,0.5,0.5]
        else:
            d_pts_rot = d_pts

        # Read the cfd point results
        d_perf_ptsN = 0

        if(self.perf_metric in ['U','P']):
            # Read the cfd results
            d_perf, d_perf_ptsN = readCFDPerfFile(perf_file, self.perf_metric, self.return_fact, self.buffered_bbox)

        # Read the SVF results
        if(self.perf_metric == 'SVF'):
            d_perf, d_perf_ptsN = readSVFPerfFile(perf_file, sort_d_sdf_byXYZ)

        # Read height map values
        if(return_hgtMap):
            hgt_map = readHeightMaps(hgt_file, maps=['glb','loc'], m_size=512)/((self.buffered_bbox[:,1]-self.buffered_bbox[:,0])[0])

        # Return values
        if(d_perf is not None):
            d_perf_ptsN = d_perf_ptsN if(not upd_idx) else len(self.idx_subset)
            #print('d_perf_ptsN',d_perf_ptsN)
            if(d_perf_ptsN == d_sdf.shape[0]):
                if(not upd_idx):
                    if(return_hgtMap):
                        return d_pts, d_pts_rot, d_sdf[:,3:], d_perf, hgt_map
                    else:
                        return d_pts, d_pts_rot, d_sdf[:,3:], d_perf
                else:
                    if(return_hgtMap):
                        #print('d_perf.shape',d_perf.shape)
                        if(matchhgtmaps):
                            cr_perf = np.concatenate([rotate_and_crop(d_perf[:,i].reshape(512,512), angle=-1*self.orien_val, crop_size=(256, 256)).reshape(256*256,1) for i in range(d_perf.shape[1])],-1)
                            return d_pts, d_pts_rot, d_sdf[:,3:], cr_perf, hgt_map[self.idx_subset]
                        else:
                            return d_pts, d_pts_rot, d_sdf[:,3:], d_perf[self.idx_subset], hgt_map[self.idx_subset]
                    else:
                        return d_pts, d_pts_rot, d_sdf[:,3:], d_perf[self.idx_subset]
            else:
                if(return_hgtMap):
                    return (None, None, None, None, None)
                else:
                    return (None, None, None, None)
        else:
            if(return_hgtMap):
                return (None, None, None, None, None)
            else:
                return (None, None, None, None)
    else:
        #print('Sampling file does not exists for: ' + self.g_name)
        if(return_hgtMap):
            return (None, None, None, None, None)
        else:
            return (None, None, None, None)

def rotate_and_crop(image, angle, crop_size):
    # Rotate the image
    height, width = image.shape
    center = (width // 2, height // 2)
    
    # Create rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform rotation
    rotated_image = cv2.warpAffine(image, M, (width, height))
    
    # Crop the image
    x_start = (width - crop_size[0]) // 2
    y_start = (height - crop_size[1]) // 2
    cropped_image = rotated_image[y_start:y_start + crop_size[1], x_start:x_start + crop_size[0]]
    
    return cropped_image

# # Function to read the CFD file
# def readCFDPerfFile(perf_file, perf_metric, return_fact, buffered_bbox, return_points=False):
#     if((os.path.exists(perf_file))and((np.load(perf_file, allow_pickle=True)['vals'].size!=1))):
#         d_sdf_cfd = np.load(perf_file)['vals']
#         # print('d_sdf_cfd',d_sdf_cfd.shape)
#         d_sdf_cfd_pts = np.load(perf_file)['pts']
#         if(d_sdf_cfd.shape[0]==d_sdf_cfd_pts.shape[0]):
#             if(return_fact):
#                 pts_Z = d_sdf_cfd_pts[:,2] 
#                 pts_Z[pts_Z<0] = 0.0000000001
#             d_sdf_cfd_pts = d_sdf_cfd_pts[:,[0,2,1]]   # Y and Z reversed in geometry
#             d_sdf_cfd_pts[:,2] = d_sdf_cfd_pts[:,2]*-1    # Points axes are flipped so we match the 
#             d_sdf_cfd_pts = normalize_points(d_sdf_cfd_pts, buffered_bbox)
#             sort_d_cfd_byXYZ = np.lexsort((d_sdf_cfd_pts[:,1],d_sdf_cfd_pts[:,2],d_sdf_cfd_pts[:,0]))
#             d_sdf_cfd_pts = d_sdf_cfd_pts[sort_d_cfd_byXYZ]
#             d_sdf_cfd = d_sdf_cfd[sort_d_cfd_byXYZ]
#             d_sdf_cfd = np.minimum(np.maximum(d_sdf_cfd, -1000000000),1000000000)            #d_sdf_cfd_101 = d_sdf_cfd.shape
#             if(perf_metric == 'U'):
#                 #print('before',d_sdf_cfd[:,0])
#                 d_sdf_cfd[:,0] = [c if((c!=-1000000000) and (c!=1000000000)) else 0. for c in d_sdf_cfd[:,0]]
#                 d_sdf_cfd[:,1] = [c if((c!=-1000000000) and (c!=1000000000)) else 0. for c in d_sdf_cfd[:,1]]
#                 d_sdf_cfd[:,2] = [c if((c!=-1000000000) and (c!=1000000000)) else 0. for c in d_sdf_cfd[:,2]]
#                 #print('after',d_sdf_cfd[:,0])
#                 if(return_fact):
#                     wSpatZ = np.expand_dims(5 * np.abs(np.log(pts_Z) / np.log(10)),-1)
#                 d_sdf_cfd = d_sdf_cfd[:,[0,2,1]]
#                 d_sdf_cfd_mag = np.expand_dims(np.linalg.norm(d_sdf_cfd,axis=1),0).T
#                 #print(perf_file + ', d_sdf_cfd_mag',np.sort(d_sdf_cfd_mag)[::-1][0])
#                 d_sdf_cfd_unit = d_sdf_cfd/(d_sdf_cfd_mag+0.000000000000001)
#                 d_sdf_cfd_mag[d_sdf_cfd_mag>1000000000] = 0.
#                 d_sdf_cfd_mag = np.maximum(0, d_sdf_cfd_mag)
#                 if(return_fact):
#                     d_sdf_cfd_mag = d_sdf_cfd_mag/wSpatZ
#                 else:
#                     d_sdf_cfd_mag = d_sdf_cfd_mag / 20 # to normalize the magnitude within a smaller range
#                 if(('_srf_' not in perf_file)and(np.sqrt(len(d_sdf_cfd_mag)).is_integer())and(int(np.sqrt(len(d_sdf_cfd_mag)))==512)):
#                     #print('d_sdf_cfd_mag',d_sdf_cfd_mag.shape)
#                     res = int(np.sqrt(len(d_sdf_cfd_mag)))
#                     vals = d_sdf_cfd_mag.reshape(res,res).astype(np.float32)
#                     early_vals = vals[0:6,0:6]
#                     median = np.percentile(early_vals, 50)
#                     v_i = np.argsort(np.mean((early_vals-median), axis=0))[0]
#                     v_j =  np.argsort(np.mean((early_vals-median), axis=1))[0]
#                     mask_base = np.repeat(np.roll(((np.arange(res+v_i)%5)!=0).astype(int),v_j)[0:res],res)
#                     mask = mask_base*mask_base.reshape(res,res).T.flatten()
#                     vals_neigh=cv2.filter2D(vals,cv2.CV_32F,np.array([[1,1,1],[1,0,1],[1,1,1],],np.float32))/8
#                     vals_comb = ((vals*mask.reshape(res,res))+((vals_neigh)*((mask==0).astype(int)).reshape(res,res)))
#                     vals_neigh=cv2.filter2D(vals_comb.astype(vals.dtype),cv2.CV_32F,np.array([[1,1,1],[1,0,1],[1,1,1],],np.float32))/8
#                     vals_comb = ((vals_comb*mask.reshape(res,res))+((vals_neigh)*((mask==0).astype(int)).reshape(res,res)))
#                     int_mask = ((mask_base.reshape(res,res)+(mask_base.reshape(res,res)).T)==0).astype(int)
#                     int_mask_neigh = (np.roll(int_mask, -1, axis=0)+np.roll(int_mask, 1, axis=0)+np.roll(int_mask, -1, axis=1)+np.roll(int_mask, 1, axis=1))
#                     vals_neigh=cv2.filter2D(vals_comb.astype(vals.dtype),cv2.CV_32F,np.array([[1,1,1],[1,0,1],[1,1,1],],np.float32))/8
#                     vals_comb = ((vals_comb*((int_mask_neigh==1).astype(int)))+((vals_neigh)*(((int_mask_neigh==0).astype(int)))))
#                     d_sdf_cfd_mag=np.expand_dims(vals_comb.flatten(),-1)
#                     #print('d_sdf_cfd_mag',d_sdf_cfd_mag.shape)
#                 d_sdf_cfd = np.concatenate([d_sdf_cfd_mag, d_sdf_cfd_unit],1)
#                 d_perf_ptsN = 0 if(np.mean(d_sdf_cfd_mag)<0.000000000001) else d_sdf_cfd_pts.shape[0]
#                             #print(str(d_sdf_cfd_pts.shape[0]) + ' vs ' + str(d_perf_ptsN) + ' : ' + perf_file)
#             else:
#                 d_sdf_cfd[d_sdf_cfd==-1000000000] = 0.
#                 d_sdf_cfd[d_sdf_cfd==1000000000] = 0.
#                 d_sdf_cfd = np.expand_dims(d_sdf_cfd,-1)
#                 d_perf_ptsN = d_sdf_cfd_pts.shape[0]
#             d_perf = d_sdf_cfd
#             #print('d_sdf_cfd_pts',d_sdf_cfd_pts.shape[0])
#             d_perf_ptsN = 0 if((perf_metric == 'U') and (np.mean(d_sdf_cfd_mag)<0.000000000001)) else d_sdf_cfd_pts.shape[0]
#             if(return_points):
#                 return d_perf, d_perf_ptsN, d_sdf_cfd_pts
#             else:
#                 return d_perf, d_perf_ptsN 
#             #print('d_perf_ptsN',d_perf_ptsN)
#         else:
#             #print('Performance file does not exist: ' + perf_file)
#             if(return_points):
#                 return None, None, None
#             else:
#                 return None, None
#     else:
#         if(return_points):
#             return None, None, None
#         else:
#             return None, None




# Function to read the CFD file
def readCFDPerfFile(perf_file, perf_metric, return_fact, buffered_bbox, return_points=False, use_postP=True):
    #print('perf_file',perf_file)
    if(os.path.exists(perf_file.replace('/Performance/','/Performance/CFD_postP/')) and use_postP):
        CFD_outputs = readCFDPostProcFile(perf_file, perf_metric, return_fact, buffered_bbox, return_points)
    else:
        CFD_outputs = readOrigCFDFile(perf_file, perf_metric, return_fact, buffered_bbox, return_points)
    if(CFD_outputs[0] is not None):
        if(return_points):
            d_perf, d_perf_ptsN, d_sdf_cfd_pts = CFD_outputs
            return d_perf, d_perf_ptsN, d_sdf_cfd_pts
        else:
            d_perf, d_perf_ptsN = CFD_outputs
            return d_perf, d_perf_ptsN
    else:
        if(return_points):
            return None, None, None
        else:
            return None, None
        
        
# Function to read the CFD file
def readOrigCFDFile(perf_file, perf_metric, return_fact, buffered_bbox, return_points=False):
    zip_ok = True
    try:
        with ZipFile(perf_file) as zf:
            zip_ok = True
    except BadZipFile:
        zip_ok = False
    if((zip_ok) and (os.path.exists(perf_file))and((np.load(perf_file, allow_pickle=True)['vals'].size!=1))):
        d_sdf_cfd = np.load(perf_file)['vals']
        print('d_sdf_cfd',d_sdf_cfd.shape)
        d_sdf_cfd_pts = np.load(perf_file)['pts']
        if(d_sdf_cfd.shape[0]==d_sdf_cfd_pts.shape[0]):
            if(return_fact):
                pts_Z = d_sdf_cfd_pts[:,2] 
                pts_Z[pts_Z<0] = 0.0000000001
            d_sdf_cfd_pts = d_sdf_cfd_pts[:,[0,2,1]]   # Y and Z reversed in geometry
            d_sdf_cfd_pts[:,2] = d_sdf_cfd_pts[:,2]*-1    # Points axes are flipped so we match the 
            d_sdf_cfd_pts = normalize_points(d_sdf_cfd_pts, buffered_bbox)
            sort_d_cfd_byXYZ = np.lexsort((d_sdf_cfd_pts[:,1],d_sdf_cfd_pts[:,2],d_sdf_cfd_pts[:,0]))
            d_sdf_cfd_pts = d_sdf_cfd_pts[sort_d_cfd_byXYZ]
            d_sdf_cfd = d_sdf_cfd[sort_d_cfd_byXYZ]
            d_sdf_cfd = np.minimum(np.maximum(d_sdf_cfd, -1000000000),1000000000)            #d_sdf_cfd_101 = d_sdf_cfd.shape
            if(perf_metric == 'U'):
                #print('before',d_sdf_cfd[:,0])
                d_sdf_cfd[:,0] = [c if((c!=-1000000000) and (c!=1000000000)) else 0. for c in d_sdf_cfd[:,0]]
                d_sdf_cfd[:,1] = [c if((c!=-1000000000) and (c!=1000000000)) else 0. for c in d_sdf_cfd[:,1]]
                d_sdf_cfd[:,2] = [c if((c!=-1000000000) and (c!=1000000000)) else 0. for c in d_sdf_cfd[:,2]]
                #print('after',d_sdf_cfd[:,0])
                if(return_fact):
                    wSpatZ = np.expand_dims(5 * np.abs(np.log(pts_Z) / np.log(10)),-1)
                d_sdf_cfd = d_sdf_cfd[:,[0,2,1]]
                d_sdf_cfd_mag = np.expand_dims(np.linalg.norm(d_sdf_cfd,axis=1),0).T
                #print(perf_file + ', d_sdf_cfd_mag',np.sort(d_sdf_cfd_mag)[::-1][0])
                d_sdf_cfd_unit = d_sdf_cfd/(d_sdf_cfd_mag+0.000000000000001)
                d_sdf_cfd_mag[d_sdf_cfd_mag>1000000000] = 0.
                d_sdf_cfd_mag = np.maximum(0, d_sdf_cfd_mag)
                if(return_fact):
                    d_sdf_cfd_mag = d_sdf_cfd_mag/wSpatZ
                else:
                    d_sdf_cfd_mag = d_sdf_cfd_mag / 20 # to normalize the magnitude within a smaller range
                if(('_XYgrid_' in perf_file)and(np.sqrt(len(d_sdf_cfd_mag)).is_integer())and(int(np.sqrt(len(d_sdf_cfd_mag)))==512)):
                    #print('d_sdf_cfd_mag',d_sdf_cfd_mag.shape)
                    res = int(np.sqrt(len(d_sdf_cfd_mag)))
                    vals = d_sdf_cfd_mag.reshape(res,res).astype(np.float32)
                    early_vals = vals[0:6,0:6]
                    median = np.percentile(early_vals, 50)
                    v_i = np.argsort(np.mean((early_vals-median), axis=0))[0]
                    v_j =  np.argsort(np.mean((early_vals-median), axis=1))[0]
                    mask_base = np.repeat(np.roll(((np.arange(res+v_i)%5)!=0).astype(int),v_j)[0:res],res)
                    mask = mask_base*mask_base.reshape(res,res).T.flatten()
                    vals_neigh=cv2.filter2D(vals,cv2.CV_32F,np.array([[1,1,1],[1,0,1],[1,1,1],],np.float32))/8
                    vals_comb = ((vals*mask.reshape(res,res))+((vals_neigh)*((mask==0).astype(int)).reshape(res,res)))
                    vals_neigh=cv2.filter2D(vals_comb.astype(vals.dtype),cv2.CV_32F,np.array([[1,1,1],[1,0,1],[1,1,1],],np.float32))/8
                    vals_comb = ((vals_comb*mask.reshape(res,res))+((vals_neigh)*((mask==0).astype(int)).reshape(res,res)))
                    int_mask = ((mask_base.reshape(res,res)+(mask_base.reshape(res,res)).T)==0).astype(int)
                    int_mask_neigh = (np.roll(int_mask, -1, axis=0)+np.roll(int_mask, 1, axis=0)+np.roll(int_mask, -1, axis=1)+np.roll(int_mask, 1, axis=1))
                    vals_neigh=cv2.filter2D(vals_comb.astype(vals.dtype),cv2.CV_32F,np.array([[1,1,1],[1,0,1],[1,1,1],],np.float32))/8
                    vals_comb = ((vals_comb*((int_mask_neigh==1).astype(int)))+((vals_neigh)*(((int_mask_neigh==0).astype(int)))))
                    d_sdf_cfd_mag=np.expand_dims(vals_comb.flatten(),-1)
                    #print('d_sdf_cfd_mag',d_sdf_cfd_mag.shape)
                d_sdf_cfd = np.concatenate([d_sdf_cfd_mag, d_sdf_cfd_unit],1)
                d_perf_ptsN = 0 if(np.mean(d_sdf_cfd_mag)<0.000000000001) else d_sdf_cfd_pts.shape[0]
                            #print(str(d_sdf_cfd_pts.shape[0]) + ' vs ' + str(d_perf_ptsN) + ' : ' + perf_file)
            else:
                d_sdf_cfd[d_sdf_cfd==-1000000000] = 0.
                d_sdf_cfd[d_sdf_cfd==1000000000] = 0.
                d_sdf_cfd = np.expand_dims(d_sdf_cfd,-1)
                d_perf_ptsN = d_sdf_cfd_pts.shape[0]
            d_perf = d_sdf_cfd
            #print('d_sdf_cfd_pts',d_sdf_cfd_pts.shape[0])
            d_perf_ptsN = 0 if((perf_metric == 'U') and (np.mean(d_sdf_cfd_mag)<0.000000000001)) else d_sdf_cfd_pts.shape[0]
            if(return_points):
                return d_perf, d_perf_ptsN, d_sdf_cfd_pts
            else:
                return d_perf, d_perf_ptsN 
            #print('d_perf_ptsN',d_perf_ptsN)
        else:
            #print('Performance file does not exist: ' + perf_file)
            if(return_points):
                return None, None, None
            else:
                return None, None
    else:
        if(return_points):
            return None, None, None
        else:
            return None, None
        
def readCFDPostProcFile(perf_file, perf_metric, return_fact, buffered_bbox, return_points=False):
    perf_file_postP = perf_file.replace('/Performance/','/Performance/CFD_postP/')
    #print('perf_file_postP',perf_file_postP)
    zip_ok = True
    try:
        with ZipFile(perf_file_postP) as zf:
            zip_ok = True
    except BadZipFile:
        zip_ok = False
    if(zip_ok and (os.path.exists(perf_file_postP))and((np.load(perf_file_postP, allow_pickle=True)['d_perf'].size!=1))):
        d_perf = np.load(perf_file_postP)['d_perf']
        d_perf_ptsN = np.load(perf_file_postP)['d_perf_ptsN']
        if(return_points):
            d_sdf_cfd_pts = np.load(perf_file_postP)['d_sdf_cfd_pts']
            return d_perf, d_perf_ptsN, d_sdf_cfd_pts
        else:
            return d_perf, d_perf_ptsN
    else:
        if(return_points):
            return None, None, None
        else:
            return None, None
    
    
def readSVFPerfFile(perf_file, sort_d_sdf_byXYZ=None):
    if(os.path.exists(perf_file)):
        if(len(np.load(perf_file))>0):
            #print('perf_file.shape',np.load(perf_file).shape)
            d_perf = np.expand_dims(np.load(perf_file),0).T
            if(d_perf.shape[0] == sort_d_sdf_byXYZ.shape[0]):
                d_perf = d_perf[sort_d_sdf_byXYZ] if(sort_d_sdf_byXYZ is not None) else d_perf
                d_perf_ptsN = d_perf.shape[0]
                return d_perf, d_perf_ptsN
            else:
                return None, None
        else:
            return None, None
    else:
        return None, None

    
def readHeightMaps(hgt_file, maps=['glb','loc'], m_size=512):
    maps_v = None
    maps_exist = (np.array([os.path.exists(hgt_file+'_'+m+'map.npy') for m in maps])).all()
    if(maps_exist):
        maps_v = [np.load(hgt_file+'_'+m+'map.npy').reshape(m_size,m_size).T.reshape(-1,1) for m in maps]
        maps_v = np.concatenate(maps_v, axis=1)
    return maps_v
    
# Dataset preparation
class XYZ_SDF_Perf_Agg(torch.utils.data.Dataset):

    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                subset='0000_0499',
                geo_types=['building'],
                mag_only = True) -> None:
    
        self.directory = directory
        self.dataset_type = dataset_type
        self.subset = subset
        self.mag_only = mag_only

        # Define aggregate performance directory
        self.aggPerf_dir = self.directory + '/Performance/aggrMetrics'
        self.parameters_dir = self.directory + '/Parameters'
        
        # Get all aggregate data for all metrics
        self.pd_permetric = {}
        for perf_metric in ['U','P','SVF']:
            p_fol = self.aggPerf_dir + '/' + dataset_type + '/' + subset + '/' + perf_metric + '/' + 'combined' 
            all_res = pd.read_csv(p_fol + '/' + perf_metric + '_mag.csv').dropna().drop(columns=['m_split']).dropna()
            sdf_res = pd.read_csv(p_fol + '/' + perf_metric + '_mag_bysdf.csv').dropna().drop(columns=['m_split']).dropna()
            quad_res = pd.read_csv(p_fol + '/' + perf_metric + '_mag_byquad.csv').dropna().drop(columns=['m_split']).dropna()
            comb_res = pd.merge(pd.merge(all_res,sdf_res, on=['geo_type', 'idx', 'orient','m_type'], how='inner'), quad_res, on=['geo_type', 'idx', 'orient','m_type'], how='inner')
            comb_res = comb_res[comb_res['geo_type'].isin(geo_types)]
            comb_res = comb_res[comb_res['orient']==0] if(perf_metric == 'SVF') else comb_res
            if((not mag_only) and (perf_metric=='U')):
                pd_ar = []
                pd_ar.append(comb_res)
                for e in ['ele','azi']:
                    all_res = pd.read_csv(p_fol + '/' + perf_metric + '_'+e+'.csv').dropna().drop(columns=['m_split']).dropna()
                    sdf_res = pd.read_csv(p_fol + '/' + perf_metric + '_'+e+'_bysdf.csv').dropna().drop(columns=['m_split']).dropna()
                    quad_res = pd.read_csv(p_fol + '/' + perf_metric + '_'+e+'_byquad.csv').dropna().drop(columns=['m_split']).dropna()
                    comb_res = pd.merge(pd.merge(all_res,sdf_res, on=['geo_type', 'idx', 'orient','m_type'], how='inner'), quad_res, on=['geo_type', 'idx', 'orient','m_type'], how='inner')
                    comb_res = comb_res[comb_res['geo_type'].isin(geo_types)]
                    pd_ar.append(comb_res)
                self.pd_permetric[perf_metric] = pd_ar
            else:
                self.pd_permetric[perf_metric] = comb_res                          
                          
        # Get the maximum size of the train/val/test splits datasets
        self.split_len = {}
        self.split_ids = {}
        train_test_splits = json.loads(open(self.parameters_dir + '/' + 'traintestParam_5000.json').read())
        for k in train_test_splits.keys():
            subKey = np.array(train_test_splits[k])
            sp_len, sp_id = [], []
            for perf_metric in ['P','U','SVF']:
                pd_cur = self.pd_permetric[perf_metric][0] if((not mag_only) and (perf_metric=='U')) else self.pd_permetric[perf_metric]
                inc_idx = pd_cur[pd_cur['idx'].isin(subKey)]
                sp_len.append(len(inc_idx))
                sp_id.append(np.array(inc_idx['idx'].values))
            self.split_len[k] = sp_len
            self.split_ids[k] = sp_id

        super().__init__()
    
    def __len__(self):
        return np.sort([len(self.pd_permetric[perf_metric][0]) if((not mag_only) and (perf_metric=='U')) else len(self.pd_permetric[perf_metric]) for perf_metric in ['U','P','SVF']])[::-1][0]
    
    def __splitlen__(self):
        return self.split_len
    
    def __splitids__(self):
        return self.split_ids

    def __getitem__(self, idx, geo_type, perf_metric, orien_val, stats_params):
        
        # Define random seed as time
        np.random.seed(int(1000 * time.time()) % 2**32)
        
        # Define the cur_directories
        idx_s = int(idx/500)
        self.set_fromIDX = "{:04d}".format(idx_s*500) + '_' + "{:04d}".format(((idx_s+1)*500)-1)
        self.p_dir = self.aggPerf_dir + '/' + self.dataset_type + '/' + self.set_fromIDX + '/' + perf_metric
        
        # Check if performance data exist for the selected performance metric
        cols_to_keep = [x for xs in [['geo_type','idx','orient'], stats_params] for x in xs] 
        stats_file = [p[cols_to_keep] for p in self.pd_permetric[perf_metric]] if((not self.mag_only) and (perf_metric=='U')) else self.pd_permetric[perf_metric][cols_to_keep]
        stats_file = [s[(s['geo_type']==geo_type) & (s['idx']==idx) & (s['orient']==orien_val)] for s in stats_file] if((not self.mag_only) and (perf_metric=='U')) else  stats_file[(stats_file['geo_type']==geo_type) & (stats_file['idx']==idx) & (stats_file['orient']==orien_val)] 
        p_exists = len(stats_file[0])>0 if((not self.mag_only) and (perf_metric=='U')) else len(stats_file)>0
        
        if(p_exists):
            # Define the performance array
            perf = np.concatenate([s[stats_params].values[0] for s in stats_file]) if((not self.mag_only) and (perf_metric=='U')) else stats_file[stats_params].values[0]
        else:
            perf = None
        
        return perf
    
class XYZ_SDF_Perf_Dataset_Agg(torch.utils.data.Dataset):
    """
    A subclass that considers the train/val/test splits and maximum number of instances.

    """
    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                subset='0000_0499',
                geo_types=['building'],
                mag_only = True,
                perf_metric = 'U',
                m_splits = ['all'],
                stat_splits = ['mean','std'],
                split_type='train', # or val or test
                max_instances = -1, #-1 to use full dataset
                geo_dataset_avail = -1, #-1 to use full dataset and ignore trained geometry datasets
                batch_size = 32,
                num_workers = 8) -> None:

        # Generate dataset
        self._data = XYZ_SDF_Perf_Agg(directory=directory,
                                dataset_type=dataset_type,
                                subset=subset,
                                geo_types=geo_types,
                                mag_only=mag_only)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.perf_metric = perf_metric
        
        # Check if performance files exist for the selected performance metric
        perf_data = self._data.pd_permetric[perf_metric][0] if((perf_metric == 'U') and (not mag_only)) else self._data.pd_permetric[perf_metric]
        m_splits_rep = np.repeat(m_splits, len(stat_splits))
        stat_splits_rep = np.repeat(stat_splits, len(m_splits)).reshape(len(stat_splits), len(m_splits)).T.flatten()
        perf_cols = [m_splits_rep[i]+'_'+stat_splits_rep[i] for i in range(len(m_splits_rep))]
        self.perf_cols_sel = [c for c in perf_cols if((c.split('_')[0] in m_splits) and (c.split('_')[1] in stat_splits))]
        perf_cols_sel_fil = self.perf_cols_sel.copy()
        perf_cols_sel_fil.append('idx')
        perf_cols_sel_fil.append('geo_type')
        perf_cols_sel_fil.append('orient')
        perf_data = perf_data[perf_cols_sel_fil]
        
        # Load the indices for the train/val/test sets ONLY
        ids_inc = np.array(json.loads(open(self._data.parameters_dir + '/' + 'traintestParam_5000.json').read())[split_type])
        
        # Filter the indices based on the type of dataset
        self.ids_exist = perf_data[perf_data['idx'].isin(ids_inc)].copy()
        
        # Define the maximum instances based on the filtered set (of available geometry and performance)
        g_name = [self.ids_exist['geo_type'].iloc[i] + '_' + str(self.ids_exist['idx'].iloc[i]) for i in range(len(self.ids_exist))]
        self.ids_exist['g_name']=g_name
        g_name_rot = [self.ids_exist['geo_type'].iloc[i] + '_' + str(self.ids_exist['idx'].iloc[i]) + '_' + str(self.ids_exist['orient'].iloc[i]) for i in range(len(self.ids_exist))]
        self.ids_exist['g_name_rot']=g_name_rot
        if(geo_dataset_avail != -1):
            if(len(geo_dataset_avail[0].split('_'))==2):
                self.ids_exist = self.ids_exist[self.ids_exist['g_name'].isin(geo_dataset_avail)]
            if((geo_dataset_avail != -1)and(len(geo_dataset_avail[0].split('_'))==3)):
                self.ids_exist = self.ids_exist[self.ids_exist['g_name_rot'].isin(geo_dataset_avail)]

        self.uniq_names = g_name_rot
        self.uniq_geo_names = self.ids_exist['g_name'].values
        self.max_instances_withrot = len(self.ids_exist)
        self.max_geo_instances = len(np.unique(self.uniq_geo_names))
        self.max_total_instances = np.minimum(len(self.uniq_names), max_instances) if(max_instances != -1) else len(self.uniq_names)
            
    def __len__(self):
        return int(np.ceil(self.max_total_instances/self.batch_size))*self.batch_size
    
    def __getlenbyGeo__(self):
        return self.max_geo_instances

    def __getlenbyGeoRot__(self):
        return self.max_instances_withrot
    
    def __getlenCurDataset__(self):
        return self.max_total_instances
    
    def __getUniqueNames__(self):
        return self.uniq_names
    
    def __getUniqueGeoNames__(self):
        return np.unique(self.uniq_geo_names[0:self.max_total_instances])
        
    def __getitemIncNone__(self, idx): 
        idx = idx % self.max_total_instances
        g_name = self.uniq_names[idx]
        outputs = self._data.__getitem__(idx=int(g_name.split('_')[1]), geo_type=g_name.split('_')[0], perf_metric=self.perf_metric, orien_val=int(g_name.split('_')[2]) if(self.perf_metric in ['U','P']) else None, stats_params = self.perf_cols_sel)
        perf = outputs
        model_input = {'idx': torch.tensor([idx]),
                       'rot': torch.tensor([int(g_name.split('_')[2])]) if(self.perf_metric in ['U','P']) else torch.tensor([0]),
                       'item_n': g_name,
                       'geo_n': '_'.join(g_name.split('_')[0:2]),
                       'geo_code': torch.tensor([0,int(g_name.split('_')[1])]) if(g_name.split('_')[0]=='building') else torch.tensor([1,int(g_name.split('_')[1])])}

        return model_input, perf
        
        
    def __getitem__(self, idx):  
        
        idx = idx % self.max_total_instances
        g_name = self.uniq_names[idx]
        
        outputs = self._data.__getitem__(idx=int(g_name.split('_')[1]), geo_type=g_name.split('_')[0], perf_metric=self.perf_metric, orien_val=int(g_name.split('_')[2]) if(self.perf_metric in ['U','P']) else None, stats_params = self.perf_cols_sel)
        #print('outputs',outputs)
        
        # If any of the outputs is None - find another random sample to replace it with - to avoid dataloader errors
        while(None in outputs):
            idx = np.random.randint(0, self.max_total_instances)
            g_name = self.uniq_names[idx]
            outputs = self._data.__getitem__(idx=int(g_name.split('_')[1]), geo_type=g_name.split('_')[0], perf_metric=self.perf_metric, orien_val=int(g_name.split('_')[2]) if(self.perf_metric in ['U','P']) else None, stats_params = self.perf_cols_sel)
        
        perf = outputs
        model_input = {'idx': torch.tensor([idx]),
                       'rot': torch.tensor([int(g_name.split('_')[2])]) if(self.perf_metric in ['U','P']) else torch.tensor([0]),
                       'item_n': g_name,
                       'geo_n': '_'.join(g_name.split('_')[0:2]),
                       'geo_code': torch.tensor([0,int(g_name.split('_')[1])]) if(g_name.split('_')[0]=='building') else torch.tensor([1,int(g_name.split('_')[1])])}

        return model_input, perf
    
    
# Dataset preparation
class XYZ_SDF_Perf_Agg_CDF(torch.utils.data.Dataset):

    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                subset='0000_0499',
                geo_types=['building','extrBuilding'],
                use_old=False) -> None:
    
        self.directory = directory
        self.dataset_type = dataset_type
        self.subset = subset

        # Define aggregate performance directory
        self.aggPerf_dir = self.directory + '/Performance/aggrMetrics'
        self.parameters_dir = self.directory + '/Parameters'

        # Get all aggregate data for all metrics
        self.pd_permetric = {}
        for perf_metric in ['U','P','SVF']:
            sp_types = ['srf'] if(perf_metric == 'P') else ['srf','grd']
            for s_t in sp_types:
                p_fol = self.aggPerf_dir + '/' + dataset_type + '/' + subset + '/' + 'splits' + '/' + perf_metric + '/' + 'combined' if(use_old) else self.aggPerf_dir + '/' + dataset_type + '/' + subset + '/' + 'splits_upd' + '/' + perf_metric + '/' + 'combined'
                all_res = pd.read_csv(p_fol + '/' + perf_metric + '_mag_' + s_t + '.csv').dropna().drop(columns=['m_split']).dropna()
                all_res.columns = np.concatenate([all_res.columns[0:4], 'all_' + all_res.columns[4:]])
                sdf_res = pd.read_csv(p_fol + '/' + perf_metric + '_mag_bysdf_' + s_t + '.csv').dropna().drop(columns=['m_split']).dropna()
                sdf_res.columns = np.concatenate([sdf_res.columns[0:4], 'sdf_' + sdf_res.columns[4:]])
                quad_res = pd.read_csv(p_fol + '/' + perf_metric + '_mag_byquad_' + s_t + '.csv').dropna().drop(columns=['m_split']).dropna()
                quad_res.columns = np.concatenate([quad_res.columns[0:4], 'quad_' + quad_res.columns[4:]])
                comb_res = pd.merge(pd.merge(all_res,sdf_res, on=['geo_type', 'idx', 'orient','m_type'], how='inner'), quad_res, on=['geo_type', 'idx', 'orient','m_type'], how='inner')
                comb_res = comb_res[comb_res['geo_type'].isin(geo_types)]
                comb_res = comb_res#comb_res[comb_res['orient']==0] if(perf_metric == 'SVF') else comb_res
                self.pd_permetric[perf_metric + '_' + s_t] = comb_res
                          
        # Get the maximum size of the train/val/test splits datasets
        self.split_len = {}
        self.split_ids = {}
        train_test_splits = json.loads(open(self.parameters_dir + '/' + 'traintestParam_5000.json').read())
        for k in train_test_splits.keys():
            subKey = np.array(train_test_splits[k])
            sp_len, sp_id = [], []
            for perf_metric in ['P','U','SVF']:
                sp_types = ['srf'] if(perf_metric == 'P') else ['srf','grd']
                for s_t in sp_types:
                    pd_cur = self.pd_permetric[perf_metric + '_' + s_t]
                    inc_idx = pd_cur[pd_cur['idx'].isin(subKey)]
                    sp_len.append(len(inc_idx))
                    sp_id.append(np.array(inc_idx['idx'].values))
            self.split_len[k] = sp_len
            self.split_ids[k] = sp_id

        super().__init__()
    
    def __len__(self):
        l_U = [len(self.pd_permetric['U_' + s_t]) for s_t in ['srf','grd']]
        l_P = [len(self.pd_permetric['P_' + s_t]) for s_t in ['srf']]
        l_SVF = [len(self.pd_permetric['SVF_' + s_t]) for s_t in ['srf','grd']]
        return np.sort([j for i in [l_U, l_P, l_SVF] for j in i])[::-1][0]
    
    def __splitlen__(self):
        return self.split_len
    
    def __splitids__(self):
        return self.split_ids

    def __getitem__(self, idx, geo_type, perf_metric, orien_val, fl_type='all', sp_type=['srf','grd']):
        
        # Define random seed as time
        np.random.seed(int(1000 * time.time()) % 2**32)
        
        # Define the cur_directories
        idx_s = int(idx/500)
        self.set_fromIDX = "{:04d}".format(idx_s*500) + '_' + "{:04d}".format(((idx_s+1)*500)-1)
        self.p_dir = self.aggPerf_dir + '/' + self.dataset_type + '/' + self.set_fromIDX + '/' + perf_metric
        
        # Check if performance data exist for the selected performance metric
        vals = []
        for s_t in sp_type:
            c_params = [c for c in self.pd_permetric[perf_metric+ '_' + s_t].columns if(fl_type in c)]
            cols_to_keep = [x for xs in [['geo_type','idx','orient'], c_params] for x in xs] 
            stats_file = self.pd_permetric[perf_metric+ '_' + s_t][cols_to_keep]
            stats_file = stats_file[(stats_file['geo_type']==geo_type) & (stats_file['idx']==idx) & (stats_file['orient']==orien_val)]
            if(len(stats_file)>0):
                vals.append(stats_file[c_params].values) 
        vals = np.concatenate(vals)#.flatten()
        
        if(len(vals)>0):
            perf = vals
        else:
            perf = None
        
        return perf
    
class XYZ_SDF_Perf_Dataset_Agg_CDF(torch.utils.data.Dataset):
    """
    A subclass that considers the train/val/test splits and maximum number of instances.

    """
    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                subset='0000_0499',
                geo_types=['building','extrBuilding'],
                perf_metric = 'U',
                fl_type = 'all',
                sp_type=['srf','grd'],
                quad_half = True,
                split_type='train', # or val or test
                max_instances = -1, #-1 to use full dataset
                geo_dataset_avail = -1,
                use_old=False) -> None:

        # Generate dataset
        self._data = XYZ_SDF_Perf_Agg_CDF(directory=directory,
                                dataset_type=dataset_type,
                                subset=subset,
                                geo_types=geo_types)
        self.perf_metric = perf_metric
        self.fl_type = fl_type
        self.sp_type = sp_type
        self.quad_half = quad_half
        
        # Check if performance files exist for the selected performance metric  
        perf_data = [self._data.pd_permetric[perf_metric + '_' + s_t] for s_t in sp_type]
        perf_cols = [c for c in perf_data[0].columns if(fl_type in c)]
        perf_cols_sel = perf_cols.copy()
        perf_cols_sel.append('idx')
        perf_cols_sel.append('geo_type')
        perf_cols_sel.append('orient')
        perf_data = [p[perf_cols_sel] for p in perf_data]
        
        # Load the indices for the train/val/test sets ONLY
        ids_inc = np.array(json.loads(open(self._data.parameters_dir + '/' + 'traintestParam_5000.json').read())[split_type])
        
        # Set variable sets
        metrics_directory = directory + '/Performance/aggrMetrics' + '/' + dataset_type + '/' + subset + '/' + 'splits_upd' if(not use_old) else directory + '/Performance/aggrMetrics' + '/' + dataset_type + '/' + subset + '/' + 'splits'
        self.set_names = np.load(metrics_directory + '/' + perf_metric + '/' + 'quad_names.npy')[4:] if((fl_type=='quad') and (quad_half)) else np.load(metrics_directory + '/' + perf_metric + '/' + 'quad_names.npy')[0:4] if((fl_type=='quad') and (not quad_half)) else np.load(metrics_directory + '/' + perf_metric + '/' + 'sdf_ranges.npy') if(fl_type=='sdf') else ['all']
        self.set_name_vals = self.set_names if(fl_type=='sdf') else [list(self.set_names).index(n)/len(self.set_names) for n in ['N','E','S','W']] if((fl_type=='quad') and (quad_half)) else [list(self.set_names).index(n)/len(self.set_names) for n in ['NE','SE','SW','NW']] if((fl_type=='quad') and (not quad_half)) else np.arange(len(self.set_names))
        self.sp_vals = np.load(metrics_directory + '/' + perf_metric + '/' + 'sp_vals.npy')
        self.min_max = np.load(metrics_directory + '/' + perf_metric + '/' + 'minmax.npy')

        # Filter the indices based on the type of dataset
        self.ids_exist = [p[p['idx'].isin(ids_inc)].copy() for p in perf_data]
        self.ids_exist_merged = pd.merge(self.ids_exist[0], self.ids_exist[1], on=['geo_type', 'idx', 'orient'], how='inner') if(len(sp_type)>1) else self.ids_exist[0]
        
        # Filter rows with potential errors
        vals_re = self.ids_exist_merged.drop(columns=['idx', 'geo_type', 'orient']).values.reshape(len(self.ids_exist_merged),-1,100)
        #filter_err = ((vals_re[:,:,-1]==1.0).astype(int).sum(-1)==vals_re.shape[-2])
        #self.ids_exist_merged = self.ids_exist_merged[filter_err]
        
        # Define the maximum instances based on the filtered set (of available geometry and performance)
        g_name = [self.ids_exist_merged['geo_type'].iloc[i] + '_' + str(self.ids_exist_merged['idx'].iloc[i]) for i in range(len(self.ids_exist_merged))]
        self.ids_exist_merged['g_name']=g_name
        g_name_rot = [self.ids_exist_merged['geo_type'].iloc[i] + '_' + str(self.ids_exist_merged['idx'].iloc[i]) + '_' + str(self.ids_exist_merged['orient'].iloc[i]) for i in range(len(self.ids_exist_merged))]
        self.ids_exist_merged['g_name_rot']=g_name_rot
        if(geo_dataset_avail != -1):
            if((geo_dataset_avail != -1)and(len(geo_dataset_avail[0].split('_'))==2)):
                self.ids_exist_merged = self.ids_exist_merged[self.ids_exist_merged['g_name'].isin(geo_dataset_avail)]
            if((geo_dataset_avail != -1)and(len(geo_dataset_avail[0].split('_'))==3)):
                self.ids_exist_merged = self.ids_exist_merged[self.ids_exist_merged['g_name_rot'].isin(geo_dataset_avail)]

        self.uniq_names = self.ids_exist_merged['g_name_rot'].values
        self.uniq_geo_names = self.ids_exist_merged['g_name'].values
        self.max_instances_withrot = len(self.ids_exist_merged)
        self.max_geo_instances = len(np.unique(self.uniq_geo_names))
        self.max_total_instances = np.minimum(len(self.uniq_names), max_instances) if(max_instances != -1) else len(self.uniq_names)
        
    def __getFullDataset__(self, shuffle=True, return_bykey=False):
        np.random.seed(int(1000 * time.time()) % 2**32)
        dataset = self.ids_exist_merged.copy()
        perf = torch.tensor(dataset.drop(columns=['g_name','g_name_rot','idx', 'geo_type', 'orient']).values.reshape(len(dataset),len(self.sp_type),-1))
        if(self.fl_type == 'quad'):
            perf = perf[:,:,len(self.set_names)*len(self.sp_vals):] if(self.quad_half) else perf[:,:,:len(self.set_names)*len(self.sp_vals)] 
        geo_ct = perf.shape[0]

        rep_val = 1
        if(perf.shape[-1]!=100):
            perf = perf.reshape(perf.shape[0],len(self.sp_type),-1,len(self.sp_vals)).permute(2,0,1,3)
            rep_val = perf.shape[0]
            perf = perf.reshape(perf.shape[0]*perf.shape[1],perf.shape[2],perf.shape[3])
        if(shuffle):
            rand_idx = np.arange(perf.shape[0])
            np.random.shuffle(rand_idx)
        else:
            rand_idx = np.arange(perf.shape[0])
        model_input = {'idx': torch.unsqueeze(torch.tile(torch.tensor(dataset['idx'].values),(rep_val,1)).flatten(),-1)[rand_idx],
                       'rot': torch.unsqueeze(torch.tile(torch.tensor(dataset['orient'].values),(rep_val,1)).flatten(), -1)[rand_idx],
                       'ref_val': None if(rep_val==1) else torch.unsqueeze(torch.tensor(np.repeat(self.set_name_vals,geo_ct)),-1)[rand_idx],
                       'item_n': list(np.repeat(dataset['g_name_rot'].values,rep_val).reshape(geo_ct, rep_val).T.flatten()[rand_idx]),
                       'geo_n': list(np.repeat(dataset['g_name'].values,rep_val).reshape(geo_ct, rep_val).T.flatten()[rand_idx]),
                       'set_names': list(self.set_names),
                       'sp_vals': torch.tensor(self.sp_vals),
                       'min_max': torch.tensor(self.min_max),
                       'geo_code': torch.tile(torch.tensor([[0,int(g.split('_')[1])] if(g.split('_')[0]=='building') else [1,int(g.split('_')[1])] for g in dataset['g_name'].values]),(rep_val,1,1)).reshape(geo_ct*rep_val,2)[rand_idx]}
        perf = perf[rand_idx]
        return model_input, perf
            
    def __len__(self):
        return self.max_total_instances
    def __getlenbyGeo__(self):
        return self.max_geo_instances

    def __getlenbyGeoRot__(self):
        return self.max_instances_withrot
    
    def __getlenCurDataset__(self):
        return self.max_total_instances
    
    def __getUniqueNames__(self):
        return self.uniq_names
    
    def __getUniqueGeoNames__(self):
        return np.unique(self.uniq_geo_names[0:self.max_total_instances])
    
    
            
# 3D point rotation - anti-clockwise rotation
# https://python.plainenglish.io/3d-affine-transformation-matrices-implementation-with-numpy-57f92058403c

def rotation_matrix_x(alpha_degree):
    alpha_radian = np.deg2rad(alpha_degree)
    rotation_alpha = [
        [1, 0, 0, 0],
        [0, np.cos(alpha_radian), -np.sin(alpha_radian), 0],
        [0, np.sin(alpha_radian), np.cos(alpha_radian), 0],
        [0, 0, 0, 1]
    ]
    return np.array(rotation_alpha)

def rotation_matrix_y(beta_degree):
    beta_radian = np.deg2rad(beta_degree)
    rotation_beta = [
        [np.cos(beta_radian), 0, np.sin(beta_radian), 0],
        [0, 1, 0, 0],
        [-np.sin(beta_radian), 0, np.cos(beta_radian), 0],
        [0, 0, 0, 1]
    ]
    return np.array(rotation_beta)

def torch_rotation_matrix_z(angle_deg: float, device=None, dtype=torch.float32):
    """Returns a 3x3 Z-axis rotation matrix for a given angle in degrees."""
    theta = torch.deg2rad(torch.tensor(angle_deg, device=device, dtype=dtype))
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    R = torch.tensor([
        [cos_t, -sin_t, 0.0],
        [sin_t,  cos_t, 0.0],
        [0.0,    0.0,  1.0]
    ], device=device, dtype=dtype)
    return R

def torch_rotation_matrix_y(angle_deg: float, device=None, dtype=torch.float32):
    """Returns a 3x3 Y-axis rotation matrix for a given angle in degrees."""
    theta = torch.deg2rad(torch.tensor(angle_deg, device=device, dtype=dtype))
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)

    R = torch.tensor([
        [cos_t, 0.0, sin_t],
        [0.0,   1.0, 0.0],
        [-sin_t, 0.0, cos_t]
    ], device=device, dtype=dtype)
    return R

def rotation_matrix_z(gamma_degree):
    gamma_radian = np.deg2rad(gamma_degree)
    rotation_gamma = [
        [np.cos(gamma_radian), -np.sin(gamma_radian), 0, 0],
        [np.sin(gamma_radian), np.cos(gamma_radian), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    return np.array(rotation_gamma)

def rotation_matrix(alpha, beta, gamma):
    return rotation_matrix_x(alpha) @ rotation_matrix_y(beta) @ rotation_matrix_z(gamma)

# Normalize and unnormalize points
def normalize_points(points, bbox):
    norm_points = (points-bbox[:,0])/(bbox[:,1]-bbox[:,0])
    return norm_points

def unnormalize_points(points, bbox):
    unnorm_points = ((points*(bbox[:,1]-bbox[:,0])))+bbox[:,0]
    return unnorm_points

def combineSort(to_sort_lists):   #Sort by all buildings then extrBuildings (and by number)
    com_list = np.array([x for row in to_sort_lists for x in row])
    c_bldg = np.array([c for c in com_list if('building' in c)])
    c_extrB = np.array([c for c in com_list if('extrBuilding' in c)])
    com_list = [cf[np.argsort([int(c.split('_')[1]) for c in cf])] for cf in [c_bldg,c_extrB]]
    com_list = np.array([x for row in com_list for x in row])
    return com_list


# Dataset preparation
class XYZ_SDF_ArbitraryD(torch.utils.data.Dataset):

    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                category = 'realbuildings',
                subsets=['Boston','London'],
                sampling_types=['rejection','surface','uniform','zeroSurface'],
                sampling_distr=[1,1,0.1,1],
                distr = [0.4,0.4,0.2],
                num_points = 50000,
                s_range = '0_1', #or N1_1
                ) -> None:
    
        self.dataset_type = dataset_type
        self.category = category
        self.subsets = subsets
        self.sampling_types = sampling_types
        self.sampling_distr = sampling_distr
        self.distr = np.array(distr)
        self.num_points = num_points
        self.s_range = s_range
        
        st_base = time.time()

        # Define geometry and sampling directories
        self.sampling_dir = directory + '/Samples'
        self.geometry_dir = directory + '/Geometry'
        self.parameters_dir = directory + '/Parameters'
        
        # Define combined sampling name
        if(len(self.sampling_types)!= 1):
            self.sampling_name = '_'.join([self.sampling_types[i][0:3] + '_' + str(int(self.sampling_distr[i]*100)) for i in range(len(self.sampling_types))])
            sampling_exists = all([os.path.exists(self.sampling_dir + '/' + self.dataset_type + '/' + self.category + '/' + s + '/combined/' + self.sampling_name) for s in self.subsets])
        else:
            self.sampling_name = self.sampling_types[0]
            sampling_exists = all([os.path.exists(self.sampling_dir + '/' + self.dataset_type + '/' + self.category + '/' + s + '/' + self.sampling_name) for s in self.subsets])
            
        if(not sampling_exists):
            print('The sampling ' + self.sampling_name + ' does not exist for all subsets.')
            
        # Load dataset stats
        data_stats_df = pd.read_csv(directory + '/Parameters/' + dataset_type + '_'+category+'_datasetStats.csv')
        sub_df = data_stats_df[(data_stats_df['subset'].isin(subsets))]
        #print(sub_df.columns)
        
        # Define dataset subset - accounting for existing geometries and corresponding sampling
        g_exists = np.array(sub_df['geometry'] == 1)
        if(len(self.sampling_types)!= 1):
            c_exists = np.array(sub_df[self.sampling_name] == 1) if(self.sampling_name in sub_df.columns) else np.repeat(0,len(sub_df))
            self.sub_df_exist = sub_df[(g_exists.astype(int) + c_exists.astype(int)) == 2][['subset','item_n']]
        else:
            s_exists = (np.sum(np.array([(sub_df[s]==1) for s in sampling_types]).T.astype(int), axis=1)==len(sampling_types))
            self.sub_df_exist = sub_df[(g_exists.astype(int) + s_exists.astype(int)) == 2][['subset','item_n']] 

        super().__init__()
    
    def __len__(self):
        return len(self.sub_df_exist)

    def __getitem__(self, subset, geo_name):

        # Define random seed as time
        np.random.seed(int(1000 * time.time()) % 2**32)
        #np.random.seed(0)
        
        # Define the cur_directories
        g_dir = self.geometry_dir + '/' + self.dataset_type +'/' + self.category + '/' + subset 
        #print('g_dir',g_dir)
        s_dir = self.sampling_dir + '/' + self.dataset_type + '/' + self.category + '/' + subset 
        #print('s_dir',s_dir)
        
        # Check if geometry, all sampling types and combined sampling exist
        geoPar_exists = (len(self.sub_df_exist[(self.sub_df_exist['item_n']==geo_name)])>0)

        st_load = time.time()
        if(geoPar_exists):
            g_name = geo_name
            
            # Read the point combined sampling files from the folder
            if(len(self.sampling_types)!= 1):
                try:
                    d_sdf_pos_cat = np.load(s_dir + '/' + 'combined' + '/' + self.sampling_name + '/' + g_name + '_pos.npy') #, mmap_mode='r'
                    d_sdf_pos_cat = d_sdf_pos_cat[np.random.randint(d_sdf_pos_cat.shape[0], size=int(self.num_points*self.distr[0]))].astype(float)
                    #print('d_sdf_pos_cat',d_sdf_pos_cat.shape)
                    d_sdf_neg_cat = np.load(s_dir + '/' + 'combined' + '/' + self.sampling_name + '/' + g_name + '_neg.npy') #, mmap_mode='r'
                    d_sdf_neg_cat = d_sdf_neg_cat[np.random.randint(d_sdf_neg_cat.shape[0], size=int(self.num_points*self.distr[1]))].astype(float)
                    #print('d_sdf_neg_cat',d_sdf_neg_cat.shape)
                    d_sdf_zero_cat = np.load(s_dir + '/' + 'combined' + '/' + self.sampling_name + '/' + g_name + '_zero.npy') #, mmap_mode='r'
                    d_sdf_zero_cat = d_sdf_zero_cat[np.random.randint(d_sdf_zero_cat.shape[0], size=int(self.num_points-(int(self.num_points*self.distr[0]))-(int(self.num_points*self.distr[1]))))].astype(float)
                    #print('d_sdf_zero_cat',d_sdf_zero_cat.shape)
                except:
                    d_sdf_pos_cat = None
                    d_sdf_neg_cat = None
                    d_sdf_zero_cat = None
            else:
                d_sdf_pos_cat = None
                d_sdf_neg_cat = None
                d_sdf_zero_cat = None
                if(self.sampling_name != 'zeroSurface'):
                    d_sdf_pos_cat = np.load(s_dir + '/' + self.sampling_name + '/' + g_name + '_' + self.sampling_name + '.npz', allow_pickle=True)['pos']
                    #print('d_sdf_pos_cat',d_sdf_pos_cat.shape)
                    d_sdf_neg_cat = np.load(s_dir + '/' + self.sampling_name + '/' + g_name + '_' + self.sampling_name + '.npz', allow_pickle=True)['neg']
                    #print('d_sdf_neg_cat',d_sdf_neg_cat.shape)
                else:
                    d_sdf_zero_cat = np.load(s_dir + '/' + self.sampling_name + '/' + g_name + '_' + self.sampling_name + '.npz', allow_pickle=True)['alls']
                    #print('d_sdf_zero_cat',d_sdf_zero_cat.shape)
            #print('st_load',time.time()-st_load)
                    
            # Combine in array
            if((d_sdf_pos_cat is not None) and (d_sdf_neg_cat is not None)):
                subset_ar = [d_sdf_pos_cat,d_sdf_neg_cat,d_sdf_zero_cat]
                z_n = True if(d_sdf_zero_cat is not None) else False
                del d_sdf_pos_cat, d_sdf_neg_cat, d_sdf_zero_cat

                #torch.from_numpy(.float()

                st_zero = time.time()
                if(z_n):
                    # Create equivalent arrays with zeros for SDF at surface, and zeros for normals outside surface
                    zero_np = np.array([0])
                    sdf_subset_ar_zero = np.concatenate((subset_ar[2][:,0:3].T,zero_np.repeat(len(subset_ar[2])).reshape(len(subset_ar[2]),1).T)).T
                    nor_subset_ar_posneg = [np.concatenate((subset_ar[i][:,0:3].T,zero_np.repeat(len(subset_ar[i])*3).reshape(len(subset_ar[i]),3).T)).T for i in range(2)]

                    # Concatenate positive and negative instances
                    sdf_subset = np.concatenate([subset_ar[0], subset_ar[1], sdf_subset_ar_zero])
                    nor_subset = np.concatenate([nor_subset_ar_posneg[0], nor_subset_ar_posneg[1], subset_ar[2]])
                    del subset_ar, nor_subset_ar_posneg, sdf_subset_ar_zero

                    # Shuffle the order of positive/negative/zero points
                    rand_pts_idx = np.random.permutation(len(sdf_subset))
                    sdf_subset = torch.from_numpy(sdf_subset[rand_pts_idx]).float()
                    nor_subset = torch.from_numpy(nor_subset[rand_pts_idx]).float()

                    # Separate coordinates and sdf values
                    xyz = sdf_subset[:,:3] if(self.s_range == '0_1') else ((sdf_subset[:,:3]*2)-1)
                    sdf = sdf_subset[:,3:4]
                    nor = nor_subset[:,3:6]
                else:
                    # Concatenate the positive and negative sets
                    sdf_subset = np.concatenate(subset_ar)
                    del subset_ar

                    # Shuffle the order of positive/negative points
                    rand_pts_idx = np.random.permutation(len(sdf_subset))
                    sdf_subset = torch.from_numpy(sdf_subset[rand_pts_idx]).float()

                    # Separate coordinates and sdf values
                    xyz = sdf_subset[:,:3] if(self.s_range == '0_1') else ((sdf_subset[:,:3]*2)-1)
                    sdf = sdf_subset[:,3:4]
                    nor = None
                #print('st_zero',time.time()-st_zero)
            else:
                xyz = None
                sdf = None
                nor = None
                
                
        else:
            print('The geometry ' + geo_type + '_' + str(idx) + ' does not exist in the dataset or does not have all sampling types.')
            xyz = None
            sdf = None
            nor = None

        # return model_input dictionary and output
        return xyz, sdf, nor



class XYZ_SDF_ArbitraryD_Dataset(torch.utils.data.Dataset):
    """
    A subclass that considers the train/val/test splits and maximum number of instances.

    """
    def __init__(self,
                directory='BuildingUrbanSDF',
                dataset_type='01_Buildings',
                category = 'realbuildings',
                subsets=['Boston','London'],
                sampling_types=['rejection','surface','uniform','zeroSurface'],
                sampling_distr=[1,1,0.1,1],
                distr = [0.4,0.4,0.2],
                num_points = 50000,
                s_range = '0_1', # or N1_1
                max_instances = -1, #-1 to use full dataset
                batch_size = 32,
                apply_rot=False,
                geo_dataset_avail = -1,
                ) -> None:

        # Generate dataset
        self._data = XYZ_SDF_ArbitraryD(directory=directory,
                          dataset_type=dataset_type,
                          category=category,
                          subsets=subsets,
                          sampling_types=sampling_types,
                          sampling_distr=sampling_distr,
                          distr=distr,
                          num_points=num_points,
                          s_range=s_range)
        self.batch_size = batch_size
        self.apply_rot = apply_rot

        self.subsets = subsets
        self._dataset = self._data.sub_df_exist
        self._dataset = self._dataset[self._dataset['item_n'].isin(geo_dataset_avail)] if(geo_dataset_avail != -1) else self._dataset
        self.set_names = self._dataset['item_n'].values
        self.set_subsets = self._dataset['subset'].values
        self.max_instances =  np.min([max_instances, len(self._dataset)])  if (max_instances != -1) else len(self._dataset)

    def __len__(self):
        return int(np.ceil(self.max_instances/self.batch_size))*self.batch_size
    
    def __getlenCurDataset__(self):
        return self.max_instances
    
    def __getTotalInstances__(self):
        return len(self._dataset)
    
    def __getUniqueNames__(self):
        return self.set_names[0:self.max_instances]

    def __getUnrotatedItem__(self, idx, returnNone = False):
        idx = idx%self.max_instances
        g_name = self.set_names[idx]
        subset = self.set_subsets[idx]
        outputs = self._data.__getitem__(subset,g_name)
        
        if((None in outputs) and (not returnNone)):
            # If any of the outputs is None - find another random sample to replace it with - to avoid dataloader errors
            while(None in outputs):
                idx = np.random.randint(0, self.max_instances)
                g_name = self.set_names[idx]
                outputs = self._data.__getitem__(g_name)
        
        if(not(None in outputs)):
            xyz, sdf, nor = outputs
            model_input = {'xyz': xyz,
                            'idx': torch.tensor([idx]),
                            'rot': torch.tensor([0]),
                            'geo_code': torch.tensor([self.subsets.index(subset),int(g_name.split('_')[1])])}

            return model_input, sdf, nor, g_name 
        else:
            return None, None, None, None 
    
    def __getRotatedItem__(self, idx, override_rot=None, returnNone = False):
        idx = idx%self.max_instances
        g_name = self.set_names[idx]
        subset = self.set_subsets[idx]
        outputs = self._data.__getitem__(subset,g_name)
        
        if((None in outputs) and (not returnNone)):
            # If any of the outputs is None - find another random sample to replace it with - to avoid dataloader errors
            while(None in outputs):
                idx = np.random.randint(0, self.max_instances)
                g_name = self.set_names[idx]
                real_idx = int(g_name.split('_')[1])
                geo_type = g_name.split('_')[0]
                outputs = self._data.__getitem__(g_name)
        
        if(not(None in outputs)):
            xyz, sdf, nor = outputs
            rot_val = np.random.randint(0,360) if(override_rot is None) else override_rot
            xyz_rot = np.concatenate([(xyz.detach().numpy() + [-0.5,-0.5,-0.5]),np.ones((xyz.detach().numpy().shape[0],1))],1)
            xyz_rot = np.dot(rotation_matrix_y(rot_val),xyz_rot.T).T[:,:3].round(6) + [0.5,0.5,0.5]

            model_input = {'xyz': torch.from_numpy(xyz_rot).float(), 
                           'idx': torch.tensor([idx]), 
                           'rot': torch.tensor([rot_val]), 
                           'geo_code': torch.tensor([self.subsets.index(subset),int(g_name.split('_')[1])])}

            return model_input, sdf, nor, g_name    
        else:
            return None, None, None, None
    
    def __getitem__(self, idx, override_rot=None):
        idx = idx%self.max_instances
        model_input, sdf, nor, g_name = self.__getUnrotatedItem__(idx) if(not self.apply_rot) else self.__getRotatedItem__(idx, override_rot, returnNone = False)

        # yield model_input dictionary and output
        return model_input, sdf, nor, g_name   
    
    def __getIncNoneItem__(self, idx, override_rot=None):
        idx = idx%self.max_instances
        model_input, sdf, nor, g_name = self.__getUnrotatedItem__(idx) if(not self.apply_rot) else self.__getRotatedItem__(idx, override_rot, returnNone = True)

        # yield model_input dictionary and output
        return model_input, sdf, nor, g_name    
    
    # def __iter__(self):
    #     # Iterate through full dataset
    #     for i in range(int(np.ceil(self.max_instances/self.batch_size))*self.batch_size):
    #         outputs = self.__getitem__(i)
    #         yield outputs