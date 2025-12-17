import torch
import torch.utils.data as data_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.autograd import Variable
from torch.utils.data import IterableDataset
from torch.func import functional_call
#from torch.nn.utils.stateless import functional_call
from building_sdf.learning.network_modules import *

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")
    
    
#### PERF

class hgtMapPerformance(nn.Module):
    def __init__(self, 
                 perf_param, 
                ):
        super().__init__()
        #########
        
        input_shape = (2,perf_param['feature_dim'],perf_param['grid_size'],perf_param['grid_size'])
        #print('input_shape',input_shape)
        self.encoder_type = perf_param['encoder_type']
        params = perf_param['encoder_params']

        if(self.encoder_type == 'unet'):
            norm_layer = get_norm_layer(norm_type=params['norm_layer_type'])
            self.encoder_model =  UnetGenerator(params['input_nc'], params['output_nc'], int(np.log(perf_param['grid_size'])/np.log(2)), params['ngf'], norm_layer=norm_layer, use_dropout=params['use_dropout'], outermost_nonlinearity=params['outermost_nonlinearity'])
        else:
            print('The encoder model does not exist: ' + self.encoder_type)
        output_shape = self.encoder_model(torch.rand(input_shape)).shape
        #print('output_shape',output_shape)

        #print('The encoder model '+self.encoder_type+', num_parameters: ', getModelParametersCount(self.encoder_model))

        
    def forward(self, inputs):
        #########
        
        h_shape = inputs['hgtmap'].shape
        hgtmap = inputs['hgtmap'].reshape((h_shape[0],int(np.sqrt(h_shape[1])),int(np.sqrt(h_shape[1])),-1)).permute(0,3,1,2)

        # Forward pass through the encoder
        outputs = self.encoder_model(hgtmap)
        outputs = outputs.permute(0,2,3,1).reshape(h_shape[0],h_shape[1],-1)
        #print('outputs',outputs.shape)

        return outputs, None
    
    
def loadModel(model_dir, model_type = 'train', load_type = 'current', epoch = None, device = 'cpu'):
    
    perf_param = loadModelParam(model_dir, param_ar = ['perf_param'])[0]
    model = hgtMapPerformance(perf_param)
    
    # Set model directories
    prefix_model = 'perf_'
    model_dir = model_dir + '/test/' if(model_type != 'train') else model_dir
    checkpoints_dir = model_dir + '/' + 'checkpoints'

    # Load the models
    model_name = 'perf_model'
    ext_str = {'current':'', 'final':'', 'epoch':'_%04d' % epoch}
    fol_base = {'current':model_dir, 'final':model_dir, 'epoch':checkpoints_dir}
    
    file_to_load = os.path.join(fol_base[load_type], model_name+'_'+load_type+ext_str[load_type]+'.pth')
    try:
        if(device == 'cpu'):
            model.load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(file_to_load))
    except:
        model = torch.nn.DataParallel(model)
        if(device == 'cpu'):
            model.load_state_dict(torch.load(file_to_load, map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(file_to_load))

    return model