import os
import json
from json import JSONEncoder
import numpy as np
import trimesh
import igl
import pyDOE as doe
from scipy.stats import gaussian_kde
import plotly.graph_objects as go
import argparse
import sys

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

# Get points density - from https://github.com/u2ni/ICML2021/blob/main/neuralImplicitTools/src/geometry.py
def density(U):
    c = gaussian_kde(np.transpose(U))(np.transpose(U))
    return c

# Bounding box calcs
def get_buffered_bbox(mesh, bbox_relative_buffer=0.05):
    bbox = np.array([[np.min(coords), np.max(coords)] for coords in mesh.vertices.T])
    bbox_extent = bbox[:, 1] - bbox[:, 0]
    buffer = bbox_extent * bbox_relative_buffer
    bbox[:, 0] -= buffer
    bbox[:, 1] += buffer
    return bbox

# Based on 4.453x - https://colab.research.google.com/drive/17skLORoyXW7EeTv821eZjk9LLUy9Yqwe
# Reduce sample size with the a larger likelihood of points close to the surface 
def accept_sample(distance, beta=5):
    acceptance_probability = np.exp(
        -beta * np.abs(distance)
    )  # the closer the point, the likelier it is to be accepted
    return np.random.binomial(1, acceptance_probability).astype(
        bool
    )  # this is a biased coin toss to figure out whether to accept the sample or not


# Get samples from mesh
# Sampling types include 'surface', 'rejection', 'uniform'
def get_sdf_samples(n_samples, mesh, bbox, sampling_type, std=2, beta=30):
    bbox_extent = bbox[:, 1] - bbox[:, 0]
    if(sampling_type != 'surface'):
        # Get samples in unit cube
        normalized_samples = doe.lhs(3, n_samples)
        # Scale unit cube samples to bbox
        samples = normalized_samples * (bbox_extent) + bbox[:, 0]
    else:
        # Get evenly distributed surface samples
        samples = trimesh.sample.sample_surface_even(centred_mesh, n_samples, radius=None)[0]
        # Add randomly distributed to samples
        samples_rand_norm = np.random.normal(loc = samples,scale = std)  
        samples = samples_rand_norm
        normalized_samples = ((samples - bbox[:,0])/(bbox_extent)) 
    # Compute distances to mesh
    distances = igl.signed_distance(samples, mesh.vertices, mesh.faces)[0]
    # Normalize distances to allow for consistent filtering irrespectively of object scale
    normalized_distances = distances / np.min(bbox_extent)
    if(sampling_type == 'rejection'):
        # Get a boolean mask of which samples to accept using accept_sample
        sample_mask = accept_sample(normalized_distances, beta=beta)
        normalized_samples = normalized_samples[sample_mask]
        normalized_distances = normalized_distances[sample_mask]
    # Concatenate coordinates and distances
    allv = np.concatenate([normalized_samples.T,np.expand_dims(normalized_distances, axis=1).T]).T
    # Separate the postive and negative sdfs
    pos = allv[allv[:,3]>0]
    neg = allv[allv[:,3]<0]
    # Return a tuple of with the coordinates and distances, separated by positive, negative and combined
    return pos, neg, allv