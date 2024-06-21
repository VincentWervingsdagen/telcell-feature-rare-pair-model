from typing import Tuple, Optional, Mapping

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import gc

import lir

import telcell.data.construct_markov_chains as MC
from telcell.data.models import Track
from telcell.models import Model

class MarkovChain(Model):
    bounding_box: list
    prior_chain: pd.DataFrame
    df_Markov_chains: pd.DataFrame()
    data: pd.DataFrame
    distance: str
    state_space_level: str
    state_space: [str]
    prior_type: str
    markov_type: str
    loops_allowed: bool
    kde_calibrator: lir.KDECalibrator()


    def __init__(
            self,
            training_set,
            cell_file,
            bounding_box,
            state_space='Omega',
            state_space_level='postal2',
            distance='frobenius',
            antenna_type = 'LTE',
            prior_type = 'uniform',
            markov_type = 'discrete',
            loops_allowed = True
    ) -> None:
        # Set global parameters
        self.cell_file = cell_file
        self.bounding_box = bounding_box
        self.distance = distance
        self.state_space_level=state_space_level
        self.markov_type=markov_type
        self.loops_allowed = loops_allowed

        # Transform the data to a dataframe with owner, device, timestamp, postal_code
        self.data = MC.transform_data(training_set,state_space_level)
        # Construct the state space.
        self.construct_state_space(state_space,state_space_level,antenna_type)
        # Construct the prior matrix, so that we can use it for estimating our movement matrices.
        self.construct_prior(prior_type)

        # Construct the Markov chains for each device.
        list_Markov_chains = []
        list_devices = []
        list_counts = []
        grouped = self.data.groupby(['device'])

        for device, track in grouped:
            list_Markov_chains.append(self.construct_markov_chain(track['cellinfo.postal_code'],markov_type,loops_allowed))
            list_devices.append(device[0])
            observed_states,count_values = np.unique(track['cellinfo.postal_code'],return_counts=True)
            count_vector = np.zeros(self.number_of_states)
            indices = np.where(np.in1d(self.state_space, observed_states))[0]
            count_vector[indices] = count_values
            list_counts.append(count_vector)

        self.df_Markov_chains = pd.DataFrame(data={'markov_chains':list_Markov_chains,'count_data':list_counts},index=list_devices)

        df_reference = self.reference_set(list_devices,distance)

        self.kde_calibrator = lir.KDECalibrator(bandwidth='silverman')
        self.kde_calibrator.fit(np.array(df_reference['score']), np.array(df_reference['hypothesis']))

        self.kde_calibrator = lir.ELUBbounder(self.kde_calibrator)
        self.kde_calibrator.fit(np.array(df_reference['score']), np.array(df_reference['hypothesis']))



    def construct_state_space(self,state_space,state_space_level,antenna_type):
        # Construct the state space
        if state_space == 'Omega':  # Select all the antennas/postal/postal3 codes in the bounding box
            self.state_space = MC.state_space_Omega(self.cell_file, self.bounding_box,antenna_type,
                                                    state_space_level)
        elif state_space == 'observations':  # Select the antennas/postal/postal3 codes that were observed by either of the phones.
            self.state_space = MC.state_space_observations(self.data)
        else:
            raise ValueError('The specified state space is not implemented. Please use Omega or observations.')
        self.number_of_states = len(self.state_space)

    def construct_prior(self,prior_type):
        # Construct the prior
        if prior_type == 'jeffrey':  # Returns a nxn matrix with each value 1/n
            self.prior_chain = MC.jeffrey_prior(number_of_states=self.number_of_states, states=self.state_space)
        elif prior_type == 'all_ones':  # Returns a nxn matrix with each value 1
            self.prior_chain = MC.all_ones_prior(number_of_states=self.number_of_states, states=self.state_space)
        elif prior_type == 'zero':  # Returns a nxn matrix based on the distance between the antennas/postal/postal3 codes.
            self.prior_chain = MC.zero_prior(states=self.state_space)
        elif prior_type == 'distance':  # Returns a nxn matrix based on the distance between the antennas/postal/postal3 codes.
            self.prior_chain = MC.distance_prior(states=self.state_space, distance_matrix_file=self.distance_matrix_file,
                                                 bounding_box=self.bounding_box)
        elif prior_type == 'population':  # Not implemented.
            self.prior_chain = MC.population_prior()
        else:
            raise ValueError(
                'The specified prior movement distribution is not implemented. Please use jeffrey, all_ones, zero, distance or population.')

    def construct_markov_chain(self,track,markov_type,loops_allowed):
        # Construct the markov chains
        if markov_type == 'discrete':
            return MC.discrete_markov_chain(track=track,prior=self.prior_chain, states=self.state_space,
                                                                      loops_allowed=loops_allowed)
        elif markov_type == 'continuous':
            MC.continuous_markov_chain()
        else:
            raise ValueError('The specified Markov chain type is not implemented. Please use discrete or continuous.')

    def calculate_score(self,distance,matrix1,matrix2,count_data):
        # Calculate the distance
        if distance == 'cut_distance':
            return MC.genetic_cut_distance(matrix_normal=matrix1,matrix_burner=matrix2)
        elif distance == 'freq_distance':
            return MC.frequent_transition_distance(matrix_normal=matrix1, matrix_burner=matrix2)
        elif distance == 'frobenius':
            return MC.frobenius_norm(matrix_normal=matrix1,matrix_burner=matrix2)
        elif distance == 'trace':
            return MC.trace_norm(matrix_normal=matrix1,matrix_burner=matrix2)
        elif distance == 'important_cut_distance_5':
            return MC.important_states_cut_distance_5(matrix_normal=matrix1,matrix_burner=matrix2)
        elif distance == 'important_cut_distance':
            return MC.important_states_cut_distance(matrix_normal=matrix1,matrix_burner=matrix2,count_data=count_data)
        else:
            raise ValueError(
                'The specified distance function is not implemented. Please use cut-distance, freq-distance, frobenius, trace or important_cut_distance.')

    def reference_set(self,list_devices,distance):
        # Group the phones together per owner, this allows for owners having multiple phones.
        owner_groups = defaultdict(list)
        for device in list_devices:
            owner, dev = device.split('_')
            owner_groups[owner].append(device)

        grouped_devices = list(owner_groups.values())
        pairs_with_labels_H_p, pairs_with_labels_H_d = MC.create_pairs(np.concatenate(grouped_devices))
        list_phones = np.vstack([*pairs_with_labels_H_p,*pairs_with_labels_H_d])
        df_reference = pd.DataFrame({'phone1':[item[0] for item in list_phones],'phone2':[item[1] for item in list_phones],
                                'hypothesis':[*len(pairs_with_labels_H_p)*[1],*len(pairs_with_labels_H_d)*[0]]}, columns=['phone1','phone2','hypothesis'])

        tqdm.pandas()
        df_reference['score'] = df_reference.progress_apply(lambda row: self.calculate_score(distance,
                                                                                             self.df_Markov_chains.loc[row['phone1']].loc['markov_chains'],
                                                                                             self.df_Markov_chains.loc[row['phone2']].loc['markov_chains'],
                                                                                             self.df_Markov_chains.loc[row['phone1']].loc['count_data']),axis=1)
        # Clear the matrices from memory
        del self.df_Markov_chains

        # Force garbage collection
        gc.collect()

        return df_reference

    def predict_lr(
            self,
            track_a: Track,
            track_b: Track,
            **kwargs,
    ) -> Tuple[Optional[float], Optional[Mapping]]:

        coords_a = [m.get_postal_value for m in track_a.measurements]
        coords_b = [m.get_postal_value for m in track_b.measurements]
        if self.state_space_level == 'antenna':
            pass
        elif self.state_space_level == 'postal':
            coords_a = [element[0:4] for element in coords_a]
            coords_b = [element[0:4] for element in coords_b]
        elif self.state_space_level == 'postal3':
            coords_a = [element[0:3] for element in coords_a]
            coords_b = [element[0:3] for element in coords_b]
        elif self.state_space_level == 'postal2':
            coords_a = [element[0:2] for element in coords_a]
            coords_b = [element[0:2] for element in coords_b]

        observed_states, count_values = np.unique(coords_a, return_counts=True)
        count_vector = np.zeros(self.number_of_states)
        indices = np.where(np.in1d(self.state_space, observed_states))[0]
        count_vector[indices] = count_values

        markov_chain_a = self.construct_markov_chain(pd.Series(coords_a),self.markov_type,self.loops_allowed)
        markov_chain_b = self.construct_markov_chain(pd.Series(coords_b),self.markov_type,self.loops_allowed)

        score = self.calculate_score(self.distance,markov_chain_a,markov_chain_b,count_vector)
        lr = self.kde_calibrator.transform(score)

        return float(lr),None



