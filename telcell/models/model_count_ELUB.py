from typing import Tuple, Optional, Mapping

from shapely.geometry import Point
from telcell.data.models import Track
from telcell.models import Model
from telcell.data.construct_markov_chains import create_pairs, transform_data

import re
import lir
from tqdm import tqdm
from collections import defaultdict, Counter
import pandas as pd
import numpy as np


class Count_ELUB(Model):
    """
    Based on the work produced by R. Longjohn, P. Smyth, and H.S. Stern, in “Likelihood
    ratios for categorical count data with applications in digital forensics,” Law,
    Probability and Risk.

    Model computes a likelihood ratio based on the categorical count data
    obtained for two CDRs originating from two phones. Model counts
    antennas found in each segmentation and determines likelihood that the two count
    vectors are produced by the same underlying multinomial distribution (with same parameters)
    vs. two different distributions. This method additionally uses an ELUB-bounder.
    """

    ELUB_bounder: lir.ELUBbounder(lir.DummyProbabilityCalibrator)

    def __init__(self,training_set):
        # Group the phones together per owner, this allows for owners having multiple phones.

        data = transform_data(training_set,'postal3')
        list_devices = np.unique(data['device'])
        owner_groups = defaultdict(list)

        for device in list_devices:
            if '_' in device:
                owner, dev = device.split('_')
            else:
                match = re.match(r"([a-zA-Z]+)(\d+)", device)
                if match:
                    owner = match.group(1)
                else:
                    TypeError('Type is not in the expected format. We expected formats like 10_2 or Sophie1.')
            owner_groups[owner].append(device)

        pairs_with_labels_H_p, pairs_with_labels_H_d = create_pairs(owner_groups)
        list_phones = np.vstack([*pairs_with_labels_H_p,*pairs_with_labels_H_d])
        df_reference = pd.DataFrame({'phone1':[item[0] for item in list_phones],'phone2':[item[1] for item in list_phones],
                                'hypothesis':[*len(pairs_with_labels_H_p)*[1],*len(pairs_with_labels_H_d)*[0]]}, columns=['phone1','phone2','hypothesis'])

        tqdm.pandas()
        df_reference['score'] = df_reference.progress_apply(lambda row: self.calculate_score(data[data['device']==row['phone1']]['cellinfo.postal_code'].values,
                                                                                             data[data['device']==row['phone2']]['cellinfo.postal_code'].values),axis=1)

        # Right now we are using a KDE calibrator, but since we already have LR's the dummy odds calibrator should be used.
        # However, this gives weird errors where float values are being assigned like p[False], where p is a float.
        kde_calibrator = lir.KDECalibrator(bandwidth='silverman').fit(np.array(df_reference['score']), np.array(df_reference['hypothesis']))
        self.ELUB_bounder = lir.ELUBbounder(kde_calibrator)
        self.ELUB_bounder.fit(np.array(df_reference['score']), np.array(df_reference['hypothesis']))

    def predict_lr(self, track_a: Track, track_b: Track, **kwargs) -> Tuple[Optional[float], Optional[Mapping]]:
        score = self.calculate_score(track_a,track_b)
        lr = self.ELUB_bounder.transform(np.array(score))
        return float(lr), None

    def calculate_score(self, track_a, track_b, **kwargs) -> float:
        if type(track_a) == pd.Series:
            pass
        elif type(track_a)==Track:
            track_a = [m.get_postal_value for m in track_a.measurements]
            track_b = [m.get_postal_value for m in track_b.measurements]

            track_a = [element[0:3] for element in track_a]
            track_b = [element[0:3] for element in track_b]

        # Union of the two vectors
        union = sorted(set(track_a) | set(track_b))

        # Create count dictionaries
        count_vector1 = Counter(track_a)
        count_vector2 = Counter(track_b)

        # Create count vectors
        count_vector1 = [count_vector1[x] for x in union]
        count_vector2 = [count_vector2[x] for x in union]

        prior = np.ones(len(union))

        lr = self.alternate_formula(prior, count_vector1, count_vector2)

        return float(lr)


    # Alternate, less computationally expensive, formula for closed form of likelihood ratio
    def alternate_formula(self, alpha, r1, r2):
        N1 = np.sum(r1)
        N2 = np.sum(r2)
        c = np.sum(alpha)
        K = len(alpha)
        left = np.ones(N2,dtype=np.float64)
        right = np.ones(N2,dtype=np.float64)
        n=0

        for k in range(0, K):
            if (r2[k] >= 1):
                for s in range(0,(r2[k])):
                    left[n] = (1+(r1[k]/(alpha[k]+s)))
                    n=n+1

        for s in range(0,(N2)):
            right[s] = (1-(N1/(c+N1+s)))

        result = np.multiply(left,right)
        lower_bound =10**(-10)+np.random.exponential(scale = 10**(-12),size=1)[0]
        upper_bound = 10**(10)-np.random.exponential(scale = 10**(8),size=1)[0]
        result = min(max(np.prod(result),lower_bound),upper_bound)
        return result