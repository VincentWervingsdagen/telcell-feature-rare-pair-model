from typing import List, Tuple, Optional, Mapping

import numpy as np
import lir

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from telcell.utils.transform import slice_track_pairs_to_intervals, create_track_pairs

from telcell.data.models import Track
from telcell.models import Model
from telcell.utils.transform import get_switches, is_colocated


def features (pairs: list):
        training_features = []
        labels = []
        for row in pairs:
                switches = get_switches(row[0],row[1])
                if (len(switches) != 0):    
                        labels += [int(is_colocated(row[0],row[1]))] * len(switches)
                        training_features += list(map(lambda x: (x.distance, x.time_difference.total_seconds()+0.000001,
                                                x.distance/(x.time_difference.total_seconds()+0.000001)), switches))

        return np.array(training_features), np.array(labels)

def binned_features(pairs: list, scores: list):
        index = 0
        binned_features = []
        labels = []
        for row in pairs:
                switches = get_switches(row[0],row[1])
                if (len(switches) != 0):
                        hist, _ = np.histogram(scores[index:index+len(switches)], bins=10,
                                range=(0, 1),density=True)
                        binned_features.append(hist)
                        labels.append(int(is_colocated(row[0],row[1])))
                        index += len(switches)

        return np.array(binned_features), np.array(labels)
      

class Regression(Model):
    
    def __init__(self, training_data: List[Track]):
        modelling, calibration = train_test_split(training_data, test_size=0.5, random_state=1)

        modelling = list(slice_track_pairs_to_intervals(create_track_pairs(modelling, all_different_source=False),
                                           interval_length_h=24))
        calibration = list(slice_track_pairs_to_intervals(create_track_pairs(calibration, all_different_source=False),
                                           interval_length_h=24))

        # Extract features from track pairs
        modelling_features, modelling_labels = features(modelling)
        calibration_features, calibration_labels = features(calibration)

        # Scale features
        self.scaler = StandardScaler()
        self.scaler.fit(modelling_features)
        self.scaler.fit(calibration_features)

        modelling_features = self.scaler.transform(modelling_features)
        calibration_features = self.scaler.transform(calibration_features)

        # First logistic regression on switches
        self.estimator1 = LogisticRegression()
        self.estimator1.fit(modelling_features, modelling_labels)
        scores = self.estimator1.predict_proba(modelling_features)[:,1]
        
        # Get binned switch scores per track
        modelling_features, modelling_labels = binned_features(modelling, scores)

        # Second logistic regression on tracks
        self.estimator2 = LogisticRegression()
        self.estimator2.fit(modelling_features, modelling_labels)

        # Apply Second model to test features to obtain final track scores for densities
        scores = self.estimator1.predict_proba(calibration_features)[:,1]
        calibration_features, calibration_labels = binned_features(calibration, scores) 
        final = self.estimator2.predict_proba(calibration_features)[:,1]

        self.calibrator = lir.ELUBbounder(lir.KDECalibrator())
        self.calibrator.fit(final, calibration_labels)
        

    def predict_lr(self, track_a: Track, track_b: Track, **kwargs) -> Tuple[float, Optional[Mapping]]:
        # read in pairs and extract features
        switches = get_switches(track_a, track_b)

        if (len(switches) <= 1):
              return None, None

        eval_features = np.array(list(map(lambda x: (x.distance, x.time_difference.total_seconds()+0.000001,
                                                     x.distance/(x.time_difference.total_seconds()+0.000001)), switches)))

        # Scale evalution features
        eval_features_scale = self.scaler.transform(eval_features)

        # Apply first regression model and bin results
        scores = self.estimator1.predict_proba(eval_features_scale)[:,1]
        hist, _ = np.histogram(scores, bins=10,
                range=(0, 1), density=True)

        # Apply second regression model to obtain final score for track pair
        X = self.estimator2.predict_proba(hist.reshape(1, -1))[:,1]
        lr = self.calibrator.transform(X)

        return float(lr), None


