from typing import Tuple, Optional, Mapping

from shapely.geometry import Point
from telcell.data.models import Track
from telcell.models import Model

import numpy as np
import geopandas as gpd


class Count(Model):
    """
    Based on the work produced by R. Longjohn, P. Smyth, and H.S. Stern, in “Likelihood 
    ratios for categorical count data with applications in digital forensics,” Law, 
    Probability and Risk

    Model computes a likelihood ratio based on the categorical count data
    obtained for two CDRs originating from two phones. Segmentation is based on first 
    4 postal code digit segmentation within a predefined bounding box. Model counts
    antennas found in each segmentation and determines likelihood that the two count
    vectors are produced by the same underlying multinomial distribution (with same parameters)
    vs. two different distributions. 
    """


    def __init__(self, postcode_file, bounding_box):
        self.postcode_df = gpd.read_file(postcode_file, bounding_box)
        self.prior = [1/len(self.postcode_df)] * len(self.postcode_df)


    def predict_lr(self, track_a: Track, track_b: Track, **kwargs) -> Tuple[float, Optional[Mapping]]:
        if not track_a or not track_b:
            return None, None

        coords_a = gpd.GeoDataFrame(geometry=[Point(m.coords.lon, m.coords.lat) for m in track_a.measurements])
        coords_a.set_crs(epsg=4326, inplace=True)

        coords_b = gpd.GeoDataFrame(geometry=[Point(m.coords.lon, m.coords.lat) for m in track_b.measurements])
        coords_b.set_crs(epsg=4326, inplace=True)

        # Spatial join
        joined_a = gpd.sjoin(coords_a, self.postcode_df, predicate='within', how='right')
        joined_b = gpd.sjoin(coords_b, self.postcode_df, predicate='within', how='right')

        # Count points per polygon
        counts_a = joined_a.groupby(self.postcode_df.geometry).size()
        counts_a = [count - 1 for count in counts_a.to_list()]
        counts_b = joined_b.groupby(self.postcode_df.geometry).size()
        counts_b = [count - 1 for count in counts_b.to_list()]
    
        lr = self.alternate_formula(self.prior, counts_a, counts_b)

        return float(lr), None

    # Alternate, less computationally expensive, formula for closed form of likelihood ratio
    def alternate_formula(self, alpha, r1, r2):
        N1 = np.sum(r1)
        N2 = np.sum(r2)
        c = np.sum(alpha)
        K = len(alpha)
        left = 1
        right = 1

        for k in range(1, K):
            if (r2[k] >= 1):
                for s in range(0,(r2[k]-1)):
                    left = left * (1+(r1[k]/(alpha[k]+s)))

        for s in range(0,(N2-1)):
            right = right*(1-(N1/(c+N1+s)))

        return left*right