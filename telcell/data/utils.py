import datetime
from typing import Iterable, List, Tuple
import pandas as pd
from tqdm import tqdm
import geopy
from geopy.extra.rate_limiter import RateLimiter
import csv


from telcell.data.models import Track


def split_track_by_interval(
        track: Track,
        start: datetime.datetime,
        end: datetime.datetime
) -> Tuple[Track, Track]:
    """
    Splits the specified `track` in two separate `Track`s: one contains only
    the `Measurement`s that fall within the specified [start, end) interval,
    the other that contains all other measurements (i.e. those that occurred
    before `start` or after `end`).

    :param track: The track to be split in two
    :param start: The start of the interval to split `track` on
    :param end: The end of the interval to split `track` on (exclusive)
    :return: Tracks of measurements within and outside the specified interval
    """
    selected_measurements = []
    remaining_measurements = []

    for measurement in track:
        if start <= measurement.timestamp < end:
            selected_measurements.append(measurement)
        else:
            remaining_measurements.append(measurement)

    selected = Track(track.owner, track.device, selected_measurements)
    remaining = Track(track.owner, track.device, remaining_measurements)
    return selected, remaining


def extract_intervals(
        timestamps: Iterable[datetime.datetime],
        start: datetime.datetime,
        duration: datetime.timedelta,
) -> List[Tuple[datetime.datetime, datetime.datetime]]:
    """
    Calculates, for a set of intervals and timestamps, which intervals have at
    least one timestamp.

    The first interval is defined by `start` and `duration`. The next interval
    starts adjacent to (before or after) the first interval, and so on. An
    interval is returned iff there is at least one timestamp in `timestamps`
    that is within the interval.

    :param timestamps: timestamps which determine which intervals are returned
    :param start: the start of the first interval
    :param duration: the duration of the intervals
    :return: a sorted list of intervals for which there is at least 1 timestamp
    """
    intervals = set()
    for ts in timestamps:
        sequence_no = (ts - start) // duration
        interval_start = start + sequence_no * duration
        # TODO: do we have to take DST into account here?
        intervals.add((interval_start, interval_start + duration))

    # TODO: yield the intervals one by one instead of taking them in memory?
    return sorted(intervals)


def get_postal_code(reverse, lat, lon):
    try:
        location = reverse((lat, lon), exactly_one=True)
        return location.raw['address']['postcode'].replace(" ", "")
    except(KeyError, AttributeError):
        return 'placeholder'


def add_postal_code(observation_file):
    df = pd.read_csv(observation_file)

    geolocator = geopy.Nominatim(user_agent='postal_code_converter1')
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=2)

    tqdm.pandas()
    unique_coordinates = df[['cellinfo.wgs84.lat', 'cellinfo.wgs84.lon']].drop_duplicates()
    unique_coordinates['cellinfo.postal_code'] = unique_coordinates.progress_apply(
        lambda row: get_postal_code(reverse=reverse, lat=row['cellinfo.wgs84.lat'], lon=row['cellinfo.wgs84.lon']), axis=1)

    df = pd.merge(df, unique_coordinates, on=['cellinfo.wgs84.lat', 'cellinfo.wgs84.lon'], how='left')
    df = df.loc[df['cellinfo.postal_code'] != 'placeholder']
    df.to_csv(observation_file)


def check_header_for_postal_code(csv_file_path):
    print(csv_file_path)
    with open(csv_file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        header = next(reader)  # Read the first row which is the header
        return 'cellinfo.postal_code' in header