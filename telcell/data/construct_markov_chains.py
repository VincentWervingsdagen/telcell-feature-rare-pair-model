import itertools
import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.linalg import eig
from tqdm import tqdm
import geopy
from geopy.extra.rate_limiter import RateLimiter


from pyproj import Transformer

from deap import base, creator, tools, algorithms


def get_postal_code(reverse, lat, lon):
    try:
        location = reverse((lat, lon),exactly_one=True)
        return location.raw['address']['postcode'].replace(" ","")
    except(KeyError,AttributeError):
        return 'placeholder'

def add_postal_code(df,observation_file):
    geolocator = geopy.Nominatim(user_agent='postal_code_converter')
    reverse = RateLimiter(geolocator.reverse, min_delay_seconds=1)

    tqdm.pandas()

    unique_coordinates = df[['cellinfo.wgs84.lat','cellinfo.wgs84.lon']].drop_duplicates()

    unique_coordinates['cellinfo.postal_code'] = unique_coordinates.progress_apply(lambda row: get_postal_code(reverse=reverse, lat=row['cellinfo.wgs84.lat'], lon=row['cellinfo.wgs84.lon']),axis=1)

    df = pd.merge(df, unique_coordinates, on=['cellinfo.wgs84.lat', 'cellinfo.wgs84.lon'], how='left')

    df = df.loc[df['cellinfo.postal_code'] != 'placeholder']

    df.to_csv(observation_file)

    return df

def transform_data(observation_file, level) -> pd.DataFrame:
    # Expects a file from the coverage.py file with columns: 'owner', 'device', 'timestamp', 'cellinfo.wgs84.lat',
    #        'cellinfo.wgs84.lon', 'cellinfo.azimuth_degrees', 'cellinfo.id' and 'cellinfo.postal_code'
    # Will return a pandas dataframe with columns owner, device, timestamp and postal_code in the defined level format.
    df_observations = pd.read_csv(observation_file)
    if 'cellinfo.postal_code' in df_observations.columns:
        df_observations = df_observations[['owner','device','cellinfo.postal_code']]
    else:
        df_observations = add_postal_code(df_observations,observation_file)

    df_observations = df_observations.dropna()  # Drop nan values
    df_observations = df_observations.reset_index()  # Reset the index
    pattern = r'^\d{4}[A-Z]{2}$'  # Check whether every item is a valid postal code.
    valid_postal_codes = df_observations['cellinfo.postal_code'].str.match(pattern)
    df_observations = df_observations[valid_postal_codes]
    if level == 'antenna':
        pass
    elif level == 'postal':
        df_observations['cellinfo.postal_code'] = [element[0:4] for element in df_observations['cellinfo.postal_code']]
    elif level == 'postal3':
        df_observations['cellinfo.postal_code'] = [element[0:3] for element in df_observations['cellinfo.postal_code']]
    elif level == 'postal2':
        df_observations['cellinfo.postal_code'] = [element[0:2] for element in df_observations['cellinfo.postal_code']]
    else:
        raise ValueError(
            'The specified state space level is not implemented. Please choose either antenna,postal,postal3')

    df_observations = df_observations.drop(columns=['index'])

    return df_observations


def state_space_Omega(cell_file,bounding_box,antenna_type,level='postal2') -> np.array:
    # Needs a file with all the antennas, the region, the antenna_type and the level that is being considered.
    # Will return a np.array with possible states that the phones can visit.
    df_cell = pd.read_csv(cell_file)
    #drop useless columns
    df_cell = df_cell.drop(['Samenvatting','Vermogen', 'Frequentie','Veilige afstand','id','objectid','WOONPLAATSNAAM','DATUM_PLAATSING',
           'DATUM_INGEBRUIKNAME','GEMNAAM','Hoogte','Hoofdstraalrichting','sat_code'],axis=1)
    #drop types except the antenna_type
    df_cell = df_cell.loc[df_cell['HOOFDSOORT'] == antenna_type]
         #Transform to wgs84
    df_cell['lat'], df_cell['lon'] = Transformer.from_crs("EPSG:28992", "EPSG:4979").transform(df_cell['X'],
                                                                                                   df_cell['Y'])
    # Only keep cell towers in bounding box
    df_cell = df_cell.loc[
            (df_cell['lon'] >= bounding_box[0]) & (df_cell['lon'] <= bounding_box[2])
            & (df_cell['lat'] >= bounding_box[1]) & (df_cell['lat'] <= bounding_box[3])]

    if level == 'antenna':
        return np.sort(np.unique(df_cell['POSTCODE'].dropna()))
    elif level == 'postal':
        df_cell = df_cell['POSTCODE'].dropna()
        array_zip_codes = [element[0:4] for element in df_cell]
        return np.sort(np.unique(array_zip_codes))
    elif level == 'postal3':
        df_cell = df_cell['POSTCODE'].dropna()
        array_zip_codes = [element[0:3] for element in df_cell]
        return np.sort(np.unique(array_zip_codes))
    elif level == 'postal2':
        df_cell = df_cell['POSTCODE'].dropna()
        array_zip_codes = [element[0:2] for element in df_cell]
        return np.sort(np.unique(array_zip_codes))
    else:
        raise ValueError('The specified state space level is not implemented. Please choose either antenna,postal,postal3')


def state_space_observations(df) -> np.array:
    # Expects a dataframe from the function transform.
    # Returns a numpy array with all seen states.
    states = np.sort(np.unique(df['cellinfo.postal_code']))
    return states


def uniform_prior(number_of_states,states) -> pd.DataFrame:
    # Expects the states of the markov chain.
    # Will return the Jeffrey prior: A number_of_states x number_of_states matrix filled with 1/number_of_states.
    return pd.DataFrame(1/number_of_states,index=states,columns=states)


def zero_prior(states) -> pd.DataFrame:
    # Expects the states of the markov chain.
    # Will return a near zero matrix.
    return pd.DataFrame(1000*np.finfo(float).eps,index=states,columns=states)


def distance_prior(distance_matrix_file,states,bounding_box) -> pd.DataFrame:
    # Expects a pre_calculated matrix from Calculate_distance_omega.py.
    # Expects the states which are used to filter the distance matrix file.
    # Return a pandas dataframe with the distance prior.
    if bounding_box == (4.1874, 51.8280, 4.593, 52.0890):
        distance_matrix = pd.read_csv(distance_matrix_file,
                                      dtype={'Unnamed: 0': str})
        distance_matrix_index = distance_matrix['Unnamed: 0']
        distance_matrix = distance_matrix.drop('Unnamed: 0', axis=1)
        distance_matrix = distance_matrix.set_index(distance_matrix_index)
        distance_matrix.columns = distance_matrix_index
        distance_matrix = distance_matrix.loc[states,states]
        return distance_matrix
    else:
        raise NotImplementedError('The distance matrix was not calculated for this bounding box. Please see Calculate_distance_omega.py.')


def population_prior():
    raise NotImplementedError


def discrete_markov_chain(track,prior,states,loops_allowed=True) -> np.array:
    # Expects the observations from a single phone, a prior distribution and the states.
    # One option is to allow self loops. Recommendation is to always put this as true.
    # Will return a matrix with the movement of the device.
    # First construct count matrix
    matrix = pd.DataFrame(0.0,index=states,columns=states)
    for current, next_ in zip(track.index[:-1], track.index[1:]):
        matrix.loc[track[current], track[next_]] += 1
    # Add prior.
    matrix = matrix + prior

    if loops_allowed == True:
        pass
    elif loops_allowed == False:
        np.fill_diagonal(matrix.values,0)
    else:
        raise ValueError('Loops allowed must be a bool variable.')
    # Normalise
    matrix = matrix/matrix.apply(func='sum',axis=0)
    matrix_normal = np.transpose(matrix)
    return matrix_normal


def continuous_markov_chain():
    raise NotImplementedError


def create_pairs(list_device) -> (list[list],list[list]):
    # Expects a list filled with items.
    # Return a list with all possible pairs of items of a list.
    # Will only return (a,b) and not (b,a), where a,b are items in the list.
    # Initialize an empty list to store the pairs and their labels
    pairs_with_labels_H_p = []
    pairs_with_labels_H_d = []

    # Iterate over each pair of devices
    for i in range(len(list_device)):
        for j in range(i + 1, len(list_device)):
            # Split the device strings to get the owner and device numbers
            owner_i, device_i = list_device[i].split('_')
            owner_j, device_j = list_device[j].split('_')

            # Determine the label based on the owners
            if owner_i == owner_j:
                pairs_with_labels_H_p.append([list_device[i], list_device[j]])
            else:
                pairs_with_labels_H_d.append([list_device[i], list_device[j]])

    return pairs_with_labels_H_p, pairs_with_labels_H_d


def cut_weight(matrix, S, T) -> float:
    # Expects a nxn matrix and two vectors S,T that are bool vectors of the state.
    # Returns the sum of the values going from set S to T.
    if (len(S) == 0) or (len(T) == 0):
        return 0
    else:
        return matrix.loc[S, T].to_numpy().sum()


def cut_distance(individual, matrix_normal, matrix_burner,states) -> tuple:
    # Expects a bool vector of length n and two nxn matrices, that need to be compared.
    # Returns the cut distance between two matrices in the form of a tuple.
    S = states[np.array(individual) == 1]
    T = states[np.array(individual) == 0]
    distance = np.abs(cut_weight(matrix_normal, S, T)-cut_weight(matrix_burner, S, T))
    return distance,


def genetic_cut_distance(matrix_normal,matrix_burner) -> float:
    # Expects two nxn matrices on the same state space.
    # Approximates the cut-distance between two matrices with a genetic algorithm.
    # This function can be quite slow depending on the number of iterations and the generation sizes.
    # Returns a float number.
    try:
        del creator.FitnessMax
        del creator.Individual
    except Exception as e:
        pass

    states = matrix_normal.index
    # Create the genetic algorithm parameters.
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", np.random.binomial, 1, np.random.uniform(0.1,0.9))
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(states))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", cut_distance, matrix_normal=matrix_normal, matrix_burner=matrix_burner, states=states)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament,tournsize=3)

    # Genetic Algorithm parameters
    population = toolbox.population(n=1000)
    ngen = 20
    cxpb = 0.5
    mutpb = 0.2

    # Run the Genetic Algorithm
    result = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

    # Extract the best individual
    best_ind = tools.selBest(population, 1)[0]
    S = [states[i] for i in range(len(best_ind)) if best_ind[i] == 1]
    T = [states[i] for i in range(len(best_ind)) if best_ind[i] == 0]

    return (1/len(states))*cut_distance(best_ind,matrix_normal, matrix_burner,states)[0]


def important_states_cut_distance_5(matrix_normal,matrix_burner) -> float:
    # Expects two matrices that need to be compared.
    # Looks at the 5 most important states based on the stationary distribution.
    # Then it calculates the maximum cut distance for all possible combinations of these 5 states.
    # Returns a float value.
    stationary_distribution = calculate_stationary_distribution(matrix_normal)
    index_important_states = np.argsort(stationary_distribution)[-5:]
    distances = []
    states = matrix_normal.index
    for element in itertools.product(range(2),repeat=5):
        individual = np.zeros(len(states))
        individual[index_important_states] = element
        distances.append(cut_distance(individual,matrix_normal,matrix_burner,states)[0])
    return max(distances)


def important_states_cut_distance(matrix_normal,matrix_burner,data_normal) -> float:
    # Expects two matrices that need to be compared.
    # First computes which states have enough observations based on rule of thumb estimation of a binomial distribution.
    # Then it calculates the maximum cut distance for the states that have good enough estimates.
    # Returns a float value.
    observation,count_observations = np.unique(data_normal['cellinfo.postal_code'],return_counts=True)
    B = stats.chi2.ppf(1-0.05 / len(observation),1)
    t = stats.norm.ppf(1-0.05/2)
    important_states = observation[count_observations>10*B/(t**2)]
    if not important_states.any():
        raise ValueError('There is not enough data to make a sufficient estimate for the markov chain.')
    distances = []
    states = matrix_normal.index
    index_important_states = np.where(states.isin(important_states))
    for element in itertools.product(range(2),repeat=len(index_important_states)):
        individual = np.zeros(len(states))
        individual[index_important_states] = element
        distances.append(cut_distance(individual,matrix_normal,matrix_burner,states)[0])
    return max(distances)


def calculate_stationary_distribution(transition_matrix) -> np.array:
    # Expects a nxn transition matrix.
    # Returns the stationary distribution.

    # Ensure the transition matrix is a square matrix
    assert transition_matrix.shape[0] == transition_matrix.shape[1], "Transition matrix must be square"

    # Compute the eigenvalues and right eigenvectors
    eigenvalues, eigenvectors = eig(transition_matrix.T)

    # Find the index of the eigenvalue 1 (stationary distribution)
    stationary_index = np.argmin(np.abs(eigenvalues - 1))

    # Extract the corresponding eigenvector
    stationary_vector = np.real(eigenvectors[:, stationary_index])

    # Normalize the stationary vector to sum to 1
    stationary_distribution = stationary_vector / stationary_vector.sum()

    return stationary_distribution


def calculate_conductance(individual,matrix,states,stationary_distribution,number_of_states) -> tuple:
    # Expects a bool vector of length n, a nxn matrix, the state space, stationary distribution and the number of states.
    # This function calculates the conductance or average number of transitions out of the bool vector into its complement.
    # Returns a tuple with one float value.
    if (np.array(individual).sum() == 0) | (np.array(individual).sum() == number_of_states):
        return 1, # Not a possible individual
    S = states[np.array(individual) == 1]
    T = states[np.array(individual) == 0]

    conductance = (stationary_distribution[np.array(individual) == 1]*matrix.loc[S, T].to_numpy().sum(axis=1)).sum()
    return conductance,


def calculate_freq_distance(matrix_normal,matrix_burner,stationary_distribution,conductance,number_of_states) -> float:
    # Expects two nxn matrices, the stationary distribution, the conductance and the number of states.
    # First it calculates which of the transitions in the first or 'normal' matrix are frequent based on the conductance.
    # Then it calculates a score only for the states that are frequent.
    # Returns a float value.

    frequent_transition_matrix_normal = np.multiply(np.transpose(number_of_states*[stationary_distribution]),(matrix_normal.to_numpy()))
    # frequent_transition_matrix_burner = np.multiply(np.transpose(number_of_states*[stationary_distribution]),(matrix_burner.to_numpy()))
    frequent_transition_matrix_normal = frequent_transition_matrix_normal>conductance
    # frequent_transition_matrix_burner = frequent_transition_matrix_burner>conductance
    # frequent_transition_matrix = (frequent_transition_matrix_normal+frequent_transition_matrix_burner)>0
    frequent_transition_matrix = frequent_transition_matrix_normal
    result = np.abs(np.ones(frequent_transition_matrix.sum())-matrix_burner.to_numpy()[frequent_transition_matrix]/matrix_normal.to_numpy()[frequent_transition_matrix])
    if len(result)==0:
        return 100000.,100000.

    return max(result)


def frequent_transition_distance(matrix_normal,matrix_burner) -> float:
    # Expects two nxn matrices.
    # Calculates the frequent transition distance by first approximating the conductance with a genetic algorithm.
    # Then it calculates the distance.
    # Returns a float.
    stationary_distribution = calculate_stationary_distribution(matrix_normal)
    states = matrix_normal.index

    try:
        del creator.FitnessMax
        del creator.Individual
    except Exception as e:
        pass

    # Create the genetic algorithm parameters.
    creator.create("FitnessMax", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    # bernoulli with parameter 0.1, since the minimum conductance is often reached by looking at one row or column.
    toolbox.register("attr_bool", np.random.binomial, 1, 0.1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(states))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", calculate_conductance, matrix=matrix_normal, states=states,stationary_distribution=stationary_distribution,number_of_states=len(states))
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament,tournsize=3)

    # Genetic Algorithm parameters
    population = toolbox.population(n=100)
    ngen = 10
    cxpb = 0.5
    mutpb = 0.2

    # Run the Genetic Algorithm
    result = algorithms.eaSimple(population, toolbox, cxpb, mutpb, ngen, verbose=False)

    # Extract the best individual
    best_ind = tools.selBest(population, 1)[0]
    conductance = calculate_conductance(best_ind,matrix_normal,states=states,stationary_distribution=stationary_distribution,number_of_states=len(states))[0]

    distance = calculate_freq_distance(matrix_normal=matrix_normal,matrix_burner=matrix_burner,stationary_distribution=stationary_distribution,conductance=conductance,number_of_states=len(states))

    return distance


def frobenius_norm(matrix_normal,matrix_burner) -> float:
    # Expects two nxn matrices.
    # Returns the frobenius norm.
    return np.linalg.norm(x=(matrix_normal-matrix_burner),ord='fro')


def trace_norm(matrix_normal,matrix_burner) -> float:
    # Expects two nxn matrices.
    # Returns the trace or nuclear norm.
    return np.linalg.norm(x=(matrix_normal-matrix_burner),ord='nuc')