import random
import statistics
import numpy as np

def visualize_data(X, Y, ax, title=None):
    if title:
        ax.set_title(title)
    color = list(map(lambda x: 'y' if x == 1 else 'k', Y))
    ax.scatter(X[:, 0], X[:, 1], color=color)

def bootstrap_sampler(data: tuple, return_indices: bool, seed: int=None):
    if seed is not None:
        random.seed(seed)

    size = data[0].shape[0]
    for single_set in data:
        assert single_set.shape[0] == size, 'All data sets must have the same size.'

    # generate random indices for the bootstrap sample
    bootstrap_indexes  = [random.randint(0, size-1) for _ in range(size)]
    out_of_bag_indexes = [idx for idx in range(size) if idx not in bootstrap_indexes]

    # create bootstrap and out-of-bag samples
    bootstrap_samples =  [single_set[bootstrap_indexes]  for single_set in data]
    out_of_bag_samples = [single_set[out_of_bag_indexes] for single_set in data]

    if return_indices:
        return (bootstrap_indexes,  *bootstrap_samples), \
               (out_of_bag_samples, *out_of_bag_samples)
    else:
        return bootstrap_samples, out_of_bag_samples

def compute_main_predictions(predictions):
    prediction_lists = {}
    for curr_predictions in predictions:
        for idx, pred in zip(curr_predictions[0], curr_predictions[1]):
            str_idx = str(idx)
            prediction_lists[str_idx] = prediction_lists.get(str_idx, []) + [pred]

    main_predictions = np.zeros(len(prediction_lists))
    for str_idx, all_predictions in prediction_lists.items():
        main_predictions[int(str_idx)] = statistics.mode(all_predictions)

    return main_predictions
