import random
import statistics
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CMAP_COLORS = {
    'red':
        [[0.0,  1.0, 1.0],
         [0.5,  0.5, 0.5],
         [1.0,  0.0, 0.0]],
    'green':
        [[0.0,  1.0, 1.0],
         [0.5,  0.5, 0.5],
         [1.0,  0.0, 0.0]],
    'blue':
        [[0.0,  0.0, 0.0],
         [0.5,  0.0, 0.0],
         [1.0,  0.0, 0.0]]
}

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

def bias_variance_estimation(clf, data, num_bootstrap_samples: int, grid: np.array):
    # unpack the data
    X, Y, Y_star = data

    # create grid predictions array to count the number of times each prediction is made
    classes = sorted(list(np.unique(Y_star)))
    if grid is not None:
        grid_predictions = np.zeros((grid.shape[0], len(classes)))

    oob_predictions = []
    for _ in range(num_bootstrap_samples):
        # create bootstrap sample
        bootstrap_samples, oob_samples = bootstrap_sampler(data, return_indices=True)

        # adjust the model
        idxs_boot, X_boot, Y_boot, Y_star_boot = bootstrap_samples
        clf.fit(X_boot, Y_boot[:, 0])

        # predictions on the out of bag samples
        idxs_oob, X_oob, Y_oob, Y_star_oob = bootstrap_samples
        curr_predictions = clf.predict(X_oob)
        oob_predictions.append((idxs_oob, curr_predictions))

        # predictions over the grid
        Y_grid = clf.predict(grid).astype(np.int32)
        for cls_idx, cls in enumerate(classes):
            grid_predictions[Y_grid == cls, cls_idx] += 1

    # get main predictions
    main_predictions = compute_main_predictions(oob_predictions)

    noises, biases, variances, losses = [], [], [], []
    for indices, predictions in oob_predictions:
        # return main predictions of the out of bag samples
        curr_main_predictions = main_predictions[indices]

        # compute the noises..
        curr_noises = (Y[indices] != Y_star[indices]).astype(np.int32)
        mean_noise = np.mean(curr_noises)

        # ... the biases, ...
        curr_biases = (Y_star[indices] != curr_main_predictions).astype(np.int32)
        mean_bias = np.mean(curr_biases)

        # ... and the variances.
        curr_variances = (predictions != curr_main_predictions).astype(np.int32)
        mean_variance = np.mean(curr_variances)

        # Compute the loss:
        # if bias == 0 ...
        bias0 = (1 - curr_biases) * (curr_noises*(1 - curr_variances) + curr_variances*(1 - curr_noises))
        # or if bias == 1
        bias1 = curr_biases*(1 - curr_noises - curr_variances + 2*curr_noises*curr_variances)

        curr_losses = bias0 + bias1
        mean_losses = np.mean(curr_losses)

        noises.append(mean_noise)
        biases.append(mean_bias)
        variances.append(mean_variance)
        losses.append(mean_losses)

    return noises, biases, variances, losses, grid_predictions

def plot_contour(grid, grid_predictions, ax):
    X_grid, Y_grid = grid
    Z_grid = np.reshape((grid_predictions[:, 0] - grid_predictions[:, 1]), (X_grid.shape[1], X_grid.shape[0])).T

    custom_cmap = LinearSegmentedColormap('batman_cmap', segmentdata=CMAP_COLORS, N=256)
    cp = ax.contourf(X_grid, Y_grid, Z_grid/len(grid_predictions), cmap=custom_cmap)

    return cp
