import random

def visualize_data(X, Y, ax, title=None):
    if title:
        ax.set_title(title)
    color = list(map(lambda x: 'y' if x == 1 else 'k', Y))
    ax.scatter(X[:, 0], X[:, 1], color=color)

def bootstrap_sampler(data: tuple, seed: int=None):
    if seed is not None:
        random.seed(seed)

    size = data[0].shape[0]
    for single_set in data:
        assert single_set.shape[0] == size, 'All data sets must have the same size.'

    # generate random indices for the bootstrap sample
    rand_ints = [random.randint(0, size-1) for _ in range(size)]
    out_of_bag_indexes = [idx for idx in range(size) if idx not in rand_ints]

    # create bootstrap and out-of-bag samples
    bootstrap_samples = [single_set[rand_ints] for single_set in data]
    out_of_bag_samples = [single_set[out_of_bag_indexes] for single_set in data]

    return bootstrap_samples, out_of_bag_samples
