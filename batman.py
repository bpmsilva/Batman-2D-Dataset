import numpy as np

# TODO: vectorize this function
def batman_upper(x):
    if x > 3:
        y = 3*(1 - (x/7) ** 2) ** 0.5
    elif x > 1:
        y = 6*(10**0.5)/7 - (0.5*abs(x)) + 1.5 - 3*(10**0.5)/7*(4-(abs(x) - 1)**2)**0.5
    elif x > 0.75:
        y = 9 - 8*abs(x)
    elif x > 0.5:
        y = 3*abs(x) + 0.75
    else:
        y = 2.55
    return y

# TODO: vectorize this function as well
def batman_lower(x):
    if x > 4:
        y = -3*(1 - (x/7) ** 2) ** 0.5
    else:
        y = abs(x/2) - (3*33**0.5 - 7)/(112)*x**2 - 3 + (1 - (abs(abs(x) - 2) - 1)**2)**0.5
    return y

def batman_curve(x, y):
    # the curve is symmetric around the y-axis
    x = abs(x)

    # if x is bigger than 7,
    # the point is outside the batman logo
    if x > 7:
        return 0

    if batman_upper(x) > y and batman_lower(x) < y:
        return 1

    return 0

def get_dataset(
    sample_size,
    noise_stdvs: int=(0.2, 0.4),
    seeds: tuple=(None, None, None)
):
    seed_data, seed_noise_x, seed_noise_y = seeds
    if seed_data is not None:
        np.random.seed(seed_data)

    # Generate the points
    X1 = 16*np.random.random(size=(sample_size, 1)) - 8
    X2 =  8*np.random.random(size=(sample_size, 1)) - 4
    X = np.concatenate((X1, X2), axis=1)

    # No noise labels
    Y_star = np.zeros((sample_size, 1))
    for i in range(len(X)):
        curr_cat = batman_curve(X[i, 0], X[i, 1])
        Y_star[i, 0] = curr_cat

    # Generate noise with different stdvs
    if seed_noise_x is not None:
        np.random.seed(seed_noise_x)
    noises_x = np.random.normal(scale=noise_stdvs[0], size=sample_size)
    if seed_noise_y is not None:
        np.random.seed(seed_noise_y)
    noises_y = np.random.normal(scale=noise_stdvs[1], size=sample_size)

    # Noisy labels
    Y = np.zeros((sample_size, 1))
    for i in range(len(X)):
        curr_cat = batman_curve(X[i, 0] + noises_x[i], X[i, 1] + noises_y[i])
        Y[i, 0] = curr_cat

    return X, Y, Y_star
