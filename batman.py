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
