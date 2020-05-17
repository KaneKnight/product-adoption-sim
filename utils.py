import random

def payoffDeltaEarlyLate(alpha, AH0, AH1, AL0, d, p, P0, P1, eta):
    """Returns difference between payoffs of adopting early vs late.
    >>> math.isclose(payoffDeltaEarlyLate(0.2, 2, 1, 0.5, 3, 0.5, 0.8, 1, 0.2), 1.19)
    True
    >>> math.isclose(payoffDeltaEarlyLate(0.2, 2, 1.5, 0.5, 3, 0.5, 0.8, 1, 0.2), 1.318)
    True
    """
    return payoffEarly(alpha, AH0, AH1, AL0, d, p, P0, eta) - payoffLate(alpha, AH1, d, p, P1)

def payoffEarly(alpha, AH0, AH1, AL0, d, p, P0, eta):
    return p * (AH0 + AH1) + (1 - p) * AL0 + eta * p * (1 - alpha) * d - P0

def payoffLate(alpha, AH1, d, p, P1):
    return p * (1 - (1 - alpha)**d) * (AH1 - P1)

def sampleDegreeFromDistribution(distribution):
    m = random.random()
    total = 0
    for k, v in distribution.items():
        total += v
        if m <= total:
            return k    

if __name__ == "__main__":
    import doctest
    import math
    doctest.testmod()    