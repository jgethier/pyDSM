import numpy as np

# Create a length 624 list to store the state of the generator
MT = [0 for i in range(624)]
index = 0

# To get last 32 bits
bitmask_1 = (2 ** 32) - 1

# To get 32. bit
bitmask_2 = 2 ** 31

# To get last 31 bits
bitmask_3 = (2 ** 31) - 1

def initialize_generator(seed):
    "Initialize the generator from a seed"
    global MT
    global bitmask_1
    MT[0] = seed
    for i in range(1,624):
        MT[i] = ((1812433253 * MT[i-1]) ^ ((MT[i-1] >> 30) + i)) & bitmask_1


def extract_number():
    """
    Extract a tempered pseudorandom number based on the index-th value,
    calling generate_numbers() every 624 numbers
    """
    global index
    global MT
    if index == 0:
        generate_numbers()
    y = MT[index]
    y ^= y >> 11
    y ^= (y << 7) & 2636928640
    y ^= (y << 15) & 4022730752
    y ^= y >> 18

    index = (index + 1) % 624
    return y

def generate_numbers():
    '''Generate an array of 624 untempered numbers'''
    global MT
    for i in range(624):
        y = (MT[i] & bitmask_2) + (MT[(i + 1 ) % 624] & bitmask_3)
        MT[i] = MT[(i + 397) % 624] ^ (y >> 1)
        if y % 2 != 0:
            MT[i] ^= 2567483615

def genrand_real3():
    '''Generates random number on (0,1)-real-interval'''
    return (extract_number() + 0.5)*(1.0/4294967296.0)


def box_muller(m,s):
    global use_last, y1, y2
    w = 1.0

    if (use_last):
        y1 = y2
        use_last=False
    else:
        while w>=1.0:
            x1 = 2.0*genrand_real3()-1.0
            x2 = 2.0*genrand_real3()-1.0
            w = x1*x1 + x2*x2
        w = np.sqrt( (-2.0*np.log(w)) / w )
        y1 = x1 * w
        y2 = x2 * w
        use_last = True

    return m+y1*s 

def gauss_distr():
    return box_muller(0.0,1.0)


