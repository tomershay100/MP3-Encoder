import numpy as np

MAX_CHANNELS = 2
MAX_GRANULES = 2
GRANULE_SIZE = 576

SBLIMIT = 32
HAN_SIZE = 512  # for loop unrolling, require that HAN_SIZE%8==0
PI = 3.14159265358979
PI36 = 0.087266462599717
PI64 = 0.049087385212
LN2 = 0.69314718

BUFFER_SIZE = 4096

BIT_RATES = [
    # MPEG version:
    # 2.5, reserved, II, I
    (-1, -1, -1, -1),
    (8, -1, 8, 32),
    (16, -1, 16, 40),
    (24, -1, 24, 48),
    (32, -1, 32, 56),
    (40, -1, 40, 64),
    (48, -1, 48, 80),
    (56, -1, 56, 96),
    (64, -1, 64, 112),
    (-1, -1, 80, 128),
    (-1, -1, 96, 160),
    (-1, -1, 112, 192),
    (-1, -1, 128, 224),
    (-1, -1, 144, 256),
    (-1, -1, 160, 320),
    (-1, -1, -1, -1)
]
SAMPLE_RATES = [
    44100, 48000, 32000,  # MPEG - I
    22050, 24000, 16000,  # MPEG - II
    11025, 12000, 8000  # MPEG - 2.5
]

MODES = {
    "STEREO": 0,
    "JOINT STEREO": 1,
    "DUAL CHANNEL": 2,
    "MONO": 3
}
MPEG_VERSIONS = {
    "MPEG_I": 3,
    "MPEG_II": 2,
    "MPEG_25": 0
}

GRANULES_PER_FRAME = [
    1,  # MPEG 2.5
    -1,  # Reserved
    1,  # MPEG II
    2  # MPEG I
]
GRANULES_SIZE = 576


def find_bitrate_index(bitrate, mpeg_version):
    for i in range(16):
        if bitrate == BIT_RATES[i][mpeg_version]:
            return i

    return -1


def find_samplerate_index(samplerate):
    for i in range(9):
        if samplerate == SAMPLE_RATES[i]:
            return i

    return -1


def find_mpeg_version(samplerate_index):
    # Pick mpeg version according to samplerate index.
    if samplerate_index < 3:
        # First 3 samplerates are for MPEG-I
        return MPEG_VERSIONS["MPEG_I"]
    elif samplerate_index < 6:
        # Then it's MPEG-II
        return MPEG_VERSIONS["MPEG_II"]
    else:
        # Finally, MPEG-2.5
        return MPEG_VERSIONS["MPEG_25"]


def mul(a, b):
    a_int64 = np.array([a], dtype='int64')
    b_int64 = np.array([b], dtype='int64')
    tmp = (a_int64[0] * b_int64[0]) >> 32
    return np.array([tmp], dtype='int32')[0]


def cmuls(are, aim, bre, bim):
    are_int64 = np.array([are], dtype='int64')
    aim_int64 = np.array([aim], dtype='int64')
    bre_int64 = np.array([bre], dtype='int64')
    bim_int64 = np.array([bim], dtype='int64')

    tre = np.array([(are_int64[0] * bre_int64[0] - aim_int64[0] * bim_int64[0]) >> 31], dtype='int32')
    dim = np.array([(are_int64[0] * bim_int64[0] + aim_int64[0] * bre_int64[0]) >> 31], dtype='int32')
    dre = tre
    return dim, dre
