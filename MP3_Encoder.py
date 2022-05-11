from dataclasses import dataclass
import math

from WAV_Reader import WavReader

MAX_CHANNELS = 2

SBLIMIT = 32
HAN_SIZE = 512  # for loop unrolling, require that HAN_SIZE%8==0
PI = 3.14159265358979
PI36 = 0.087266462599717
PI64 = 0.049087385212
LN2 = 0.69314718


@dataclass
class Subband:
    off: []
    fl: []
    x: []

    def __init__(self):
        self.off = [0] * MAX_CHANNELS
        self.fl = [[0] * 64] * SBLIMIT
        self.x = [[0] * HAN_SIZE] * MAX_CHANNELS


@dataclass
class MDCT:
    cos_l: []

    def __init__(self):
        self.cos_l = [[0] * 36] * 18


class MP3Encoder:
    def __init__(self, wav_file: WavReader):
        self.__subband_initialise()
        self.__mdct_initialise()
        pass

    def __subband_initialise(self):
        self.__subband = Subband()
        for i in range(MAX_CHANNELS - 1, -1, -1):
            self.__subband.off[i] = 0

        for i in range(SBLIMIT - 1, -1, -1):
            for j in range(64 - 1, -1, -1):
                filter = 1e9 * math.cos(((2 * i + 1) * (16 - j) * PI64))
                if filter >= 0:
                    filter = math.modf(filter + 0.5)[1]
                else:
                    filter = math.modf(filter - 0.5)[1]
                # scale and convert to fixed point before storing
                self.__subband.fl[i][j] = int(filter * 0x7fffffff * 1e-9)

    def __mdct_initialise(self):
        self.__mdct = MDCT()
        # prepare the mdct coefficients
        for m in range(18 - 1, -1, -1):
            for k in range(36 - 1, -1, -1):
                # combine window and mdct coefficients into a single table
                # scale and convert to fixed point before storing
                self.__mdct.cos_l[m][k] = int(
                    math.sin(PI36 * (k + 0.5)) * math.cos((PI / 72) * (2 * k + 19) * (2 * m + 1)) * 0x7fffffff)
