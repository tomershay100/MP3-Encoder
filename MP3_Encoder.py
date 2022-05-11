from dataclasses import dataclass
import math

from WAV_Reader import WavReader
import util


@dataclass
class Subband:
    off: []
    fl: []
    x: []

    def __init__(self):
        self.off = [0] * util.MAX_CHANNELS
        self.fl = [[0] * 64] * util.SBLIMIT
        self.x = [[0] * util.HAN_SIZE] * util.MAX_CHANNELS


@dataclass
class MDCT:
    cos_l: []

    def __init__(self):
        self.cos_l = [[0] * 36] * 18


@dataclass
class MPEG:
    version: int = 0
    layer: int = 0
    granules_per_frame: int = 0
    mode: int = 0
    bitrate: int = 0
    emphasis: int = 0
    padding: int = 0
    bits_per_frame: int = 0
    bits_per_slot: int = 0
    frac_slots_per_frame: float = 0.0
    slot_lag: float = 0.0
    whole_slots_per_frame: int = 0
    bitrate_index: int = 0
    samplerate_index: int = 0
    crc: int = 0
    ext: int = 0
    mode_ext: int = 0
    copyright: int = 0
    original: int = 0


class MP3Encoder:
    def __init__(self, wav_file: WavReader):
        self.__subband_initialise()
        self.__mdct_initialise()

        self.__mpeg = MPEG()
        self.__mpeg.mode = wav_file.mpeg_mode
        self.__mpeg.bitrate = wav_file.bitrate
        self.__mpeg.emphasis = wav_file.emphasis
        self.__mpeg.copyright = wav_file.copyright
        self.__mpeg.original = wav_file.original

        self.__resv_max = 0
        self.__resv_size = 0

        self.__mpeg.layer = 3  # Only Layer III currently implemented.
        self.__mpeg.crc = 0
        self.__mpeg.ext = 0
        self.__mpeg.mode_ext = 0
        self.__mpeg.bits_per_slot = 8

        self.__mpeg.samplerate_index = util.find_samplerate_index(wav_file.samplerate)
        self.__mpeg.version = util.find_mpeg_version(self.__mpeg.samplerate_index)


        pass

    def __subband_initialise(self):
        self.__subband = Subband()
        for i in range(util.MAX_CHANNELS - 1, -1, -1):
            self.__subband.off[i] = 0

        for i in range(util.SBLIMIT - 1, -1, -1):
            for j in range(64 - 1, -1, -1):
                filter = 1e9 * math.cos(((2 * i + 1) * (16 - j) * util.PI64))
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
                self.__mdct.cos_l[m][k] = int(math.sin(util.PI36 * (k + 0.5)) * math.cos(
                    (util.PI / 72) * (2 * k + 19) * (2 * m + 1)) * 0x7fffffff)
