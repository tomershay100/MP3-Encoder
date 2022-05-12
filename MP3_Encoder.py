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
    mode: int = 0  # Stereo mode
    bitrate: int = 0
    emphasis: int = 0  # De-emphasis
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


@dataclass
class BitstreamStruct:
    data: []  # Processed data
    data_size: int = 0  # Total data size
    data_position: int = 0  # Data position
    cache: int = 0  # bit stream cache
    cache_bits: int = 0  # free bits in cache


@dataclass
class GrInfo:
    table_select: []
    slen: []
    part2_3_length: int = 0
    big_values: int = 0
    count1: int = 0
    global_gain: int = 0
    scalefac_compress: int = 0
    region0_count: int = 0
    region1_count: int = 0
    preflag: int = 0
    scalefac_scale: int = 0
    count1table_select: int = 0
    part2_length: int = 0
    sfb_lmax: int = 0
    address1: int = 0
    address2: int = 0
    address3: int = 0
    quantizerStepSize: int = 0

    def __init__(self):
        self.table_select = [0] * 3
        self.slen = [0] * 4


@dataclass
class CH:
    tt: GrInfo

    def __init__(self):
        self.tt = GrInfo()


@dataclass
class GR:
    ch: []

    def __init__(self):
        self.ch = [CH()] * util.MAX_CHANNELS


@dataclass
class SideInfo:
    scfsi: []
    gr: []
    private_bits: int = 0
    resvDrain: int = 0

    def __init__(self):
        self.gr = [GR()] * util.MAX_GRANULES
        self.scfsi = [[0] * 4] * util.MAX_CHANNELS


@dataclass
class ScaleFactor:
    l: []  # [cb]
    s: []  # [window][cb]

    def __init__(self):
        self.l = [[[0] * 22] * util.MAX_CHANNELS] * util.MAX_CHANNELS
        self.s = [[[[0] * 3] * 13] * util.MAX_CHANNELS] * util.MAX_CHANNELS


@dataclass
class L3Loop:
    xr: int  # magnitudes of the spectral values
    xrsq: []  # xr squared
    xrabs: []  # xr absolute
    xrmax: int  # maximum of xrabs array
    en_tot: []  # gr
    en: []
    xm: []
    xrmaxl: []
    steptab: []  # 2**(-x/4)  for x = -127..0
    steptabi: []  # 2**(-x/4)  for x = -127..0
    int2idx: []  # x**(3/4)   for x = 0..9999

    def __init__(self):
        self.xrsq = [0] * util.GRANULE_SIZE
        self.xrabs = [0] * util.GRANULE_SIZE
        self.en_tot = [0] * util.GRANULE_SIZE
        self.en = [[0] * 21] * util.MAX_GRANULES
        self.xm = [[0] * 21] * util.MAX_GRANULES
        self.xrmaxl = [0] * util.MAX_GRANULES
        self.steptab = [0.0] * 128
        self.steptabi = [0] * 128
        self.int2idx = [0] * 10000


class MP3Encoder:
    def __init__(self, wav_file: WavReader):
        self.__wav_file = wav_file
        # Compute default encoding values.
        self.__mean_bits = 0
        self.__ratio = [[[0.0] * 21] * util.MAX_CHANNELS] * util.MAX_CHANNELS
        self.__scalefactor = ScaleFactor()
        self.__buffer = [None] * util.MAX_CHANNELS
        self.__pe = [[0] * util.MAX_GRANULES] * util.MAX_CHANNELS
        self.__l3_enc = [[[0] * util.GRANULE_SIZE] * util.MAX_GRANULES] * util.MAX_CHANNELS
        self.__l3_sb_sample = [[[[0] * util.SBLIMIT] * 18] * (util.MAX_GRANULES + 1)] * util.MAX_CHANNELS
        self.__mdct_freq = [[[0] * util.GRANULE_SIZE] * util.MAX_GRANULES] * util.MAX_CHANNELS
        self.__l3loop = L3Loop()
        self.__mdct = MDCT()
        self.__subband = Subband()
        self.__side_info = SideInfo()
        self.__mpeg = MPEG()

        self.__subband_initialise()
        self.__mdct_initialise()
        self.__loop_initialise()

        self.__mpeg.mode = wav_file.mpeg_mode
        self.__mpeg.bitrate = wav_file.bitrate
        self.__mpeg.emphasis = wav_file.emphasis
        self.__mpeg.copyright = wav_file.copyright
        self.__mpeg.original = wav_file.original

        #  Set default values.
        self.__resv_max = 0
        self.__resv_size = 0
        self.__mpeg.layer = 1  # Only Layer III currently implemented.
        self.__mpeg.crc = 0
        self.__mpeg.ext = 0
        self.__mpeg.mode_ext = 0
        self.__mpeg.bits_per_slot = 8

        self.__mpeg.samplerate_index = util.find_samplerate_index(wav_file.samplerate)
        self.__mpeg.version = util.find_mpeg_version(self.__mpeg.samplerate_index)
        self.__mpeg.bitrate_index = util.find_bitrate_index(self.__mpeg.bitrate, self.__mpeg.version)
        self.__mpeg.granules_per_frame = util.GRANULES_PER_FRAME[self.__mpeg.version]

        # Figure average number of 'slots' per frame.
        avg_slots_per_frame = (float(self.__mpeg.granules_per_frame) * util.GRANULES_SIZE / float(
            wav_file.samplerate)) * (1000 * float(self.__mpeg.bitrate) / float(self.__mpeg.bits_per_slot))

        self.__mpeg.whole_slots_per_frame = int(avg_slots_per_frame)

        self.__mpeg.frac_slots_per_frame = avg_slots_per_frame - float(self.__mpeg.whole_slots_per_frame)
        self.__mpeg.slot_lag = - self.__mpeg.frac_slots_per_frame

        if self.__mpeg.frac_slots_per_frame == 0:
            self.__mpeg.padding = 0

        self.__bitstream = BitstreamStruct([b''] * util.BUFFER_SIZE, util.BUFFER_SIZE, 0, 0, 32)

        # determine the mean bitrate for main data
        if self.__mpeg.granules_per_frame == 2:  # MPEG 1
            self.__side_info_len = 8 * ((4 + 17) if wav_file.num_of_channels == 1 else (4 + 32))
        else:  # MPEG 2
            self.__side_info_len = 8 * ((4 + 9) if wav_file.num_of_channels == 1 else (4 + 17))

    def __subband_initialise(self):
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
        # prepare the mdct coefficients
        for m in range(18 - 1, -1, -1):
            for k in range(36 - 1, -1, -1):
                # combine window and mdct coefficients into a single table
                # scale and convert to fixed point before storing
                self.__mdct.cos_l[m][k] = int(math.sin(util.PI36 * (k + 0.5)) * math.cos(
                    (util.PI / 72) * (2 * k + 19) * (2 * m + 1)) * 0x7fffffff)

    # Calculates the look up tables used by the iteration loop.
    def __loop_initialise(self):
        # quantize: stepsize conversion, fourth root of 2 table.
        # The table is inverted (negative power) from the equation given
        # in the spec because it is quicker to do x*y than x/y.
        # The 0.5 is for rounding.
        for i in range(128 - 1, -1, -1):
            self.__l3loop.steptab[i] = 2.0 ** (float(127 - i) / 4)
            if self.__l3loop.steptab[i] * 2 > 0x7fffffff:  # MAXINT = 2**31 = 2**(124/4)
                self.__l3loop.steptabi[i] = 0x7fffffff
            else:
                # The table is multiplied by 2 to give an extra bit of accuracy.
                # In quantize, the long multiply does not shift it's result left one
                # bit to compensate.
                self.__l3loop.steptabi[i] = int(self.__l3loop.steptab[i] * 2 + 0.5)

        # quantize: vector conversion, three quarter power table.
        # The 0.5 is for rounding, the .0946 comes from the spec.
        for i in range(128 - 1, -1, -1):
            self.__l3loop.int2idx[i] = int(math.sqrt(math.sqrt(float(i)) * float(i)) - 0.0946 + 0.5)

    def print_info(self):
        # Print some info about the file about to be created
        version_names = ["2.5", "reserved", "II", "I"]
        mode_names = ["stereo", "joint-stereo", "dual-channel", "mono"]
        demp_names = ["none", "50/15us", "", "CITT"]

        print(f"MPEG-{version_names[self.__mpeg.version]} layer III, {mode_names[self.__mpeg.mode]}"
              f" Psychoacoustic Model: Shine")
        print(f"Bitrate: {self.__mpeg.bitrate} kbps ", end='')
        print(f"De-emphasis: {demp_names[self.__mpeg.emphasis]}\t{'Original' if self.__mpeg.original else ''}\t"
              f"{'(C)' if self.__mpeg.copyright else ''}")
        print(f"Encoding \"{self.__wav_file.file_path}\" to \"{self.__wav_file.file_path[:-3]}mp3\"\n")
