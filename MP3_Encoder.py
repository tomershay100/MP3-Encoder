from copy import copy
from dataclasses import dataclass
import math

import tables
from WAV_Reader import WavReader
import util
import numpy as np


@dataclass
class Subband:
    off: []
    fl: []
    x: []

    def __init__(self):
        self.off = np.zeros(util.MAX_CHANNELS, dtype=np.int32)
        self.fl = np.zeros((util.SBLIMIT, 64), dtype=np.int32)
        self.x = np.zeros((util.MAX_CHANNELS, util.HAN_SIZE), dtype=np.int32)


@dataclass
class MDCT:
    cos_l: []

    def __init__(self):
        self.cos_l = np.zeros((18, 36), dtype=np.int32)


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
        self.table_select = np.zeros(3, dtype=np.int32)
        self.slen = np.zeros(4, dtype=np.int32)


@dataclass
class CH:
    tt: GrInfo

    def __init__(self):
        self.tt = GrInfo()


@dataclass
class GR:
    ch: []

    def __init__(self):
        self.ch = [CH() for _ in range(util.MAX_CHANNELS)]


@dataclass
class SideInfo:
    scfsi: []
    gr: []
    private_bits: int = 0
    resvDrain: int = 0

    def __init__(self):
        self.gr = [GR() for _ in range(util.MAX_GRANULES)]
        self.scfsi = np.zeros((util.MAX_CHANNELS, 4), dtype=np.int32)


@dataclass
class ScaleFactor:
    l: []  # [cb]
    s: []  # [window][cb]

    def __init__(self):
        self.l = np.zeros((util.MAX_GRANULES, util.MAX_CHANNELS, 22), dtype=np.int32)  # [cb]
        self.s = np.zeros((util.MAX_GRANULES, util.MAX_CHANNELS, 13, 3), dtype=np.int32)  # [window][cb]


@dataclass
class L3Loop:
    xr: np.array  # a pointer of the magnitudes of the spectral values
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
        self.xrsq = np.zeros(util.GRANULE_SIZE, dtype=np.int32)
        self.xrabs = np.zeros(util.GRANULE_SIZE, dtype=np.int32)
        self.en_tot = np.zeros(util.MAX_GRANULES, dtype=np.int32)
        self.en = np.zeros((util.MAX_GRANULES, 21), dtype=np.int32)
        self.xm = np.zeros((util.MAX_GRANULES, 21), dtype=np.int32)
        self.xrmaxl = np.zeros(util.MAX_GRANULES, dtype=np.int32)
        self.steptab = np.zeros(128, dtype=np.double)
        self.steptabi = np.zeros(128, dtype=np.int32)
        self.int2idx = np.zeros(10000, dtype=np.int32)


class MP3Encoder:
    def __init__(self, wav_file: WavReader):
        self.__wav_file = wav_file
        # Compute default encoding values.
        self.__mean_bits = 0
        self.__ratio = np.zeros((util.MAX_GRANULES, util.MAX_CHANNELS, 21), dtype=np.double)
        self.__scalefactor = ScaleFactor()
        self.__buffer = np.zeros((util.MAX_CHANNELS, self.__wav_file.num_of_samples), dtype=np.int16)
        self.__pe = np.zeros((util.MAX_CHANNELS, util.MAX_GRANULES), dtype=np.double)
        self.__l3_enc = np.zeros((util.MAX_CHANNELS, util.MAX_GRANULES, util.GRANULE_SIZE), dtype=np.int32)
        self.__l3_sb_sample = np.zeros((util.MAX_CHANNELS, util.MAX_GRANULES + 1, 18, util.SBLIMIT),
                                       dtype=np.int32)
        self.__mdct_freq = np.zeros((util.MAX_CHANNELS, util.MAX_GRANULES, util.GRANULE_SIZE), dtype=np.int32)
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

        self.__bitstream = BitstreamStruct([b'' for _ in range(util.BUFFER_SIZE)], util.BUFFER_SIZE, 0, 0, 32)

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

    def encode(self):
        samples_per_pass = self.__samples_per_pass() * self.__wav_file.num_of_channels

        # All the magic happens here
        total_sample_count = self.__wav_file.num_of_samples * self.__wav_file.num_of_channels
        count = total_sample_count // samples_per_pass

        for i in range(count):
            data = self.__encode_buffer_internal()

    def __samples_per_pass(self):
        return self.__mpeg.granules_per_frame * util.GRANULE_SIZE

    def __encode_buffer_internal(self):
        if self.__mpeg.frac_slots_per_frame:
            self.__mpeg.padding = (1 if self.__mpeg.slot_lag <= (self.__mpeg.frac_slots_per_frame - 1.0) else 0)
            self.__mpeg.slot_lag += self.__mpeg.padding - self.__mpeg.frac_slots_per_frame

        self.__mpeg.bits_per_frame = 8 * self.__mpeg.whole_slots_per_frame + self.__mpeg.padding
        self.__mpeg.mean_bits = (self.__mpeg.bits_per_frame - self.__side_info_len) / self.__mpeg.granules_per_frame

        # apply mdct to the polyphase output
        self.__mdct_sub()  # TODO check for validity

        # bit and noise allocation

        # write the frame to the bitstream
        self.__format_bitstream()  # TODO check for validity
        pass

    def __mdct_sub(self):
        # note. we wish to access the array 'config->mdct_freq[2][2][576]' as
        # [2][2][32][18]. (32*18=576),
        self.__mdct_freq = self.__mdct_freq.reshape((2, 2, 32, 18))
        mdct_in = np.zeros(36, dtype=np.int32)

        for ch in range(self.__wav_file.num_of_channels - 1, -1, -1):
            for gr in range(self.__mpeg.granules_per_frame):
                # set up pointer to the part of config->mdct_freq we're using
                # mdct_enc = self.__mdct_freq[ch][gr]

                # polyphase filtering
                for k in range(0, 2, 18):
                    self.__l3_sb_sample[ch][gr + 1][k] = self.__window_filter_subband(
                        self.__l3_sb_sample[ch][gr + 1][k], ch)
                    self.__l3_sb_sample[ch][gr + 1][k + 1] = self.__window_filter_subband(
                        self.__l3_sb_sample[ch][gr + 1][k + 1], ch)

                    # Compensate for inversion in the analysis filter
                    # (every odd index of band AND k)
                    for band in range(1, 32, 2):
                        self.__l3_sb_sample[ch][gr + 1][k + 1][band] *= -1

                # Perform imdct of 18 previous subband samples + 18 current subband samples
                for band in range(0, 32, 1):
                    for k in range(18 - 1, -1, -1):
                        mdct_in[k] = self.__l3_sb_sample[ch][gr][k][band]
                        mdct_in[k + 18] = self.__l3_sb_sample[ch][gr + 1][k][band]

                    # Calculation of the MDCT
                    # In the case of long blocks ( block_type 0,1,3 ) there are
                    # 36 coefficients in the time domain and 18 in the frequency domain.
                    for k in range(18 - 1, -1, -1):
                        vm = util.mul(mdct_in[35], self.__mdct.cos_l[k][35])
                        for j in range(35, 0, -7):
                            vm += util.mul(mdct_in[j - 1], self.__mdct.cos_l[k][j - 1])
                            vm += util.mul(mdct_in[j - 2], self.__mdct.cos_l[k][j - 2])
                            vm += util.mul(mdct_in[j - 3], self.__mdct.cos_l[k][j - 3])
                            vm += util.mul(mdct_in[j - 4], self.__mdct.cos_l[k][j - 4])
                            vm += util.mul(mdct_in[j - 5], self.__mdct.cos_l[k][j - 5])
                            vm += util.mul(mdct_in[j - 6], self.__mdct.cos_l[k][j - 6])
                            vm += util.mul(mdct_in[j - 7], self.__mdct.cos_l[k][j - 7])
                        self.__mdct_freq[ch][gr][band][k] = vm

                    # Perform aliasing reduction butterfly
                    if band != 0:
                        self.__mdct_freq[ch][gr][band][0], self.__mdct_freq[ch][gr][band - 1][17 - 0] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][0], self.__mdct_freq[ch][gr][band - 1][17 - 0],
                            tables.MDCT_CS0, tables.MDCT_CA0)
                        self.__mdct_freq[ch][gr][band][1], self.__mdct_freq[ch][gr][band - 1][17 - 1] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][1], self.__mdct_freq[ch][gr][band - 1][17 - 1],
                            tables.MDCT_CS1, tables.MDCT_CA1)
                        self.__mdct_freq[ch][gr][band][2], self.__mdct_freq[ch][gr][band - 1][17 - 2] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][2], self.__mdct_freq[ch][gr][band - 1][17 - 2],
                            tables.MDCT_CS2, tables.MDCT_CA2)
                        self.__mdct_freq[ch][gr][band][3], self.__mdct_freq[ch][gr][band - 1][17 - 3] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][3], self.__mdct_freq[ch][gr][band - 1][17 - 3],
                            tables.MDCT_CS3, tables.MDCT_CA3)
                        self.__mdct_freq[ch][gr][band][4], self.__mdct_freq[ch][gr][band - 1][17 - 4] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][4], self.__mdct_freq[ch][gr][band - 1][17 - 4],
                            tables.MDCT_CS4, tables.MDCT_CA4)
                        self.__mdct_freq[ch][gr][band][5], self.__mdct_freq[ch][gr][band - 1][17 - 5] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][5], self.__mdct_freq[ch][gr][band - 1][17 - 5],
                            tables.MDCT_CS5, tables.MDCT_CA5)
                        self.__mdct_freq[ch][gr][band][6], self.__mdct_freq[ch][gr][band - 1][17 - 6] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][6], self.__mdct_freq[ch][gr][band - 1][17 - 6],
                            tables.MDCT_CS6, tables.MDCT_CA6)
                        self.__mdct_freq[ch][gr][band][7], self.__mdct_freq[ch][gr][band - 1][17 - 7] = util.cmuls(
                            self.__mdct_freq[ch][gr][band][7], self.__mdct_freq[ch][gr][band - 1][17 - 7],
                            tables.MDCT_CS7, tables.MDCT_CA7)

            # Save latest granule's subband samples to be used in the next mdct call
            self.__l3_sb_sample[ch][0] = copy(self.__l3_sb_sample[ch][self.__mpeg.granules_per_frame])

        self.__mdct_freq = self.__mdct_freq.reshape((util.MAX_CHANNELS, util.MAX_GRANULES, util.GRANULE_SIZE))

    def __window_filter_subband(self, s, ch):
        buffer = self.__wav_file.buffer[self.__wav_file.get_buffer_pos(ch):]
        y = np.zeros(64, dtype=np.int32)
        # replace 32 oldest samples with 32 new samples
        for i in range(32 - 1, -1, -1):
            self.__subband.x[ch][i + self.__subband.off[ch]] = int(buffer[0]) << 16
            self.__wav_file.set_buffer_pos(ch, 2)
            buffer = self.__wav_file.buffer[self.__wav_file.get_buffer_pos(ch):]

        for i in range(64 - 1, -1, -1):
            s_value = util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (0 << 6)) & (util.HAN_SIZE - 1)],
                               tables.enwindow[i + (0 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (1 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (1 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (2 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (2 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (3 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (3 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (4 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (4 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (5 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (5 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (6 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (6 << 6)])
            s_value += util.mul(self.__subband.x[ch][(self.__subband.off[ch] + i + (7 << 6)) & (util.HAN_SIZE - 1)],
                                tables.enwindow[i + (7 << 6)])

            y[i] = s_value

        self.__subband.off[ch] = (self.__subband.off[ch] + 480) & (util.HAN_SIZE - 1)  # offset is modulo (HAN_SIZE)

        for i in range(util.SBLIMIT - 1, -1, -1):
            s_value = util.mul(self.__subband.fl[i][63], y[63])
            for j in range(63, 0, -7):
                s_value += util.mul(self.__subband.fl[i][j - 1], y[j - 1])
                s_value += util.mul(self.__subband.fl[i][j - 2], y[j - 2])
                s_value += util.mul(self.__subband.fl[i][j - 3], y[j - 3])
                s_value += util.mul(self.__subband.fl[i][j - 4], y[j - 4])
                s_value += util.mul(self.__subband.fl[i][j - 5], y[j - 5])
                s_value += util.mul(self.__subband.fl[i][j - 6], y[j - 6])
                s_value += util.mul(self.__subband.fl[i][j - 7], y[j - 7])
            s[i] = s_value

        return s

    # bit and noise allocation
    def iteration_loop(self):
        l3_xmin = np.zeros((util.MAX_GRANULES, util.MAX_CHANNELS, 21), dtype=np.double)

        for ch in range(self.__wav_file.num_of_channels - 1, -1, -1):
            for gr in range(self.__mpeg.granules_per_frame):
                # setup pointers
                ix = self.__l3_enc[ch][gr]
                self.__l3loop.xr = self.__mdct_freq[ch][gr]

                # Precalculate the square, abs, and maximum, for us later on.
                self.__l3loop.xrmax = 0
                for i in range(util.GRANULE_SIZE - 1, -1, -1):
                    self.__l3loop.xrsq[i] = util.mulsr(self.__l3loop.xr[i], self.__l3loop.xr[i])
                    self.__l3loop.xrabs[i] = util.labs(self.__l3loop.xr[i])
                    if self.__l3loop.xrabs[i] > self.__l3loop.xrmax:
                        self.__l3loop.xrmax = self.__l3loop.xrabs[i]

                cod_info = self.__side_info.gr[gr].ch[ch]
                cod_info.sfb_lmax = util.SFB_LMAX - 1  # gr_deco

                l3_xmin[gr, ch, 0:cod_info.sfb_lmax] = 0

                if self.__mpeg.version == util.MPEG_VERSIONS.MPEG_I:
                    self.__calc_scfsi(l3_xmin, ch, gr)

    # calculation of the scalefactor select information ( scfsi )
    def __calc_scfsi(self, l3_xmin, ch, gr):
        l3_side = self.__side_info

        # This is the scfsi_band table from 2.4.2.7 of the IS
        scfsi_band_long = [0, 6, 11, 16, 21]
        condition = 0

        scalefac_band_long = util.scale_fact_band_index[self.__mpeg.samplerate_index][0]

        self.__l3loop.xrmaxl[gr] = self.__l3loop.xrmax
        scfsi_set = 0

        # the total energy of the granule
        temp = 0
        for i in range(util.GRANULE_SIZE - 1, -1, -1):
            temp += self.__l3loop.xrsq[i] >> 10  # a bit of scaling to avoid overflow

        if temp:
            self.__l3loop.en_tot[gr] = np.log(np.double(temp * 4.768371584e-7)) / util.LN2

    def __format_bitstream(self):
        for ch in range(self.__wav_file.num_of_channels):
            for gr in range(self.__mpeg.granules_per_frame):
                for i in range(util.GRANULE_SIZE):
                    if self.__mdct_freq[ch][gr][i] < 0 and self.__l3_enc[ch][gr][i] > 0:
                        self.__l3_enc[ch][gr][i] *= -1

        self.__encode_side_info()
        self.__encode_main_data()

    def __encode_side_info(self):

        self.__putbits(0x7ff, 11)
        self.__putbits(self.__mpeg.version, 2)
        self.__putbits(self.__mpeg.layer, 2)
        self.__putbits((0 if self.__mpeg.crc else 1), 1)
        self.__putbits(self.__mpeg.bitrate_index, 4)
        self.__putbits(self.__mpeg.samplerate_index % 3, 2)
        self.__putbits(self.__mpeg.padding, 1)
        self.__putbits(self.__mpeg.ext, 1)
        self.__putbits(self.__mpeg.mode, 2)
        self.__putbits(self.__mpeg.mode_ext, 2)
        self.__putbits(self.__mpeg.copyright, 1)
        self.__putbits(self.__mpeg.original, 1)
        self.__putbits(self.__mpeg.emphasis, 2)

        if self.__mpeg.version == 3:
            self.__putbits(0, 9)
            if self.__wav_file.num_of_channels == 2:
                self.__putbits(self.__side_info.private_bits, 3)
            else:
                self.__putbits(self.__side_info.private_bits, 5)
        else:
            self.__putbits(0, 8)
            if self.__wav_file.num_of_channels == 2:
                self.__putbits(self.__side_info.private_bits, 2)
            else:
                self.__putbits(self.__side_info.private_bits, 1)

        if self.__mpeg.version == 3:
            for ch in range(self.__wav_file.num_of_channels):
                for scfsi_band in range(4):
                    self.__putbits(self.__side_info.scfsi[ch][scfsi_band], 1)

        for gr in range(self.__mpeg.granules_per_frame):
            for ch in range(self.__wav_file.num_of_channels):
                # gi = self.__side_info.gr[gr].ch[ch].tt
                self.__putbits(self.__side_info.gr[gr].ch[ch].tt.part2_3_length, 12)
                self.__putbits(self.__side_info.gr[gr].ch[ch].tt.big_values, 9)
                self.__putbits(self.__side_info.gr[gr].ch[ch].tt.global_gain, 8)
                if self.__mpeg.version == 3:
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.scalefac_compress, 4)
                else:
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.scalefac_compress, 9)
                self.__putbits(0, 1)
                for region in range(3):
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.table_select[region], 5)

                self.__putbits(self.__side_info.gr[gr].ch[ch].tt.region0_count, 4)
                self.__putbits(self.__side_info.gr[gr].ch[ch].tt.region1_count, 3)

                if self.__mpeg.version == 3:
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.preflag, 1)
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.scalefac_scale, 1)
                    self.__putbits(self.__side_info.gr[gr].ch[ch].tt.count1table_select, 1)

    def __encode_main_data(self):
        for gr in range(self.__mpeg.granules_per_frame):
            for ch in range(self.__wav_file.num_of_channels):
                slen1 = tables.slen1_tab[self.__side_info.gr[gr].ch[ch].tt.scalefac_compress]
                slen2 = tables.slen2_tab[self.__side_info.gr[gr].ch[ch].tt.scalefac_compress]
                if gr == 0 or self.__side_info.scfsi[ch][0] == 0:
                    for sfb in range(6):
                        self.__putbits(self.__scalefactor.l[gr][ch][sfb], slen1)
                if gr == 0 or self.__side_info.scfsi[ch][1] == 0:
                    for sfb in range(6, 11, 1):
                        self.__putbits(self.__scalefactor.l[gr][ch][sfb], slen1)
                if gr == 0 or self.__side_info.scfsi[ch][2] == 0:
                    for sfb in range(11, 16, 1):
                        self.__putbits(self.__scalefactor.l[gr][ch][sfb], slen2)
                if gr == 0 or self.__side_info.scfsi[ch][3] == 0:
                    for sfb in range(16, 21, 1):
                        self.__putbits(self.__scalefactor.l[gr][ch][sfb], slen2)

                self.__huffman_code_bits(gr, ch)

    # write N bits into the bit stream.
    # bs = bit stream structure
    # val = value to write into the buffer
    # N = number of bits of val
    def __putbits(self, val, N):
        if self.__bitstream.cache_bits > N:
            self.__bitstream.cache_bits -= N
            self.__bitstream.cache |= val << self.__bitstream.cache_bits
        else:
            if self.__bitstream.data_position + 1 >= self.__bitstream.data_size:
                if not self.__bitstream.data:
                    self.__bitstream.data = np.zeros(self.__bitstream.data_size + (self.__bitstream.data_size // 2))
                else:
                    self.__bitstream.data = np.append(self.__bitstream.data, np.zeros(self.__bitstream.data_size // 2))
                self.__bitstream.data_size += self.__bitstream.data_size // 2
            N -= self.__bitstream.cache_bits
            self.__bitstream.cache |= val >> N

            self.__bitstream.data_position += 1
            self.__bitstream.cache_bits = 32 - N
            if N != 0:
                self.__bitstream.cache = val << self.__bitstream.cache_bits
            else:
                self.__bitstream.cache = 0

    def __huffman_code_bits(self, gr, ch):
        scale_fac = tables.scale_fact_band_index[self.__mpeg.samplerate_index][0]

        bits = util.get_bits_count(self.__bitstream)

        # 1: Write the bigvalues
        big_values = self.__side_info.gr[gr].ch[ch].tt.big_values << 1

        scalefac_index = self.__side_info.gr[gr].ch[ch].tt.region0_count + 1
        region1_start = scale_fac[scalefac_index]
        scalefac_index += self.__side_info.gr[gr].ch[ch].tt.region1_count + 1
        region2_start = scale_fac[scalefac_index]

        for i in range(0, big_values, 2):
            # get table pointer
            idx = (i >= region1_start) + (i >= region2_start)
            table_index = self.__side_info.gr[gr].ch[ch].tt.table_select[idx]
            # get huffman code
            if table_index:
                x = self.__l3_enc[ch][gr][0][i]
                y = self.__l3_enc[ch][gr][0][i + 1]
                self.__huffman_code(table_index, x, y)

        # 2: Write count1 area
        h = tables.huffman_table[self.__side_info.gr[gr].ch[ch].tt.count1table_select + 32]
        count1_end = big_values + (self.__side_info.gr[gr].ch[ch].tt.count1 << 2)
        for i in range(big_values, count1_end, 4):
            v = self.__l3_enc[ch][gr][0][i]
            w = self.__l3_enc[ch][gr][0][i + 1]
            x = self.__l3_enc[ch][gr][0][i + 2]
            y = self.__l3_enc[ch][gr][0][i + 3]
            self.__huffman_coder_count1(h, v, w, x, y)

        bits = util.get_bits_count(self.__bitstream) - bits
        bits = self.__side_info.gr[gr].ch[ch].tt.part2_3_length - self.__side_info.gr[gr].ch[ch].tt.part2_length - bits
        if bits:
            stuffing_words = bits // 32
            remaining_bits = bits % 32

            # Due to the nature of the Huffman code tables, we will pad with ones * /
            while stuffing_words:
                self.__putbits(~0, 32)
                if remaining_bits:
                    self.__putbits((np.int64(1) << remaining_bits) - 1, remaining_bits)

                stuffing_words -= 1

    def __huffman_code(self, table_select, x, y):
        ext = 0
        x_bits = 0

        x, sign_x = util.abs_and_sign(x)
        y, sign_y = util.abs_and_sign(y)

        h = tables.huffman_table[table_select]
        y_len = h.ylen
        if table_select > 15:  # ESC-table is used
            lin_bits_x = 0
            lin_bits_y = 0
            lin_bits = h.linbits
            if x > 14:
                lin_bits_x = x - 15
                x = 15

            if y > 14:
                lin_bits_y = y - 15
                y = 15

            idx = (x * y_len) + y
            code = h.table[idx]
            c_bits = h.hlen[idx]

            if x > 14:
                ext |= lin_bits_x
                x_bits += lin_bits

            if x != 0:
                ext <<= 1
                ext |= sign_x
                x_bits += 1

            if y > 14:
                ext <<= lin_bits
                ext |= lin_bits_y
                x_bits += lin_bits
            if y != 0:
                ext <<= 1
                ext |= sign_y
                x_bits += 1

            self.__putbits(code, c_bits)
            self.__putbits(ext, x_bits)
        else:  # No ESC-words
            idx = (x * y_len) + y
            code = h.table[idx]
            c_bits = h.hlen[idx]
            if x != 0:
                code <<= 1
                code |= sign_x
                c_bits += 1

            if y != 0:
                code <<= 1
                code |= sign_y
                c_bits += 1

            self.__putbits(code, c_bits)

    def __huffman_coder_count1(self, h, v, w, x, y):
        code = 0
        cbits = 0

        v, signv = util.abs_and_sign(v)
        w, signw = util.abs_and_sign(w)
        x, signx = util.abs_and_sign(x)
        y, signy = util.abs_and_sign(y)

        p = v + (w << 1) + (x << 2) + (y << 3)
        self.__putbits(h.table[p], h.hlen[p])

        if v:
            code = signv
            cbits = 1
        if w:
            code = (code << 1) | signw
            cbits += 1
        if x:
            code = (code << 1) | signx
            cbits += 1
        if y:
            code = (code << 1) | signy
            cbits += 1
        self.__putbits(code, cbits)
