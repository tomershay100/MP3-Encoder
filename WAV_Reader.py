import struct
import sys
import util


class WavReader:
    def __init__(self, file_path, bit_rate=64):
        self.__file_path = file_path
        self.__bitrate = bit_rate
        self.__file = open(self.__file_path, 'rb')

        self.__read_header()

        self.check_bitrate_index()
        self.check_samplerate_index()

    # Reads the header information to check if it is a valid file with PCM audio samples.
    def __read_header(self):
        buffer = self.__file.read(128)

        idx = buffer.find(b'RIFF')  # bytes 1 - 4
        if idx == -1:
            sys.exit('Bad WAVE file.')
        idx += 4
        self.__chunk_size = struct.unpack('<I', buffer[idx:idx + 4])[0]  # bytes 5 - 8

        idx = buffer.find(b'WAVE')  # bytes 9 - 12
        if idx == -1:
            sys.exit('Bad WAVE file.')
        idx = buffer.find(b'fmt ')  # bytes 13 - 16 (format chunk marker)
        if idx == -1:
            sys.exit('Bad WAVE file.')

        idx += 4
        sub_chunk1_size = struct.unpack('<I', buffer[idx:idx + 4])[0]  # bytes 17 - 20, size of fmt section
        if sub_chunk1_size != 16:
            sys.exit('Unsupported WAVE file, compression used instead of PCM.')

        idx += 4
        format_type = struct.unpack('<H', buffer[idx:idx + 2])[0]  # bytes 21 - 22
        if format_type != 1:  # 1 for PCM
            sys.exit('Unsupported WAVE file, compression used instead of PCM.')

        idx += 2
        self.__num_of_ch = struct.unpack('<H', buffer[idx:idx + 2])[0]  # bytes 23 - 24
        if self.__num_of_ch > 1:  # Set to stereo mode if wave data is stereo, mono otherwise.
            self.__mpeg_mode = self.__num_of_ch
        else:
            self.__mpeg_mode = util.MODES["MONO"]

        idx += 2
        self.__samplerate = struct.unpack('<I', buffer[idx:idx + 4])[0]  # bytes 25 - 28
        if self.__samplerate not in (32000, 44100, 48000):
            sys.exit('Unsupported sampling frequency.')
        self.__sample_rate_code = {44100: 0b00, 48000: 0b01, 32000: 0b10}.get(self.__samplerate)

        idx += 4
        self.__byte_rate = struct.unpack('<I', buffer[idx:idx + 4])[0]  # bytes 29 - 32
        # ByteRate = (SampleRate * BitsPerSample * Channels) / 8

        idx += 4
        self.__block_align = struct.unpack('<H', buffer[idx:idx + 2])[0]  # bytes 33 - 34
        # BlockAlign = BitsPerSample * Channels / 8

        idx += 2
        self.__bits_per_sample = struct.unpack('<H', buffer[idx:idx + 2])[0]  # bytes 35 - 36
        if not (self.__bits_per_sample in (8, 16, 32)):
            sys.exit('Unsupported WAVE file, samples not int8, int16 or int32 type.')

        idx = buffer.find(b'data')  # bytes 37 - 40
        if idx == -1:
            sys.exit('Bad WAVE file.')

        idx += 4
        sub_chunk2_size = struct.unpack('<I', buffer[idx:idx + 4])[0]  # bytes 41 - 44, size of data section
        self.__num_of_samples = int(sub_chunk2_size * 8 / self.__bits_per_sample / self.__num_of_ch)

        self.__file.seek(idx + 4)

        self.__num_of_slots = 12 * self.__bitrate * 1000 // self.__samplerate
        self.__copyright = 0
        self.__original = 0
        self.__chmode = 0b11 if self.__num_of_ch == 1 else 0b10
        self.__modext = 0b10
        self.__sync_word = 0b11111111111
        self.__mpeg_version = 0b11
        self.__layer = 0b11
        self.__crc = 0b1
        self.__emphasis = 0b00
        self.__pad_bit = 0
        self.__rest = 0

        self.__header = (self.__sync_word << 21 | self.__mpeg_version << 19 |
                         self.__layer << 17 | self.__crc << 16 |
                         self.__bitrate << 7 | self.__sample_rate_code << 10 |
                         self.__pad_bit << 9 | self.__chmode << 6 |
                         self.__modext << 4 | self.__copyright << 3 |
                         self.__original << 2 | self.__emphasis)

    def check_bitrate_index(self):
        if util.find_bitrate_index(self.__bitrate, self.__mpeg_version) <= 0:
            sys.exit("Unsupported bitrate configuration.")  # error - not a valid bitrate for encoder

    def check_samplerate_index(self):
        if util.find_samplerate_index(self.__samplerate) <= 0:
            sys.exit("Unsupported samplerate configuration.")  # error - not a valid samplerate for encoder

    @property
    def mpeg_mode(self):
        return self.__mpeg_mode

    @property
    def bitrate(self):
        return self.__bitrate

    @property
    def emphasis(self):
        return self.__emphasis

    @property
    def copyright(self):
        return self.__copyright

    @property
    def original(self):
        return self.__original

    @property
    def samplerate(self):
        return self.__samplerate
