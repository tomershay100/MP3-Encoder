import os
import sys

import util
from MP3_Encoder import MP3Encoder
from WAV_Reader import WavReader

if __name__ == "__main__":
    if len(sys.argv) > 2:
        sys.exit('Unexpected number of arguments.')
    if len(sys.argv) < 2:
        sys.exit('No directory specified.')
    file_path = sys.argv[1]

    if not os.path.exists(file_path):
        sys.exit('File not found.')
    # default_mpeg = {"bit_rate": 64, "emp": None, "copyright": 0, "original": 1}
    wav_file = WavReader(file_path)
    encoder = MP3Encoder(wav_file)
    encoder.print_info()
    encoder.encode()
