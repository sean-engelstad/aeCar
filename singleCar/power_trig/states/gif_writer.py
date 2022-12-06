import utils
from utils.gifs import GifWriter
import os

my_writer = GifWriter(frames_per_second=5)
my_writer(gif_filename="power_trig_state.gif", path=os.getcwd())