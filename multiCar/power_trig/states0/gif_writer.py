import utils
from utils.gifs import GifWriter
import os

my_writer = GifWriter(frames_per_second=2)
my_writer(gif_filename="mpt_state0.gif", path=os.getcwd())