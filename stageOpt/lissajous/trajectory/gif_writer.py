import utils
from utils.gifs import GifWriter
import os

my_writer = GifWriter(frames_per_second=5)
my_writer(gif_filename="lissajous_traj.gif", path=os.getcwd())