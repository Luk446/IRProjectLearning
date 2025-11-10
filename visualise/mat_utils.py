import os
import colorsys
import re
from datetime import datetime

def get_latest_robot_position_file(dir):
    files = [f for f in os.listdir(dir) if re.match(r"robot_position_\d{8}-\d{6}\.csv$", f)]
    return max(files, key=lambda f: datetime.strptime(f.split("_")[2].split(".")[0], "%Y%m%d-%H%M%S")) if files else None

def brighten_color(color, factor=0.3):
    r, g, b, a = color
    h, light, s = colorsys.rgb_to_hls(r, g, b)
    light = min(1, light + factor * (1 - light))  # smoothly increase lightness
    r, g, b = colorsys.hls_to_rgb(h, light, s)
    return (r, g, b, a)
