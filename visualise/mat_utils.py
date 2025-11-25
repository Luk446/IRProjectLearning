import os
import colorsys
import re
from datetime import datetime


def get_latest_robot_position_folder(dir, id=0):
    files = [f for f in os.listdir(dir) if re.match(r"\d{8}-\d{6}$", f)]
    files.sort(
        key=lambda f: datetime.strptime(f, "%Y%m%d-%H%M%S"), reverse=True
    )
    if id != 0:
        return files[id - 1] if len(files) >= id else None
    return files[0] if files else None


def brighten_color(color, factor=0.3):
    r, g, b, a = color
    h, light, s = colorsys.rgb_to_hls(r, g, b)
    light = min(1, light + factor * (1 - light))  # smoothly increase lightness
    r, g, b = colorsys.hls_to_rgb(h, light, s)
    return (r, g, b, a)
