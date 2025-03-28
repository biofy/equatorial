import os

os.environ["EARTH2STUDIO_CACHE"] = "/workspace/data/earth2cache"

from earth2studio.data import GFS
from earth2studio.models.px.sfno import VARIABLES
from earth2studio.utils.time import to_time_array

if __name__ == "__main__":
    gfs = GFS()
    gfs(to_time_array(["2025-03-03"]), VARIABLES)
