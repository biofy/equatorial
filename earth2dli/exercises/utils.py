import os
from datetime import datetime, timedelta
from math import ceil, floor

import numpy as np
import torch
from loguru import logger

LAT_LONS = {
    "Houston": (29.75, 264.75),
    "New Orleans": (30.0, 270.0),
    "San Francisco": (37.75, 237.5),
    "San Jose": (37.5, 238.0),
    "Tampa": (28.0, 277.5),
    "Paris": (48.75, 2.25),
    "London": (51.5, 0.0),
    "Munich": (48.0, 11.5),
    "Cairo": (30.0, 31.25),
    "Nairobi": (-1.25, 36.75),
    "Cape Town": (-34.0, 18.5),
    "Caracas": (10.5, 293.0),
    "Rio de Janeiro": (-23.0, 316.75),
    "Lima": (-12.0, 283.0),
    "Bangkok": (13.75, 100.5),
    "Taipei": (25.0, 121.5),
    "Tokyo": (35.75, 139.75),
    "Manila": (14.5, 121.0),
    "Vientiane": (18.0, 102.5),
    "Melbourne": (-37.75, 145.0),
    "Wellington": (-41.25, 174.75),
    "Suva": (-18.25, 178.5),
}

TCS = {
    # name: (recommended start, recommended location)
    "Harvey": ("2017-08-24 12:00:00", "Houston"),
    "Ida": ("2021-08-28 12:00:00", "New Orleans"),
    "Chanthu": ("2021-09-12 00:00:00", "Taipei"),
    "Ian": ("2022-09-26 12:00:00", "Tampa"),
    "Noru": ("2022-09-26 12:00:00", "Vientiane"),
    "Beryl": ("2024-07-04 00:00:00", "Houston"),
    "Geemi": ("2024-07-23 00:00:00", "Taipei"),
    "Helene": ("2024-09-25 00:00:00", "Tampa"),
    "Kong-Rey": ("2024-10-30 00:00:00", "Taipei"),
}


def get_locations() -> list[str]:
    return list(LAT_LONS)


def get_lat_lon(loc: str) -> tuple[float, float]:
    if loc not in LAT_LONS:
        logger.error("Could not find '%s', returning Houston coordinates", loc)
        loc = "Houston"
    return LAT_LONS[loc]


def check_cds() -> None:
    cds_api = os.path.join(os.path.expanduser("~"), ".cdsapirc")
    if not os.path.exists(cds_api):
        key = input(
            "Enter CDS access token (e.g., 12345678-1234-1234-1234-123456123456):"
        )
        with open(cds_api, "w") as f:
            f.write("url: https://cds.climate.copernicus.eu/api\n")
            f.write(f"key: {key}\n")


def get_recent_time() -> np.datetime64:
    tmp = (datetime.now() - timedelta(hours=6)).replace(
        minute=0, second=0, microsecond=0
    )
    return np.datetime64(tmp.replace(hour=(tmp.hour // 6) * 6).isoformat())


@torch.no_grad()
def specific_to_relative(
    q: torch.Tensor, p: torch.Tensor, T: torch.Tensor
) -> torch.Tensor:
    # See also
    # https://nvidia.github.io/earth2studio/examples/extend/03_custom_datasource.html
    epsilon = 0.621981

    e = (p * q * (1.0 / epsilon)) / (1 + q * (1.0 / (epsilon) - 1))

    es_w = 611.21 * torch.exp(17.502 * (T - 273.16) / (T - 32.19))
    es_i = 611.21 * torch.exp(22.587 * (T - 273.16) / (T + 0.7))

    alpha = torch.clip((T - 250.16) / (273.16 - 250.16), 0, 1.2) ** 2
    es = alpha * es_w + (1 - alpha) * es_i
    return 100 * e / es


def make_quarter_degree(coord_from: float, coord_to: float) -> np.ndarray:
    start = floor(coord_from * 4) / 4
    stop = ceil(coord_to * 4) / 4
    num = int(1 + 4 * (stop - start))
    return np.linspace(start, stop, num, endpoint=True)
