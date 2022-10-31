import argparse
import gzip
import io
import platform
import warnings
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import dask
import intake
import logging
import numpy as np
import pandas as pd
import urllib3
import xarray as xr
from dask.distributed import Client, LocalCluster
from pydantic import ValidationError

from ldndctools.misc.geohash import coords2geohash_dec
from ldndctools.misc.types import BoundingBox

log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


@dataclass
class ClimateSiteStats:
    id: int
    latitude: float
    longitude: float

    wind_speed: float
    annual_precipitation: float
    temperature_average: float
    temperature_amplitude: float

    elevation: Optional[int] = -1
    cloudiness: Optional[float] = 0.5
    rainfall_intensity: Optional[float] = 5.0


def subset_climate_data(
    *,
    bbox: Optional[BoundingBox] = None,
    mask: Optional[xr.DataArray] = None,
    date_min: Optional[str] = None,
    date_max: Optional[str] = None,
) -> xr.Dataset:

    with resources.path("data", "catalog.yml") as cat:
        catalog = intake.open_catalog(str(cat))
        ds = catalog["climate_era5land_hr"].to_dask()

        if mask is not None:
            mask = mask.interp(lat=ds["lat"], lon=ds["lon"])
            ds = ds.where(mask>0)

        if bbox:
            ds = ds.sel(lat=slice(bbox.y1, bbox.y2), lon=slice(bbox.x1, bbox.x2))

        if date_min:
            ds = ds.sel(time=slice(date_min, None))

        if date_max:
            ds = ds.sel(time=slice(None, date_max))

    return ds


def amplitude(da):
    return (da.max(dim="time") - da.min(dim="time")) / 2


def fill_header_global(time: pd.Timestamp) -> str:
    txt = f"""
%global
time     = "{time}/1"
%cuefile = "cuefile.txt"

"""
    return txt


def fill_header(d: ClimateSiteStats) -> str:

    txt = f"""
%climate
name = "climate_{d.id}.txt.gz"
archive = "ERA5 Land (resampled to LDNDC HR)"
id   = "{d.id}"

%attributes
elevation  = {d.elevation}
latitude   = {d.latitude:.6f}
longitude  = {d.longitude:.6f}

wind speed = {d.wind_speed:.2f}
cloudiness = {d.cloudiness:.1f}
rainfall intensity = {d.rainfall_intensity}

annual precipitation  = {d.annual_precipitation:.1f}
temperature average   = {d.temperature_average:.2f}
temperature amplitude = {d.temperature_amplitude:.2f}

%data
*       *       tavg    tmin    tmax    grad    prec    rh      wind
"""
    return txt


def writer(
    df: pd.DataFrame, pid: int, *, args: Any = None
) -> Iterable[int]:

    buffer = io.StringIO()

    header_global = fill_header_global(df.time.min().date())
    buffer.write(header_global)

    all_hashes: Iterable[int] = []
    for geohash, gdf in df.groupby(df.index):
        geohash = int(geohash)


        #stats["tavg"] = ds.tavg.groupby("time.year").mean(dim="time").mean(dim="year")
        #stats["tamp"] = ds.tavg.groupby("time.year").apply(amplitude).mean(dim="year")
        #stats["prec"] = ds.prec.groupby("time.year").sum(dim="time").mean(dim="year")
        #stats["wind"] = ds.wind.groupby("time.year").mean(dim="time").mean(dim="year")
        yearly_mean = gdf.groupby([gdf.time.dt.year]).mean()
        yearly_sum = gdf.groupby([gdf.time.dt.year]).sum()
        #yearly_amp = gdf.groupby([gdf.time.dt.year]).apply(amplitude)
        data = ClimateSiteStats(
                                id=int(geohash),
                                latitude=float( gdf.lat.mean()),
                                longitude=float( gdf.lon.mean()),
                                wind_speed=float( yearly_mean.wind.mean()),
                                annual_precipitation=float( yearly_sum.prec.mean()),
                                temperature_average=float( yearly_mean.tavg.mean()),
                                temperature_amplitude=float(1.0))
        header = fill_header(data)
        buffer.write(header)

        gdf = gdf.sort_values(by="time")
        cols = ["year", "jday", "tavg", "tmin", "tmax", "grad", "prec", "rh", "wind"]
        values = [
            gdf.time.dt.year,
            gdf.time.dt.dayofyear,
            gdf.tavg,
            gdf.tmin,
            gdf.tmax,
            gdf.rad,
            gdf.prec,
            gdf.rh,
            gdf.wind,
        ]

        fmt = "%-8.0f%-8.0f%-8.2f%-8.2f%-8.2f%-8.2f%-8.1f%-8.1f%-8.2f"
        np.savetxt(buffer, pd.DataFrame(dict(zip(cols, values))), fmt=fmt)

        all_hashes.append(geohash)

        buffer.write("\n\n")
    buffer.seek(0)

    with gzip.open(args.outfolder / f"climdata-{pid:03}.txt.gz", "wt") as f:
        f.write(buffer.read())

    return all_hashes


@np.vectorize
def inner_func(x, lat, lon):
    return -1 if np.isnan(x) else coords2geohash_dec(lat=lat, lon=lon)


def geohash_xr(mask: xr.DataArray) -> xr.DataArray:
    #mask = mask.load()
    lon_xr = mask.lon.broadcast_like(mask)
    lat_xr = mask.lat.broadcast_like(mask)
    # TODO: allow  dask and when fixed remove mask.load())
    data = xr.apply_ufunc(inner_func, mask, lat_xr, lon_xr, output_dtypes=[np.int64], dask='allowed') 
    assert data.dtype == np.int64
    return data


def get_boundingbox(bbox_flag: str) -> Union[None, BoundingBox]:
    if not bbox_flag:
        return None

    x1, y1, x2, y2 = [float(x) for x in bbox_flag.split(",")]
    try:
        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
    except ValidationError:
        print("Ilegal bounding box coordinates specified.")
        print("required: x1,y1,x2,y2 [cond: x1<x2, y1<y2]")
        print(f"given:    x1={x1},y1={y1},x2={x2},y2={y2}")
        exit(1)
    return bbox


def get_mask(mask_flag: str) -> xr.DataArray:
    if not mask_flag:
        return None

    filepath, var = mask_flag.split(":")
    da = xr.open_dataset(filepath)[var].load()
    return xr.ones_like(da).where(da > 0)


def conf():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "outfolder",
        nargs="?",
        type=lambda p: Path(p).absolute(),
        default=Path.cwd() / "output",
        help="outpath for climate archive files",
    )

    parser.add_argument(
        "-b",
        "--bbox",
        dest="bbox",
        default=None,
        metavar="[X1,Y1,X2,Y2]",
        help="bounding box",
    )

    parser.add_argument(
        "-m",
        "--mask",
        dest="mask",
        default=None,
        help="netcdf file with mask variable [format: filename.nc:var]",
    )

    parser.add_argument(
        "--dmin",
        dest="date_min",
        default=None,
        type=str,
        help="minimum date to consider [format: 2000-01-01]",
    )

    parser.add_argument(
        "--dmax",
        dest="date_max",
        default=None,
        type=str,
        help="maximum date to consider [format: 2001-12-31]",
    )

    parser.add_argument(
        "-p",
        "--partitions",
        dest="npart",
        default=20,
        type=int,
        help="subdivision for output",
    )

    args = parser.parse_args()
    args.outfolder.mkdir(parents=True, exist_ok=True)

    # create stream handler and set level to debug
    sh = logging.StreamHandler()
    sh.setLevel( logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('[%(asctime)s - %(name)s - %(levelname)s - %(message)s]')

    # add formatter to ch
    sh.setFormatter( formatter)
 
    # add stream handler to logger
    log.addHandler( sh)

    return args


def main():

    args = conf()

    cluster = LocalCluster( n_workers=10, memory_limit="15 GiB", dashboard_address=":1234")
    client = Client( cluster)

    log.info(f"NOTE: You can see progress at {platform.node()}:1234 if bokeh is installed")

    bbox = get_boundingbox(args.bbox)
    mask = get_mask(args.mask)

    # mask = xr.open_dataset("VN_MISC5_V2.nc")["rice_rot"]
    # mask = xr.where(mask > 0, 1, np.nan)

    ##get data, dask array
    ds = subset_climate_data(
        bbox=bbox,  # BoundingBox(x1=101.5, x2=109.5, y1=8.0, y2=23.5),
        mask=mask,
        date_min=args.date_min,
        date_max=args.date_max,
    )

    log.debug(f"function done: subset_climate_data")

    if mask is not None:
        ds["mask"] = mask
        ## check that coords are close, then use those of reference file
        #coords_close = (
        #    np.allclose(mask.lat.values, stats.lat.values, atol=1e-05) *
        #    np.allclose(mask.lon.values, stats.lon.values, atol=1e-05)
        #    )
        #if coords_close:
        #    stats = stats.assign_coords({"lat": mask.lat, "lon": mask.lon})
        #else:
        #    raise ValueError("Coords from climate and ref datasets are not close enough.")
        
        #stats["mask"] = xr.ones_like(stats.tavg.load())
        #stats["mask"][:] = mask.values
        ##stats["mask"] = mask.reindex_like(stats.tavg, method="nearest", tolerance=1e-3)
    else:
        tavg = ds.tavg.groupby("time.year").mean(dim="time").mean(dim="year")
        ds["mask"] = xr.ones_like( tavg).where( tavg > -100)

    ds["geohash"] = geohash_xr( ds.mask)
    ds["geohash"].attrs["_FillValue"] = -1
    ds["geohash"].attrs["missing_value"] = -1
    ds = ds.stack(location=("lon", "lat"))
    ds = ds.swap_dims({"location": "geohash"})
    ds = ds.set_coords("geohash")
    del ds["location"]

    log.debug(f"location done")

    ddf = ds.to_dask_dataframe(dim_order=["geohash", "time"])
    ddf = ddf.loc[ddf.geohash > 0, :]
    ddf = ddf.set_index("geohash")
    ddf = ddf.repartition(npartitions=args.npart)

    log.debug(f"repartition done")

    # ignore precip for now???
    ddf = ddf.dropna(subset=["tavg", "tmin", "tmax", "rad", "rh", "wind"], how="all")
    partitions = ddf.to_delayed(optimize_graph=True)

    log.debug(f"partitions done")

    formatted = [
        dask.delayed(writer)(
            part, i, args=dask.delayed(args)
        )
        for i, part in enumerate(partitions)
    ]

    log.debug(f"formatted done")

    processed_geohashs = dask.compute(*formatted)

    log.debug(f"compute done")

    with open(args.outfolder / "ids.txt", "w") as out:
        for chunk_geohashs in processed_geohashs:
            out.write(" ".join([f"{ghash}" for ghash in chunk_geohashs]) + "\n")

if __name__ == "__main__":
    main()
