#!/usr/bin/env python3
# Dynamic LandscapeDNDC Sitefile Creator (DLSC)
#
# Use this tool to build XML LDNDC site files
# __________________________________________________
# 2019/10/13, christian.werner@kit.edu
#
# descr: dynamically select regions and create (arable) LDNDC
#        site.xml file
#
#
# Christian Werner (IMK-IFU, KIT)
# christian.werner@kit.edu

import logging
import os
from importlib import resources
from pathlib import Path

import intake
import numpy as np
from pydantic import ValidationError
from tqdm import tqdm

from ldndctools.cli.cli import cli
from ldndctools.cli.selector import ask_for_resolution, CoordinateSelection, Selector
from ldndctools.extra import get_config, set_config
from ldndctools.misc.create_data import create_dataset
from ldndctools.misc.types import BoundingBox, RES
from ldndctools.sources.soil.soil_iscricwise import ISRICWISE_SoilDataset
from ldndctools.sources.soil.soil_national import NATIONAL_SoilDataset

log = logging.getLogger(__name__)
log.setLevel("INFO")

NODATA = "-99.99"

# TODO: there has to be a better way here...
#       also, tqdm takes no effect
#with resources.path("data", "") as dpath:
#    DPATH = Path(dpath)


def main( **kwargs):
    # parse args
    args = cli()

    # read config
    if 'config' in kwargs:
        cfg = kwargs['config']
    else:
        cfg = get_config(args.config)

    # write config
    #if args.storeconfig:
    #    set_config(cfg)


    # def _get_cfg_item(group, item, save="na"):
    #     return cfg[group].get(item, save)

    # TODO: move this to file
    # BASEINFO = dict(
    #     AUTHOR=_get_cfg_item("info", "author"),
    #     EMAIL=_get_cfg_item("info", "email"),
    #     DATE=str(datetime.datetime.now()),
    #     DATASET=_get_cfg_item("project", "dataset"),
    #     VERSION=_get_cfg_item("project", "version", save="0.1"),
    #     SOURCE=_get_cfg_item("project", "source"),
    # )

    if False: #(args.rcode is not None) or (args.file is not None):
        log.info("Non-interactive mode...")
        cfg["interactive"] = False
    else:
        pass
        #log.info("Interactive mode...")

    # query environment or command flags for selection (non-interactive mode)
    args.rcode = os.environ.get("DLSC_REGION", args.rcode)
    rcode = args.rcode.split("+") if args.rcode else None

    if not RES.contains(args.resolution):
        log.error(f"Wrong resolution: {args.resolution}. Use HR, MR or LR.")
        exit(-1)

    res = RES[cfg["resolution"]]


    bbox = None
    if  ('lat' in cfg) and ('lon' in cfg):
       cfg['bbox'] = [float(cfg["lon"]-1.0), float(cfg["lat"]-1.0), 
                      float(cfg["lon"]+1.0), float(cfg["lat"]+1.0)]
       log.info("Creating bounding box for coordinates specified.")

    if cfg["bbox"]:
        if type( cfg["bbox"]) == list:
            x1, y1, x2, y2 = cfg["bbox"]
        else:
            x1, y1, x2, y2 = [float(x) for x in cfg["bbox"].split(",")]
        try:
            bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        except ValidationError:
            log.info("Ilegal bounding box coordinates specified.")
            log.info("required: x1,y1,x2,y2 [cond: x1<x2, y1<y2]")
            log.info(f"given:    x1={x1},y1={y1},x2={x2},y2={y2}")
            exit(1)
        log.info(f"given:    x1={x1},y1={y1},x2={x2},y2={y2}")
        
    if "outfile" not in cfg:
        cfg["outfile"] = f"sites_{res.name}.xml"

    if not cfg["outfile"].endswith(".xml"):
        cfg["outfile"] = f'{cfg["outfile"]}.xml'

    log.info(f"Soil resolution: {res.name} {res.value}")
    log.info(f'Outfile name:    {cfg["outfile"]}')

    res_scale_mapper = {RES.LR: 50, RES.MR: 50, RES.HR: 10}

    with resources.path("data", "catalog.yml") as cat:
        catalog = intake.open_catalog(str(cat))

    df = catalog.admin(scale=res_scale_mapper[res]).read()
    
    print("config ", cfg)
    if "soil" in cfg and cfg["soil"] == "national":
        soil_raw = catalog.soil_national(res=res.name, port=8082).read()
        print("soil_raw ",soil_raw)
        soil = NATIONAL_SoilDataset( soil_raw)
    else:
        soil_raw = catalog.soil(res=res.name, port=8082).read()
        soil = ISRICWISE_SoilDataset( soil_raw)

    if False: #args.file:
        selector = CoordinateSelection(args.file)
    else:
        selector = Selector(df)

    if False: #args.interactive:
        res = ask_for_resolution(cfg)
        selector.ask()
    #else:
    #    if rcode:
    #        selector.set_region(rcode)

    if bbox:
        log.info(f"Setting bounding box to {bbox}")
        selector.set_bbox(bbox)
        log.info(f"Setting bounding box to {bbox}")
    else:
        if isinstance(selector, Selector):
            log.info("Adjusting bounding box to selection extent")
            extent = selector.gdf_mask.bounds.iloc[0]

            new_bbox = BoundingBox(
                x1=np.floor(extent.minx).astype("float").item(),
                x2=np.ceil(extent.maxx).astype("float").item(),
                y1=np.floor(extent.miny).astype("float").item(),
                y2=np.ceil(extent.maxy).astype("float").item(),
            )
            selector.set_bbox(new_bbox)

    #log.info(selector.selected)

    with tqdm(total=1) as progressbar:
        xml, nc = create_dataset(soil, selector, res, cfg, progressbar)

    if ('output' in cfg) and cfg['output'] == 'stream':
        return xml
    else:
        open( cfg["outfile"], "w").write(xml)
        ENCODING = {
            "siteid": {"dtype": "int32", "_FillValue": -1, "zlib": True},
            "soilmask": {"dtype": "int32", "_FillValue": -1, "zlib": True},
        }
        nc.to_netcdf(cfg["outfile"].replace(".xml", ".nc"), encoding=ENCODING)

if __name__ == "__main__":
    main()
