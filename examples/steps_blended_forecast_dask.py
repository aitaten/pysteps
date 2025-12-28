# -*- coding: utf-8 -*-
"""
Blended forecast with Dask-backed NWP input
===========================================

Run with:
    python steps_blended_forecast_dask.py --mode raw-dask
    python steps_blended_forecast_dask.py --mode cascades-provider

This example mirrors the original blended forecast example but passes the NWP
inputs to STEPS using either a Dask array (raw-dask) or a lazy cascade provider
(cascades-provider) to avoid loading the full forecast into memory at once.
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

import pysteps
from pysteps import blending, cascade, io, nowcasts, rcparams, utils
from pysteps.visualization import plot_precip_field

try:
    import dask
    import dask.array as da
except ImportError:
    dask = None
    da = None


def main():
    ################################################################################
    # Read the radar images and the NWP forecast
    # ------------------------------------------
    #
    # First, we import a sequence of 3 images of 10-minute radar composites
    # and the corresponding NWP rainfall forecast that was available at that time.
    #
    # You need the pysteps-data archive downloaded and the pystepsrc file
    # configured with the data_source paths pointing to data folders.
    # Additionally, the pysteps-nwp-importers plugin needs to be installed, see
    # https://github.com/pySTEPS/pysteps-nwp-importers.
    #
    # Selected case
    date_radar = datetime.strptime("202010310400", "%Y%m%d%H%M")
    # The last NWP forecast was issued at 00:00
    date_nwp = datetime.strptime("202010310000", "%Y%m%d%H%M")
    radar_data_source = rcparams.data_sources["bom"]
    nwp_data_source = rcparams.data_sources["bom_nwp"]

    ###############################################################################
    # Load the data from the archive
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #
    root_path = radar_data_source["root_path"]
    path_fmt = "prcp-c10/66/%Y/%m/%d"
    fn_pattern = "66_%Y%m%d_%H%M00.prcp-c10"
    fn_ext = radar_data_source["fn_ext"]
    importer_name = radar_data_source["importer"]
    importer_kwargs = radar_data_source["importer_kwargs"]
    timestep = 10.0
    #
    # Find the radar files in the archive
    fns = io.find_by_date(
        date_radar, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_prev_files=2
    )
    #
    # Read the radar composites
    importer = io.get_method(importer_name, "importer")
    radar_precip, _, radar_metadata = io.read_timeseries(
        fns, importer, **importer_kwargs
    )
    #
    # Import the NWP data
    filename = os.path.join(
        nwp_data_source["root_path"],
        datetime.strftime(date_nwp, nwp_data_source["path_fmt"]),
        datetime.strftime(date_nwp, nwp_data_source["fn_pattern"])
        + "."
        + nwp_data_source["fn_ext"],
    )
    #
    nwp_importer = io.get_method("bom_nwp", "importer")
    nwp_precip, _, nwp_metadata = nwp_importer(filename)

    # Only keep the NWP forecasts from the last radar observation time (2020-10-31 04:00)
    # onwards

    nwp_precip = nwp_precip[24:43, :, :]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["raw-dask", "cascades-provider"],
        default="raw-dask",
        help="How to provide precip_models to STEPS",
    )
    args = parser.parse_args()

    if dask is None or da is None:
        print("This example requires dask and dask.array. Please install dask first.")
        sys.exit(1)

    ################################################################################
    # Pre-processing steps
    # --------------------

    # Make sure the units are in mm/h
    converter = pysteps.utils.get_method("mm/h")
    radar_precip, radar_metadata = converter(radar_precip, radar_metadata)
    nwp_precip, nwp_metadata = converter(nwp_precip, nwp_metadata)

    # Threshold the data
    radar_precip[radar_precip < 0.1] = 0.0
    nwp_precip[nwp_precip < 0.1] = 0.0

    # Plot the radar rainfall field and the first time step of the NWP forecast.
    date_str = datetime.strftime(date_radar, "%Y-%m-%d %H:%M")
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plot_precip_field(
        radar_precip[-1, :, :],
        geodata=radar_metadata,
        title=f"Radar observation at {date_str}",
        colorscale="STEPS-NL",
    )
    plt.subplot(122)
    plot_precip_field(
        nwp_precip[0, :, :],
        geodata=nwp_metadata,
        title=f"NWP forecast at {date_str}",
        colorscale="STEPS-NL",
    )
    plt.tight_layout()
    plt.show()

    # transform the data to dB
    transformer = pysteps.utils.get_method("dB")
    radar_precip, radar_metadata = transformer(
        radar_precip, radar_metadata, threshold=0.1
    )
    nwp_precip, nwp_metadata = transformer(nwp_precip, nwp_metadata, threshold=0.1)

    # r_nwp has to be four dimentional (n_models, time, y, x).
    # If we only use one model:
    if nwp_precip.ndim == 3:
        nwp_precip = nwp_precip[None, :]

    ###############################################################################
    # For the initial time step (t=0), the NWP rainfall forecast is not that different
    # from the observed radar rainfall, but it misses some of the locations and
    # shapes of the observed rainfall fields. Therefore, the NWP rainfall forecast will
    # initially get a low weight in the blending process.
    #
    # Determine the velocity fields
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    oflow_method = pysteps.motion.get_method("lucaskanade")

    # First for the radar images
    velocity_radar = oflow_method(radar_precip)

    # Then for the NWP forecast
    velocity_nwp = []
    # Loop through the models
    for n_model in range(nwp_precip.shape[0]):
        # Loop through the timesteps. We need two images to construct a motion
        # field, so we can start from timestep 1. Timestep 0 will be the same
        # as timestep 1.
        _v_nwp_ = []
        for t in range(1, nwp_precip.shape[1]):
            v_nwp_ = oflow_method(nwp_precip[n_model, t - 1 : t + 1, :])
            _v_nwp_.append(v_nwp_)
            v_nwp_ = None
        # Add the velocity field at time step 1 to time step 0.
        _v_nwp_ = np.insert(_v_nwp_, 0, _v_nwp_[0], axis=0)
        velocity_nwp.append(_v_nwp_)
    velocity_nwp = np.stack(velocity_nwp)

    ################################################################################
    # Prepare precip_models for STEPS blending
    # ----------------------------------------
    #
    # Two modes are available:
    #   raw-dask: pass the NWP forecasts as a Dask array (shape: n_models, T, M, N)
    #   cascades-provider: pass a callable that returns a cascade dict for
    #                      a given (model, timestep) without stacking everything.

    n_models, n_timesteps, m, n = nwp_precip.shape
    if n_models < 1 or n_timesteps < 2:
        raise ValueError("NWP precipitation needs at least one model and two timesteps")

    precip_models = nwp_precip
    if args.mode == "raw-dask":
        # Chunk on model and timestep so STEPS can grab slices lazily.
        precip_models = da.from_array(
            nwp_precip, chunks=(1, 1, m, n)
        )  # only timestep/model slices should compute as needed
        print(f"Using raw-dask mode with precip_models chunks {precip_models.chunks}")

    elif args.mode == "cascades-provider":
        # Use the same decomposition method/bandpass filter as STEPS defaults.
        filter_method = cascade.get_method("gaussian")
        bandpass_filter = filter_method((m, n), n_levels=6)
        decomposition_method, _ = cascade.get_method("fft")
        fft_method = utils.get_method("numpy", shape=(m, n))

        cascade_cache = {}
        for model_idx in range(n_models):
            for timestep_idx in range(n_timesteps):
                cascade_cache[(model_idx, timestep_idx)] = decomposition_method(
                    field=nwp_precip[model_idx, timestep_idx],
                    bp_filter=bandpass_filter,
                    n_levels=6,
                    mask=None,
                    method="fft",
                    fft_method=fft_method,
                    output_domain="spatial",
                    compute_stats=True,
                    normalize=True,
                    compact_output=True,
                )

        def cascade_provider(model_idx: int, timestep_idx: int):
            return cascade_cache[(model_idx, timestep_idx)]

        precip_models = cascade_provider
        print("Using cascades-provider mode (lazy lookup from per-timestep cache).")

    precip_forecast = blending.steps.forecast(
        precip=radar_precip,
        precip_models=precip_models,
        velocity=velocity_radar,
        velocity_models=velocity_nwp,
        timesteps=18,
        timestep=timestep,
        issuetime=date_radar,
        n_ens_members=1,
        precip_thr=radar_metadata["threshold"],
        kmperpixel=radar_metadata["xpixelsize"] / 1000.0,
        noise_stddev_adj="auto",
        vel_pert_method=None,
    )

    # Transform the data back into mm/h
    precip_forecast, _ = converter(precip_forecast, radar_metadata)
    radar_precip_mmh, _ = converter(radar_precip, radar_metadata)
    nwp_precip_mmh, _ = converter(nwp_precip, nwp_metadata)

    ################################################################################
    # Visualize the output
    # ~~~~~~~~~~~~~~~~~~~~
    #
    # The NWP rainfall forecast has a lower weight than the radar-based extrapolation
    # forecast at the issue time of the forecast (+0 min). Therefore, the first time
    # steps consist mostly of the extrapolation.
    # However, near the end of the forecast (+180 min), the NWP share in the blended
    # forecast has become more important and the forecast starts to resemble the
    # NWP forecast more.

    fig = plt.figure(figsize=(5, 12))

    leadtimes_min = [30, 60, 90, 120, 150, 180]
    n_leadtimes = len(leadtimes_min)
    for n_lead, leadtime in enumerate(leadtimes_min):
        # Nowcast with blending into NWP
        ax1 = plt.subplot(n_leadtimes, 2, n_lead * 2 + 1)
        plot_precip_field(
            precip_forecast[0, int(leadtime / timestep) - 1, :, :],
            geodata=radar_metadata,
            title=f"Nowcast +{leadtime} min",
            axis="off",
            colorscale="STEPS-NL",
            colorbar=False,
        )
        ax1.axis("off")

        # Raw NWP forecast
        plt.subplot(n_leadtimes, 2, n_lead * 2 + 2)
        ax2 = plot_precip_field(
            nwp_precip_mmh[0, int(leadtime / timestep) - 1, :, :],
            geodata=nwp_metadata,
            title=f"NWP +{leadtime} min",
            axis="off",
            colorscale="STEPS-NL",
            colorbar=False,
        )
        ax2.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
