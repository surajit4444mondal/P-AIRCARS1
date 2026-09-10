import pytest
import sunpy
import os
import numpy as np
from astropy.io import fits
from unittest.mock import patch, MagicMock
from paircars.utils.mwa_ploting_utils import *


@patch("paircars.utils.mwa_ploting_utils.os.system")
@patch("paircars.utils.mwa_ploting_utils.drop_cache")
@patch("paircars.utils.mwa_ploting_utils.run_shadems")
@patch("paircars.utils.mwa_ploting_utils.glob.glob")
@patch("paircars.utils.mwa_ploting_utils.Image.open")
@patch("paircars.utils.mwa_ploting_utils.get_ms_scan_size")
@patch("casatools.ms")
@patch("paircars.utils.mwa_ploting_utils.msmetadata")
def test_plot_ms_diagnostics(
    mock_msmetadata,
    mock_casamstool,
    mock_get_ms_scan_size,
    mock_Image_open,
    mock_glob,
    mock_run_shadems,
    mock_drop_cache,
    mock_os_system,
    tmp_path,
):
    vm = MagicMock()
    vm.available = 8 * 1024**3
    mstool = MagicMock()
    mstool.nrow.return_value = 1000
    mock_casamstool.return_value = mstool
    msmd = MagicMock()
    msmd.ncorrforpol.return_value = [4]
    msmd.scannumbers.return_value = [1, 2, 3]
    mock_msmetadata.return_value = msmd
    mock_get_ms_scan_size.side_effect = [300, 400, 300]

    def glob_side_effect(pattern):
        if pattern.endswith("_plots*.pdf"):
            return []
        if "amp" in pattern and pattern.endswith("*.png"):
            return ["amp1.png", "amp2.png"]
        if "phase" in pattern and pattern.endswith("*.png"):
            return ["phase1.png"]
        if "real" in pattern and pattern.endswith("*.png"):
            return []
        if "imag" in pattern and pattern.endswith("*.png"):
            return []
        return []

    mock_glob.side_effect = glob_side_effect
    mock_img_converted_primary = MagicMock()
    mock_img_converted_extra = MagicMock()

    def image_open_side_effect(path):
        m = MagicMock()
        if os.path.basename(path) in {"amp1.png", "phase1.png"}:
            m.convert.return_value = mock_img_converted_primary
        else:
            m.convert.return_value = mock_img_converted_extra
        return m

    mock_Image_open.side_effect = image_open_side_effect
    outdir = tmp_path / "out"
    outdir.mkdir()
    code, output_pdf_list = plot_ms_diagnostics(
        "test.ms",
        str(outdir),
    )
    assert code == 0
    assert isinstance(output_pdf_list, list)
    assert len(output_pdf_list) >= 1
    assert mock_img_converted_primary.save.call_count >= 1
    assert mock_run_shadems.call_count > 0
    rm_calls = [c for c in mock_os_system.call_args_list if "rm -rf" in c.args[0]]
    assert len(rm_calls) == 4
    mock_drop_cache.assert_called_once_with("test.ms")
    mstool.open.assert_called_once_with("test.ms")
    mstool.close.assert_called_once()
    msmd.open.assert_called_once_with("test.ms")
    msmd.close.assert_called_once()


def test_get_mwamap(dummy_image):
    result = get_mwamap(dummy_image)
    assert isinstance(result, sunpy.map.GenericMap)


def test_save_in_hpc(dummy_image):
    outdir = os.path.dirname(os.path.abspath(dummy_image))
    outfile = f"{outdir}/{os.path.basename(dummy_image).split('.fits')[0]}_HPC.fits"
    result = save_in_hpc(dummy_image)
    assert result == outfile
    assert os.path.exists(outfile) == True
    header = fits.getheader(outfile)
    assert header["CTYPE1"] == "HPLN-TAN"
    assert header["CTYPE2"] == "HPLT-TAN"
    os.system(f"rm -rf {outfile}")
    assert os.path.exists(outfile) == False


def test_plot_in_hpc(dummy_image):
    imagelist, sunmap = plot_in_hpc(dummy_image, extensions=["png"])
    assert len(imagelist) == 1
    assert imagelist[0][-4:] == ".png"
    assert os.path.exists(imagelist[0]) == True
    assert isinstance(sunmap, sunpy.map.GenericMap)
    os.system(f"rm -rf {imagelist[0]}")
    assert os.path.exists(imagelist[0]) == False


@patch("paircars.utils.mwa_ploting_utils.get_ms_scans", return_value=[1])
def test_plot_goes_full_timeseries(mock_get_validscans, dummy_msname):
    workdir = os.getcwd()
    plot_file_prefix = "test_goes_tseries"
    result = plot_goes_full_timeseries(
        dummy_msname,
        str(workdir),
        plot_file_prefix=plot_file_prefix,
    )
    mock_get_validscans.assert_called_once_with(dummy_msname)
    assert str(result) == f"{workdir}/{plot_file_prefix}.png"
    assert os.path.exists(result) == True
    assert result[-4:] == ".png"
    os.system(f"rm -rf {result}")
    assert os.path.exists(result) == False


@pytest.mark.parametrize(
    "imagename, expected_suffix",
    [
        ("mock_MFS_image.fits", "_MFS.fits"),  # Should include _MFS
        ("mock_image.fits", ".fits"),  # No _MFS
    ],
)
@patch("paircars.utils.mwa_ploting_utils.make_mwa_overlay")
@patch("paircars.utils.mwa_ploting_utils.plot_in_hpc")
@patch("paircars.utils.mwa_ploting_utils.save_in_hpc")
@patch("paircars.utils.mwa_ploting_utils.os.makedirs")
@patch("paircars.utils.mwa_ploting_utils.os.system")
@patch("paircars.utils.mwa_ploting_utils.fits.open")
@patch("paircars.utils.mwa_ploting_utils.fits.getheader")
@patch("paircars.utils.mwa_ploting_utils.SkyCoord")
@patch("paircars.utils.mwa_ploting_utils.Time")
@patch("paircars.utils.mwa_ploting_utils.cutout_image")
@patch("paircars.utils.mwa_ploting_utils.calc_solar_image_stat")
def test_rename_mwasolar_image(
    mock_calc_stats,
    mock_cutout,
    mock_Time,
    mock_SkyCoord,
    mock_getheader,
    mock_fits_open,
    mock_os_system,
    mock_os_makedirs,
    mock_save_hpc,
    mock_plot_hpc,
    mock_overlay,
    imagename,
    expected_suffix,
):
    # Setup dummy image output name
    mock_cutout.return_value = imagename

    # Setup header
    mock_header = {"DATE-OBS": "2021-01-01T12:00:00", "CRVAL3": 1.4e9}
    mock_getheader.return_value = mock_header

    # Setup FITS open context manager
    mock_hdul = MagicMock()
    mock_fits_open.return_value.__enter__.return_value = mock_hdul
    mock_hdul.__getitem__.return_value.header = {}

    # Setup astropy Time
    mock_Time.return_value.jd = 2459215.0
    mock_coords = MagicMock()
    mock_coords.ra.deg = 100.0
    mock_coords.dec.deg = 45.0
    mock_SkyCoord.return_value = mock_coords

    # Setup image stats
    mock_calc_stats.return_value = (10, -1, 1.0, 50, 5.0, 3.0, 2.0, 8.0)

    # Call rename_mwasolar_image
    result = rename_mwasolar_image(imagename, imagedir="/tmp/images")

    # Check suffix
    assert result.endswith(expected_suffix)


@pytest.mark.parametrize("plot_quantity", ["TB", "flux"])
@patch("paircars.utils.mwa_ploting_utils.interpolate_nans", side_effect=lambda x: x)
@patch("paircars.utils.mwa_ploting_utils.ImageNormalize")
@patch("paircars.utils.mwa_ploting_utils.LogStretch")
@patch("paircars.utils.mwa_ploting_utils.plt")
@patch("paircars.utils.mwa_ploting_utils.matplotlib")
@patch("paircars.utils.mwa_ploting_utils.os.system")
@patch("paircars.utils.mwa_ploting_utils.TimeSeries")
@patch("paircars.utils.mwa_ploting_utils.Fido")
@patch("paircars.utils.mwa_ploting_utils.np.load")
def test_make_ds_plot(
    m_npload,
    m_Fido,
    m_TimeSeries,
    m_os_system,
    m_matplotlib,
    m_plt,
    m_LogStretch,
    m_ImageNormalize,
    m_interp,
    plot_quantity,
):
    freqs = np.linspace(100, 110, 5)
    times = np.arange(10)
    timestamps = np.array([f"2025-01-01T00:{i:02d}:00" for i in range(10)])
    T_data = np.random.rand(5, 10) * 1e7
    S_data = np.random.rand(5, 10)
    flags = np.zeros((5, 10), dtype=bool)
    m_npload.side_effect = [
        (freqs, times, timestamps, T_data, S_data, flags),
        (freqs, times, timestamps, T_data, S_data, flags),
    ]
    m_Fido.search.return_value = "search_results"
    m_Fido.fetch.return_value = ["goes_file.fits"]
    fake_goes = MagicMock()
    fake_goes.truncate.return_value = fake_goes
    fake_goes.time.to_datetime.return_value = np.array(
        ["2025-01-01T00:00:00", "2025-01-01T00:09:00"]
    )
    m_TimeSeries.return_value = fake_goes
    fake_fig = MagicMock()
    fake_ax = MagicMock()
    fake_cax = MagicMock()
    m_plt.figure.return_value = fake_fig
    fake_fig.add_subplot.side_effect = [
        fake_ax,  # ax_spec
        fake_ax,  # ax_ts
        fake_ax,  # ax_goes
        fake_cax,  # cax
    ]
    fake_ax.get_yticks.return_value = np.arange(5)
    fake_fig.colorbar.return_value = MagicMock()
    output = make_ds_plot(
        dsfiles="fake_ds.npy",
        plot_file="output.png",
        plot_quantity=plot_quantity,
        showgui=False,
    )
    assert output == "output.png"
    assert m_npload.call_count == 2
    m_Fido.search.assert_called_once()
    m_Fido.fetch.assert_called_once()
    m_TimeSeries.assert_called_once()
    m_plt.savefig.assert_called_once_with("output.png", bbox_inches="tight")
    m_plt.show.assert_not_called()
    assert m_plt.close.call_count >= 1
