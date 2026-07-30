import numpy as np
import numexpr as ne
import traceback
import glob
import os
import subprocess
from casatools import msmetadata
from astropy.io import fits
from .basic_utils import (
    suppress_output,
    ra_dec_to_hms_dms,
    weighted_mean,
)
from .resource_utils import limit_threads
from .flagging import do_flag_backup, flag_quartical_table
from .solarflagger import flagger
from .calibration import (
    fluxcal_caltable,
    uvrange_casa_to_quartical,
    quartical_matrix_normalize,
    get_cal_flag_info,
)
from .imaging import calc_sun_dia
from .image_utils import (
    create_circular_mask,
    create_circular_mask_array,
    calc_dyn_range,
    generate_tb_map,
    make_timeavg_image,
    make_stokes_wsclean_imagecube,
)
from .udocker_utils import run_wsclean, run_quartical
from .sunpos_utils import determine_quiet_disk, cal_apparent_solarcenter


def cal_crossphase(imagename):
    """
    Function to calculate Stokes U, V leakage through correlation analysis

    Parameters
    ----------
    imagename : str
        FITS image

    Returns
    -------
    float
        Cross hand phase
    """
    data = fits.getdata(imagename)
    u_data = data[2, 0, ...].astype(np.float64)
    v_data = data[3, 0, ...].astype(np.float64)
    max_pos = np.where(np.abs(v_data) == np.nanmax(np.abs(v_data)))
    peak_v = v_data[max_pos][0]
    crossphase_list = np.arange(-180, 180, 1)
    psi = np.deg2rad(crossphase_list)
    if u_data.size > 1 and v_data.size > 1:
        cospsi = np.cos(psi)[:, None, None]
        sinpsi = np.sin(psi)[:, None, None]
        u = u_data[None, ...]
        v = v_data[None, ...]
        # rotation using numexpr
        new_u = ne.evaluate("cospsi*u + sinpsi*v")
        new_v = ne.evaluate("-sinpsi*u + cospsi*v")
        # compute correlation coefficient for each psi
        cc_list = []
        for i in range(len(crossphase_list)):
            cc = abs(np.corrcoef(new_u[i].ravel(), new_v[i].ravel())[1, 0])
            cc_list.append(cc)
        cc_list = np.array(cc_list)
    pos = np.argsort(cc_list)
    cross_phases = crossphase_list[pos[0:4]]
    psi = np.deg2rad(cross_phases)
    cospsi = np.cos(psi)[:, None, None]
    sinpsi = np.sin(psi)[:, None, None]
    u = u_data[None, ...]
    v = v_data[None, ...]
    new_u = ne.evaluate("cospsi*u + sinpsi*v")
    new_v = ne.evaluate("-sinpsi*u + cospsi*v")
    x = np.nanmax(np.abs(new_u), axis=(-2, -1))
    pos = np.argsort(x)
    for i in pos[0:2]:
        peak_new_v = new_v[i][max_pos]
        if np.sign(peak_new_v) == np.sign(peak_v):
            cross_phase = cross_phases[i]
            return cross_phase


def leakage_fitting(leakage_file_list):
    """
    Fit a 1D polynomial to Stokes I to Stokes Q leakage spectral variation
    
    Parameters
    ----------
    leakage_file_list : list
        Leakage file list 
        Note: numpy file with format [freq in MHz, dict {selfcal_iter:[q_leakage, u_leakage, v_leakage, q_err, u_err, v_err]}]
    
    Returns
    -------
    numpy.array
        Stokes I to Q leakage polynomial
    numpy.array
        Stokes I to U leakage polynomial
    numpy.array
        Stokes I to V leakage polynomial
    """
    freq_list = []
    q_list = []
    u_list = []
    v_list = []
    leakage_file_list = sorted(leakage_file_list)
    for leakage_file in leakage_file_list:
        freq, leakage_dic = np.load(leakage_file,allow_pickle=True)
        freq_list.append(freq)
        max_iter = max(leakage_dic.keys())
        (
            q_leakage,
            u_leakage,
            v_leakage,
            res_q_leakage,
            res_u_leakage,
            res_v_leakage,
        ) = leakage_dic[max_iter]
        q_list.append(q_leakage)
        u_list.append(u_leakage)
        v_list.append(v_leakage)
    if len(freq_list)==0:
        q_poly = [] 
        u_poly = []
        v_poly = []
    elif len(freq_list)==1:
        q_poly = q_list 
        u_poly = u_list 
        v_poly = v_list
    elif len(freq_list)==2:
        q_poly = np.polyfit(freq_list, q_list, deg=1)
        u_poly = np.polyfit(freq_list, u_list, deg=1)
        v_poly = np.polyfit(freq_list, v_list, deg=1)
    elif len(freq_list)==3:
        q_poly = np.polyfit(freq_list, q_list, deg=2)
        u_poly = np.polyfit(freq_list, u_list, deg=2)
        v_poly = np.polyfit(freq_list, v_list, deg=2)
    elif len(freq_list)>3:
        q_poly = np.polyfit(freq_list, q_list, deg=3)
        u_poly = np.polyfit(freq_list, u_list, deg=3)
        v_poly = np.polyfit(freq_list, v_list, deg=3)   
    return q_poly, u_poly, v_poly    
   

def do_uvsub_flag(msname, threshold_list=[10, 7, 5], ncpu=1):
    """
    Perform uv-sub flags

    Parameters
    ----------
    msname : str
        Measurement set
    threshold_list: list, optional
        Threshold list
    ncpu: int, optional
        Number of CPU threads to use
    """
    for threshold in threshold_list:
        result, n_final_flagged, n_additional_flagged = flagger(
            msname,
            "residual",
            threshold=threshold,
            num_processes=ncpu,
            flagbackup=False,
        )
        

def get_quiet_sun_flux(freq):
    """
    Get quiet Sun flux density in Jy.

    Parameters
    ----------
    freq : float
        Frequency in MHz

    Returns
    -------
    float
        Flux density in Jy
    """
    p = np.poly1d([-1.93715165e-06, 7.84627718e-04, -3.15744433e-02, 2.32834400e-01])
    flux = p(freq) * 10**4  # Polynomial return in SFU
    return flux


def make_qs_model(msname, clname="quiet_sun.cl"):
    """
    Make CASA component list of quiet Sun model

    Parameters
    ----------
    msname : str
        Name of the measurement set
    clname : str, optional
        Name of the component list

    Returns
    -------
    str
        Name of the component list file
    """
    from casatools import componentlist

    msmd = msmetadata()
    msmd.open(msname)
    freq = msmd.meanfreq(0, unit="MHz")
    phasecenter = msmd.phasecenter(0)
    msmd.close()

    radeg = np.rad2deg(phasecenter["m0"]["value"])
    decdeg = np.rad2deg(phasecenter["m1"]["value"])
    rahms, decdms = ra_dec_to_hms_dms(radeg, decdeg)
    radec_str = f"J2000 {rahms} {decdms}"
    sun_size = calc_sun_dia(freq)  # In arcmin
    QS_flux = get_quiet_sun_flux(freq)  # In Jy

    # Make sure the component list does not already exist. The tool will complain otherwise.
    os.system("rm -rf " + clname)
    cl = componentlist()
    cl.addcomponent(
        dir=radec_str,
        flux=QS_flux,  # For a gaussian, this is the integrated area.
        fluxunit="Jy",
        freq=f"{freq}MHz",
        shape="gaussian",  ## Gaussian
        majoraxis=f"{sun_size}arcmin",
        minoraxis=f"{sun_size}arcmin",
        positionangle="0deg",
        spectrumtype="spectral index",
        index=0.0,
    )
    # Save the file
    cl.rename(filename=clname)
    cl.done()
    return clname


def quiet_sun_selfcal(msname, logger, selfcaldir, refant="1", solint="int"):
    """
    Perform quiet Sun Gaussian model based self-calibration

    Parameters
    ----------
    msname : str
        Measurement set
    logger : str
        Python logger
    selfcaldir : str
        Self-calibration directory
    refant : str, optional
        Reference antenna
    solint : str, optional
        Solution interval

    Returns
    -------
    int
        Success message
    str
        Caltable name
    """
    from casatasks import ft, delmod, gaincal, applycal, flagmanager
    prefix = (
        selfcaldir + "/" + os.path.basename(msname).split(".ms")[0] + "_selfcal_present"
    )
    bpass_caltable = prefix.replace("present", f"{0}") + ".gcal"
    if os.path.exists(bpass_caltable):
        os.system("rm -rf " + bpass_caltable)
    do_flag_backup(msname, flagtype="qs_selfcal")

    try:
        ###################################
        # Import simulated QS model
        ###################################
        qs_model = make_qs_model(
            msname, clname=f"{os.path.basename(msname).split('.ms')[0]}_qs.cl"
        )
        delmod(vis=msname, otf=True, scr=True)
        ft(vis=msname, complist=qs_model, usescratch=True)
        os.system(f"rm -rf {qs_model}")

        #####################
        # Perform calibration
        #####################
        logger.info(
            f"gaincal(vis='{msname}',caltable='{bpass_caltable}',uvrange='<100lambda',refant='{refant}',solint='{solint}',minsnr=3,calmode='p')\n"
        )
        with suppress_output():
            gaincal(
                vis=msname,
                caltable=bpass_caltable,
                uvrange="<100lambda",
                refant=refant,
                minsnr=3,
                solint=f"{solint}",
                solnorm=True,
                calmode="p",
            )
        if not os.path.exists(bpass_caltable):
            logger.info("No gain solutions are found.\n")
            msg = 1
            bpass_caltable = ""
        else:
            ########################
            # Applying solutions
            ########################
            logger.info(
                f"applycal(vis={msname},gaintable=[{bpass_caltable}],interp=['linear'],applymode='calonly',calwt=[False])\n"
            )
            with suppress_output():
                applycal(
                    vis=msname,
                    gaintable=[bpass_caltable],
                    interp=["linear"],
                    applymode="calonly",
                    calwt=[False],
                )
            msg = 0
    except Exception:
        logger.exception(traceback.print_exc())
        msg = 2
        bpass_caltable = ""
    finally:
        with suppress_output():
            flagmanager(vis=msname, mode="restore", versionname="qs_selfcal_1")
            flagmanager(vis=msname, mode="delete", versionname="qs_selfcal_1")
        return msg, bpass_caltable


def check_valid_image(imagename):
    """
    Check whether the image is valid or not

    Parameters
    ----------
    imagename : str
        Image name

    Returns
    -------
    bool
        Whether valid image or not
    """
    data = fits.getdata(imagename)
    if np.nansum(data) == 0:
        return False
    else:
        return True


def calc_leakage(imagename, threshold=5, disc_size=50):
    """
    Calculate Stokes I to Q, U, V leakages

    Parameters
    ----------
    imagename : str
        Image name
    threshold : float
        Threshold to choose region with Stokes I detection
    disc_size : float
        Solar disc area in arcminute to mask for calculating rms
        N.B.: Chosen slightly larger to avoid any off-coronal emission from CMEs

    Returns
    -------
    float
        Stokes I to Q leakage
    float
        Stokes I to U leakage
    float
        Stokes I to V leakage
    float
        Stokes I to Q leakage error
    float
        Stokes I to U leakage error
    float
        Stokes I to V leakage error
    """
    valid_image = check_valid_image(imagename)
    disk_detected = determine_quiet_disk(imagename)
    if valid_image is False or disk_detected is False:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    tb_map = generate_tb_map(imagename)
    tb_data = fits.getdata(tb_map)[0, 0, ...] / 10**6  # in MK
    data = fits.getdata(imagename)
    header = fits.getheader(imagename)
    pix_size = abs(header["CDELT1"]) * 3600.0  # In arcsec
    radius = int((disc_size * 60) / pix_size)
    i_data = data[0, 0, ...]
    q_data = data[1, 0, ...]
    u_data = data[2, 0, ...]
    v_data = data[3, 0, ...]
    #############################
    # Calculating image rms
    #############################
    msg, _, _, center_x, center_y= cal_apparent_solarcenter(imagename)
    if msg==0:
        mask = create_circular_mask_array(i_data, radius, center_x=center_x, center_y=center_y)
    else:
        mask = create_circular_mask_array(i_data, radius)
    i_rms = np.nanstd(i_data[~mask])
    i_thresh = threshold * i_rms
    ##############################################
    # Estimating regions for leakage calculation
    ##############################################
    pos = np.where((i_data < i_thresh) | (tb_data > 1.0))
    q_data[pos] = np.nan
    u_data[pos] = np.nan
    v_data[pos] = np.nan
    q_by_i = q_data / i_data
    u_by_i = u_data / i_data
    v_by_i = v_data / i_data
    q_by_i = q_by_i[~np.isnan(q_by_i)].flatten()
    u_by_i = u_by_i[~np.isnan(u_by_i)].flatten()
    v_by_i = v_by_i[~np.isnan(v_by_i)].flatten()

    #########################################
    # Estimating leakage and leakage errors
    #########################################
    q_leakage = round(np.nanmedian(q_by_i), 4)
    u_leakage = round(np.nanmedian(u_by_i), 4)
    v_leakage = round(np.nanmedian(v_by_i), 4)
    
    q_cor = q_data - (q_leakage*i_data)
    u_cor = u_data - (u_leakage*i_data)
    v_cor = v_data - (v_leakage*i_data)

    q_leakage_err = round((3*np.nanstd(q_cor))/np.nanmax(i_data),6)
    u_leakage_err = round((3*np.nanstd(u_cor))/np.nanmax(i_data),6)
    v_leakage_err = round((3*np.nanstd(v_cor))/np.nanmax(i_data),6)
    os.system(f"rm -rf {tb_map}")
    
    return q_leakage, u_leakage, v_leakage, q_leakage_err, u_leakage_err, v_leakage_err


def correct_leakage(
    imagename,
    modelname="",
    q_leakage=0.0,
    u_leakage=0.0,
    v_leakage=0.0,
    threshold=5,
    disc_size=50,
):
    """
    Correct leakages in image plane

    Parameters
    ----------
    imagename : str
        Image name
    modelname : str, optional
        Model name
    q_leakage : float, optional
        Q leakage
    u_leakage : float, optional
        U leakage
    v_leakage : float, optional
        V leakage
    threshold : float
        Threshold to choose region with Stokes I detection
    disc_size : float
        Solar disc area in arcminute to mask for calculating rms
        N.B.: Chosen slightly larger to avoid any off-coronal emission from CMEs

    Returns
    -------
    str
        Leakage corrected imagename
    str
        Leakage corrected modelname
    """
    #######################
    # Read image data
    #######################
    imagedata = fits.getdata(imagename)
    image_I = imagedata[0, 0, ...]
    image_Q = imagedata[1, 0, ...]
    image_U = imagedata[2, 0, ...]
    image_V = imagedata[3, 0, ...]

    if os.path.exists(modelname):
        correct_model = True
    else:
        correct_model = False
    if correct_model:
        ##########################
        # Read model data
        ##########################
        modeldata = fits.getdata(modelname)
        model_I = modeldata[0, 0, ...]
        model_Q = modeldata[1, 0, ...]
        model_U = modeldata[2, 0, ...]
        model_V = modeldata[3, 0, ...]
        modelheader = fits.getheader(modelname)

    ###################################
    # Creating mask
    ####################################
    imageheader = fits.getheader(imagename)
    center_y, center_x = np.where(imagedata[0,0,...]==np.nanmax(imagedata[0,0,...]))
    pix_size = abs(imageheader["CDELT1"]) * 3600.0  # In arcsec
    radius = int((disc_size * 60) / pix_size)
    mask = create_circular_mask_array(image_I, radius, center_x = center_x[0], center_y = center_y[0])

    ####################################
    # Calculate rms
    ####################################
    q_rms = np.nanstd(image_Q[~mask])
    u_rms = np.nanstd(image_U[~mask])
    v_rms = np.nanstd(image_V[~mask])

    ###################################
    # Correcting images
    ###################################
    image_Q = image_Q - (q_leakage * image_I)
    image_U = image_U - (u_leakage * image_I)
    image_V = image_V - (v_leakage * image_I)
    posq = np.where(abs(image_Q) < threshold * q_rms)
    posu = np.where(abs(image_U) < threshold * u_rms)
    posv = np.where(abs(image_V) < threshold * v_rms)
    imagedata[1, 0, ...] = image_Q
    imagedata[2, 0, ...] = image_U
    imagedata[3, 0, ...] = image_V
    fits.writeto(
        imagename.split(".fits")[0] + "_leakagecor.fits",
        data=imagedata,
        header=imageheader,
        overwrite=True,
    )

    if correct_model:
        ####################################
        # Correcting model images
        ####################################
        model_Q = model_Q - (q_leakage * model_I)
        model_U = model_U - (u_leakage * model_I)
        model_V = model_V - (v_leakage * model_I)
        model_Q[posq] = 0.0
        model_U[posu] = 0.0
        model_V[posv] = 0.0
        modeldata[1, 0, ...] = model_Q
        modeldata[2, 0, ...] = model_U
        modeldata[3, 0, ...] = model_V
        modeldata[np.isinf(modeldata)] = 0.0
        modeldata[np.isnan(modeldata)] = 0.0
        fits.writeto(
            modelname.split(".fits")[0] + "_leakagecor.fits",
            data=modeldata,
            header=modelheader,
            overwrite=True,
        )

    if correct_model:
        return (
            imagename.split(".fits")[0] + "_leakagecor.fits",
            modelname.split(".fits")[0] + "_leakagecor.fits",
        )
    else:
        return (
            imagename.split(".fits")[0] + "_leakagecor.fits",
            None,
        )


def correct_pbcor_leakage(
    imagename,
    modelname,
    metafits,
    pbcor=True,
    leakagecor=True,
    pbuncor=True,
    leakage_info=[],
    ncpu=1,
):
    """
    Perform primary beam and leakage correction

    Parameters
    ----------
    imagename : str
        Image name
    modelname : str
        Model image name
    metafits : str
        Metafits file
    pbcor : bool, optional
        Perform primary beam correction
    leakagecor : bool, optional
        Perform image based residual leakage correction
    pbuncor : bool, optional
        Undo primary beam correction
    leakage_info : list, optional
        User provided leakages (no leakage calculation will be done)
    ncpu : int, optional
        Number of CPU threads

    Returns
    -------
    str
        Final image
    str
        Final model
    list
        Leakage and leakage error list
    """
    ncpu = max(1, ncpu)
    leakage_info = []
    freq = fits.getheader(imagename)["CRVAL3"]
    pbfile = f"freq_{freq}_pb.npy"
    if pbcor is False:
        pbcor_image = imagename
        pbcor_model = modelname
    else:
        ####################################
        # Correcting image
        ####################################
        pbcor_image = imagename.split(".fits")[0] + "_pbcor.fits"
        pbcor_cmds = [
            "run-mwa-singlepbcor",
            imagename,
            metafits,
            pbcor_image,
            "--interpolated",
            "--num_threads",
            f"{ncpu}",
            "--pb_jones_file",
            f"{pbfile}",
        ]
        if os.path.exists(pbfile) is False:
            pbcor_cmds.append("--save_pb")
        subprocess.run(
            pbcor_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

        #########################################
        # Correcting model
        #########################################
        pbcor_model = modelname.split(".fits")[0] + "_pbcor.fits"
        pbcor_cmds = [
            "run-mwa-singlepbcor",
            modelname,
            metafits,
            pbcor_model,
            "--interpolated",
            "--num_threads",
            f"{ncpu}",
            "--pb_jones_file",
            f"{pbfile}",
        ]
        subprocess.run(
            pbcor_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    if leakagecor is False:
        leakagecor_image = pbcor_image
        leakagecor_model = pbcor_model
    else:
        ########################################
        # Estimating and correcting leakage
        ########################################
        if len(leakage_info) == 0:
            (
                q_leakage,
                u_leakage,
                v_leakage,
                q_leakage_err,
                u_leakage_err,
                v_leakage_err,
            ) = calc_leakage(pbcor_image)
            if np.isnan(q_leakage):
                q_leakage = 0.0
                q_leakage_err = 0.0
            if np.isnan(u_leakage):
                u_leakage = 0.0
                u_leakage_err = 0.0
            if np.isnan(v_leakage):
                v_leakage = 0.0
                v_leakage_err = 0.0
            leakage_info = [
                q_leakage,
                u_leakage,
                v_leakage,
                q_leakage_err,
                u_leakage_err,
                v_leakage_err,
            ]
        else:
            q_leakage,
            u_leakage,
            v_leakage,
            q_leakage_err,
            u_leakage_err,
            v_leakage_err = leakage_info
        leakagecor_image, leakagecor_model = correct_leakage(
            pbcor_image,
            modelname=pbcor_model,
            q_leakage=q_leakage,
            u_leakage=u_leakage,
            v_leakage=v_leakage,
        )

    if pbuncor is False:
        final_image = leakagecor_image
        final_model = leakagecor_model
    else:
        ##########################################
        # Restore primary beam corrections
        ###########################################
        # For image
        ###############
        final_image = imagename.split(".fits")[0] + "_pbuncor.fits"
        pbcor_cmds = [
            "run-mwa-singlepbcor",
            leakagecor_image,
            metafits,
            final_image,
            "--interpolated",
            "--num_threads",
            f"{ncpu}",
            "--pb_jones_file",
            f"{pbfile}",
            "--restore",
        ]
        if os.path.exists(pbfile) is False:
            pbcor_cmds.append("--save_pb")
        subprocess.run(
            pbcor_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        #################
        # For model
        #################
        final_model = modelname.split(".fits")[0] + "_pbuncor.fits"
        pbcor_cmds = [
            "run-mwa-singlepbcor",
            leakagecor_model,
            metafits,
            final_model,
            "--interpolated",
            "--num_threads",
            f"{ncpu}",
            "--pb_jones_file",
            f"{pbfile}",
            "--restore",
        ]
        if os.path.exists(pbfile) is False:
            pbcor_cmds.append("--save_pb")
        subprocess.run(
            pbcor_cmds,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    return final_image, final_model, leakage_info


def update_leakage(
    wsclean_images,
    wsclean_models,
    image_cube,
    model_cube,
    metafits,
    pbcor=True,
    leakagecor=True,
    leakage_info=[],
    pbuncor=True,
    ncpu=-1,
):
    """
    Update leakage of a single set of wsclean Stokes image

    Parameters
    ----------
    wsclean_images : list
        List of wsclean Stokes images for the image cube
    wsclean_models : list
        List of wsclean Stokes models for the model cube
    image_cube : str
        Stokes image cube name
    model_cube : str
        Stokes model cube name
    metafits : str
        Metafits file
    pbcor : bool, optional
        Perform primary beam correction or not
    leakagecor : bool, optional
        Perform leakage correction or not
    leakage_info : list, optional
        Use provided leakage info
    pbuncor : bool, optional
        Undo primary beam correction or not
    ncpu : int, optional
        NUmber of CPU threads to use

    Returns
    -------
    list
        Leakage informations
    """
    ncpu = max(1, ncpu)
    valid_image = check_valid_image(image_cube)
    if valid_image:
        cor_imagename, cor_modelname, leakage_info = correct_pbcor_leakage(
            image_cube,
            model_cube,
            metafits,
            pbcor=pbcor,
            leakagecor=leakagecor,
            pbuncor=pbuncor,
            leakage_info=leakage_info,
            ncpu=ncpu,
        )
        image_data = fits.getdata(cor_imagename)
        image_data[np.isinf(image_data)] = 0.0
        image_data[np.isnan(image_data)] = 0.0
        model_data = fits.getdata(cor_modelname)
        model_data[np.isinf(model_data)] = 0.0
        model_data[np.isnan(model_data)] = 0.0
        for i in range(len(wsclean_images)):
            spectro_image_header = fits.getheader(wsclean_images[i])
            spectro_image_data = fits.getdata(wsclean_images[i])
            spectro_image_data[0, 0, ...] = image_data[i, 0, ...]
            fits.writeto(
                wsclean_images[i],
                data=spectro_image_data,
                header=spectro_image_header,
                overwrite=True,
            )
            spectro_model_header = fits.getheader(wsclean_models[i])
            spectro_model_data = fits.getdata(wsclean_models[i])
            spectro_model_data[0, 0, ...] = model_data[i, 0, ...]
            fits.writeto(
                wsclean_models[i],
                data=spectro_model_data,
                header=spectro_model_header,
                overwrite=True,
            )
        return leakage_info
    else:
        return


def correct_spectrosnap_pbleak(
    image_dic,
    model_dic,
    metafits,
    logger,
    pbcor=True,
    leakagecor=True,
    pbuncor=True,
    leakage_info_polynomial=[],
    ncpu=-1,
):
    """
    Correct spectrocopic snapshot images for primary beam and leakage

    Parameters
    ----------
    image_dic : dict
        Image dictionary
    model_dic : dict
        Model dictionary
    metafits : str
        Metafits file
    logger : logger
        Python logger
    pbcor : bool, optional
        Perform primary beam correction
    leakagecor : bool, optional
        Leakage correction
    pbuncor : bool, optional
        Undo primary beam correction
    leakage_info_polynomial : list, optional
        Leakage info polynomial provided by user [[q_leakage poly, u_leakage poly, v_leakage poly]]
    ncpu : int, optional
        Number of CPU threads to use

    Returns
    -------
    list
        Leakage information list
    """
    ncpu = max(1, ncpu)
    images = list(image_dic.keys())
    models = list(model_dic.keys())
    leakage_info_list = []
    leakage_info_dic = {}
    ################################
    # If leakage is provided by user
    ################################
    if len(leakage_info_polynomial)==3:
        q_leakage_poly = np.poly1d(leakage_info_polynomial[0])
        u_leakage_poly = np.poly1d(leakage_info_polynomial[1])
        v_leakage_poly = np.poly1d(leakage_info_polynomial[2])
        for i in range(len(images)):
            imagename = images[i]
            modelname = models[i]
            header = fits.getheader(imagename)
            freq = header["CRVAL3"]
            q_leakage = q_leakage_poly(freq)
            u_leakage = u_leakage_poly(freq)
            v_leakage = v_leakage_poly(freq)
            if "MFS" not in imagename:
                wsclean_images = sorted(image_dic[imagename])
                wsclean_models = sorted(model_dic[modelname])
                valid_image = check_valid_image(imagename)
                if valid_image:  
                    leakage_info = update_leakage(
                            wsclean_images,
                            wsclean_models,
                            imagename,
                            modelname,
                            metafits,
                            pbcor=pbcor,
                            leakagecor=leakagecor,
                            pbuncor=pbuncor,
                            leakage_info = [q_leakage, u_leakage, v_leakage, 0, 0, 0],
                            ncpu=ncpu,
                        )  
                    leakage_info_list.append(leakage_info)     
        return leakage_info_list   
    else:
        #####################################
        # Leakage is estimated and corrected
        #####################################
        no_disk_detected_images = []
        for i in range(len(images)):
            imagename = images[i]
            modelname = models[i]
            if "MFS" not in imagename:
                wsclean_images = sorted(image_dic[imagename])
                wsclean_models = sorted(model_dic[modelname])
                valid_image = check_valid_image(imagename)
                if valid_image:            
                    disk_detected, disk_size = determine_quiet_disk(wsclean_images[0])
                    if disk_detected is False:
                        logger.warning(f"Solar disk is not detected for: {wsclean_images[0]}")
                        no_disk_detected_images.append([wsclean_images, wsclean_models])
                    else:
                        ########################################################
                        # Updating images for leakage if solar disk is detected
                        ########################################################
                        leakage_info = update_leakage(
                            wsclean_images,
                            wsclean_models,
                            imagename,
                            modelname,
                            metafits,
                            pbcor=pbcor,
                            leakagecor=leakagecor,
                            pbuncor=pbuncor,
                            ncpu=ncpu,
                        )
                        if leakage_info is None:
                            logger.warning(
                                f"leakage can not be calculated for: {wsclean_images[0]}"
                            )
                            no_disk_detected_images.append([wsclean_images, wsclean_models])
                        else:
                            leakage_info_list.append(leakage_info)
                            freq = float(fits.getheader(wsclean_images[0])["CRVAL3"])
                            freq_keys = leakage_info_dic.keys()
                            if freq not in freq_keys:
                                leakage_info_dic[freq] = leakage_info
                            else:
                                old_leakage_info = leakage_info_dic[freq]
                                temp_leakage_info = [old_leakage_info, leakage_info]
                                temp_leakage_info = np.array(temp_leakage_info)
                                Q = temp_leakage_info[:, 0]
                                U = temp_leakage_info[:, 1]
                                V = temp_leakage_info[:, 2]
                                Qe = temp_leakage_info[:, 3]
                                Ue = temp_leakage_info[:, 4]
                                Ve = temp_leakage_info[:, 5]
                                q_leakage, q_err = weighted_mean(Q, Qe)
                                u_leakage, u_err = weighted_mean(U, Ue)
                                v_leakage, v_err = weighted_mean(V, Ve)
                                leakage_info_dic[freq] = [
                                    q_leakage,
                                    u_leakage,
                                    v_leakage,
                                    q_err,
                                    u_err,
                                    v_err,
                                ]
                                
        ######################################################
        # Correcting images where solar disk were not detected
        ######################################################
        if len(no_disk_detected_images) > 0:
            freq_keys = np.array(leakage_info_dic.keys())
            for non_disk in no_disk_detected_images:
                wsclean_images = non_disk[0]
                wsclean_models = non_disk[1]
                freq = fits.getheader(wsclean_images[0])["CRVAL3"]
                pos = np.argmin(np.abs(freq_keys - freq))
                key = freq_keys[pos]
                old_leakage_info = leakage_info_dic[key]
                leakage_info = update_leakage(
                    wsclean_images,
                    wsclean_models,
                    imagename,
                    modelname,
                    metafits,
                    pbcor=pbcor,
                    leakagecor=leakagecor,
                    leakage_info=old_leakage_info,
                    pbuncor=pbuncor,
                    ncpu=ncpu,
                )
                leakage_info_list.append(leakage_info)
    os.system("rm -rf *_pbcor.fits *_leakagecor.fits *_pbuncor.fits *pb.npy")
    return leakage_info_list


def selfcal_round(
    msname,
    metafits,
    logger,
    selfcaldir,
    cellsize,
    imsize,
    round_number=0,
    uvrange="",
    minuv=0,
    calmode="ap",
    solint="30s",
    solnorm=True,
    refant="1",
    do_bandpass=True,
    applymode="calonly",
    threshold=3,
    weight="briggs",
    robust=0.0,
    multiscale_scales=[],
    scale_bias=0.6,
    use_previous_model=False,
    use_solar_mask=True,
    mask_radius=40,
    nchans=1,
    nintervals=1,
    fluxscale_mwa=False,
    solar_attn=10,
    pbcor=True,
    leakagecor=True,
    pbuncor=True,
    do_intensity_cal=False,
    do_polcal=False,
    solve_array_leakage=False,
    leakage_info_polynomial=[],
    pol_solnorm=False,
    do_flag=False,
    restore_flag=True,
    ncpu=-1,
    mem=-1,
):
    """
    A single self-calibration round

    Parameters
    ----------
    msname : str
        Name of the measurement set
    metafits : str
        Metafits file
    logger : logger
        Python logger
    selfcaldir : str
        Self-calibration directory
    cellsize : float
        Cellsize in arcsec
    imsize :  int
        Image pixel size
    round_number : int, optional
        Selfcal iteration number
    uvrange : float, optional
       UV range for calibration
    minuv : float, optional
        Minimum uv in lambda
    calmode : str, optional
        Calibration mode ('p' or 'ap')
    solint : str, optional
        Solution intervals
    solnorm : bool, optional
        Solution normalisation
    refant : str, optional
        Reference antenna
    do_bandpass: bool, optional
        Perform bandpass calibration
    applymode : str, optional
        Solution apply mode (calonly or calflag)
    threshold : float, optional
        Imaging and auto-masking threshold
    weight : str, optional
        Image weighting
    robust : float, optional
        Robust parameter for briggs weighting
    multiscale_scales : list, optional
        Multiscale scales to use
    scale_bias : float, optional
        Multiscale scale bias
    use_previous_model : bool, optional
        Use previous model
    use_solar_mask : bool, optional
        Use solar disk mask or not
    mask_radius : float, optional
        Mask radius in arcminute
    nchans : int, optional
        Number of spectral channels
    nintervals : int, optional
        Number of temporal intervals
    fluxscale_mwa : bool, optional
        Fluxscale caltable using reference bandpass
    solar_attn : float, optional
        Solar attenuation in dB (only used if fluxscale_mwa is True)
    pbcor : bool, optional
        Primary beam correction
    leakagecor : bool, optional
        Leakage correction
    pbuncor : bool, optional
        Undo primary beam correction
    do_intensity_cal : bool, optional
        Perform intensity self-calibration
    do_polcal : bool, optional
        Perform polarisation calibration or not
    solve_array_leakage : bool, optional
        Perform a single leakage correction over the entire array
    leakage_info_polynomial : list, optional
        User provided leaakage info polynomial [q_leakage poly, u_leakage poly, v_leakage poly]
    pol_solnorm : bool, optional
        Normalise quartical solutions or not
    do_flag : bool, optional
        Perform UVsub flagging
    restore_flag : bool, optional
        Restore last round flags or not
    ncpu : int, optional
        Number of CPUs to use in WSClean
    mem : float, optional
        Memory usage limit in WSClean

    Returns
    -------
    int
        Success message
    list
        Caltable name list
    float
        RMS based dynamic range
    float
        RMS of the image
    str
        Image name
    str
        Model image name
    str
        Residual image name
    list
        Leakage informations [Q_leakage, U_leakage, V_leakage, Q_leakage_error, U_leakage_error, V_leakage_error]
    bool
        Quiet solar disk is detected or not 
    """
    ncpu = max(1, ncpu)
    mem = max(1, mem)

    with limit_threads(n_threads=ncpu):
        from casatasks import gaincal, bandpass, applycal, flagdata, delmod, flagmanager
        from casatools import table

    cwd = os.getcwd()
    msname = msname.rstrip("/")
    msname = os.path.abspath(msname)
    os.chdir(selfcaldir)
    disk_detected = False
    polcal_datacolumn="DATA"
    
    #########################
    # UV range in casa format
    #########################
    if minuv > 0:
        if uvrange == "" or uvrange.startswith(">"):
            uvrange = f">{minuv}lambda"
        elif uvrange.startswith("<"):
            maxuv = uvrange.split("lambda")[0].split("<")[1]
            uvrange = f"{minuv}~{maxuv}lambda"
        elif "~" in uvrange:
            maxuv = uvrange.split("lambda")[0].split("~")[-1]
            uvrange = f"{minuv}~{maxuv}lambda"

    if not use_previous_model:
        delmod(vis=msname, otf=True, scr=True)
    prefix = (
        selfcaldir + "/" + os.path.basename(msname).split(".ms")[0] + "_selfcal_present"
    )
    os.system(f"rm -rf {prefix}*image.fits {prefix}*residual.fits")
    
    applycal_gaintable = []
    interp = []
    leakage_info_list = [[0.0] * 6]
    
    try:
        if weight == "briggs":
            weight += " " + str(robust)
        wsclean_args = [
            "-quiet",
            f"-scale {cellsize}asec",
            f"-size {imsize} {imsize}",
            "-no-dirty",
            "-gridder wgridder",
            f"-weight {weight}",
            "-niter 10000",
            "-mgain 0.85",
            "-nmiter 5",
            "-gain 0.1",
            f"-minuv-l {minuv}",
            f"-j {ncpu}",
            f"-abs-mem {mem}",
            f"-auto-mask {threshold + 0.1}",
            f"-auto-threshold {threshold}",
        ]
        if do_polcal is False:
            wsclean_args.append("-pol I")
            pol = "I"
            if calmode == "p":
                wsclean_args.append("-no-negative")
        else:
            wsclean_args.append("-pol IQUV")
            pol = "IQUV"

        ngrid = max(1, int(ncpu / 2))
        if ngrid > 1:
            wsclean_args.append(f"-parallel-gridding {ngrid}")

        ################################################
        # Creating and using solar mask
        ################################################
        fits_mask = msname.split(".ms")[0] + "_solar-mask.fits"
        if not os.path.exists(fits_mask):
            logger.info(f"Creating solar mask of size: {mask_radius} arcmin.\n")
            fits_mask = create_circular_mask(
                msname, cellsize, imsize, mask_radius=mask_radius
            )
        if fits_mask is not None and os.path.exists(fits_mask) and use_solar_mask:
            wsclean_args.append(f"-fits-mask {fits_mask}")

        #########################################
        # Multi-scale parameters
        #########################################
        if len(multiscale_scales)>0:
            wsclean_args.append("-multiscale")
            wsclean_args.append("-multiscale-gain 0.1")
            wsclean_args.append(
                "-multiscale-scales " + ",".join([str(s) for s in multiscale_scales])
            )
            wsclean_args.append(f"-multiscale-scale-bias {scale_bias}")
            if imsize >= 1024 and 4 * max(multiscale_scales) < 512:
                wsclean_args.append("-parallel-deconvolution 512")
        elif imsize >= 1024:
            wsclean_args.append("-parallel-deconvolution 512")    

        #####################################
        # Temporal imaging configuration
        #####################################
        if nintervals > 1:
            wsclean_args.append(f"-intervals-out {nintervals}")
        if nchans > 1:
            wsclean_args.append(f"-channels-out {nchans}")
            wsclean_args.append("-gap-channel-division")
            wsclean_args.append("-no-mf-weighting")

        #####################################
        # Figuring out previous round images
        #####################################
        wsclean_args.append(f"-name {prefix}")
        if use_previous_model and do_polcal is False:
            previous_models = glob.glob(f"{prefix}*model.fits")
            total_models_expected = nintervals * nchans
            if len(previous_models) == total_models_expected:
                wsclean_args.append("-continue")
            else:
                os.system(f"rm -rf {prefix}*")

        ###################
        # WSClean imaging
        ###################
        wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " " + msname
        logger.info(f"\nWSClean command: {wsclean_cmd}\n")
        msg = run_wsclean(wsclean_cmd, "paircarswsclean", verbose=False)
        if msg != 0:
            logger.error("Imaging is not successful.\n")
            return 1, applycal_gaintable, 0, 0, "", "", "", [], disk_detected

        ##########################################
        # If polarisation calibration is requested
        # Full Stokes image arrangement
        # Primary beam and leakage correction 
        ##########################################
        if do_polcal:
            prediction_failed=False
            #######################################
            # Making stokes cube
            #######################################
            pollist = list(pol)
            wsclean_images_dic = {}
            wsclean_models_dic = {}
            wsclean_residuals_dic = {}
            for suffix in ["image", "model", "residual"]:
                stokeslist = []
                for p in pollist:
                    if pollist == ["I"]:
                        stokeslist.append(
                            sorted(glob.glob(prefix + "*" + f"-{suffix}.fits"))
                        )
                    else:
                        stokeslist.append(
                            sorted(glob.glob(prefix + "*-" + p + f"-{suffix}.fits"))
                        )
                for i in range(len(stokeslist[0])):
                    wsclean_images = sorted([stokeslist[k][i] for k in range(len(pollist))])
                    image_prefix = (
                        selfcaldir
                        + "/"
                        + os.path.basename(wsclean_images[0])
                        .split(f"-{suffix}")[0]
                        .split("-I")[0]
                    )
                    image_cube = make_stokes_wsclean_imagecube(
                        wsclean_images,
                        image_prefix + f"-IQUV-{suffix}.fits",
                        keep_wsclean_images=True,
                    )
                    if suffix == "image":
                        wsclean_images_dic[image_cube] = wsclean_images
                    elif suffix == "model":
                        wsclean_models_dic[image_cube] = wsclean_images
                    elif suffix == "residual":
                        wsclean_residuals_dic[image_cube] = wsclean_images

            ################################
            # Leakage correction
            ################################
            if pbcor is True or leakagecor is True or pbuncor is True:
                if len(leakage_info_polynomial)==3:
                    logger.info("Leakage correction is done using pre-determined leakage polynomial.")
                else:
                    leakage_info_polynomial = []
                leakage_info_list = correct_spectrosnap_pbleak(
                    wsclean_images_dic,
                    wsclean_models_dic,
                    metafits,
                    logger,
                    pbcor=pbcor,
                    leakagecor=leakagecor,
                    pbuncor=pbuncor,
                    leakage_info_polynomial=leakage_info_polynomial,
                    ncpu=ncpu,
                )
                
                ##########################################################
                # Predict models if image is leakage corrected
                ##########################################################
                logger.debug("Re-predicting corrected models.")
                delmod(vis=msname, otf=True, scr=True)
                wsclean_cmd = "wsclean " + " ".join(wsclean_args) + " -predict " + msname
                logger.info(f"\nWSClean command: {wsclean_cmd}\n")
                prediction_msg = run_wsclean(wsclean_cmd, "paircarswsclean", verbose=False)
                if prediction_msg != 0:
                    prediction_failed = True
                    logger.warning("Re-prediction is failed.")

            #######################################
            # Remove chunk files
            #######################################
            images = list(wsclean_images_dic.keys())
            models = list(wsclean_models_dic.keys())
            residuals = list(wsclean_residuals_dic.keys())
            for i in range(len(images)):
                imagename = images[i]
                modelname = models[i]
                residualname = residuals[i]
                wsclean_images = wsclean_images_dic[imagename]
                wsclean_models = wsclean_models_dic[modelname]
                wsclean_residuals = wsclean_residuals_dic[residualname]
                for img in wsclean_images:
                    os.system(f"rm -rf {img}")
                for mod in wsclean_models:
                    os.system(f"rm -rf {mod}")
                for res in wsclean_residuals:
                    os.system(f"rm -rf {res}")

        #####################################
        # Analyzing images
        #####################################
        wsclean_files = {}
        for suffix in ["image", "model", "residual"]:
            files = glob.glob(prefix + f"*MFS-{suffix}.fits")
            if not files:
                files = glob.glob(prefix + f"*{suffix}.fits")
            wsclean_files[suffix] = files

        wsclean_images = wsclean_files["image"]
        wsclean_models = wsclean_files["model"]
        wsclean_residuals = wsclean_files["residual"]
        
        #######################################
        # Disk detection
        #######################################
        if pol == "I":
            stokesI_images = sorted(glob.glob(f"{prefix}*-image.fits"))
        else:
            stokesI_images = sorted(glob.glob(f"{prefix}*-I-image.fits"))
        for imagename in stokesI_images:
            detected, size = determine_quiet_disk(imagename)
            if detected and disk_detected is False:
                disk_detected = True
                break

        #######################################################################
        # Final frequency averaged images for backup or calculating dynamic ranges
        #######################################################################
        if do_polcal:
            keep_wsclean_images = False
        else:
            keep_wsclean_images = True
        final_image = (
            prefix.replace("present", f"{round_number}") + f"_{pol}_image.fits"
        )
        final_model = (
            prefix.replace("present", f"{round_number}") + f"_{pol}_model.fits"
        )
        final_residual = (
            prefix.replace("present", f"{round_number}") + f"_{pol}_residual.fits"
        )

        if len(wsclean_images) == 0:
            logger.error("No image is made.")
            return 1, applycal_gaintable, 0, 0, "", "", "", [], disk_detected
        elif len(wsclean_images) == 1:
            os.system(f"cp -r {wsclean_images[0]} {final_image}")
        else:
            final_image = make_timeavg_image(
                wsclean_images, final_image, keep_wsclean_images=keep_wsclean_images
            )
        if len(wsclean_models) == 1:
            os.system(f"cp -r {wsclean_models[0]} {final_model}")
        else:
            final_model = make_timeavg_image(
                wsclean_models, final_model, keep_wsclean_images=keep_wsclean_images
            )
        if len(wsclean_residuals) == 1:
            os.system(f"cp -r {wsclean_residuals[0]} {final_residual}")
        else:
            final_residual = make_timeavg_image(
                wsclean_residuals,
                final_residual,
                keep_wsclean_images=keep_wsclean_images,
            )
        os.system("rm -rf *psf.fits")

        #########################################
        # Restoring previous round flags
        #########################################
        with suppress_output():
            flags = flagmanager(vis=msname, mode="list")
        keys = flags.keys()
        for k in keys:
            if k == "MS":
                pass
            else:
                version = flags[0]["name"]
                if "selfcal" in version:
                    try:
                        with suppress_output():
                            if restore_flag:
                                flagmanager(
                                    vis=msname, mode="restore", versionname=version
                                )
                            flagmanager(vis=msname, mode="delete", versionname=version)
                    except BaseException:
                        pass

        #####################################
        # Calculating dynamic ranges
        ######################################
        model_flux, rms_DR, rms = calc_dyn_range(
            final_image,
            final_model,
            final_residual,
            fits_mask=fits_mask,
        )
        if model_flux == 0:
            logger.error("No model flux.\n")
            return 1, applycal_gaintable, 0, 0, "", "", "", [], disk_detected

        ############################
        # Flag backup before selfcal
        ############################
        do_flag_backup(msname, flagtype="selfcal")

        ########################################
        # Check if any calibration is requested
        ########################################
        if do_intensity_cal is False and do_polcal is False:
            logger.info("No calibration is requested. Returing only previous state.")
            return (
                2,
                applycal_gaintable,
                rms_DR,
                rms,
                final_image,
                final_model,
                final_residual,
                [],
                disk_detected,
            )

        #########################################
        # If model prediction failed in polcal
        #########################################
        if do_polcal:
            if prediction_failed:
                logger.error("Error in predicting model.")
                return (
                    3,
                    applycal_gaintable,
                    rms_DR,
                    rms,
                    final_image,
                    final_model,
                    final_residual,
                    [],
                    disk_detected,
                )

        ##############################
        # Perform intensity selfcal
        ##############################
        if do_intensity_cal:
            if fluxscale_mwa:
                solnorm = True
            ##########################
            # Perform gain calibration
            ##########################
            gain_caltable = prefix.replace("present", f"{round_number}") + ".gcal"
            if os.path.exists(gain_caltable):
                os.system("rm -rf " + gain_caltable)

            logger.info(
                f"gaincal(vis='{msname}',caltable='{gain_caltable}',uvrange='{uvrange}',refant='{refant}',solint='{solint}',calmode='{calmode}',minsnr=3,solnorm={solnorm})\n"
            )
            with suppress_output():
                gaincal(
                    vis=msname,
                    caltable=gain_caltable,
                    uvrange=uvrange,
                    refant=refant,
                    minsnr=3,
                    calmode=calmode,
                    solint=f"{solint}",
                    solnorm=solnorm,
                )

            if not os.path.exists(gain_caltable):
                logger.error("No gain solutions are found.\n")
                return 3, applycal_gaintable, 0, 0, "", "", "", [], disk_detected
            applycal_gaintable.append(gain_caltable)
            interp.append("linear")

            #################################
            # Gaincal flagging
            #################################
            (
                _,
                _,
                _,
                pre_flag_frac,
                pre_chan_flag_frac,
                pre_ant_flag_frac,
                pre_time_flag_frac,
            ) = get_cal_flag_info(gain_caltable)
            do_flag_backup(gain_caltable, flagtype="gainflag")
            with suppress_output():
                flagdata(
                    vis=gain_caltable,
                    mode="rflag",
                    datacolumn="CPARAM",
                    timedevscale=5.0,
                    freqdevscale=5.0,
                    flagbackup=False,
                )
            (
                _,
                _,
                _,
                flag_frac,
                chan_flag_frac,
                ant_flag_frac,
                time_flag_frac,
            ) = get_cal_flag_info(gain_caltable)
            if (
                flag_frac - pre_flag_frac > 0.5
                or ant_flag_frac - pre_ant_flag_frac > 0.5
                or time_flag_frac - pre_time_flag_frac > 0.5
            ):
                logger.info("Restoring flags of gaincal solutions.")
                flagmanager(
                    vis=gain_caltable,
                    mode="restore",
                    versionname="gainflag_1",
                )
            else:
                tb = table()
                tb.open(gain_caltable)
                gain = tb.getcol("CPARAM")
                flag = tb.getcol("FLAG")
                tb.close()
                gain[flag] = np.nan
                tb.open(gain_caltable, nomodify=False)
                new_gain = tb.getcol("CPARAM")
                shape = new_gain.shape
                for i in range(shape[0]):
                    avg = np.nanmedian(np.abs(gain[i, ...]))
                    new_gain[i, ...] = new_gain[i, ...] / avg
                tb.putcol("CPARAM", new_gain)
                tb.flush()
                tb.close()
            with suppress_output():
                flagmanager(vis=gain_caltable, mode="delete", versionname="gainflag_1")
            if not do_bandpass and fluxscale_mwa:
                logger.info("Flux scaled gain caltable using MWA reference bandpass.")
                fluxcal_caltable(gain_caltable, attn=solar_attn)

            ##################################
            # Perform bandpass calibration
            ##################################
            if calmode == "ap" and do_bandpass:
                bpass_caltable = prefix.replace("present", f"{round_number}") + ".bcal"
                if os.path.exists(bpass_caltable):
                    os.system("rm -rf " + bpass_caltable)
                logger.info(
                  f"bandpass(vis='{msname}',caltable='{bpass_caltable}',uvrange='{uvrange}',refant='{refant}',"
                  f"solint='inf',gaintable=['{gain_caltable}'],interp={interp},minsnr=3,solnorm=True)\n"
                )
                with suppress_output():
                    bandpass(
                        vis=msname,
                        caltable=bpass_caltable,
                        uvrange=uvrange,
                        refant=refant,
                        minsnr=3,
                        solint="inf",
                        interp=interp,
                        gaintable=[gain_caltable],
                        solnorm=True,
                    )
                if not os.path.exists(bpass_caltable):
                    logger.error("No bandpass solutions are found.\n")
                    if fluxscale_mwa:
                        logger.info(
                            "Flux scaled gain caltable using MWA reference bandpass."
                        )
                        fluxcal_caltable(gain_caltable, attn=solar_attn)
                else:
                    applycal_gaintable.append(bpass_caltable)
                    interp.append("linear,linear")

                    #############################
                    # Bandpass flagging
                    #############################
                    (
                        _,
                        _,
                        _,
                        pre_flag_frac,
                        pre_chan_flag_frac,
                        pre_ant_flag_frac,
                        pre_time_flag_frac,
                    ) = get_cal_flag_info(bpass_caltable)
                    do_flag_backup(bpass_caltable, flagtype="bpassflag")
                    with suppress_output():
                        flagdata(
                            vis=bpass_caltable,
                            mode="rflag",
                            datacolumn="CPARAM",
                            timedevscale=5.0,
                            freqdevscale=5.0,
                            flagbackup=False,
                        )
                    if (
                        flag_frac - pre_flag_frac > 0.5
                        or ant_flag_frac - pre_ant_flag_frac > 0.5
                        or chan_flag_frac - pre_chan_flag_frac > 0.5
                    ):
                        logger.info("Restoring flags of bandpass solutions.")
                        flagmanager(
                            vis=bpass_caltable,
                            mode="restore",
                            versionname="bpassflag_1",
                        )
                    else:
                        tb = table()
                        tb.open(bpass_caltable)
                        gain = tb.getcol("CPARAM")
                        flag = tb.getcol("FLAG")
                        tb.close()
                        gain[flag] = np.nan
                        tb.open(bpass_caltable, nomodify=False)
                        new_gain = tb.getcol("CPARAM")
                        shape = new_gain.shape
                        for i in range(shape[0]):
                            avg = np.nanmedian(np.abs(gain[i, ...]))
                            new_gain[i, ...] = new_gain[i, ...] / avg
                        tb.putcol("CPARAM", new_gain)
                        tb.flush()
                        tb.close()
                    with suppress_output():
                        flagmanager(
                            vis=bpass_caltable, mode="delete", versionname="bpassflag_1"
                        )
                    if fluxscale_mwa:
                        logger.info(
                            "Flux scaled bandpass caltable using MWA reference bandpass."
                        )
                        fluxcal_caltable(bpass_caltable, attn=solar_attn)

            logger.info(
                f"applycal(vis='{msname}',gaintable={applycal_gaintable},interp={interp},applymode='{applymode}',calwt=[False],flagbackup=False)\n"
            )
            with suppress_output():
                applycal(
                    vis=msname,
                    gaintable=applycal_gaintable,
                    interp=interp,
                    applymode=applymode,
                    calwt=[False],
                    flagbackup=False,
                )
            polcal_datacolumn="CORRECTED_DATA"

        ###################################################
        # Perform polarisation calibration using quartical
        ###################################################
        if do_polcal:
            pol_caltable = prefix.replace("present", f"{round_number}") + ".dcal"
            quartical_log = prefix.replace("present", f"{round_number}") + ".qclog"
            if os.path.exists(pol_caltable):
                os.system(f"rm -rf {pol_caltable}")
            minuv, maxuv = uvrange_casa_to_quartical(msname, uvrange)
            ##############################################################################
            # If intensity calibration is also requested, calibrating using corrected data
            ##############################################################################
            if polcal_datacolumn=="CORRECTED_DATA":
                tb = table()
                tb.open(msname)
                col_names = tb.colnames()
                tb.close()
                if "CORRECTED_DATA" not in col_names:
                    polcal_datacolumn="DATA"
                    
            quartical_args = [
                "goquartical",
                f"input_ms.path={msname}",
                f"input_ms.data_column={polcal_datacolumn}",
                f"input_ms.select_uv_range=[{minuv},{maxuv}]",
                "input_model.recipe=MODEL_DATA",
                f"output.gain_directory={pol_caltable}",
                f"solver.reference_antenna={refant}",
                "output.overwrite=True",
                "output.log_to_terminal=True",
                f"output.log_directory={quartical_log}",
                "solver.terms=[D]",
                "solver.iter_recipe=[200]",
                "solver.propagate_flags=True",
                f"solver.threads={ncpu}",
                "dask.threads=1",
                "D.type=complex",
            ]
            if solint == "inf":
                quartical_args.append("D.time_interval=1")
            elif solint != "int":
                quartical_args.append(f"D.time_interval={solint}")
            else:
                quartical_args.append(f"D.time_interval={nintervals}")
            if do_bandpass:
                msmd = msmetadata()
                msmd.open(msname)
                freqres = int(msmd.chanres(0,unit="kHz")[0])
                msmd.close()
                quartical_args.append(f"D.freq_interval={freqres}kHz")
            else:
                quartical_args.append("D.freq_interval=1")
            if solve_array_leakage:
                quartical_args.append("D.solve_per=array")
            quartical_cmd = " ".join(quartical_args)
            logger.info(f"\nQuartical cmd: {quartical_cmd}\n")
            quartical_msg = run_quartical(
                quartical_cmd, "paircarsquartical", verbose=False
            )
            os.system(f"rm -rf {quartical_log}")
            if quartical_msg != 0 or os.path.exists(pol_caltable) is False:
                logger.error("Quartical calibration is not successful.\n")
                return 3, [], 0, 0, "", "", "", [], disk_detected
            applycal_gaintable.append(pol_caltable)

            ######################################
            # Flagging quartical table
            ######################################
            pol_caltable = flag_quartical_table(pol_caltable)

            ######################################
            # Caltable normalisation
            ######################################
            if pol_solnorm:
                pol_caltable = quartical_matrix_normalize(pol_caltable, overwrite=True)

            ######################################
            # Applying quartical solutions
            ######################################
            temp_pol_caltable = (
                prefix.replace("present", f"{round_number}") + "_temp.dcal"
            )
            quartical_args = [
                "goquartical",
                f"input_ms.path={msname}",
                f"input_ms.data_column={polcal_datacolumn}",
                "output.log_to_terminal=True",
                f"output.log_directory={quartical_log}",
                f"output.gain_directory={temp_pol_caltable}",
                "output.overwrite=True",
                "output.products=[corrected_data]",
                "output.columns=[CORRECTED_DATA]",
                "output.flags=True",
                "solver.terms=[D]",
                "solver.iter_recipe=[0]",
                "solver.propagate_flags=True",
                f"solver.threads={ncpu}",
                "dask.threads=1",
                "D.type=complex",
                f"D.load_from={pol_caltable}/D",
            ]
            quartical_cmd = " ".join(quartical_args)
            logger.info(f"\nQuartical cmd: {quartical_cmd}\n")
            quartical_msg = run_quartical(
                quartical_cmd, "paircarsquartical", verbose=False
            )
            os.system(f"rm -rf {quartical_log} {temp_pol_caltable}")
            if quartical_msg != 0:
                logger.error(
                    "Quartical calibration applying solutions is not successful.\n"
                )
                return 3, [], 0, 0, "", "", "", [], disk_detected

        #####################################
        # Flag zeros
        #####################################
        with suppress_output():
            flagdata(
                vis=msname,
                mode="clip",
                clipzeros=True,
                datacolumn="corrected",
                flagbackup=False,
            )

        ######################################
        # UVsub flagging
        ######################################
        try:
            if do_flag:
                logger.info("Flagging in uv-domain data.\n")
                do_uvsub_flag(msname, threshold_list=[10, 7, 5], ncpu=max(1, ncpu))
        except Exception:
            logger.exception(traceback.print_exc())
        return (
            0,
            applycal_gaintable,
            rms_DR,
            rms,
            final_image,
            final_model,
            final_residual,
            leakage_info_list,
            disk_detected,
        )
    except Exception:
        logger.exception(traceback.print_exc())
        return 4, applycal_gaintable, 0, 0, "", "", "", [], disk_detected
    finally:
        os.chdir(cwd)
