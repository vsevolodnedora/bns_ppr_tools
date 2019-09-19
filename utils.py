#######################################################################
#
# Default settings, paths, filenames and constants
# used in the analysis
#
#
#
# (Author Vsevolod Nedora)
#
########################################################################

import numpy as np

class Paths:

    scripts  =  '/data01/numrel/vsevolod.nedora/scripts_server/'
    ppr_sims =  '/data01/numrel/vsevolod.nedora/postprocessed4/'
    lorene =    '/data/numrel/Lorene/Lorene_TABEOS/GW170817/'
    # lorene =    '/data01/numrel/vsevolod.nedora/Data/Lorene/'
    TOVs =      '/data01/numrel/vsevolod.nedora/Data/TOVs/'
    gw170817 =   '/data1/numrel/WhiskyTHC/Backup/2018/GW170817/' # "/data01/numrel/vsevolod.nedora/tmp/" # '/data1/numrel/WhiskyTHC/Backup/2018/SLy4_M130130_SR_physics/'
    skynet =    '/data01/numrel/vsevolod.nedora/Data/skynet/'
    output =    '/data01/numrel/vsevolod.nedora/output/'
    plots =     '/data01/numrel/vsevolod.nedora/figs/'
    mkn =       '/data01/numrel/vsevolod.nedora/macrokilonova_bayes/mk_source/source/'
    home =      '/data01/numrel/vsevolod.nedora/bns_ppr_scripts/'
    SLy4_hydo=  '/data01/numrel/vsevolod.nedora/Data/EOS/SLy4/SLy4_hydro_14-Dec-2017.h5'
    # skynet =   '/data01/numrel/vsevolod.nedora/scripts_server/ejecta/skynet/'

    @staticmethod
    def get_eos_fname_from_curr_dir(sim):

        if sim.__contains__("SLy4"):
            fname = "/data01/numrel/vsevolod.nedora/Data/EOS/SLy4/SLy4_hydro_14-Dec-2017.h5"
        elif sim.__contains__("LS220"):
            fname = "/data01/numrel/vsevolod.nedora/Data/EOS/LS220/LS_220_hydro_27-Sep-2014.h5"
        elif sim.__contains__("DD2"):
            fname = "/data01/numrel/vsevolod.nedora/Data/EOS/DD2/DD2_DD2_hydro_30-Mar-2015.h5"
        elif sim.__contains__("SFHo"):
            fname = "/data01/numrel/vsevolod.nedora/Data/EOS/SFHo/SFHo_hydro_29-Jun-2015.h5"
        elif sim.__contains__("BLh"):
            fname = "/data01/numrel/vsevolod.nedora/Data/EOS/SFHo+BL/BLH_new_hydro_10-Jun-2019.h5"
        else:
            raise NameError("Current dir does not contain a hint to what EOS to use: \n{}"
                            .format(sim))
        return fname


class Files:
    it_time     = 'collated/rho.maximum.asc'
    models      = 'models.csv'
    models_empty= 'models_tmp2.csv'

    # collated2
    disk_mass   = 'disk_mass.asc'

    # collated
    dens_unb    = 'dens_unbnd.norm1.asc'
    dens_unb_bern = 'dens_unbnd_bernoulli.norm1.asc'
    dens        = 'dens.norm1.asc'

    # ejecta
    total_flux  = 'total_flux.dat'
    hist_theta  = 'hist_theta.dat'
    hist_ye     = 'hist_ye.dat'
    hist_entropy= 'hist_entropy.dat'
    hist_vel_inf= 'hist_vel_inf.dat'

    # nucle
    yields      = 'yields.h5'
    solar_r     = 'solar_r.dat'

    # waveforms
    l2_m2       = 'waveform_l2_m2.dat'
    tmerg       = 'tmerger.dat'

    # ejecta profile:
    ejecta_profile="ejecta_profile.dat"
    ejecta_profile_bern= "ejecta_profile_bern.dat"

    # mkn
    mkn_model   = 'mkn_model.h5'
    filt_at2017gfo= 'AT2017gfo.h5'


class Lists:

    chosen_sims = [
        "DD2_M13641364_M0_SR"
        "DD2_M13641364_M0_LK_SR_R04"
    ]


    eos = ["DD2", "LS220", "SFHo", "SLy4", "BLh"]
    res = ["VLR", "LR", "SR", "HR"]
    neut= ["M0", "M1"]
    visc= ["LK"]
    # q   = [1.0, 1.05, 1.1, 1.11, 1.13, 1.16, 1.19, 1.2, 1.22, 1.29]
    # q   = [1.0, 1.053, 1.102, 1.106, 1.132, 1.159, 1.185, 1.201, 1.222, 1.285]
    colors_q={1.000:"firebrick", 1.053:"red", 1.102:"gold",
              1.106:"darkkhaki", 1.132:"olivedrab", 1.159:"darkgreen",
              1.185:"lightseagreen", 1.201:"darkgreen", 1.222:"royalblue", 1.285:"navy"}

    dyn_not_pas = [
                  "DD2_M13641364_M0_HR_R04", # not in sim2
                  "DD2_M13641364_M0_HR", # not in sim2
                  "DD2_M14861254_M0_HR", # not in sim2
                  "LS220_M14001330_M0_HR", # not in sim2
                  "LS220_M14351298_M0_HR", # in sim2
                  "SLy4_M13641364_M0_HR", # not in sim2
                  "SLy4_M13641364_M0_LR", # in sim2 BUT problem with rocketing dynimical ejecta
                  ]

    bern_pass=[
                  "LS220_M13641364_M0_LK_SR",
                  "LS220_M13641364_M0_SR",
                  "SFHo_M14521283_M0_LR",
                  "SFHo_M14521283_M0_LK_SR",
                  "SFHo_M13641364_M0_LK_SR_2019pizza",
                  "SFHo_M13641364_M0_LK_SR",
                  "SFHo_M14521283_M0_HR"
    ]

    collate_list = [
        "bnstrackergen-bns_positions..asc",
        "dens.norm1.asc",
        "dens_unbnd.norm1.asc",
        "dens_unbnd_bernoulli.norm1.asc",
        "dens_unbnd_garching.norm1.asc",
        "H.norm2.asc",
        "luminosity_nua.norm1.asc",
        "luminosity_nue.norm1.asc",
        "luminosity_nux.norm1.asc",
        "mp_Psi4_l0_m0_r400.00.asc",
        "mp_Psi4_l1_m0_r400.00.asc",
        "mp_Psi4_l1_m1_r400.00.asc",
        "mp_Psi4_l2_m0_r400.00.asc",
        "mp_Psi4_l2_m1_r400.00.asc",
        "mp_Psi4_l2_m2_r400.00.asc",
        "mp_Psi4_l3_m0_r400.00.asc",
        "mp_Psi4_l3_m1_r400.00.asc",
        "mp_Psi4_l3_m2_r400.00.asc",
        "mp_Psi4_l3_m3_r400.00.asc",
        "mp_Psi4_l4_m0_r400.00.asc",
        "mp_Psi4_l4_m1_r400.00.asc",
        "mp_Psi4_l4_m2_r400.00.asc",
        "mp_Psi4_l4_m3_r400.00.asc",
        "mp_Psi4_l4_m4_r400.00.asc",
        "outflow_det_0.asc",
        "outflow_det_1.asc",
        "outflow_det_2.asc",
        "outflow_det_3.asc",
        "rho.maximum.asc",
        "temperature.maximum.asc",
        "thc_leakagem0-thc_leakage_m0_flux..asc"
    ]

    tarball = [
        # "bnstrackergen - bns_positions..asc",
        "dens.norm1.asc",
        "dens_unbnd.norm1.asc",
        "dens_unbnd_bernoulli.norm1.asc",
        "dens_unbnd_garching.norm1.asc",
        "H.norm2.asc",
        "luminosity_nua.norm1.asc",
        "luminosity_nue.norm1.asc",
        "luminosity_nux.norm1.asc",
        "mp_Psi4_l0_m0_r400.00.asc",
        "mp_Psi4_l1_m0_r400.00.asc",
        "mp_Psi4_l1_m1_r400.00.asc",
        "mp_Psi4_l2_m0_r400.00.asc",
        "mp_Psi4_l2_m1_r400.00.asc",
        "mp_Psi4_l2_m2_r400.00.asc",
        "mp_Psi4_l3_m0_r400.00.asc",
        "mp_Psi4_l3_m1_r400.00.asc",
        "mp_Psi4_l3_m2_r400.00.asc",
        "mp_Psi4_l3_m3_r400.00.asc",
        "mp_Psi4_l4_m0_r400.00.asc",
        "mp_Psi4_l4_m1_r400.00.asc",
        "mp_Psi4_l4_m2_r400.00.asc",
        "mp_Psi4_l4_m3_r400.00.asc",
        "mp_Psi4_l4_m4_r400.00.asc",
        "outflow_det_0.asc",
        "outflow_det_1.asc",
        "outflow_det_2.asc",
        "outflow_det_3.asc",
        "rho.maximum.asc",
        "temperature.maximum.asc",
        # "thc_leakagem0 - thc_leakage_m0_flux..asc",
        "outflow_surface_det_0_fluxdens.asc",
        "outflow_surface_det_1_fluxdens.asc",
        # "BH_diagnostics.ah1.gp"
    ]

    outflow = [
        "corr_vel_inf_bern_theta.h5", #
        "corr_vel_inf_theta.h5",
        "corr_ye_entropy.h5",
        "corr_ye_entropy.png",
        "corr_ye_theta.h5",
        "ejecta.h5",
        "ejecta_profile_bern.dat", #
        "ejecta_profile.dat",
        "hist_entropy.dat",
        "hist_entropy.xg",
        "hist_log_rho.dat",
        "hist_log_rho.xg",
        "hist_vel_inf_bern.dat", #
        "hist_temperature.dat",
        "hist_temperature.xg",
        "hist_theta.dat",
        "hist_theta.xg",
        "hist_vel.dat",
        "hist_vel_inf.dat",
        "hist_vel_inf.xg",
        "hist_vel.xg",
        "hist_ye.dat",
        "hist_ye.xg",
        "mass_averages.dat",
        "profile_entropy.xg",
        "profile_flux.xg",
        "profile_rho.xg",
        "profile_temperature.xg",
        "profile_vel_inf.xg",
        "profile_vel.xg",
        "profile_ye.xg",
        "theta_75.dat",
        "total_flux.dat",
        "yields.h5"
    ]

    gw = [
        "E_GW_dot_l2.png",
        "E_GW_dot.png",
        "E_GW_l2.png",
        "E_GW.png",
        "EJ.dat",
        "fpeak.dat",
        "postmerger_psd_l2_m2.dat",
        "postmerger_strain_l2_m2.dat",
        # "psd_l0_m0.dat",
        # "psd_l1_m0.dat",
        # "psd_l1_m1.dat",
        # "psd_l2_m0.dat",
        # "psd_l2_m1.dat",
        "psd_l2_m2.dat",
        # "psd_l3_m0.dat",
        # "psd_l3_m1.dat",
        # "psd_l3_m2.dat",
        # "psd_l3_m3.dat",
        # "psd_l4_m0.dat",
        # "psd_l4_m1.dat",
        # "psd_l4_m2.dat",
        # "psd_l4_m3.dat",
        # "psd_l4_m4.dat",
        # "psi4_l0_m0.dat",
        # "psi4_l1_m0.dat",
        # "psi4_l1_m1.dat",
        # "psi4_l2_m0.dat",
        # "psi4_l2_m1.dat",
        "psi4_l2_m2.dat",
        # "psi4_l3_m0.dat",
        # "psi4_l3_m1.dat",
        # "psi4_l3_m2.dat",
        # "psi4_l3_m3.dat",
        # "psi4_l4_m0.dat",
        # "psi4_l4_m1.dat",
        # "psi4_l4_m2.dat",
        # "psi4_l4_m3.dat",
        # "psi4_l4_m4.dat",
        # "strain_l0_m0.dat",
        # "strain_l1_m0.dat",
        # "strain_l1_m1.dat",
        # "strain_l2_m0.dat",
        # "strain_l2_m1.dat",
        "strain_l2_m2.dat",
        # "strain_l3_m0.dat",
        # "strain_l3_m1.dat",
        # "strain_l3_m2.dat",
        # "strain_l3_m3.dat",
        # "strain_l4_m0.dat",
        # "strain_l4_m1.dat",
        # "strain_l4_m2.dat",
        # "strain_l4_m3.dat",
        # "strain_l4_m4.dat",
        "waveform_l2_m2.dat",
        "tmerger.dat",
        "tcoll.dat"
    ]

    h5_disk = [
        "rho.file *.h5",
        "temperature.file *.h5",
        "Y_e.file *.h5",
        "volform.file *.h5",
        "w_lorentz.file *.h5"
    ]


class Labels:
    def __init__(self):
        pass

    @staticmethod
    def labels(v_n):
        # solar

        if v_n == 'theta':
            return r"Angle from orbital plane"

        elif v_n == 'phi':
            return r"Azimuthal angle"

        elif v_n == 'mass':
            return r'normed $M_{\rm{ej}}$'

        elif v_n == 'ejmass':
            return r'$M_{\rm{ej}}$ $[10^{-2}M_{\odot}]$'

        elif v_n == 'ejmass3':
            return r'$M_{\rm{ej}}$ $[10^{-3}M_{\odot}]$'

        elif v_n == "vel_inf":
            return r"$\upsilon_{\infty}$ [c]"

        elif v_n == "Y_e" or v_n == "ye" or v_n == "Ye":
            return r"$Y_e$"

        elif v_n == 'flux':
            return r'$\dot{M}$'

        elif v_n == 'time':
            return r'$t$ [ms]'

        elif v_n == "Y_final":
            return r'Relative final abundances'

        elif v_n == "A":
            return r"Mass number, A"

        elif v_n == 't-tmerg':
            return r'$t-t_{merg}$ [ms]'

        elif v_n == 'entropy' or v_n == 's':
            return r'$s$'

        elif v_n == 't_eff' or v_n == 'T_eff':
            return r'$\log($T$_{eff}/K)$'

        else:
            raise NameError("No label found for v_n:{}"
                            .format(v_n))


class Limits:

    @staticmethod
    def lim(v_n):
        if v_n in ["Y_e", "ye", "Ye"]:
            return 0., 0.5
        elif v_n in  ["vel_inf", "vinf", "velinf"]:
            return 0, 1.
        elif v_n in ["theta"]:
            return 0, 90.
        elif v_n in ["phi"]:
            return 0., 360
        elif v_n in ["entropy", "s"]:
            return 0, 200.
        else:
            raise NameError("limit for v_n:{} is not found"
                            .format(v_n))

    @staticmethod
    def in_dic(dic):
        # if "v_n" in dic.keys():
        #     if dic["v_n"] != None:
        #         lim1, lim2 = Limits.lim(dic["v_n"])
        #         dic["zmin"], dic["zmax"] = lim1, lim2

        if "v_n_x" in dic.keys():
            if dic["v_n_x"] != None:
                lim1, lim2 = Limits.lim(dic["v_n_x"])
                dic["xmin"], dic["xmax"] = lim1, lim2

        if "v_n_y" in dic.keys():
            if dic["v_n_y"] != None:
                lim1, lim2 = Limits.lim(dic["v_n_y"])
                dic["ymin"], dic["ymax"] = lim1, lim2

        return dic


class Constants:

    ns_rho = 1.6191004634e-5
    time_constant = 0.004925794970773136  # to to to ms
    energy_constant = 1787.5521500932314
    volume_constant = 2048


class Printcolor:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self):
        pass

    @staticmethod
    def red(text, comma=False):
        if comma:
            print(Printcolor.FAIL + text + Printcolor.ENDC),
        else:
            print(Printcolor.FAIL + text + Printcolor.ENDC)

    @staticmethod
    def yellow(text, comma=False):
        if comma:
            print(Printcolor.WARNING + text + Printcolor.ENDC),
        else:
            print(Printcolor.WARNING + text + Printcolor.ENDC)

    @staticmethod
    def blue(text, comma=False):
        if comma:
            print(Printcolor.OKBLUE + text + Printcolor.ENDC),
        else:
            print(Printcolor.OKBLUE + text + Printcolor.ENDC)

    @staticmethod
    def green(text, comma=False):
        if comma:
            print(Printcolor.OKGREEN + text + Printcolor.ENDC),
        else:
            print(Printcolor.OKGREEN + text + Printcolor.ENDC)

    @staticmethod
    def bold(text, comma=False):
        if comma:
            print(Printcolor.BOLD + text + Printcolor.ENDC),
        else:
            print(Printcolor.BOLD + text + Printcolor.ENDC)

    @staticmethod
    def print_colored_string(parts, colors, comma=False):
        assert len(parts) == len(colors)
        for color in colors:
            assert color in ["", "blue", "red", "yellow", "green"]

        for part, color in zip(parts, colors):
            if color == "":
                print(part),
            elif color == "blue":
                Printcolor.blue(part, comma=True)
            elif color == "green":
                Printcolor.green(part, comma=True)
            elif color == "red":
                Printcolor.red(part, comma=True)
            elif color == "yellow":
                Printcolor.yellow(part, comma=True)
            else:
                raise NameError("wrong color: {}".format(color))
        if comma:
            print(''),
        else:
            print('')


class FORMULAS:

    def __init__(self):
        pass

    @staticmethod
    def r(x, y):
        return np.sqrt(x ** 2 + y ** 2)# + z ** 2)

    @staticmethod
    def density(rho, w_lorentz, vol):
        return rho * w_lorentz * vol

    @staticmethod
    def vup(velx, vely, velz):
        return [velx, vely, velz]

    @staticmethod
    def metric(gxx, gxy, gxz, gyy, gyz, gzz):
        return [[gxx, gxy, gxz], [gxy, gyy, gyz], [gxz, gyz, gzz]]

    @staticmethod
    def enthalpy(eps, press, rho):
        return 1 + eps + (press / rho)

    @staticmethod
    def shift(betax, betay, betaz):
        return [betax, betay, betaz]

    @staticmethod
    def shvel(shift, vlow):
        shvel = np.zeros(shift[0].shape)
        for i in range(len(shift)):
            shvel += shift[i] * vlow[i]
        return shvel

    @staticmethod
    def u_0(w_lorentz, shvel, lapse):
        return w_lorentz * (shvel - lapse)

    @staticmethod
    def vlow(metric, vup):
        vlow = [np.zeros_like(vv) for vv in [vup[0], vup[1], vup[2]]]
        for i in range(3):  # for x, y
            for j in range(3):
                vlow[i][:] += metric[i][j][:] * vup[j][:]  # v_i = g_ij * v^j (lowering index) for x y
        return vlow

    @staticmethod
    def vphi(x, y, vlow):
        return -y * vlow[0] + x * vlow[1]

    @staticmethod
    def vr(x, y, r, vup):
        # r = np.sqrt(x ** 2 + y ** 2)
        # print("x: {}".format(x.shape))
        # print("y: {}".format(y.shape))
        # print("r: {}".format(y.shape))
        # print("vup[0]: {}".format(vup[0].shape))

        return (x / r) * vup[0] + (y / r) * vup[1]

    @staticmethod
    def theta(r, z):
        # r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        return np.arccos(z/r)

    @staticmethod
    def phi(x, y):
        return np.arctan2(y, x)

    @staticmethod
    def dens_unb_geo(u_0, rho, w_lorentz, vol):

        c_geo = -u_0 - 1.0
        mask_geo = (c_geo > 0).astype(int)  # 1 or 0
        rho_unbnd_geo = rho * mask_geo  # if 1 -> same, if 0 -> masked
        dens_unbnd_geo = rho_unbnd_geo * w_lorentz * vol

        return dens_unbnd_geo

    @staticmethod
    def dens_unb_bern(enthalpy, u_0, rho, w_lorentz, vol):

        c_ber = -enthalpy * u_0 - 1.0
        mask_ber = (c_ber > 0).astype(int)
        rho_unbnd_bernoulli = rho * mask_ber
        density_unbnd_bernoulli = rho_unbnd_bernoulli * w_lorentz * vol

        return density_unbnd_bernoulli

    @staticmethod
    def dens_unb_garch(enthalpy, u_0, lapse, press, rho, w_lorentz, vol):

        c_ber = -enthalpy * u_0 - 1.0
        c_gar = c_ber - (lapse / w_lorentz) * (press / rho)
        mask_gar = (c_gar > 0).astype(int)
        rho_unbnd_garching = rho * mask_gar
        density_unbnd_garching = rho_unbnd_garching * w_lorentz * vol

        return density_unbnd_garching

    @staticmethod
    def ang_mom(rho, eps, press, w_lorentz, vol, vphi):
        return (rho * (1 + eps) + press) * w_lorentz * w_lorentz * vol * vphi

    @staticmethod
    def ang_mom_flux(ang_mom, lapse, vr):
        return ang_mom * lapse * vr

    # data manipulation methods
    @staticmethod
    def get_slice(x3d, y3d, z3d, data3d, slice='xy'):

        if slice == 'yz':
            ix0 = np.argmin(np.abs(x3d[:, 0, 0]))
            if abs(x3d[ix0, 0, 0]) < 1e-15:
                res = data3d[ix0, :, :]
            else:
                if x3d[ix0, 0, 0] > 0:
                    ix0 -= 1
                res = 0.5 * (data3d[ix0, :, :] + data3d[ix0 + 1, :, :])
        elif slice == 'xz':
            iy0 = np.argmin(np.abs(y3d[0, :, 0]))
            if abs(y3d[0, iy0, 0]) < 1e-15:
                res = data3d[:, iy0, :]
            else:
                if y3d[0, iy0, 0] > 0:
                    iy0 -= 1
                res = 0.5 * (data3d[:, iy0, :] + data3d[:, iy0 + 1, :])
        elif slice == 'xy':
            iz0 = np.argmin(np.abs(z3d[0, 0, :]))
            if abs(z3d[0, 0, iz0]) < 1e-15:
                res = data3d[:, :, iz0]
            else:
                if z3d[0, 0, iz0] > 0 and iz0 > 0:
                    iz0 -= 1
                res = 0.5 * (data3d[:, :, iz0] + data3d[:, :, iz0 + 1])
        else:
            raise ValueError("slice:{} not recognized. Use 'xy', 'yz' or 'xz' to get a slice")
        return res


    # --------- OUTFLOWED -----

    @staticmethod
    def vinf(eninf):
        return np.sqrt(2 * eninf)

    @staticmethod
    def vinf_bern(eninf, enthalpy):
        return np.sqrt(2*(enthalpy*(eninf + 1) - 1))

    @staticmethod
    def vel(w_lorentz):
        return np.sqrt(1 - 1 / (w_lorentz**2))

    @staticmethod
    def get_tau(rho, vel, radius, lrho_b):

        rho_b = 10 ** lrho_b
        tau_0 = 0.5 * 2.71828182845904523536 * (radius / vel) * (0.004925794970773136) # in ms
        tau_b = tau_0 * ((rho/rho_b) ** (1.0 / 3.0))
        return tau_b # ms