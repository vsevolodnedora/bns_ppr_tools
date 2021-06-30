import numpy as np

rho_const = 6.176269145886162e+17

# === Profile ===
def get_reflev_borders(rl):
    ''' boundaries of different reflevels '''
    if rl == 6:
        xmin, xmax = -14, 14
        ymin, ymax = -14, 14
        zmin, zmax = 0, 14
    elif rl == 5:
        xmin, xmax = -28, 28
        ymin, ymax = -28, 28
        zmin, zmax = 0, 28
    elif rl == 4:
        xmin, xmax = -48, 48
        ymin, ymax = -48, +48
        zmin, zmax = 0, 48
    elif rl == 3:
        xmin, xmax = -88, 88
        ymin, ymax = -88, 88
        zmin, zmax = 0, 88
    elif rl == 2:
        xmin, xmax = -178, 178
        ymin, ymax = -178, +178
        zmin, zmax = 0, 178
    elif rl == 1:
        xmin, xmax = -354, 354
        ymin, ymax = -354, +354
        zmin, zmax = 0, 354
    elif rl == 0:
        xmin, xmax = -1044, 1044
        ymin, ymax = -1044, 1044
        zmin, zmax = 0, 1044
    else:
        # pass
        raise IOError("Set limits for rl:{}".format(rl))

    return (xmin, xmax, ymin, ymax, zmin, zmax)

def get_hist_bins(mask, v_n):
    """ for correlation and histogram analysis """
    if v_n == "hu_0": bins = np.linspace(-1.2, -0.8, 500)
    elif v_n == 'u_0': bins = np.linspace(-1.2, -0.8, 500)
    #elif v_n == "ang_mom": raise NameError("bins are not implemented: v_n:{} mask:{}".format(v_n, mask))
    elif v_n == "Ye": bins = np.linspace(0, 0.5, 500)
    #elif v_n == "ang_mom_flux":  raise NameError("bins are not implemented: v_n:{} mask:{}".format(v_n, mask))
    #elif v_n == "inv_ang_mom_flux": return NameError("")
    elif v_n == "temp": bins = 10.0 ** np.linspace(-2, 2, 300)
    elif v_n == "entr":
        if mask == 'remnant': bins = np.linspace(0., 30., 500)
        else: bins = np.linspace(0., 200., 500)
        #elif mask == 'remnant': bins = np.linspace(0., 30., 500)
        #else: raise NameError("bins are not implemented: v_n:{} mask:{}".format(v_n, mask))
    elif v_n == "r": bins = np.linspace(0, 50, 500)
    elif v_n == "phi": bins = np.linspace(-np.pi, np.pi, 500)
    elif v_n == "rho":
        if mask == "disk": bins = 10.0 ** np.linspace(4.0, 13.0, 500) / rho_const
        elif mask == "remnant": bins = 10.0 ** np.linspace(10.0, 17.00, 500) / rho_const
        else: bins = 10.0 ** np.linspace(5.0, 16.00, 500) / rho_const
        #else: raise NameError("bins are not implemented: v_n:{} mask:{}".format(v_n, mask))
    elif v_n == "velz": bins = np.linspace(-1., 1., 500)
    elif v_n == "theta": bins = np.linspace(0, 0.5 * np.pi, 500)
    elif v_n == "dens_unb_bern": bins = 10.0 ** np.linspace(-12., -6., 500)
    elif v_n == "press": bins = 10.0 ** np.linspace(-13., 5., 300)
    elif v_n == "Q_eff_nua": bins = 10.0 ** np.linspace(-15., -10., 500)
    elif v_n == "Q_eff_nua_over_density": bins = 10.0 ** np.linspace(-10., -2., 500)
    else:
        raise NameError("bins are not implemented: v_n:{} mask:{}".format(v_n, mask))

    return bins

def get_corr_dic(mask, v_ns):
    if v_ns == "rho_r":
        corr_task_dic = [
            {"v_n": "rho", "edges": get_hist_bins(mask, 'rho')},  # not in CGS :^
            {"v_n": "r", "edges": get_hist_bins(mask, 'r')}
        ]
    elif v_ns == "r_Ye":
        corr_task_dic = [
            {"v_n": "r", "edges": get_hist_bins(mask, 'r')},  # not in CGS :^
            {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')}
        ]
    elif v_ns == "rho_Ye":
        corr_task_dic = [
            {"v_n": "rho", "edges": get_hist_bins(mask, 'rho')},  # not in CGS :^
            {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')}
        ]
    elif v_ns == "Ye_entr":
        corr_task_dic = [
            {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')},  # not in CGS :^
            {"v_n": "entr", "edges": get_hist_bins(mask, 'entr')}
        ]
    elif v_ns == "temp_Ye":
        corr_task_dic = [
            {"v_n": "temp", "edges": get_hist_bins(mask, 'temp')},  # not in CGS :^
            {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')}
        ]
    elif v_ns == "velz_Ye":
        corr_task_dic = [
            {"v_n": "velz", "edges": get_hist_bins(mask, 'velz')},  # not in CGS :^
            {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')}
        ]
    elif v_ns == "rho_theta":
        corr_task_dic = [
            {"v_n": "rho", "edges": get_hist_bins(mask, 'rho')},  # not in CGS :^
            {"v_n": "theta", "edges": get_hist_bins(mask, 'theta')}
        ]
    elif v_ns == "velz_theta":
        corr_task_dic = [
            {"v_n": "velz", "edges": get_hist_bins(mask, 'velz')},  # not in CGS :^
            {"v_n": "theta", "edges": get_hist_bins(mask, 'theta')}
        ]
    elif v_ns == "rho_temp":
        corr_task_dic = [
            {"v_n": "rho", "edges": get_hist_bins(mask, 'rho')},  # not in CGS :^
            {"v_n": "temp", "edges": get_hist_bins(mask, 'temp')}
        ]
    elif v_ns == "rho_ang_mom":
        corr_task_dic = [
            {"v_n": "rho", "edges": get_hist_bins(mask, 'rho')},  # not in CGS :^
            {"v_n": "ang_mom", "points": 500, "scale": "log", "min":1e-9}
        ]
    elif v_ns == "rho_ang_mom_flux":
        corr_task_dic = [
            {"v_n": "rho", "edges": get_hist_bins(mask, 'rho')},  # not in CGS :^
            {"v_n": "ang_mom_flux", "points": 500, "scale": "log", "min":1e-12}
        ]
    elif v_ns == "Ye_dens_unb_bern":
        corr_task_dic = [
            {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": get_hist_bins(mask, 'dens_unb_bern')}
        ]
    elif v_ns == "rho_dens_unb_bern":
        corr_task_dic = [
            {"v_n": "rho", "edges": get_hist_bins(mask, 'rho')},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": get_hist_bins(mask, 'dens_unb_bern')}
        ]
    elif v_ns == "velz_dens_unb_bern":
        corr_task_dic = [
            {"v_n": "velz", "edges": get_hist_bins(mask, 'velz')},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": get_hist_bins(mask, 'dens_unb_bern')}
        ]
    elif v_ns == "theta_dens_unb_bern":
        corr_task_dic = [
            {"v_n": "theta", "edges": get_hist_bins(mask, 'theta')},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": get_hist_bins(mask, 'dens_unb_bern')}
        ]
    elif v_ns == "ang_mom_flux_theta":
        corr_task_dic = [
            {"v_n": "ang_mom_flux", "points": 500, "scale": "log", "min":1e-12},  # not in CGS :^
            {"v_n": "theta", "edges": get_hist_bins(mask, 'theta')}
        ]
    elif v_ns == "ang_mom_flux_dens_unb_bern":
        corr_task_dic = [
            {"v_n": "ang_mom_flux", "points": 500, "scale": "log", "min": 1e-12},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": get_hist_bins(mask, 'dens_unb_bern')}
        ]
    elif v_ns == "inv_ang_mom_flux_dens_unb_bern":
        corr_task_dic = [
            {"v_n": "inv_ang_mom_flux", "points": 500, "scale": "log", "min":1e-12},  # not in CGS :^
            {"v_n": "dens_unb_bern", "edges": get_hist_bins(mask, 'dens_unb_bern')}
        ]
    elif v_ns == "hu_0_ang_mom":
        corr_task_dic = [
            {"v_n": "hu_0", "edges": get_hist_bins(mask, 'hu_0')},  # not in CGS :^
            {"v_n": "ang_mom", "points": 500, "scale": "log", "min":1e-9}
        ]
    elif v_ns == "hu_0_ang_mom_flux":
        corr_task_dic = [
            {"v_n": "hu_0", "edges": get_hist_bins(mask, 'hu_0')},  # not in CGS :^
            {"v_n": "ang_mom_flux", "points": 300, "scale": "log", "min":1e-12}
        ]
    elif v_ns == "hu_0_Ye":
        corr_task_dic = [
            {"v_n": "hu_0", "edges": get_hist_bins(mask, 'hu_0')},  # not in CGS :^
            {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')}
        ]
    elif v_ns == "hu_0_temp":
        corr_task_dic = [
            {"v_n": "hu_0", "edges": get_hist_bins(mask, 'hu_0')},  # not in CGS :^
            {"v_n": "temp", "edges": get_hist_bins(mask, 'temp')}
        ]
    elif v_ns == "hu_0_entr":
        corr_task_dic = [
            {"v_n": "hu_0", "edges": get_hist_bins(mask, 'hu_0')},  # not in CGS :^
            {"v_n": "entr", "edges": get_hist_bins(mask, 'entr')}
        ]
    elif v_ns == "Q_eff_nua_dens_unb_bern":
        # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_dens_unb_bern
        corr_task_dic = [
            {"v_n": "Q_eff_nua", "edges": get_hist_bins(mask, 'Q_eff_nua')},
            {"v_n": "dens_unb_bern", "edges": get_hist_bins(mask, 'dens_unb_bern')}
        ]
    elif v_ns == "Q_eff_nua_over_density_hu_0":
        # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_over_D_hu_0
        corr_task_dic = [
            {"v_n": "Q_eff_nua_over_density", "edges": get_hist_bins(mask, 'Q_eff_nua_over_density')},
            {"v_n": "hu_0", "edges": get_hist_bins(mask, 'hu_0')}
        ]
    elif v_ns == "Q_eff_nua_over_density_theta":
        # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_over_D_theta
        corr_task_dic = [
            {"v_n": "Q_eff_nua_over_density", "edges": get_hist_bins(mask, 'Q_eff_nua_over_density')},
            {"v_n": "theta", "edges": get_hist_bins(mask, 'theta')}
        ]
    elif v_ns == "Q_eff_nua_over_density_Ye":
        # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_over_D_Ye
        corr_task_dic = [
            {"v_n": "Q_eff_nua_over_density", "edges": get_hist_bins(mask, 'Q_eff_nua_over_density')},
            {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')}
        ]
    elif v_ns == "Q_eff_nua_u_0":
        # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_u_0
        corr_task_dic = [
            {"v_n": "Q_eff_nua", "edges": get_hist_bins(mask, 'Q_eff_nua')},
            {"v_n": "u_0", "edges": get_hist_bins(mask, 'u_0')}
        ]
    elif v_ns == "Q_eff_nua_Ye":
        # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_ye
        corr_task_dic = [
            {"v_n": "Q_eff_nua", "edges": get_hist_bins(mask, 'Q_eff_nua')},
            {"v_n": "Ye", "edges": get_hist_bins(mask, 'Ye')}
        ]
    elif v_ns == "Q_eff_nua_hu_0":
        # corr_task_dic = d3slice.corr_task_dic_q_eff_nua_hu_0
        corr_task_dic = [
            {"v_n": "Q_eff_nua", "edges": get_hist_bins(mask, 'Q_eff_nua')},
            {"v_n": "hu_0", "edges": get_hist_bins(mask, 'hu_0')}
        ]
    else:
        raise NameError("unknown task for correlation computation: {}".format(v_ns))

    return corr_task_dic

# === Ejecta ===
def get_hist_bins_ej(v_n):
    """ For ejecta histograms """
    if v_n == "Y_e":
        return np.linspace(0.035, 0.55, 100)
    elif v_n == "theta":
        return np.linspace(0.031, 3.111, 50)
    elif v_n == "phi":
        return np.linspace(0.06, 6.29, 93)
    elif v_n == "vel_inf" or v_n == "vel_inf_bern":
        return np.linspace(0., 1., 50)
    elif v_n == "entropy":
        return np.linspace(0, 200, 100)
    elif v_n == "temperature":
        return np.linspace(0, 5, 100)
    # elif v_n == "rho": return histogram_rho
    elif v_n == "logrho":
        return np.linspace(-16.0, -8.0, 200)
    else:
        raise NameError("no hist edges found for v_n:{}".format(v_n))