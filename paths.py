"""
    Set default paths
"""

# path to dir, inside of which the 'simulation_name/' dir with output
# default_ppr_dir = '/data01/numrel/vsevolod.nedora/postprocessed5/'
# default_ppr_dir = '/data01/numrel/vsevolod.nedora/postprocessed_phase_trans/PSRJ1829_2456/'
default_ppr_dir = '/data01/numrel/vsevolod.nedora/postprocessed_SLy/'
# path to dir where to look for 'simulation_name/' dir with data e.g., 'output-xxxx/' folders
# default_data_dir = '/data1/numrel/WhiskyTHC/Backup/2018/GW170817/'
# default_data_dir = '/data1/numrel/WhiskyTHC/Backup/2018/GW170817/Ejecta_Data_Quarks/PSRJ1829_2456/'
default_data_dir = '/data1/numrel/WhiskyTHC/Backup/2018/Multiphys/'

# path to dir with skynet files for nucleosyntehsis
skynet =    '/data01/numrel/vsevolod.nedora/Data/skynet/'

# path to TOV siquences (for initial data extraction only)
TOVs =      '/data01/numrel/vsevolod.nedora/Data/TOVs/'

# path where to look for .tat.gs of initial data (for initial data extraction only)
lorene =    '/data/numrel/Lorene/Lorene_TABEOS/GW170817/'

# location of "12345.h5" profiles *inside* the simulation 'default_data_dir/simulation_name/'
default_profile_dic = 'profiles/3d/'

# where to look for hydro EOS files for different simulations (get EOS name from simulation name)
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
    elif sim.__contains__("BHBlp"):
        fname = "/data01/numrel/vsevolod.nedora/Data/EOS/BHB/BHB_lp_hydro_10-May-2016.h5"
    elif sim.__contains__("BLQ"):
        fname = "/data/numrel/WhiskyTHC/EOS/SFHo+BLQ/BLh_gibbs_180_0.35_new_hydro_08-Nov-2019.h5"
    else:
        raise NameError("Current dir does not contain a hint to what EOS to use: \n{}"
                        .format(sim))
    return fname


debug = True # Don't touch this