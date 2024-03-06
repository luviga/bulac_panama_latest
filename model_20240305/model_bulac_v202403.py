# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13, 2021
Last updated: Aug. 06, 2023

@author: Climate Lead Group; Luis Victor-Gallardo, Jairo Quirós-Tortós
Suggested citation: UNEP (2022). Is Natural Gas a Good Investment for Latin 
                                America and the Caribbean? From Economic to 
                                Employment and Climate Impacts of the Power
                                Sector. https://wedocs.unep.org/handle/20.500.11822/40923
                                
"""

import pandas as pd
import pickle
import sys
from copy import deepcopy
import math
import numpy as np
import time
import os


# Import functions that support this model:
from model_bulac_funcs import intersection_2, interpolation_to_end, \
    fun_reverse_dict_data, fun_extract_new_dict_data, fun_dem_model_projtype, \
    fun_dem_proj, fun_unpack_costs, fun_unpack_taxes, \
    interpolation_non_linear_final, unpack_values_df_2
    
        
def compute_delta_for_technology(total_list, time_vector, list_life):
    delta_list = [0]
    for y in range(1, len(time_vector)):
        delta_list.append(total_list[y] - total_list[y-1])
    for y in range(int(list_life[0]), len(time_vector)):
        delta_list[y] += delta_list[y - int(list_life[y])]
    return delta_list   

def discounted_values(values_list, discount_rate):
    return [fv / (1 + discount_rate) ** n for n, fv in enumerate(values_list, 0)]

pd.options.mode.chained_assignment = None  # default='warn'

# > Define booleans to control process:
# if True, it overwrites energy projections with transport model
overwrite_transport_model = True
model_agro_and_waste = True
model_rac = True

##############################################################################
# SIMULATION: Implement the equations designed in "model_design" #
# Recording initial time of execution
start_1 = time.time()
di_nam = 'data_inputs_202403.xlsx'

###############################################################################
# 0) open the reference data bases.

dict_database = pickle.load(open('dict_db.pickle', 'rb'))

# print('Check the dict database.')
# sys.exit()

###############################################################################
# Will save all files in one outputs folder. 
# Folder is created if it does not exist
cwd = os.getcwd()
path = cwd + "/outputs"

if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)

###############################################################################
# Here we relate power plants and production per carrier in the base year:
'''
#> Select the technologies that exist in the base year:
PP_Offshore_Wind: -
PP_Onshore_Wind: base year
PP_PV Utility_Solar: base year
PP_PV DistComm_Solar: -
PP_PV DistResi_Solar: -
PP_CSP_Solar: -
PP_Geothermal: base year
PP_Hydro: base year
PP_Nuclear: base year
PP_Thermal.re_Sugar cane and derivatives: base year
PP_PV Utility+Battery_Solar: -
PP_Thermal_Coal: base year
PP_Thermal_Natural Gas: base year
ST_Utility Scale Battery: -
ST_Commercial Battery: -
ST_Residential Battery: -
ST_Pumped Hydro: -
PP_Other: -
PP_Thermal_Fuel oil: - (we cannot distinguish Diesel/Fuel Oil with our data)
PP_Thermal_Diesel: base year
'''

'''
#> Select the fuels that exist in the base year:
Coal: base year
Oil: base year
Natural gas: base year
Biofuels: base year
Waste: -
Nuclear: base year
Hydro: base year
Geothermal: base year
Solar PV: base year
Solar thermal: -
Wind: base year
Tide: -
Other sources: -
'''

dict_equiv_country = { \
    'Argentina':'Argentina',  # this works for iea dataframe
    'Barbados':'Barbados',
    'Belize':'Belice',
    'Bolivia':'Bolivia',
    'Brazil':'Brasil',
    'Chile':'Chile',
    'Colombia':'Colombia',
    'Costa Rica':'Costa Rica',
    'Cuba':'Cuba',
    'Ecuador':'Ecuador ',
    'El Salvador':'El Salvador',
    'Grenada':'Grenada',
    'Guatemala':'Guatemala',
    'Guyana':'Guyana ',
    'Haiti':'Haiti',
    'Honduras':'Honduras',
    'Jamaica':'Jamaica',
    'Mexico':'Mexico ',
    'Nicaragua':'Nicaragua',
    'Panama':'Panama',
    'Paraguay':'Paraguay',
    'Peru':'Peru',
    'Dominican Republic':'Republica Dominicana',
    'Suriname':'Suriname',
    'Trinidad and Tobago':'Trinidad & Tobago',
    'Uruguay':'Uruguay',
    'Venezuela':'Venuezuela'}

dict_equiv_country_2 = { \
    'Argentina':'Argentina',  # this works for iea dataframe
    'Barbados':'Barbados',
    'Belize':'Belize',
    'Bolivia':'Bolivia',
    'Brazil':'Brazil',
    'Chile':'Chile',
    'Colombia':'Colombia',
    'Costa Rica':'Costa Rica',
    'Cuba':'Cuba',
    'Ecuador':'Ecuador',
    'El Salvador':'El Salvador',
    'Grenada':'Grenada',
    'Guatemala':'Guatemala',
    'Guyana':'Guyana',
    'Haiti':'Haiti',
    'Honduras':'Honduras',
    'Jamaica':'Jamaica',
    'Mexico':'Mexico',
    'Nicaragua':'Nicaragua',
    'Panama':'Panama',
    'Paraguay':'Paraguay',
    'Peru':'Peru',
    'Dominican Republic':'Dominican Republic',
    'Suriname':'Suriname',
    'Trinidad and Tobago':'Trinidad & Tobago',
    'Uruguay':'Uruguay',
    'Venezuela':'Venezuela'}

# Find the common countries per region:
unique_reg = []
dict_regs_and_countries_raw = {}
dict_regs_and_countries = {}

k1_count = 0
k1_list = []
for k1 in list(dict_database.keys()):  # across databases
    k1_list.append(k1)
    for k2 in list(dict_database[k1].keys()):  # across regions
        dummy_list = list(dict_database[k1][k2].keys())
        add_dummy = {k1:dummy_list}
        if k1_count == 0:
            dict_regs_and_countries_raw.update({k2:add_dummy})
        else:
            if k2 == '2_CA':
                k2 = '2_Central America'
            else:
                pass
            dict_regs_and_countries_raw[k2].update(add_dummy)
    k1_count += 1

for reg in list(dict_regs_and_countries_raw.keys()):
    if 'Trinidad & Tobago' in dict_regs_and_countries_raw[reg][k1_list[0]]:
        fix_idx = dict_regs_and_countries_raw[reg][k1_list[0]].index('Trinidad & Tobago')
        dict_regs_and_countries_raw[reg][k1_list[0]][fix_idx] = 'Trinidad and Tobago'
    country_list = intersection_2(dict_regs_and_countries_raw[reg][k1_list[0]],
                                  dict_regs_and_countries_raw[reg][k1_list[1]])
    dict_regs_and_countries.update({reg:country_list})

###############################################################################
# *Input parameters are listed below*:
# Capacity: (initial condition by power plant Tech, OLADE, dict_database)
# Production: (initial condition by power plant Tech, OLADE, dict_database (total supply))
# Imports: (initial condition by power plant Tech, OLADE, dict_database (total supply))
# Exports: (initial condition by power plant Tech, OLADE, dict_database (total supply))
# Externality cost: (by Fuel, IMF, data_inputs => costs_externalities)
# CAPEX (capital unit cost per unit of capacity, ATB, data_inputs => costs_power_techs)
# CAU (max. activity production per unit of capacity, ATB, data_inputs => costs_power_techs)
# Fixed FOM (fixed FOM unit cost per unit of capacity, ATB, data_inputs => costs_power_techs)
# Grid connection cost (GCC unit cost per unit of capacity, ATB, data_inputs => costs_power_techs)
# Heat Rate (IAR (only for fossil-based plants), ATB, data_inputs => costs_power_techs)
    # WARNING! may need IAR for renewables
# Net capacity factor: (real activity per max. activity possible, ATB, data_inputs => costs_power_techs)
# Operational life (by power plant Tech, ATB, data_inputs => costs_power_techs)
# Variable FOM (by power plant activity, ATB, data_inputs => costs_power_techs)
# Emission factor: (by Fuel, type == consumption, see README of data_inputs)
# Demand energy intensity (by Demand sector == Tech, data_inputs => scenarios)
# Distribution of end-use consumption (by Fuel, data_inputs => scenarios)
# Distribution of new electrical energy generation (by power plant Tech, data_inputs => scenarios)
# %Imports (by Fuel, data_inputs => scenarios)
# %Exports (by Fuel, data_inputs => scenarios)
# GDP growth (combine with energy intensity, data_inputs => scenarios)
# GDP (grab from historic data, and combine with GDP growth)
# Export price (by Fuel) # WARNING! MISSING CORRECT CALIBRATION!
# Import price (by Fuel) # WARNING! MISSING CORRECT CALIBRATION!

# *Output parameters are listed below*:
# Energy Demand by Fuel and Demand sector (Tech)
# New capacity: endogenous // f(demand, scenarios, net capacity factor)
# CAPEX (by power plant Tech)
# Fixed OPEX (by power plant Tech)
# Var OPEX (by Fuel)
# Imports expenses (by Fuel)
# Exports revenue (by Fuel)
# Emissions (by Fuel)
# Externality Global Warming (by Fuel)
# Externality Local Pollution (by Fuel)

# Review the EB structure:
# Capacity: Cap > region > country > tech (power plant) > year (2019)
# Demand: EB > region > country > Energy consumption > Demand sector > Fuel > year str(2019)
# Transformation: EB > region > country > Total transformation > Demand sector > Fuel > year str(2019)
    # here, negative: use // positive: produce
# Local production: EB > Total supply > Exports/Imports/Production > Fuel > year str(2019)

# Then, simply produce a table with inputs and outputs for each dimension combination, but make sure to have the calculations done first.

###############################################################################
# 3) implementing...
# 3a) open the "data_inputs.xlsx", each one of the sheets into dfs:

'''
Name definitions:
di_nam: data_inputs_name
'''

# General sheets:
df1_general = pd.read_excel(di_nam, sheet_name="2_general")

# Calibraton and sets sheets:
'''
This code the energy balances introduced by the user:
'''
df2_fuel_eq = pd.read_excel(di_nam, sheet_name="3_FUEQ")
df2_EB = pd.read_excel(di_nam, sheet_name="4_EB")
df2_InsCap = pd.read_excel(di_nam, sheet_name="5_InsCap")
df2_scen_sets = pd.read_excel(di_nam, sheet_name="6_scen_sets")
df2_sets2pp = pd.read_excel(di_nam, sheet_name="7_set2pp")
df2_trans_sets = pd.read_excel(di_nam, sheet_name="8_trans_sets")
df2_trans_sets_eq = pd.read_excel(di_nam, sheet_name="9_trans_sets_eq")
df2_agr_sets_eq = pd.read_excel(di_nam, sheet_name="10_agro_sets")
df2_res_sets_eq = pd.read_excel(di_nam, sheet_name="11_res_sets")

# Scenarios sheets:
df3_scen = pd.read_excel(di_nam, sheet_name="12_scen")
df3_scen_dems = pd.read_excel(di_nam, sheet_name="13_scen_dems")
df3_tpt_data = pd.read_excel(di_nam, sheet_name="14_trans_data")
df3_agr_data = pd.read_excel(di_nam, sheet_name="15_agro_data")
df3_res_data = pd.read_excel(di_nam, sheet_name="16_res_data")

# Technical sheets:
df4_rac_data = pd.read_excel(di_nam, sheet_name="17_rac_data")  # nueva!
df4_ef_agro_res = \
    pd.read_excel(di_nam, sheet_name="18_agro_res_emissions")
df4_ar_emi = pd.read_excel(di_nam, sheet_name="19_ar_emissions")  # nueva!
df4_cfs = pd.read_excel(di_nam, sheet_name="20_cfs")
df4_ef = pd.read_excel(di_nam, sheet_name="21_emissions")
df4_rac_emi = pd.read_excel(di_nam, sheet_name="22_rac_emissions")  # nueva!
df4_job_fac = pd.read_excel(di_nam, sheet_name="23_job_fac")
df4_tran_dist_fac = pd.read_excel(di_nam, sheet_name="24_t&d")
df4_caps_rest = pd.read_excel(di_nam, sheet_name="25_cap_rest")

# Economic sheets:
df5_ext = pd.read_excel(di_nam, sheet_name="26_ext")
d5_res = pd.read_excel(di_nam, sheet_name="27_res_cost")
d5_power_techs = pd.read_excel(di_nam, sheet_name="28_power_cost")
d5_tpt = pd.read_excel(di_nam, sheet_name="29_trans_cost")
d5_agr = pd.read_excel(di_nam, sheet_name="30_agro_cost")
d5_rac = pd.read_excel(di_nam, sheet_name="31_rac_cost")  # nueva!
d5_tax = pd.read_excel(di_nam, sheet_name="32_tax")

##############################################################################
# Process the content of the general sheet (1_general):
df1_general.set_index('Parameter', inplace = True)
dict_general_inp = df1_general.to_dict('index')
'''
The columns of the general dictionary are:
['Value', 'Year', 'Attribute', 'Unit', 'Source', 'Description']

The keys (variables) of the dictionary are:
 ['ini_year', 'fin_year', 'country', 'gdp', 'discount_rate', 'discount_year']
'''

# Call years
per_first_yr = dict_general_inp['ini_year']['Value']
per_last_yr = dict_general_inp['fin_year']['Value']
time_vector = [i for i in range(per_first_yr, per_last_yr+1)]

# Call countries
dict_gen_all_param_names = list(dict_general_inp.keys())
dict_gen_country_params = [i for i in dict_gen_all_param_names
                           if 'country' in i]

general_country_list, gen_cntry_list_param_idx = [], []
for cntry_idx in dict_gen_country_params:
    general_country_list.append(dict_general_inp[cntry_idx]['Value'])
    gen_cntry_list_param_idx.append(cntry_idx)

# Get the regions of interest and store them for future use:
regions_list = []
all_regions = dict_regs_and_countries.keys()
country_2_reg = {}
for areg in all_regions:
    all_country_list = dict_regs_and_countries[areg]
    for cntry in all_country_list:
        country_2_reg.update({cntry:areg})

        # Store useful regions for future use:
        if areg not in regions_list and cntry in general_country_list:
            regions_list.append(areg)

# Call GDP and population.
gdp_dict = {}  # per country index
popbase_dict = {}
popfinal_dict = {}
popproj_dict = {}
for cntry in general_country_list:
    cntry_list_idx = general_country_list.index(cntry)
    cntry_idx = gen_cntry_list_param_idx[cntry_list_idx]
    gdp_idx = 'gdp_' + str(cntry_idx.split('_')[-1])
    popbase_idx = 'pop_base_' + str(cntry_idx.split('_')[-1])
    popfinal_idx = 'pop_final_' + str(cntry_idx.split('_')[-1])
    popproj_idx = 'pop_proj_' + str(cntry_idx.split('_')[-1])

    if gdp_idx in list(dict_general_inp.keys()):
        gdp_value = dict_general_inp[gdp_idx]['Value']
        gdp_dict.update({cntry: gdp_value})

        popbase_value = dict_general_inp[popbase_idx]['Value']
        popbase_dict.update({cntry: popbase_value})

        popfinal_value = dict_general_inp[popfinal_idx]['Value']
        popfinal_dict.update({cntry: popfinal_value})

        popproj_value = dict_general_inp[popproj_idx]['Value']
        popproj_dict.update({cntry: popproj_value})
    else:
        print('There is no GDP value defined for: ' + cntry)

'''
Development note: only introduce 1 GDP year for the future. The rest of the
years should be controlled by the GDP growth parameter.

The population is introduced for two years: first and last. The interpolation
is linear. This can change of other data is provided.
'''

# Call information for discounting:
r_rate = dict_general_inp['discount_rate']['Value']
r_year = dict_general_inp['discount_year']['Value']
ini_simu_yr = dict_general_inp['ini_simu_yr']['Value']

##############################################################################
# Process the content of structural sheets:

# This code extracts the sets used for energy balancing:
list_scen_fuels = df2_scen_sets['Fuel'].tolist()
list_scen_fuel_primary_and_secondary = \
    df2_scen_sets['Primary, Secondary or Power'].tolist()
list_scen_fuels_u = list(dict.fromkeys(list_scen_fuels))
list_scen_fuels_u_prim_and_sec = []
list_scen_fuels_cat_u = []
for af in list_scen_fuels_u:
    this_fuel_idx = list_scen_fuels.index(af)
    this_fuel_cat = list_scen_fuel_primary_and_secondary[this_fuel_idx]
    list_scen_fuels_cat_u.append(this_fuel_cat)
    if this_fuel_cat in ['Primary', 'Secondary']:
        list_scen_fuels_u_prim_and_sec.append(af)

# This code extracts sets to connect power plants to energy balance:
dict_equiv_pp_fuel = {}
for n in range(len(df2_sets2pp['Technology'])):
    dict_equiv_pp_fuel.update({df2_sets2pp['Technology'][n]:\
        df2_sets2pp['Fuel'][n]})

# This code extracts the transport sets and its structure:
list_trn_type = df2_trans_sets['Type'].tolist()
list_trn_fuel = df2_trans_sets['Fuel'].tolist()
list_trn_type_and_fuel = []
for n in range(len(list_trn_type)):
    this_type, this_fuel = list_trn_type[n], list_trn_fuel[n]
    if this_fuel != '-':
        this_type_and_fuel = this_type + '_' + this_fuel
    else:
        this_type_and_fuel = this_type
    list_trn_type_and_fuel.append(this_type_and_fuel)

list_trn_lvl1_u_raw = df2_trans_sets['Demand set level 1'].tolist()
list_trn_lvl2_u_raw = df2_trans_sets['Demand set level 2'].tolist()
list_trn_lvl1_u = \
    [i for i in list(dict.fromkeys(list_trn_lvl1_u_raw)) if '-' != i]
list_trn_lvl2_u = \
    [i for i in list(dict.fromkeys(list_trn_lvl2_u_raw)) if '-' != i]
# The level 2 list only applies to passenger vehicles
dict_trn_nest = {}
for l1 in range(len(list_trn_lvl1_u)):
    this_l1 = list_trn_lvl1_u[l1]
    dict_trn_nest.update({this_l1:{}})
    if this_l1 != 'Passenger':
        this_l2 = 'All'
        mask_trans_t_and_f = \
            (df2_trans_sets['Demand set level 1'] == this_l1) & \
            (df2_trans_sets['Fuel'] == '-')
        df_transport_t_and_f = df2_trans_sets.loc[mask_trans_t_and_f]
        list_trn_types = df_transport_t_and_f['Type'].tolist()
        dict_trn_nest[this_l1].update({this_l2:deepcopy(list_trn_types)})
    else:
        for l2 in range(len(list_trn_lvl2_u)):
            this_l2 = list_trn_lvl2_u[l2]
            mask_trans_t_and_f = \
                (df2_trans_sets['Demand set level 1'] == this_l1) & \
                (df2_trans_sets['Demand set level 2'] == this_l2) & \
                (df2_trans_sets['Fuel'] == '-')
            df_transport_t_and_f = df2_trans_sets.loc[mask_trans_t_and_f]
            list_trn_types = df_transport_t_and_f['Type'].tolist()
            dict_trn_nest[this_l1].update({this_l2:deepcopy(list_trn_types)})

# This code extracts set change equivalence:
pack_fe = {'new2old':{}, 'old2new':{}}
for n in range(len(df2_fuel_eq['OLADE_structure'].tolist())):
    old_struc = df2_fuel_eq['OLADE_structure'].tolist()[n]
    new_struc = df2_fuel_eq['New_structure'].tolist()[n]
    pack_fe['new2old'].update({new_struc: old_struc})
    pack_fe['old2new'].update({old_struc: new_struc})

# we have to open the first data frame:
# 1) list all the unique elements from the "df3_tpt_data" parameters:
tr_list_scenarios = df3_tpt_data['Scenario'].tolist()
tr_list_scenarios_u = list(dict.fromkeys(tr_list_scenarios))

tr_list_app_countries = df3_tpt_data['Application_Countries'].tolist()
tr_list_app_countries_u = list(dict.fromkeys(tr_list_app_countries))

tr_list_parameters = df3_tpt_data['Parameter'].tolist()
tr_list_parameters_u = list(dict.fromkeys(tr_list_parameters))

tr_list_type_and_fuel = df3_tpt_data['Type & Fuel ID'].tolist()
tr_list_type_and_fuel_u = list(dict.fromkeys(tr_list_type_and_fuel))

tr_list_type = df3_tpt_data['Type'].tolist()
tr_list_type_u = list(dict.fromkeys(tr_list_type))

tr_list_fuel = df3_tpt_data['Fuel'].tolist()
tr_list_fuel_u = list(dict.fromkeys(tr_list_fuel))

tr_list_projection = df3_tpt_data['projection'].tolist()
tr_list_projection_u = list(dict.fromkeys(tr_list_projection))

# We must overwrite the dict-database based on OLADE for a user_defined
# input to avoid compatibility issues.
use_original_pickle = True
if use_original_pickle is True:
    pass
else:
    dict_database_freeze = deepcopy(dict_database)
    # fun_reverse_dict_data(dict_database_freeze, '5_Southern Cone',
    #                       'Uruguay', True,
    #                       list_scen_fuels_u_prim_and_sec, pack_fe)
    print('We must re-write the base data. This can take a while.')
    # We must use the reference EB and InstCap sheets from data_inputs
    # agile_mode = True
    agile_mode = False
    if agile_mode is False:
        dict_ref_EB, dict_ref_InstCap = \
            fun_extract_new_dict_data(df2_EB, df2_InsCap, per_first_yr)
        with open('dict_ref_EB.pickle', 'wb') as handle1:
            pickle.dump(dict_ref_EB, handle1,
                        protocol=pickle.HIGHEST_PROTOCOL)
        handle1.close()
        with open('dict_ref_InstCap.pickle', 'wb') as handle2:
            pickle.dump(dict_ref_InstCap, handle2,
                        protocol=pickle.HIGHEST_PROTOCOL)
        handle2.close()
    else:
        dict_ref_EB = pickle.load(open('dict_ref_EB.pickle', 'rb'))
        dict_ref_InstCap = \
            pickle.load(open('dict_ref_InstCap.pickle', 'rb'))

    # We must replace the dictionaries:
    dict_database['EB'] = deepcopy(dict_ref_EB)
    dict_database['Cap'] = deepcopy(dict_ref_InstCap)

# print('Review the sets and general inputs')
# sys.exit()

#######################################################################

# 3b) create the nesting structure to iterate across:
    # future > scenario > region > country > yeat
    # WARNING! Inlcude only 1 future in this script, i.e., here we only produce future 0 inputs & outputs

# ... extracting the unique list of future...
scenario_list = list(set(df3_scen['Scenario'].tolist()))
scenario_list.remove('ALL')


scenario_list.sort()

dict_test_transport_model = {}

# ... we will work with a single dictionary containing all simulations:
# RULE: most values are time_vector-long lists, except externality unit costs (by country):
ext_by_country = {}

count_under_zero = 0

base_year = str(per_first_yr)

print('PROCESS 1 - RUNNING THE SIMULATIONS')
dict_scen = {}  # fill and append to final dict later
idict_net_cap_factor_by_scen_by_country = {}
store_percent_BAU = {}
for s in range(len(scenario_list)):
    this_scen = scenario_list[s]
    print('# 1 - ', this_scen)

    dict_test_transport_model.update({this_scen:{}})

    dict_local_reg = {}
    idict_net_cap_factor_by_scen_by_country.update({this_scen:{}})

    for r in range(len(regions_list)):
        this_reg = regions_list[r]
        print('   # 2 - ', this_reg)

        country_list = dict_regs_and_countries[this_reg]
        country_list.sort()
        dict_local_country = {}

        # Add a filter to include countries with transport data only:
        country_list = [c for c in country_list if c in tr_list_app_countries_u]

        for c in range(len(country_list)):
            this_country = country_list[c]
            print('      # 3 - ', this_country)

            dict_test_transport_model[this_scen].update({this_country:{}})

            # ...store the capacity factor by country:
            idict_net_cap_factor_by_scen_by_country[this_scen].update({this_country:{}})

            # ...call the GDP of the base year
            this_gdp_base = gdp_dict[this_country]

            # ...call and make population projection
            this_pop_base = popbase_dict[this_country]
            this_pop_final = popfinal_dict[this_country]
            this_pop_proj = popproj_dict[this_country]
            this_pop_vector_known = ['' for y in range(len(time_vector))]
            this_pop_vector_known[0] = this_pop_base
            this_pop_vector_known[-1] = this_pop_final
            if this_pop_proj == 'Linear':
                this_pop_vector = \
                    interpolation_to_end(time_vector, ini_simu_yr, \
                        this_pop_vector_known, 'last', this_scen, 'Population')

            # ...subselect the scenario dataframe you will use
            mask_scen = \
                (df3_scen['Scenario'] == this_scen) | \
                (df3_scen['Scenario'] == 'ALL')
            df_scen = df3_scen.loc[mask_scen] #  _rc is for "region" and "country"
            df_scen.reset_index(drop=True, inplace=True)

            indices_df_scen = df_scen.index.tolist()
            list_application_countries_all = \
                df_scen['Application_Countries'].tolist()
            list_application_countries = \
                list(set(df_scen['Application_Countries'].tolist()))

            for ac in list_application_countries:
                if this_country in ac.split(' ; '):
                    select_app_countries = deepcopy(ac)

            indices_df_scen_select = [i for i in range(len(indices_df_scen))
                                      if (list_application_countries_all[i]
                                          == select_app_countries) or
                                         (list_application_countries_all[i]
                                          == 'ALL') or
                                         (this_country in
                                          list_application_countries_all[i])
                                          ]

            df_scen_rc = df_scen.iloc[indices_df_scen_select]
            df_scen_rc.reset_index(drop=True, inplace=True)

            # 3c) create the demands per fuel per country in a single dictionary (follow section 2 for structure)
            # This depends on the EB dictionary, the GDP, and the scenarios' <Demand energy intensity>
            # From EB, extract the elements by demand and fuel:

            this_country_2 = dict_equiv_country_2[this_country]

            dict_base_energy_demand = \
                dict_database['EB'][this_reg][this_country_2]['Energy consumption']
            list_demand_sector_techs = list(dict_base_energy_demand.keys())
            list_demand_sector_techs.remove('none')

            list_fuel_raw = list(dict_base_energy_demand['none'].keys())
            list_fuel = [e for e in list_fuel_raw if ('Total' not in e and
                                                      'Non-' not in e)]

            # We must now create a dictionary with the parameter, the technology, and the fuel.
            # By default, demand technologies consume the fuel.
            param_related = 'Demand energy intensity'  # this is in "scenarios"
            param_related_2 = 'GDP growth'  # this is in "scenarios"
            param_related_3 = 'Distribution of end-use consumption'  # this is in "scenarios"

            # Select the "param_related"
            mask_param_related = (df_scen_rc['Parameter'] == param_related)
            df_param_related = df_scen_rc.loc[mask_param_related]
            df_param_related.reset_index(drop=True, inplace=True)

            # Select the "param_related_2"
            mask_param_related_2 = (df_scen_rc['Parameter'] == param_related_2)
            df_param_related_2 = df_scen_rc.loc[mask_param_related_2]
            df_param_related_2.reset_index(drop=True, inplace=True)

            # Select the "param_related_3"
            mask_param_related_3 = (df_scen_rc['Parameter'] == param_related_3)
            df_param_related_3 = df_scen_rc.loc[mask_param_related_3]
            df_param_related_3.reset_index(drop=True, inplace=True)

            # ...select an alternative "param_related_3" where scenarios can be managed easily
            mask_scen_3 = \
                (df3_scen_dems['Scenario'] == this_scen) | \
                (df3_scen_dems['Scenario'] == 'ALL')
            df_scen_3 = df3_scen_dems.loc[mask_scen_3]
            df_scen_3.reset_index(drop=True, inplace=True)

            indices_df_scen_spec = df_scen_3.index.tolist()
            list_application_countries_spec = \
                df_scen_3['Application_Countries'].tolist()

            indices_df_scen_select = [i for i in range(len(indices_df_scen_spec))
                                      if (list_application_countries_spec[i]
                                          == select_app_countries) or
                                         (list_application_countries_spec[i]
                                          == 'ALL') or
                                         (this_country in
                                          list_application_countries_spec[i])
                                          ]

            df_scen_3_spec = df_scen_3.iloc[indices_df_scen_select]
            df_scen_3_spec.reset_index(drop=True, inplace=True)  # this should be ready to use
            ###################################################################

            # ...acting for "GDP growth"
            this_gdp_growth_projection = df_param_related_2.iloc[0]['projection']
            this_gdp_growth_value_type = df_param_related_2.iloc[0]['value']
            this_gdp_growth_vals_raw = []
            this_gdp_growth_vals = []
            for y in time_vector:
                this_gdp_growth_vals_raw.append(df_param_related_2.iloc[0][y])
            if (this_gdp_growth_projection == 'flat' and
                    this_gdp_growth_value_type == 'constant'):
                for y in range(len(time_vector)):
                    this_gdp_growth_vals.append(this_gdp_growth_vals_raw[0])
            if this_gdp_growth_projection == 'user_defined':
                this_gdp_growth_vals = deepcopy(this_gdp_growth_vals_raw)

            # ...acting for GDP and GDP per capita:
            this_gdp_vals = []
            this_gdp_per_cap_vals = []
            this_gdp_pc_growth_vals = []
            this_pop_growth_vals = []
            for y in range(len(time_vector)):
                if y == 0:
                    this_gdp_vals.append(this_gdp_base)
                else:
                    this_growth = this_gdp_growth_vals[y]
                    next_year_gdp = this_gdp_vals[-1]*(1+this_growth/100)
                    this_gdp_vals.append(next_year_gdp)

                this_y_gdp_per_capita = \
                    this_gdp_vals[-1]/(this_pop_vector[y]*1e6)
                this_gdp_per_cap_vals.append(this_y_gdp_per_capita)
                if y != 0:
                    # Calculate the growth of the GDP per capita
                    gdp_pc_last = this_gdp_per_cap_vals[y-1]
                    gdp_pc_present = this_gdp_per_cap_vals[y]
                    this_gdp_pc_growth = \
                        100*(gdp_pc_present - gdp_pc_last)/gdp_pc_last
                    this_gdp_pc_growth_vals.append(this_gdp_pc_growth)

                    # Calculate the growth of the population
                    pop_last = this_pop_vector[y-1]
                    pop_present = this_pop_vector[y]
                    this_pop_growth = 100*(pop_present - pop_last)/pop_last
                    this_pop_growth_vals.append(this_pop_growth)
                else:
                    this_gdp_pc_growth_vals.append(0)
                    this_pop_growth_vals.append(0)

            # Create the energy demand dictionary (has a projection):
            dict_energy_demand = {}  # by sector
            dict_energy_intensity = {}  # by sector
            dict_energy_demand_by_fuel = {}  # by fuel
            tech_counter = 0
            for tech in list_demand_sector_techs:

                tech_idx = df_param_related['Demand sector'].tolist().index(tech)
                tech_counter += 1

                # ...acting for "Demand energy intensity" (_dei)
                this_tech_dei_df_param_related = df_param_related.iloc[tech_idx]
                this_tech_dei_projection = this_tech_dei_df_param_related['projection']
                this_tech_dei_value_type = this_tech_dei_df_param_related['value']
                this_tech_dei_known_vals_raw = []
                this_tech_dei_known_vals = []

                ref_energy_consumption = \
                    dict_base_energy_demand[tech]['Total'][base_year]

                y_count = 0
                for y in time_vector:
                    this_tech_dei_known_vals_raw.append(this_tech_dei_df_param_related[y])
                    # Act already by attending "endogenous" calls:
                    if this_tech_dei_df_param_related[y] == 'endogenous':
                        add_value = ref_energy_consumption*1e9/this_gdp_vals[y_count]  # MJ/USD
                    elif math.isnan(float(this_tech_dei_df_param_related[y])) is True and y_count >= 1:
                        add_value = ''
                    elif (float(this_tech_dei_df_param_related[y]) != 0.0 and
                            this_tech_dei_value_type == 'rel_by'):
                        add_value = \
                            this_tech_dei_known_vals[0]*this_tech_dei_df_param_related[y]
                    this_tech_dei_known_vals.append(add_value)
                    y_count += 1

                this_tech_dei_vals = \
                    interpolation_to_end(time_vector, ini_simu_yr, \
                        this_tech_dei_known_vals, 'last', this_scen, '')

                # ...since we have the intensities, we can obtain the demands:
                this_tech_ed_vals = []
                y_count = 0
                for y in time_vector:
                    add_value = \
                        this_tech_dei_vals[y_count]*this_gdp_vals[y_count]/1e9  # PJ
                    this_tech_ed_vals.append(add_value)
                    y_count += 1

                # store the total energy demand:
                dict_energy_demand.update({tech:{'Total':this_tech_ed_vals}})
                dict_energy_intensity.update({tech:{'Total':this_tech_dei_vals}})

                # ...we can also obtain the demands per fuel, which can vary depending on the "apply_type"
                # we will now iterate across fuels to find demands:

                total_sector_demand_baseyear = 0  # this applies for a *demand technology*
                for loc_fuel in list_fuel:  # sum across al fuels for a denominator
                    if 'Total' not in loc_fuel and 'Non-' not in loc_fuel:
                        total_sector_demand_baseyear += \
                            dict_base_energy_demand[tech][loc_fuel][base_year]

                        if tech_counter == 1:  # we must calculate the total energy demand by fuel
                            zero_list = [0 for y in range(len(time_vector))]
                            dict_energy_demand_by_fuel.update({loc_fuel:zero_list})

                # ...these are variables are needed for internal distribution of demands
                check_percent = False
                store_fush = {}  # "fush"  means "fuel shares"
                store_fush_rem = {}

                for fuel in list_fuel:
                    #if 'Total' not in fuel and 'Other' not in fuel and 'Non-' not in fuel:
                    if 'Total' not in fuel and 'Non-' not in fuel:
                        fuel_idx = df_param_related_3['Fuel'].tolist().index(fuel)

                        # ...acting for "Distribution of end-use consumption" (_deuc)
                        this_fuel_deuc_df_param_related = df_param_related_3.iloc[tech_idx]
                        this_fuel_deuc_projection = this_fuel_deuc_df_param_related['projection']
                        this_fuel_deuc_value_type = this_fuel_deuc_df_param_related['value']
                        this_fuel_deuc_known_vals_raw = []
                        this_fuel_deuc_known_vals = []

                        # ...our goal here: obtain final demand by fuel:
                        this_tech_fuel_ed_vals = []

                        # ...here, seek the EB proportion and keep it constant using "fuel demand" (_fd)
                        # ...we also need to extract the total fuel demand, which is the correct denominator
                        total_fuel_demand_baseyear = 0
                        total_fuel_demand_baseyear_2 = 0
                        for tech_internal in list_demand_sector_techs:
                            total_fuel_demand_baseyear += \
                                dict_base_energy_demand[tech_internal][fuel][base_year]
                        #
                        num_fd = dict_base_energy_demand[tech][fuel][base_year]
                        if total_sector_demand_baseyear != 0:
                            den_fd = total_sector_demand_baseyear
                        else:
                            den_fd = 1
                        endo_share = num_fd/den_fd

                        if this_fuel_deuc_projection == 'keep_proportions':
                            y_count = 0
                            for y in time_vector:
                                # ...start by summing across all demands:
                                add_value = \
                                    endo_share*this_tech_ed_vals[y_count]
                                this_tech_fuel_ed_vals.append(add_value)

                                # ...be sure to add the fuel demand too:
                                dict_energy_demand_by_fuel[fuel][y_count] += add_value

                                y_count += 1

                        elif this_fuel_deuc_projection == 'redistribute':  # here we need to change the demand relationships by sector, in a smooth manner (i.e., interpolate)
                            mask_3_spec_tech = \
                                ((df_scen_3_spec['Demand sector'] == tech) &
                                 (df_scen_3_spec['Fuel'] == fuel))
                            df_redistribute_data = \
                                df_scen_3_spec.loc[mask_3_spec_tech]

                            if df_redistribute_data['value'].iloc[0] == 'percent':
                                check_percent = True

                            this_fush_known_vals_raw = []
                            this_fush_known_vals = []
                            if check_percent is True:  # is compatible with "interpolate"
                                for y in time_vector:
                                    add_value = \
                                        df_redistribute_data[y].iloc[0]

                                    this_fush_known_vals_raw.append(add_value)
                                    if type(add_value) is int:
                                        if math.isnan(add_value) is False:
                                            this_fush_known_vals.append(add_value/100)
                                        else:
                                            pass
                                    elif str(y) == str(base_year):
                                        this_fush_known_vals.append(endo_share)
                                    else:
                                        this_fush_known_vals.append('')

                                if add_value != 'rem':
                                    this_fush_vals = \
                                        interpolation_to_end(time_vector, 
                                                             ini_simu_yr,
                                                             this_fush_known_vals,
                                                             'last',
                                                             this_scen, '')

                                else:  # we need to fill later                               
                                    this_fush_vals = \
                                        [0 for y in range(len(time_vector))]
                                    store_fush_rem.update({fuel:this_fush_vals})
                                store_fush.update({fuel:this_fush_vals})

                                y_count = 0
                                for y in time_vector:
                                    add_value = \
                                        this_fush_vals[y_count]*this_tech_ed_vals[y_count]
                                    this_tech_fuel_ed_vals.append(add_value)
                                    dict_energy_demand_by_fuel[fuel][y_count] += add_value

                                    y_count += 1

                            if check_percent is not True:  # should do the same as in 'keep_proportions'
                                y_count = 0
                                for y in time_vector:
                                    add_value = \
                                        endo_share*this_tech_ed_vals[y_count]
                                    this_tech_fuel_ed_vals.append(add_value)
                                    dict_energy_demand_by_fuel[fuel][y_count] += add_value
                                    y_count += 1

                        dict_energy_demand[tech].update({fuel:this_tech_fuel_ed_vals})

                # ...here we need to run the remainder if necessary:
                if check_percent is True:
                    fuel_rem = list(store_fush_rem.keys())[0]
                    oneminus_rem_list_fush = store_fush_rem[fuel_rem]
                    for fuel in list_fuel:
                        if fuel != fuel_rem:
                            for y in range(len(time_vector)):
                                oneminus_rem_list_fush[y] += store_fush[fuel][y]

                    this_tech_fuel_ed_vals = []
                    for y in range(len(time_vector)):
                        store_fush[fuel_rem][y] = 1-oneminus_rem_list_fush[y]
                        add_value = \
                            store_fush[fuel_rem][y]*this_tech_ed_vals[y]
                        this_tech_fuel_ed_vals.append(add_value)
                        dict_energy_demand_by_fuel[fuel_rem][y] += add_value

                    dict_energy_demand[tech].update({fuel_rem:this_tech_fuel_ed_vals})

            # Now, let's store the energy demand projections in the country dictionary:
            # parameters: 'Energy demand by sector', 'Energy intensity by sector'
            dict_local_country.update({this_country:{'Energy demand by sector': dict_energy_demand}})
            dict_local_country[this_country].update({'Energy intensity by sector': dict_energy_intensity})
            dict_local_country[this_country].update({'Energy demand by fuel': dict_energy_demand_by_fuel})

            """
            INSTRUCTIONS:
            1) Perform the agro and waste estimations
            2) Store the variables for print
            """
            if model_agro_and_waste:
                # Dataframes with sets:
                # df2_agr_sets_eq
                # df2_res_sets_eq

                # Dataframes with scenarios:
                # df3_agr_data
                # df3_res_data

                # Dataframes with emission factors:
                # df4_ef_agro_res
                # df4_ar_emi

                # Dataframes with techno-economic data:
                # d5_agr
                # d5_res

                # Now we will model according to the diagrams, and we
                # will model emissions and costs simultaneously

                # General data used:
                # Population: this_pop_vector
                # GDP per capita: this_gdp_per_cap_vals

                ### Agriculture
                #-
                # Produccion pecuaria y fermentacion enterica
                # Calculate future activity data:
                mask_scen = (df3_agr_data['Scenario'] == this_scen)
                df3_agr_data_scen = df3_agr_data.loc[mask_scen]

                mask_scen = (df4_ef_agro_res['Scenario'] == this_scen)
                df4_ef_agro_res_scen = df4_ef_agro_res.loc[mask_scen]

                # Project the demand:
                mask_dem = (df3_agr_data_scen['Parameter'] == 'Demand')
                df3_agr_data_scen_dem = df3_agr_data_scen.loc[mask_dem]

                types_livestock = df3_agr_data_scen_dem['Type'].tolist()
                types_projection = df3_agr_data_scen_dem['Projection'].tolist()
                types_by_vals_dem = df3_agr_data_scen_dem[time_vector[0]].tolist()
                

                dem_liv_proj_dict = {}
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_dem[l]
                    this_val_list = []

                    if this_proj == 'grow_gdp_pc':
                        for y in range(len(time_vector)):
                            gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                            if y == 0:
                                this_val_list.append(this_by_val)
                            else:
                                next_val = this_val_list[-1] * (1 + gdp_pc_gr)
                                this_val_list.append(next_val)

                    dem_liv_proj_dict.update({this_live: this_val_list}) #Megatoneladas

                # Include imports and exports (Megatoneladas o Millones de tonaladas)
                mask_imp = (df3_agr_data_scen['Parameter'] == 'Imports') &\
                    (df3_agr_data_scen['Param_ID'] == 5)
                df3_agr_data_scen_imp = df3_agr_data_scen.loc[mask_imp]
                types_livestock = df3_agr_data_scen_imp['Type'].tolist()
                types_projection = df3_agr_data_scen_imp['Projection'].tolist()
                types_by_vals_imp = df3_agr_data_scen_imp[time_vector[0]].tolist()
                
                imp_proj_dict = {}
                all_vals_dict  = {}
                for y in range(len(time_vector)):
                    all_vals_dict.update({time_vector[y]:df3_agr_data_scen_imp[time_vector[y]].tolist()})
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_imp[l]
                    this_val_list = []    
                    if  this_proj == 'grow_gdp_pc':
                        total_imp_list = []
                        gen_imp_pc = []
                        for y in range(len(time_vector)):
                            gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                            if y == 0:
                                gen_imp_pc.append(this_by_val/this_pop_vector[0]) 
                                total_imp_list.append(this_by_val)
                            else:
                                next_val_gen_pc = gen_imp_pc[-1] * (1 + gdp_pc_gr)
                                gen_imp_pc.append(next_val_gen_pc)
                                next_val_total = next_val_gen_pc*this_pop_vector[y]
                                total_imp_list.append(next_val_total)
                        imp_proj_dict.update({this_live: total_imp_list})
                    elif this_proj == 'flat':
                        this_val_list = [this_by_val] * len(time_vector)
                        imp_proj_dict.update({this_live: total_imp_list})
                    elif this_proj == 'user_defined':
                        for y in range(len(time_vector)):
                            this_val_list.append(all_vals_dict[time_vector[y]][l])
                        imp_proj_dict.update({this_live: total_imp_list})
                        

                # Grabbing OPEX:
                d5_liv_imp_opex_mask = (d5_agr['Tech'] == 'Livestock Imports') & \
                    (d5_agr['Scenario'] == this_scen)
                d5_liv_imp_opex = d5_agr.loc[d5_liv_imp_opex_mask]
                d5_liv_imp_opex_by = d5_liv_imp_opex[time_vector[0]].iloc[0]
                d5_liv_imp_opex_proj = d5_liv_imp_opex['Projection'].iloc[0]
                types_imp_projection_opex = d5_liv_imp_opex['Projection'].tolist()
                types_by_imp_vals_opex = d5_liv_imp_opex[time_vector[0]].iloc[0]
                
                
                list_liv_imp_opex = []
                if d5_liv_imp_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_liv_imp_opex.append(types_by_imp_vals_opex)    
                            
                opex_liv_imp_proj_dict = {}
                opex_liv_imp_proj_dict_disc = {}
                # Estimate the cost for livestok (MUSD):
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    list_opex_liv_imp_proj, list_opex_liv_imp_proj_dict  = [], []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        local_out_opex_liv_proj = \
                            list_liv_imp_opex[y] * imp_proj_dict[this_live][y] 
                        list_opex_liv_imp_proj.append(local_out_opex_liv_proj)
                        local_out_opex_liv_proj_disc = \
                            list_liv_imp_opex[y] * imp_proj_dict[this_live][y]* disc_constant 
                        list_opex_liv_imp_proj_dict.append(local_out_opex_liv_proj_disc)
                    opex_liv_imp_proj_dict.update({this_live: list_opex_liv_imp_proj})
                    opex_liv_imp_proj_dict_disc.update({this_live: list_opex_liv_imp_proj_dict})                
                        
                #Storing livestock import costs results 
                dict_local_country[this_country].update({'OPEX de importación ganadera [MUSD]': deepcopy(opex_liv_imp_proj_dict)})
                dict_local_country[this_country].update({'OPEX de importación ganadera [MUSD] (disc)': deepcopy(opex_liv_imp_proj_dict_disc)})
                
                #Exports (Megatoneladas o Millones de tonaladas)
                mask_exp = (df3_agr_data_scen['Parameter'] == 'Exports') &\
                        (df3_agr_data_scen['Param_ID'] == 7)
                df3_agr_data_scen_exp = df3_agr_data_scen.loc[mask_exp]
                types_livestock = df3_agr_data_scen_exp['Type'].tolist()
                types_projection = df3_agr_data_scen_exp['Projection'].tolist()
                types_by_vals_exp = df3_agr_data_scen_exp[time_vector[0]].tolist()
                    
                exp_proj_dict = {}
                all_vals_dict  = {}
                for y in range(len(time_vector)):
                    all_vals_dict.update({time_vector[y]:df3_agr_data_scen_exp[time_vector[y]].tolist()})
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_exp[l]
                    this_val_list = []    
                    if  this_proj == 'grow_gdp_pc':
                        total_exp_list = []
                        gen_exp_pc = []
                        for y in range(len(time_vector)):
                            gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                            if y == 0:
                                gen_exp_pc.append(this_by_val/this_pop_vector[0]) 
                                total_exp_list.append(this_by_val)
                            else:
                                next_val_gen_pc = gen_exp_pc[-1] * (1 + gdp_pc_gr)
                                gen_exp_pc.append(next_val_gen_pc)
                                next_val_total = next_val_gen_pc*this_pop_vector[y]
                                total_exp_list.append(next_val_total)
                        exp_proj_dict.update({this_live: total_exp_list})
                    elif this_proj == 'flat':
                        this_val_list = [this_by_val] * len(time_vector)
                        exp_proj_dict.update({this_live: total_exp_list})
                    elif this_proj == 'user_defined':
                        for y in range(len(time_vector)):
                            this_val_list.append(all_vals_dict[time_vector[y]][l])
                        exp_proj_dict.update({this_live: total_exp_list})   
                        
                # Grabbing OPEX:
                d5_liv_exp_opex_mask = (d5_agr['Tech'] == 'Livestock Exports') & \
                    (d5_agr['Scenario'] == this_scen)
                d5_liv_exp_opex = d5_agr.loc[d5_liv_exp_opex_mask]
                d5_liv_exp_opex_by = d5_liv_exp_opex[time_vector[0]].iloc[0]
                d5_liv_exp_opex_proj = d5_liv_exp_opex['Projection'].iloc[0]
                types_by_exp_vals_opex = d5_liv_exp_opex[time_vector[0]].iloc[0]
                
                
                list_liv_exp_opex = []
                if d5_liv_exp_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_liv_exp_opex.append(types_by_exp_vals_opex)    
                            
                opex_liv_exp_proj_dict = {}
                opex_liv_exp_proj_dict_disc = {}
                # Estimate the cost for livestok (MUSD):
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    list_opex_liv_exp_proj, list_opex_liv_exp_proj_disc = [], []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        local_out_opex_liv_proj = \
                            list_liv_exp_opex[y] * exp_proj_dict[this_live][y] 
                        list_opex_liv_exp_proj.append(local_out_opex_liv_proj)
                        local_out_opex_liv_proj_disc = \
                            list_liv_imp_opex[y] * exp_proj_dict[this_live][y]* disc_constant 
                        list_opex_liv_exp_proj_disc.append(local_out_opex_liv_proj_disc)
                    opex_liv_exp_proj_dict.update({this_live: list_opex_liv_exp_proj})
                    opex_liv_exp_proj_dict_disc.update({this_live: list_opex_liv_exp_proj_disc})               
                        
                #Storing livestock import costs results 
                dict_local_country[this_country].update({'OPEX de exportación ganadera [MUSD]': deepcopy(opex_liv_exp_proj_dict)})
                dict_local_country[this_country].update({'OPEX de exportación ganadera [MUSD] (disc)': deepcopy(opex_liv_exp_proj_dict_disc)})
                
                

                # Obtain heads with average weight:
                mask_heads = (df3_agr_data_scen['Parameter'] == 'Heads')
                df3_agr_data_scen_heads = df3_agr_data_scen.loc[mask_heads]

                mask_aw = (df3_agr_data_scen['Parameter'] == 'Average weight')
                df3_agr_data_scen_aw = df3_agr_data_scen.loc[mask_aw]

                types_livestock = df3_agr_data_scen_aw['Type'].tolist()
                types_projection = df3_agr_data_scen_aw['Projection'].tolist()
                types_by_vals_aw = df3_agr_data_scen_aw[time_vector[0]].tolist()

                all_vals_aw_dict = {}
                for y in range(len(time_vector)):
                    all_vals_aw_dict.update({time_vector[y]:df3_agr_data_scen_aw[time_vector[y]].tolist()})

                dem_aw_proj_dict = {}
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_aw[l]
                    this_val_list = []

                    if this_proj == 'flat':
                        this_val_list = [this_by_val] * len(time_vector)
                        dem_aw_proj_dict.update({this_live: this_val_list})
                    elif this_proj == 'user_defined':
                        this_val_list = []
                        for y in range(len(time_vector)):
                            this_val_list.append(all_vals_aw_dict[time_vector[y]][l])
                        dem_aw_proj_dict.update({this_live: this_val_list})


                types_livestock = df3_agr_data_scen_heads['Type'].tolist()
                types_projection = df3_agr_data_scen_heads['Projection'].tolist()
                types_by_vals_head = df3_agr_data_scen_heads[time_vector[0]].tolist()
                
               
                prod_head_proj_dict = {}
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_head[l]
                    this_val_list = []

                    if this_proj == 'endogenous':
                        for y in range(len(time_vector)):
                            if y == 0:
                                this_val_list.append(this_by_val)
                            else:
                                next_val = \
                                    (dem_liv_proj_dict[this_live][y] -\
                                     imp_proj_dict[this_live][y] + \
                                         exp_proj_dict[this_live][y]) * 1e9 / \
                                     (dem_aw_proj_dict[this_live][y])      
                                this_val_list.append(next_val)

                    prod_head_proj_dict.update({this_live: this_val_list})
                    
                
                # Gather the emissions factor of enteric fermentation:
                mask_fe = (df4_ef_agro_res_scen['Group'] == 'Fermentación entérica')
                df4_ef_agro_res_fe = df4_ef_agro_res_scen.loc[mask_fe]

                types_livestock = df4_ef_agro_res_fe['Type'].tolist()
                types_projection = df4_ef_agro_res_fe['Projection'].tolist()
                types_by_vals_fe = df4_ef_agro_res_fe[time_vector[0]].tolist()

                all_vals_fe_dict = {}
                for y in range(len(time_vector)):
                    all_vals_fe_dict.update({time_vector[y]:df4_ef_agro_res_fe[time_vector[y]].tolist()})

                emisfac_fe_proj_dict = {}
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_fe[l]
                    this_val_list = []

                    if this_proj == 'flat':
                        this_val_list = [this_by_val] * len(time_vector)
                        emisfac_fe_proj_dict.update({this_live: this_val_list})
                    elif this_proj == 'user_defined':
                        this_val_list = []
                        for y in range(len(time_vector)):
                            this_val_list.append(all_vals_fe_dict[time_vector[y]][l])
                        emisfac_fe_proj_dict.update({this_live: this_val_list})

                # Estimate the enteric fermentation emissions:
                out_emis_fe_proj_dict = {}
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    list_out_emis_fe_proj = []
                    for y in range(len(time_vector)):
                        local_out_emis_fe_proj = \
                            emisfac_fe_proj_dict[this_live][y] * \
                            prod_head_proj_dict[this_live][y] / 1e6
                        list_out_emis_fe_proj.append(local_out_emis_fe_proj)
                    out_emis_fe_proj_dict.update({this_live:
                        list_out_emis_fe_proj})

                
                # Calculating Costs:
                list_liv_capex = []
                list_liv_olife = []
                list_liv_opex = []

                # Grabbing CAPEX:
                
                d5_liv_capex_mask = (d5_agr['Tech'] == 'Lifestock Production_CAPEX') & \
                    (d5_agr['Parameter'] == 'ganado') & (d5_agr['Scenario'] == this_scen)
                d5_liv_capex = d5_agr.loc[d5_liv_capex_mask]
                d5_liv_capex_by = d5_liv_capex[time_vector[0]].iloc[0]
                d5_liv_capex_proj = d5_liv_capex['Projection'].iloc[0]
                if d5_liv_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_liv_capex.append(d5_liv_capex_by)
                # Grabbing operational life:
                d5_liv_ol_mask = (d5_agr['Tech'] == 'Lifestock Production') & \
                    (d5_agr['Parameter'] == 'Operational life') & \
                        (d5_agr['Scenario'] == this_scen)
                d5_liv_ol = d5_agr.loc[d5_liv_ol_mask]
                d5_liv_ol_by = d5_liv_ol[time_vector[0]].iloc[0]
                d5_liv_ol_proj = d5_liv_ol['Projection'].iloc[0]
                if d5_liv_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_liv_olife.append(d5_liv_ol_by)
                # Grabbing OPEX:
                d5_liv_opex_mask = (d5_agr['Tech'] == 'Lifestock Production_OPEX') & \
                    (d5_agr['Scenario'] == this_scen)
                d5_liv_opex = d5_agr.loc[d5_liv_opex_mask]
                d5_liv_opex_by = d5_liv_opex[time_vector[0]].iloc[0]
                d5_liv_opex_proj = d5_liv_opex['Projection'].iloc[0]
                types_projection_opex = d5_liv_opex['Projection'].tolist()
                types_liv_opex = d5_liv_opex['Parameter'].tolist()
                types_by_vals_opex = d5_liv_opex[time_vector[0]].tolist()
                
                all_vals_opex_dict = {}
                for y in range(len(time_vector)):
                    all_vals_opex_dict.update({time_vector[y]:d5_liv_opex[time_vector[y]].tolist()})
                    
                for l in range(len(types_liv_opex)):
                    if d5_liv_opex_proj == 'flat':
                        for y in range(len(time_vector)):
                            list_liv_opex.append(d5_liv_opex_by)
                            
                opex_liv_proj_dict = {}
                for l in range(len(types_liv_opex)):
                    this_live = types_liv_opex[l]
                    this_proj = types_projection_opex[l]
                    this_by_val = types_by_vals_opex[l]
                    this_val_list = []

                    if this_proj == 'flat':
                        this_val_list = [this_by_val] * len(time_vector)
                        opex_liv_proj_dict.update({this_live: this_val_list})
                    elif this_proj == 'user_defined':
                        this_val_list = []
                        for y in range(len(time_vector)):
                            this_val_list.append(all_vals_opex_dict[time_vector[y]][l])
                        opex_liv_proj_dict.update({this_live: this_val_list})
                        
                # Calculate investment requirements:
                livestock_heads_total = [sum(values) for values in \
                                         zip(*prod_head_proj_dict.values())]
            
                total_livestock_delta = [0]
                for y in range(1, len(time_vector)):
                    total_livestock_delta.append(livestock_heads_total[y] - livestock_heads_total[y-1])
                for y in range(int(list_liv_olife[0]), len(time_vector)):
                    total_livestock_delta[y] += total_livestock_delta[y - int(list_liv_olife[y])]
                
                liv_capex = [(ucost * act)/1e6 for ucost, act in \
                              zip(list_liv_capex, livestock_heads_total)]
                
                liv_capex_disc = deepcopy(liv_capex)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    liv_capex_disc[y] *= disc_constant
                    

                # Estimate the cost for livestok:
                out_opex_liv_proj_dict = {}
                out_opex_liv_proj_dict_disc = {}
                for l in range(len(types_liv_opex)):
                    this_live = types_liv_opex[l]
                    list_opex_liv_proj = []
                    list_opex_liv_proj_disc = []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        local_out_opex_liv_proj = \
                            opex_liv_proj_dict[this_live][y] * \
                            livestock_heads_total[y] / 1e6
                        local_out_opex_liv_proj_disc = \
                            opex_liv_proj_dict[this_live][y] * \
                            livestock_heads_total[y]* \
                              disc_constant  / 1e6
                        list_opex_liv_proj.append(local_out_opex_liv_proj)
                        list_opex_liv_proj_disc.append(local_out_opex_liv_proj_disc)
                    out_opex_liv_proj_dict.update({this_live:
                        list_opex_liv_proj})
                    out_opex_liv_proj_dict_disc.update({this_live:
                        list_opex_liv_proj_disc})
                            
                            
                #Storing livestock import costs results 
                dict_local_country[this_country].update({'CAPEX de ganadera [MUSD]': deepcopy(liv_capex)})
                dict_local_country[this_country].update({'CAPEX de ganadera [MUSD] (disc)': deepcopy(liv_capex_disc)})
                dict_local_country[this_country].update({'OPEX de ganadera [MUSD]': deepcopy(out_opex_liv_proj_dict)})
                dict_local_country[this_country].update({'OPEX de ganadera [MUSD] (disc)': deepcopy(out_opex_liv_proj_dict_disc)})
                
                # Sistema de gestion de estiercol
                # Gather the emissions factor of enteric fermentation:
                mask_sge = (df4_ef_agro_res_scen['Group'] == 'Sistema de Gestión de Estiércol')
                df4_ef_agro_res_sge = df4_ef_agro_res_scen.loc[mask_sge]

                types_livestock = df4_ef_agro_res_sge['Type'].tolist()
                types_projection = df4_ef_agro_res_sge['Projection'].tolist()
                types_by_vals_sge = df4_ef_agro_res_sge[time_vector[0]].tolist()
                
                emisfac_sge_proj_dict = {}
                all_sge_dict = {}
                for y in range(len(time_vector)):
                    all_sge_dict.update({time_vector[y]:df4_ef_agro_res_sge[time_vector[y]].tolist()})
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_sge[l]
                    this_val_list = []

                    if this_proj == 'flat':
                        this_val_list = [this_by_val] * len(time_vector)
                        emisfac_sge_proj_dict.update({this_live: this_val_list})
                    elif this_proj == 'user_defined':
                        for y in range(len(time_vector)):
                            this_val_list.append(all_sge_dict[time_vector[y]][l])
                        emisfac_sge_proj_dict.update({this_live: this_val_list})

                # Estimate the manure management emissions:
                out_emis_sge_proj_dict = {}
                for l in range(len(types_livestock)):
                    this_live = types_livestock[l]
                    list_out_emis_sge_proj = []
                    for y in range(len(time_vector)):
                        local_out_emis_sge_proj = \
                            emisfac_sge_proj_dict[this_live][y] * \
                            prod_head_proj_dict[this_live][y] / 1e6
                        list_out_emis_sge_proj.append(local_out_emis_sge_proj)
                    out_emis_sge_proj_dict.update({this_live:
                        list_out_emis_sge_proj})
                
                # Calculating Costs:
                sge_capex_proj_dict = {}
                sge_capex_out_dict = {}
                sge_capex_out_dict_disc = {}
                list_sge_olife_dict = {}
                sge_fopex_proj_dict = {}
                sge_vopex_proj_dict = {}
                all_sge_capex_dict = {}
                all_sge_vopex_dict = {}
                all_sge_fopex_dict = {}
                
                # Grabbing CAPEX:
                d5_mask_sge_cost = (d5_agr['Tech'] == 'SGE') & \
                    (d5_agr['Parameter']=='CAPEX') & \
                        (d5_agr['Scenario'] == this_scen)
                d5_sge_capex = d5_agr.loc[d5_mask_sge_cost]
                d5_sge_capex_by = d5_sge_capex[time_vector[0]].iloc[0]
                d5_sge_capex_proj = d5_sge_capex['Projection'].iloc[0]
                types_sge_capex = d5_sge_capex['Type'].tolist()
                types_sge_projection = d5_sge_capex['Projection'].tolist()
                types_by_vals_sge = d5_sge_capex[time_vector[0]].tolist()
                
                for y in range(len(time_vector)):
                    all_sge_capex_dict.update({time_vector[y]:d5_sge_capex[time_vector[y]].tolist()})
                for l in range(len(types_sge_capex)):
                    this_capex = types_sge_capex[l]
                    this_proj = types_sge_projection[l]
                    this_by_val = types_by_vals_sge[l]
                    this_val_list = []
                    
                    if this_proj == 'flat':
                        for y in range(len(time_vector)):
                            this_val_list = [this_by_val]*len(time_vector)
                            sge_capex_proj_dict.update({this_capex: this_val_list})
                    elif this_proj == 'user_defined':
                        for y in range(len(time_vector)):
                            this_val_list.append(all_sge_capex_dict[time_vector[y]][l])
                            sge_capex_proj_dict.update({this_capex: this_val_list})
                
                        
                # Grabbing operational life:
                d5_sge_ol_mask = (d5_agr['Tech'] == 'SGE') & \
                    (d5_agr['Parameter'] == 'Operational life') & \
                        (d5_agr['Scenario'] == this_scen)
                d5_sge_ol = d5_agr.loc[d5_sge_ol_mask]
                d5_sge_fl_ol_by = d5_sge_ol[time_vector[0]].iloc[0]
                d5_seg_ol_proj = d5_sge_ol['Projection'].iloc[0]
                types_sge_ol = d5_sge_ol['Type'].tolist()
                types_sge_ol_projection = d5_sge_ol['Projection'].tolist()
                types_by_vals_sge_ol = d5_sge_ol[time_vector[0]].tolist()
                
                for l in range(len(types_sge_ol)):
                    this_ol = types_sge_ol[l]
                    this_proj = types_sge_ol_projection[l]
                    this_by_val = types_by_vals_sge_ol[l]
                    this_val_list = []
                    
                    if this_proj == 'flat':
                        for y in range(len(time_vector)):
                            this_val_list = [this_by_val]*len(time_vector)
                            list_sge_olife_dict.update({this_ol: this_val_list})
                        
                # Grabbing fixed OPEX:
                d5_sge_fopex_mask = (d5_agr['Tech'] == 'SGE') & \
                    (d5_agr['Parameter']== 'Fixed FOM') & \
                        (d5_agr['Scenario'] == this_scen)
                d5_sge_fopex = d5_agr.loc[d5_sge_fopex_mask]
                d5_sge_fopex_by = d5_sge_fopex[time_vector[0]].iloc[0]
                d5_sge_fopex_proj = d5_sge_fopex['Projection'].iloc[0]
                types_sge_fopex = d5_sge_fopex['Type'].tolist()
                types_sge_projection_fopex = d5_sge_fopex['Projection'].tolist()
                types_by_vals_sge_fopex = d5_sge_fopex[time_vector[0]].tolist()
                
                
                for y in range(len(time_vector)):
                    all_sge_fopex_dict.update({time_vector[y]:d5_sge_fopex[time_vector[y]].tolist()}) 
                for l in range(len(types_sge_fopex)):
                    this_opex = types_sge_fopex[l]
                    this_proj = types_sge_projection_fopex[l]
                    this_by_val = types_by_vals_sge_fopex[l]
                    this_val_list = []
                    
                    if this_proj == 'flat':
                        for y in range(len(time_vector)):
                            this_val_list = [this_by_val]*len(time_vector)
                            sge_fopex_proj_dict.update({this_opex: this_val_list})
                    elif this_proj == 'user_defined':
                        for y in range(len(time_vector)):
                            this_val_list.append(all_sge_fopex_dict[time_vector[y]][l])
                            sge_fopex_proj_dict.update({this_opex: this_val_list})
                            
                # Grabbing variable OPEX:
                d5_sge_vopex_mask = (d5_agr['Tech'] == 'SGE') & \
                    (d5_agr['Parameter']== 'Variable FOM') &\
                        (d5_agr['Scenario'] == this_scen)
                d5_sge_vopex = d5_agr.loc[d5_sge_vopex_mask]
                d5_sge_vopex_by = d5_sge_vopex[time_vector[0]].iloc[0]
                d5_sge_vopex_proj = d5_sge_vopex['Projection'].iloc[0]
                types_sge_vopex = d5_sge_vopex['Type'].tolist()
                types_sge_projection_vopex = d5_sge_vopex['Projection'].tolist()
                types_by_vals_sge_vopex = d5_sge_vopex[time_vector[0]].tolist()
                
                for y in range(len(time_vector)):
                    all_sge_vopex_dict.update({time_vector[y]:d5_sge_vopex[time_vector[y]].tolist()})
                for l in range(len(types_sge_vopex)):
                    this_opex = types_sge_vopex[l]
                    this_proj = types_sge_projection_vopex[l]
                    this_by_val = types_by_vals_sge_vopex[l]
                    this_val_list = []
                    
                    if this_proj == 'flat':
                        this_val_list = [this_by_val]*len(time_vector)
                        sge_vopex_proj_dict.update({this_opex: this_val_list})    
                    elif this_proj == 'user_defined':
                        for y in range(len(time_vector)):
                            this_val_list.append(all_sge_vopex_dict[time_vector[y]][l])
                            sge_vopex_proj_dict.update({this_opex: this_val_list})
                            
                            
                    #Calculating for investment requirement (MUSD)        
                    # Compute delta for this technology
                    tech_life_list = list_sge_olife_dict[types_sge_vopex[l]]  # This list needs to be provided for each technology type
                    this_live = prod_head_proj_dict[types_sge_vopex[l]]
                    delta_list = compute_delta_for_technology(this_live, time_vector, tech_life_list)
                    capex_sge = [(ucost * act)/1e6 for ucost, act in zip(sge_capex_proj_dict[types_sge_vopex[l]], delta_list)]
                    sge_capex_out_dict.update({this_opex: this_val_list})
                    discount_rate = r_rate/100
                    for key, values in sge_capex_out_dict.items():
                        sge_capex_out_dict_disc[key] = discounted_values(values, discount_rate)
                    
                
                #Estimate the fixed opex sge (MUSD)
                out_fopex_proj_dict = {}
                out_fopex_proj_dict_disc = {}
                for l in range(len(types_sge_fopex)):
                    this_fopex = types_sge_fopex[l]
                    list_sge_fopex_proj, list_sge_fopex_proj_disc= [], []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year)) 
                        local_out_sge_fopex_proj = \
                            sge_fopex_proj_dict[this_fopex][y] * \
                            prod_head_proj_dict[this_fopex][y]/1e6
                        local_out_sge_fopex_proj_disc = \
                            sge_fopex_proj_dict[this_fopex][y] * \
                            prod_head_proj_dict[this_fopex][y]*\
                                disc_constant/1e6
                        list_sge_fopex_proj.append(local_out_sge_fopex_proj)
                        list_sge_fopex_proj_disc.append(local_out_sge_fopex_proj_disc)
                    out_fopex_proj_dict.update({this_fopex:list_sge_fopex_proj})
                    out_fopex_proj_dict_disc.update({this_fopex:list_sge_fopex_proj_disc})
               
                #Estimate the variable opex sge (MUSD)
                out_vopex_proj_dict = {}
                out_vopex_proj_dict_disc = {}
                for l in range(len(types_sge_vopex)):
                    this_vopex = types_sge_vopex[l]
                    list_sge_vopex_proj,list_sge_vopex_proj_disc  = [], []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        local_out_sge_vopex_proj = \
                            sge_vopex_proj_dict[this_vopex][y] * \
                            prod_head_proj_dict[this_vopex][y]/1e6
                        local_out_sge_vopex_proj_disc = \
                             sge_vopex_proj_dict[this_vopex][y] * \
                             prod_head_proj_dict[this_vopex][y]*\
                                 disc_constant/1e6
                        list_sge_vopex_proj.append(local_out_sge_vopex_proj)
                        list_sge_vopex_proj_disc.append(local_out_sge_vopex_proj_disc)
                    out_vopex_proj_dict.update({this_vopex:list_sge_vopex_proj})
                    out_vopex_proj_dict_disc.update({this_vopex:list_sge_vopex_proj_disc})
                    

                #Storing SGE costs results 
                dict_local_country[this_country].update({'CAPEX por Sistema de tratemiento de estiércol de ganado [MUSD]': deepcopy(sge_capex_out_dict)})
                dict_local_country[this_country].update({'OPEX fijo por Sistema de tratemiento de estiércol de ganado [MUSD]': deepcopy(out_fopex_proj_dict)})
                dict_local_country[this_country].update({'OPEX variable por Sistema de tratemiento de estiércol de ganado [MUSD]': deepcopy(out_vopex_proj_dict)})
                dict_local_country[this_country].update({'CAPEX por Sistema de tratemiento de estiércol de ganado [MUSD] (disc)': deepcopy(sge_capex_out_dict_disc)})
                dict_local_country[this_country].update({'OPEX fijo por Sistema de tratemiento de estiércol de ganado [MUSD](disc)': deepcopy(out_fopex_proj_dict_disc)})
                dict_local_country[this_country].update({'OPEX variable por Sistema de tratemiento de estiércol de ganado [MUSD](disc)': deepcopy(out_vopex_proj_dict_disc)})
            
                # Cultivo de arroz
                #Demanda de arroz 
                mask_rice = (df3_agr_data_scen['Parameter'] == 'Demand')  &\
                    (df3_agr_data_scen['Type']== 'Rice') # unit: Mton
                df3_agr_data_scen_rice = df3_agr_data_scen.loc[mask_rice]
                types_rice = df3_agr_data_scen_rice['Type'].tolist()
                types_projection_rice = df3_agr_data_scen_rice['Projection'].iloc[0]
                types_by_vals_rice = df3_agr_data_scen_rice[time_vector[0]].iloc[0]
                
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df3_agr_data_scen_rice[time_vector[y]].iloc[0]})
                    
                if  types_projection_rice == 'grow_gdp_pc':
                    total_rice_list = []
                    gen_rice_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_rice_pc.append(types_by_vals_rice/this_pop_vector[0]) 
                            total_rice_list.append(types_by_vals_rice)
                        else:
                            next_val_gen_pc = gen_rice_pc[-1] * (1 + gdp_pc_gr)
                            gen_rice_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_rice_list.append(next_val_total)
                elif  types_projection_rice == 'user_defined':
                    total_rice_list = []
                    gen_rice_pc = []
                    for y in range(len(time_vector)):
                        gen_rice_pc.append(all_vals_gen_pc_dict[time_vector[y]])
                        total_rice_list.append(gen_rice_pc[-1]*this_pop_vector[y])
                elif types_projection_rice == 'flat':
                    total_rice_list = [types_by_vals_rice] * len(time_vector)
                    
                    
                #Importación de arroz 
                mask_rice_imp = (df3_agr_data_scen['Parameter'] == 'Imports')  &\
                    (df3_agr_data_scen['Type']== 'Rice') # unit: Mton
                df3_agr_data_scen_rice_imp = df3_agr_data_scen.loc[mask_rice_imp]
                types_rice_imp = df3_agr_data_scen_rice_imp['Type'].tolist()
                types_projection_rice_imp = df3_agr_data_scen_rice_imp['Projection'].iloc[0]
                types_by_vals_rice_imp = df3_agr_data_scen_rice_imp[time_vector[0]].iloc[0]
                
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df3_agr_data_scen_rice[time_vector[y]].iloc[0]})
                    
                if  types_projection_rice_imp == 'grow_gdp_pc':
                    total_rice_imp_list = []
                    gen_rice_imp_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_rice_imp_pc.append(types_by_vals_rice_imp/this_pop_vector[0]) #quemas per cápita 
                            total_rice_imp_list.append(types_by_vals_rice_imp)
                        else:
                            next_val_gen_pc = gen_rice_imp_pc[-1] * (1 + gdp_pc_gr)
                            gen_rice_imp_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_rice_imp_list.append(next_val_total)
                elif  types_projection_rice_imp == 'user_defined':
                    total_rice_imp_list = []
                    gen_rice_imp_pc = []
                    for y in range(len(time_vector)):
                        gen_rice_imp_pc.append(all_vals_gen_pc_dict[time_vector[y]])
                        total_rice_imp_list.append(gen_rice_imp_pc[-1]*this_pop_vector[y])
                elif types_projection_rice_imp == 'flat':
                    total_rice_imp_list = [types_by_vals_rice_imp] * len(time_vector)
                    
                 # Calculating Costs:
                list_rice_imp_opex = []


                # Grabbing OPEX:
                d5_rice_imp_opex_mask = (d5_agr['Tech'] == 'Rice Imports') & \
                    (d5_agr['Parameter'] == 'OPEX') &\
                        (d5_agr['Scenario'] == this_scen)
                d5_rice_imp_opex = d5_agr.loc[d5_rice_imp_opex_mask]
                d5_rice_imp_opex_by = d5_rice_imp_opex[time_vector[0]].iloc[0]
                d5_rice_imp_opex_proj = d5_rice_imp_opex['Projection'].iloc[0]
                if d5_rice_imp_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_rice_imp_opex.append(d5_rice_imp_opex_by)        
               
                
                rice_imp_opex = [(ucost * act) for ucost, act in zip(list_rice_imp_opex, total_rice_imp_list)] 
                
                
                #Estimating discounted cost
                rice_imp_opex_disc = deepcopy(rice_imp_opex)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    rice_imp_opex_disc[y] *= disc_constant
                
                #Exportación de arroz 
                mask_rice_exp = (df3_agr_data_scen['Parameter'] == 'Exports')  &\
                    (df3_agr_data_scen['Type'] == 'Rice') # unit: Mton
                df3_agr_data_scen_rice_exp = df3_agr_data_scen.loc[mask_rice_exp]
                types_rice_exp = df3_agr_data_scen_rice_exp['Type'].tolist()
                types_projection_rice_exp = df3_agr_data_scen_rice_exp['Projection'].iloc[0]
                types_by_vals_rice_exp = df3_agr_data_scen_rice_exp[time_vector[0]].iloc[0]
                
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df3_agr_data_scen_rice[time_vector[y]].iloc[0]})
                    
                if  types_projection_rice_exp == 'grow_gdp_pc':
                    total_rice_exp_list = []
                    gen_rice_exp_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_rice_exp_pc.append(types_by_vals_rice_exp/this_pop_vector[0]) #quemas per cápita 
                            total_rice_exp_list.append(types_by_vals_rice_exp)
                        else:
                            next_val_gen_pc = gen_rice_exp_pc[-1] * (1 + gdp_pc_gr)
                            gen_rice_exp_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_rice_exp_list.append(next_val_total)
                elif  types_projection_rice_exp == 'user_defined':
                    total_rice_exp_list = []
                    gen_rice_exp_pc = []
                    for y in range(len(time_vector)):
                        gen_rice_exp_pc.append(all_vals_gen_pc_dict[time_vector[y]])
                        total_rice_exp_list.append(gen_rice_exp_pc[-1]*this_pop_vector[y])
                elif types_projection_rice_exp == 'flat':
                    total_rice_exp_list = [types_by_vals_rice_exp] * len(time_vector)
                
                
                 # Calculating Costs:
                list_rice_exp_opex = []


                # Grabbing OPEX:
                d5_rice_exp_opex_mask = (d5_agr['Tech'] == 'Rice Exports') & \
                    (d5_agr['Parameter'] == 'OPEX') &\
                        (d5_agr['Scenario'] == this_scen)
                d5_rice_exp_opex = d5_agr.loc[d5_rice_exp_opex_mask]
                d5_rice_exp_opex_by = d5_rice_exp_opex[time_vector[0]].iloc[0]
                d5_rice_exp_opex_proj = d5_rice_exp_opex['Projection'].iloc[0]
                if d5_rice_exp_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_rice_exp_opex.append(d5_rice_exp_opex_by)       
               
                
                rice_exp_opex = [(ucost * act) for ucost, act in zip(list_rice_exp_opex, total_rice_exp_list)] 
                rice_exp_opex_disc = deepcopy(rice_exp_opex)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    rice_exp_opex_disc[y] *= disc_constant
                
                #Storing rice exports and imports costs results 
                dict_local_country[this_country].update({'OPEX para importación de arroz [MUSD]': deepcopy(rice_imp_opex)})
                dict_local_country[this_country].update({'OPEX para exportación de arroz [MUSD]': deepcopy(rice_exp_opex)})
                dict_local_country[this_country].update({'OPEX para importación de arroz [MUSD](disc)': deepcopy(rice_imp_opex_disc)})
                dict_local_country[this_country].update({'OPEX para exportación de arroz [MUSD](disc)': deepcopy(rice_exp_opex_disc)})
                
                
                #data for floated rice
                mask_rice_fl = (df3_agr_data_scen['Type'] == 'Inundación')  # unit: ha
                df3_rice_fl = df3_agr_data_scen.loc[mask_rice_fl]
                types_projection_rice_fl = df3_rice_fl['Projection'].iloc[0]
                rice_fl_proj = df3_rice_fl['Projection'].iloc[0]
                rice_fl_by = df3_rice_fl[time_vector[0]].iloc[0]
    
                
                # Projection growth type:
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df3_agr_data_scen_rice[time_vector[y]].iloc[0]})
                    
                if  rice_fl_proj == 'grow_gdp_pc':
                    total_rice_fl_list = []
                    gen_rice_fl_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_rice_fl_pc.append(rice_fl_by/this_pop_vector[0]) #quemas per cápita 
                            total_rice_fl_list.append(rice_fl_by)
                        else:
                            next_val_gen_pc = gen_rice_fl_pc[-1] * (1 + gdp_pc_gr)
                            gen_rice_fl_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_rice_fl_list.append(next_val_total)
                elif  rice_fl_proj == 'user_defined':
                    total_rice_fl_list = []
                    gen_rice_fl_pc = []
                    for y in range(len(time_vector)):
                        total_rice_fl_list.append(df3_rice_fl[time_vector[y]].iloc[0])
                elif rice_fl_proj == 'flat':
                    total_rice_fl_list = [rice_fl_by] * len(time_vector)
                    
                mask_fe_rice_fl = (df4_ef_agro_res_scen['Group'] == 'Cultivo de arroz') &\
                    (df4_ef_agro_res_scen['Type'] == 'Cultivo de arroz inundado')
                df4_fe_rice_fl = df4_ef_agro_res_scen.loc[mask_fe_rice_fl]
                types_projection_fe_rice_fl = df4_fe_rice_fl['Projection'].iloc[0]
                types_by_vals_fe_rice_fl = df4_fe_rice_fl[time_vector[0]].iloc[0]
                
                
                if types_projection_fe_rice_fl == 'flat':
                    fe_rice_fl_list = [types_by_vals_fe_rice_fl] * len(time_vector)
                elif types_projection_fe_rice_fl == 'user_defined':
                    for y in range(len(time_vector)):
                        fe_rice_fl_list.append(df4_fe_rice_fl[time_vector[y]].iloc[0])
                
                #Emission estimation (kton CH4)
                rice_fl_emis = [(ef * act)/1e6 for ef, act in zip(fe_rice_fl_list, total_rice_fl_list)]   
                
                # Calculating Costs:
                list_rice_fl_capex = []
                list_rice_fl_olife = []
                list_rice_fl_opex = []

                # Grabbing CAPEX:
                d5_rice_fl_capex_mask = (d5_agr['Tech'] == 'Cultivo_arroz_ CAPEX') &\
                    (d5_agr['Scenario'] == this_scen)
                d5_rice_fl_capex = d5_agr.loc[d5_rice_fl_capex_mask]
                d5_rice_fl_capex_by = d5_rice_fl_capex[time_vector[0]].iloc[0]
                d5_rice_fl_capex_proj = d5_rice_fl_capex['Projection'].iloc[0]
                types_rice_capex = d5_rice_fl_capex['Parameter'].tolist()
                types_rice_projection = d5_rice_fl_capex['Projection'].tolist()
                types_by_vals_fl_rice = d5_rice_fl_capex[time_vector[0]].tolist()
                
                rice_capex_proj_dict = {}
                for l in range(len(types_rice_capex)):
                    this_capex = types_rice_capex[l]
                    this_proj = types_rice_projection[l]
                    this_by_val = types_by_vals_fl_rice[l]
                    this_val_list = []
                    
                    if this_proj == 'flat':
                        #for y in range(len(time_vector)):
                        this_val_list = [this_by_val]*len(time_vector)
                        rice_capex_proj_dict.update({this_capex: this_val_list})
                 
                
                # Grabbing operational life:
                d5_rice_fl_ol_mask = (d5_agr['Tech'] == 'Cultivo_arroz_Op') &\
                    (d5_agr['Scenario'] == this_scen)
                d5_rice_fl_ol = d5_agr.loc[d5_rice_fl_ol_mask]
                d5_rice_fl_ol_by = d5_rice_fl_ol[time_vector[0]].iloc[0]
                d5_rice_fl_ol_proj = d5_rice_fl_ol['Projection'].iloc[0]
                if d5_rice_fl_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_rice_fl_olife.append(d5_rice_fl_ol_by)
                        
                # Grabbing OPEX:
                d5_rice_fl_opex_mask = (d5_agr['Tech'] == 'Cultivo_arroz_OPEX') &\
                    (d5_agr['Scenario'] == this_scen)
                d5_rice_fl_opex = d5_agr.loc[d5_rice_fl_opex_mask]
                d5_rice_fl_opex_by = d5_rice_fl_opex[time_vector[0]].iloc[0]
                d5_rice_fl_opex_proj = d5_rice_fl_opex['Projection'].iloc[0]
                types_rice_opex = d5_rice_fl_opex['Parameter'].tolist()
                types_rice_projection_opex = d5_rice_fl_opex['Projection'].tolist()
                types_by_vals_rice_opex = d5_rice_fl_opex[time_vector[0]].tolist()
                
                
                rice_opex_proj_dict = {}
                for l in range(len(types_rice_opex)):
                    this_opex = types_rice_opex[l]
                    this_proj = types_rice_projection_opex[l]
                    this_by_val = types_by_vals_rice_opex[l]
                    this_val_list = []
                    
                    if this_proj == 'flat':
                        for y in range(len(time_vector)):
                            this_val_list = [this_by_val]*len(time_vector)
                            rice_opex_proj_dict.update({this_opex: this_val_list})
                        
                
                # Calculate investment requirements (ha):
                total_rice_fl_list_delta = [0]
                for y in range(1, len(time_vector)):
                    total_rice_fl_list_delta.append(total_rice_fl_list[y] - total_rice_fl_list[y-1])
                for y in range(int(list_rice_fl_olife[0]), len(time_vector)):
                    total_rice_fl_list_delta[y] += total_rice_fl_list_delta[y - int(list_rice_fl_olife[y])]
                
                #Estimate the investment for floated rice (MUSD)
                out_capex_rice_proj_dict = {}
                out_capex_rice_proj_dict_disc = {}
                for l in range(len(types_rice_capex)):
                    this_capex = types_rice_capex[l]
                    list_capex_rice_proj = []
                    list_capex_rice_proj_disc = []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        if total_rice_fl_list_delta[y] < 0:
                            local_out_capex_rice_proj = 0
                            local_out_capex_rice_proj_disc = 0
                        else:
                            local_out_capex_rice_proj = \
                            rice_capex_proj_dict[this_capex][y] * \
                            total_rice_fl_list_delta[y]/1e6
                            local_out_capex_rice_proj_disc = \
                            rice_capex_proj_dict[this_capex][y] * \
                            total_rice_fl_list_delta[y]*\
                                disc_constant/1e6
                        list_capex_rice_proj.append(local_out_capex_rice_proj)
                        list_capex_rice_proj_disc.append(local_out_capex_rice_proj_disc)
                    out_capex_rice_proj_dict.update({this_capex:list_capex_rice_proj})
                    out_capex_rice_proj_dict_disc.update({this_capex:list_capex_rice_proj_disc})
                
                
                # Estimate the cost for floated rice (MUSD):
                out_opex_rice_fl_proj_dict = {}
                out_opex_rice_fl_proj_dict_disc = {}
                for l in range(len(types_rice_opex)):
                    this_opex = types_rice_opex[l]
                    list_opex_rice_proj = []
                    list_opex_rice_proj_disc = []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        local_out_opex_rice_proj = \
                            rice_opex_proj_dict[this_opex][y] * \
                            total_rice_fl_list[y] / 1e6
                        local_out_opex_rice_proj_disc = \
                            rice_opex_proj_dict[this_opex][y] * \
                            total_rice_fl_list[y] *\
                                disc_constant/ 1e6
                        list_opex_rice_proj.append(local_out_opex_rice_proj)
                        list_opex_rice_proj_disc.append(local_out_opex_rice_proj_disc)
                    out_opex_rice_fl_proj_dict.update({this_opex:
                        list_opex_rice_proj})
                    out_opex_rice_fl_proj_dict_disc.update({this_opex:
                        list_opex_rice_proj_disc})


                #Storing rice floated emissions and costs results 
                dict_local_country[this_country].update({'Emisiones de cultivo de arroz por inundación [kton CH4]': deepcopy(rice_fl_emis)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por inundación [MUSD]': deepcopy(out_opex_rice_fl_proj_dict)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por inundación [MUSD]': deepcopy(out_capex_rice_proj_dict)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por inundación [MUSD](disc)': deepcopy(out_opex_rice_fl_proj_dict_disc)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por inundación [MUSD](disc)': deepcopy(out_capex_rice_proj_dict)})


                #data for irrigated rice
                mask_rice_ir = (df3_agr_data_scen['Type'] == 'Irrigado')  # unit: ha
                df3_rice_ir = df3_agr_data_scen.loc[mask_rice_ir]
                types_projection_rice_ir = df3_rice_ir['Projection'].iloc[0]
                rice_ir_proj = df3_rice_ir['Projection'].iloc[0]
                rice_ir_by = df3_rice_ir[time_vector[0]].iloc[0]
                
                # Projection growth type:
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df3_agr_data_scen_rice[time_vector[y]].iloc[0]})
                    
                if  rice_ir_proj == 'grow_gdp_pc':
                    total_rice_ir_list = []
                    gen_rice_ir_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_rice_ir_pc.append(rice_ir_by/this_pop_vector[0]) #quemas per cápita 
                            total_rice_ir_list.append(rice_ir_by)
                        else:
                            next_val_gen_pc = gen_rice_ir_pc[-1] * (1 + gdp_pc_gr)
                            gen_rice_ir_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_rice_ir_list.append(next_val_total)
                elif  rice_ir_proj == 'user_defined':
                    total_rice_ir_list = []
                    gen_rice_ir_pc = []
                    for y in range(len(time_vector)):
                        #gen_rice_ir_pc.append(all_vals_gen_pc_dict[time_vector[y]])
                        total_rice_ir_list.append(df3_rice_ir[time_vector[y]].iloc[0])
                elif rice_ir_proj == 'flat':
                    total_rice_ir_list = [rice_ir_by] * len(time_vector)
                    
                    
                mask_fe_rice_ir = (df4_ef_agro_res_scen['Group'] == 'Cultivo de arroz') &\
                    (df4_ef_agro_res_scen['Type'] == 'Cultivo de arroz irrigado')
                df4_fe_rice_ir = df4_ef_agro_res_scen.loc[mask_fe_rice_ir]
                types_projection_fe_rice_ir = df4_fe_rice_ir['Projection'].iloc[0]
                types_by_vals_fe_rice_ir = df4_fe_rice_ir[time_vector[0]].iloc[0]
                
                
                if types_projection_fe_rice_ir == 'flat':
                    fe_rice_ir_list = [types_by_vals_fe_rice_ir] * len(time_vector)
                
                #Emission estimation (kton CH4)
                rice_ir_emis = [(ef * act)/1e6 for ef, act in zip(fe_rice_ir_list, total_rice_ir_list)]   
                
                
                # Calculating Costs:
                list_rice_ir_olife = []

                # Grabbing CAPEX:
                d5_rice_ir_capex_mask = (d5_agr['Tech'] == 'Cultivo_arroz_ CAPEX') &\
                    (d5_agr['Scenario'] == this_scen)
                d5_rice_ir_capex = d5_agr.loc[d5_rice_ir_capex_mask]
                d5_rice_ir_capex_by = d5_rice_ir_capex[time_vector[0]].iloc[0]
                d5_rice_ir_capex_proj = d5_rice_ir_capex['Projection'].iloc[0]
                types_rice_capex = d5_rice_ir_capex['Parameter'].tolist()
                types_rice_projection = d5_rice_ir_capex['Projection'].tolist()
                types_by_vals_rice = d5_rice_ir_capex[time_vector[0]].tolist()
                
                
                rice_capex_proj_dict = {}
                for l in range(len(types_rice_capex)):
                    this_capex = types_rice_capex[l]
                    this_proj = types_rice_projection[l]
                    this_by_val = types_by_vals_rice[l]
                    this_val_list = []
                    
                    if this_proj == 'flat':
                        for y in range(len(time_vector)):
                            this_val_list = [this_by_val]*len(time_vector)
                            rice_capex_proj_dict.update({this_capex: this_val_list}) 
                
                
                # Grabbing operational life:
                d5_rice_ir_ol_mask = (d5_agr['Tech'] == 'Cultivo_arroz_Op') & \
                    (d5_agr['Scenario'] == this_scen)
                d5_rice_ir_ol = d5_agr.loc[d5_rice_ir_ol_mask]
                d5_rice_ir_ol_by = d5_rice_ir_ol[time_vector[0]].iloc[0]
                d5_rice_ir_ol_proj = d5_rice_ir_ol['Projection'].iloc[0]
                if d5_rice_ir_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_rice_ir_olife.append(d5_rice_ir_ol_by)
                    
                        
                # Grabbing OPEX:
                d5_rice_ir_opex_mask = (d5_agr['Tech'] == 'Cultivo_arroz_OPEX') &\
                    (d5_agr['Scenario'] == this_scen)
                d5_rice_ir_opex = d5_agr.loc[d5_rice_ir_opex_mask]
                d5_rice_ir_opex_by = d5_rice_ir_opex[time_vector[0]].iloc[0]
                d5_rice_ir_opex_proj = d5_rice_ir_opex['Projection'].iloc[0]
                types_rice_opex = d5_rice_ir_opex['Parameter'].tolist()
                types_rice_projection_opex = d5_rice_ir_opex['Projection'].tolist()
                types_by_vals_rice_opex = d5_rice_ir_opex[time_vector[0]].tolist()
                
                
                rice_opex_proj_dict = {}
                for l in range(len(types_rice_opex)):
                    this_opex = types_rice_opex[l]
                    this_proj = types_rice_projection_opex[l]
                    this_by_val = types_by_vals_rice_opex[l]
                    this_val_list = []
                    
                    if this_proj == 'flat':
                        for y in range(len(time_vector)):
                            this_val_list = [this_by_val]*len(time_vector)
                            rice_opex_proj_dict.update({this_opex: this_val_list})
                        
               
                # Calculate investment requirements (ha):
                total_rice_ir_list_delta = [0]
                for y in range(1, len(time_vector)):
                    total_rice_ir_list_delta.append(total_rice_ir_list[y] - total_rice_ir_list[y-1])
                for y in range(int(list_rice_ir_olife[0]), len(time_vector)):
                    total_rice_ir_list_delta[y] += total_rice_ir_list_delta[y - int(list_rice_ir_olife[y])]
                
                #Estimate the investment for floated rice (MUSD)
                out_capex_rice_proj_dict = {}
                out_capex_rice_proj_dict_disc = {}
                for l in range(len(types_rice_capex)):
                    this_capex = types_rice_capex[l]
                    list_capex_rice_proj = []
                    list_capex_rice_proj_disc = []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        if total_rice_ir_list_delta[y] < 0:
                            local_out_capex_rice_proj = 0
                            local_out_capex_rice_proj_disc = 0
                        else: 
                            local_out_capex_rice_proj = \
                            rice_capex_proj_dict[this_capex][y] * \
                                total_rice_ir_list_delta[y]/1e6
                            local_out_capex_rice_proj_disc = \
                            rice_capex_proj_dict[this_capex][y] * \
                                total_rice_ir_list_delta[y] *\
                                    disc_constant/1e6
                        list_capex_rice_proj.append(local_out_capex_rice_proj)
                        list_capex_rice_proj_disc.append(local_out_capex_rice_proj_disc)
                    out_capex_rice_proj_dict.update({this_capex:
                                                     list_capex_rice_proj})
                    out_capex_rice_proj_dict_disc.update({this_capex:
                                                     list_capex_rice_proj_disc})
                  

                # Estimate the cost for floated rice (MUSD):
                out_opex_rice_proj_dict = {}
                out_opex_rice_proj_dict_disc = {}
                for l in range(len(types_rice_opex)):
                    this_opex = types_rice_opex[l]
                    list_opex_rice_proj = []
                    list_opex_rice_proj_disc = []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))    
                        local_out_opex_rice_proj = \
                            rice_opex_proj_dict[this_opex][y] * \
                            total_rice_ir_list[y] / 1e6
                        local_out_opex_rice_proj_disc = \
                                rice_opex_proj_dict[this_opex][y] * \
                                total_rice_ir_list[y]*\
                                    disc_constant/ 1e6
                        list_opex_rice_proj.append(local_out_opex_rice_proj)
                        list_opex_rice_proj_disc.append(local_out_opex_rice_proj_disc)
                    out_opex_rice_proj_dict.update({this_opex:
                        list_opex_rice_proj})
                    out_opex_rice_proj_dict_disc.update({this_opex:
                        list_opex_rice_proj_disc})
                

                 
                #Storing rice irrigated emissions and costs results 
                dict_local_country[this_country].update({'Emisiones de cultivo de arroz por irrigación [kton CH4]': deepcopy(rice_ir_emis)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por irrigación [MUSD]': deepcopy(out_opex_rice_proj_dict)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por irrigación [MUSD]': deepcopy(out_capex_rice_proj_dict)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por irrigación [MUSD](disc)': deepcopy(out_opex_rice_proj_dict_disc)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por irrigación [MUSD](disc)': deepcopy(out_capex_rice_proj_dict_disc)})
                
                
                #data for aerated rice
                mask_rice_ar = (df3_agr_data_scen['Type'] == 'Aireado')  # unit: ha
                df3_rice_ar = df3_agr_data_scen.loc[mask_rice_ar]
                types_projection_rice_ar = df3_rice_ar['Projection'].iloc[0]
                rice_ar_proj = df3_rice_ar['Projection'].iloc[0]
                rice_ar_by = df3_rice_ar[time_vector[0]].iloc[0]
                
                # Projection growth type:
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df3_agr_data_scen_rice[time_vector[y]].iloc[0]})
                    
                if  rice_ar_proj == 'grow_gdp_pc':
                    total_rice_ar_list = []
                    gen_rice_ar_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_rice_ar_pc.append(rice_ar_by/this_pop_vector[0]) #quemas per cápita 
                            total_rice_ar_list.append(rice_ar_by)
                        else:
                            next_val_gen_pc = gen_rice_ar_pc[-1] * (1 + gdp_pc_gr)
                            gen_rice_ar_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_rice_ar_list.append(next_val_total)
                elif  rice_ar_proj == 'user_defined':
                    total_rice_ar_list = []
                    gen_rice_ar_pc = []
                    for y in range(len(time_vector)):
                        #gen_rice_ar_pc.append(all_vals_gen_pc_dict[time_vector[y]])
                        total_rice_ar_list.append(df3_rice_ar[time_vector[y]].iloc[0])
                elif rice_ar_proj == 'flat':
                    total_rice_ar_list = [rice_ar_by] * len(time_vector)
                
                mask_fe_rice_ar = (df4_ef_agro_res_scen['Group'] == 'Cultivo de arroz') &\
                    (df4_ef_agro_res_scen['Type'] == 'Cultivo de arroz aereado')
                df4_fe_rice_ar = df4_ef_agro_res_scen.loc[mask_fe_rice_ar]
                types_projection_fe_rice_ar = df4_fe_rice_ar['Projection'].iloc[0]
                types_by_vals_fe_rice_ar = df4_fe_rice_ar[time_vector[0]].iloc[0]
                
                
                if types_projection_fe_rice_ar == 'flat':
                    fe_rice_ar_list = [types_by_vals_fe_rice_ar] * len(time_vector)
                
                
                #Emission estimation (kton CH4)
                rice_ar_emis = [(ef * act)/1e6 for ef, act in zip(fe_rice_ar_list, total_rice_ar_list)]   
                
                
                # Calculating Costs:
                list_rice_ar_olife = []


                # Grabbing CAPEX:
                d5_rice_ar_capex_mask = (d5_agr['Tech'] == 'Cultivo_arroz_aireado_CAPEX') &\
                    (d5_agr['Scenario'] == this_scen)
                d5_rice_ar_capex = d5_agr.loc[d5_rice_ar_capex_mask]
                d5_rice_ar_capex_by = d5_rice_ar_capex[time_vector[0]].iloc[0]
                d5_rice_ar_capex_proj = d5_rice_ar_capex['Projection'].iloc[0]
                type_rice_capex = d5_rice_ar_capex['Parameter'].tolist()
                types_rice_projection = d5_rice_ar_capex['Projection'].tolist()
                types_by_vals_rice = d5_rice_ar_capex[time_vector[0]].tolist()
                  

                rice_capex_proj_dict = {}
                for l in range(len(type_rice_capex)):
                    this_capex = type_rice_capex[l]
                    this_proj = types_rice_projection[l]
                    this_by_val = types_by_vals_rice[l]
                    this_val_list = []
                    if this_proj == 'flat':
                        for y in range(len(time_vector)):
                            this_val_list = [this_by_val]*len(time_vector)
                            rice_capex_proj_dict.update({this_capex: this_val_list})
                 
                
                # Grabbing operational life:
                d5_rice_ar_ol_mask = (d5_agr['Tech'] == 'Cultivo_arroz_aireado') &\
                    (d5_agr['Scenario'] == this_scen)
                d5_rice_ar_ol = d5_agr.loc[d5_rice_ar_ol_mask]
                d5_rice_ar_ol_by = d5_rice_ar_ol[time_vector[0]].iloc[0]
                d5_rice_ar_ol_proj = d5_rice_ar_ol['Projection'].iloc[0]
                if d5_rice_ar_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_rice_ar_olife.append(d5_rice_ar_ol_by)
                        
     
                # Grabbing OPEX:
                d5_rice_ar_opex_mask = (d5_agr['Tech'] == 'Cultivo_arroz_aireado_OPEX') &\
                    (d5_agr['Scenario'] == this_scen)
                d5_rice_ar_opex = d5_agr.loc[d5_rice_ar_opex_mask]
                d5_rice_ar_opex_by = d5_rice_ar_opex[time_vector[0]].iloc[0]
                d5_rice_ar_opex_proj = d5_rice_ar_opex['Projection'].iloc[0]
                types_rice_opex_2 = d5_rice_ar_opex['Parameter'].tolist()
                types_rice_projection_opex = d5_rice_ar_opex['Projection'].tolist()
                types_by_vals_rice_opex = d5_rice_ar_opex[time_vector[0]].tolist()
                
                
                rice_opex_proj_dict = {}
                for l in range(len(types_rice_opex_2)):
                    this_opex = types_rice_opex_2[l]
                    this_proj = types_rice_projection_opex[l]
                    this_by_val = types_by_vals_rice_opex[l]
                    this_val_list = []
                    if this_proj == 'flat':
                        for y in range(len(time_vector)):
                            this_val_list = [this_by_val]*len(time_vector)
                            rice_opex_proj_dict.update({this_opex: this_val_list})
                        
                # Calculate investment requirements (ha):
                total_rice_ar_list_delta = [0]
                for y in range(1, len(time_vector)):
                    total_rice_ar_list_delta.append(total_rice_ar_list[y] - total_rice_ar_list[y-1])
                for y in range(int(list_rice_ar_olife[0]), len(time_vector)):
                    total_rice_ar_list_delta[y] += total_rice_ar_list_delta[y - int(list_rice_ar_olife[y])]
                
                
                #Estimate the investment for aerated rice (MUSD)
                out_capex_rice_proj_dict = {}
                out_capex_rice_proj_dict_disc = {}
                for l in range(len(type_rice_capex)):
                    this_capex = type_rice_capex[l]
                    list_capex_rice_proj = []
                    list_capex_rice_proj_disc = []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        local_out_capex_rice_proj = \
                            rice_capex_proj_dict[this_capex][y] * \
                            total_rice_ar_list_delta[y]/1e6
                        local_out_capex_rice_proj_disc = \
                            rice_capex_proj_dict[this_capex][y] * \
                            total_rice_ar_list_delta[y]*\
                                disc_constant/1e6
                        list_capex_rice_proj.append(local_out_capex_rice_proj)
                        list_capex_rice_proj_disc.append(local_out_capex_rice_proj_disc)
                    out_capex_rice_proj_dict.update({this_capex:list_capex_rice_proj})
                    out_capex_rice_proj_dict_disc.update({this_capex:list_capex_rice_proj_disc})
                 
                    
                # Estimate the cost for aerated rice (MUSD):
                out_opex_rice_proj_dict = {}
                out_opex_rice_proj_dict_disc = {}
                for l in range(len(types_rice_opex_2)):
                    this_opex = types_rice_opex_2[l]
                    list_opex_rice_proj = []
                    list_opex_rice_proj_disc = []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        local_out_opex_rice_proj = \
                            rice_opex_proj_dict[this_opex][y] * \
                            total_rice_ar_list[y] / 1e6
                        local_out_opex_rice_proj_disc = \
                            rice_opex_proj_dict[this_opex][y] * \
                            total_rice_ar_list[y]*\
                                disc_constant/ 1e6
                        list_opex_rice_proj.append(local_out_opex_rice_proj)
                        list_opex_rice_proj_disc.append(local_out_opex_rice_proj_disc)
                    out_opex_rice_proj_dict.update({this_opex:
                        list_opex_rice_proj})
                    out_opex_rice_proj_dict_disc.update({this_opex:
                        list_opex_rice_proj_disc})
                
                        
    
                #Storing aerated rice emissions and costs results 
                dict_local_country[this_country].update({'Emisiones de cultivo de arroz por aireado [kton CH4]': deepcopy(rice_ar_emis)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por aireado [MUSD]': deepcopy(out_opex_rice_proj_dict)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por aireado [MUSD]': deepcopy(out_capex_rice_proj_dict)})
                dict_local_country[this_country].update({'OPEX para cultivo de arroz por aireado [MUSD](disc)': deepcopy(out_opex_rice_proj_dict_disc)})
                dict_local_country[this_country].update({'CAPEX para cultivo de arroz por aireado [MUSD](disc)': deepcopy(out_capex_rice_proj_dict_disc)})
                 
                

                # Quema de sabanas
                mask_burns = (df3_agr_data_scen['Parameter'] == 'Quema de sabanas')  # unit: ha
                df3_agr_data_scen_burns = df3_agr_data_scen.loc[mask_burns]
                types_burns = df3_agr_data_scen_burns['Type'].tolist()
                types_projection_burns = df3_agr_data_scen_burns['Projection'].iloc[0]
                types_by_vals_burns = df3_agr_data_scen_burns[time_vector[0]].iloc[0]
            
                # Projection growth type:
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df3_agr_data_scen_burns[time_vector[y]].iloc[0]})
                if  types_projection_burns == 'grow_gdp_pc':
                    total_burns_list = []
                    gen_burns_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_burns_pc.append(types_by_vals_burns/this_pop_vector[0]) #quemas per cápita 
                            total_burns_list.append(types_by_vals_burns)
                        else:
                            next_val_gen_pc = gen_burns_pc[-1] * (1 + gdp_pc_gr)
                            gen_burns_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_burns_list.append(next_val_total)
                elif  types_projection_burns == 'user_defined':
                    total_burns_list = []
                    gen_burns_pc = []
                    for y in range(len(time_vector)):
                        #gen_burns_pc.append(all_vals_gen_pc_dict[time_vector[y]])
                        total_burns_list.append(all_vals_gen_pc_dict[time_vector[y]])    
                
                #--------            
                
                # Tomanto fracción de compostaje [adim]:
                mask_agr_comp = (df3_agr_data_scen['Parameter'] == 'Tratamiento de residuos agricolas')
                agr_comp_df = df3_agr_data_scen.loc[mask_agr_comp]
                arg_comp_proj = agr_comp_df['Projection'].iloc[0]
                agr_comp_by = agr_comp_df[time_vector[0]].iloc[0]
                
                agr_comp_list = []
                if arg_comp_proj == 'flat':
                    agr_comp_list = [agr_comp_by/100]*len(time_vector)
                elif arg_comp_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        agr_comp_list.append(agr_comp_df[time_vector[y]].iloc[0]/100)
                    
                    
                
                #Nueva área de quema de sabana
                total_sabana_burns = [(sabana*(1- comp)) for sabana, comp \
                                 in zip(total_burns_list, agr_comp_list)]
                
                    
                # tasa de generación de biomasa/residuo agrícola por área [ton/ha]
                mask_agr_biomass = (df3_agr_data_scen['Parameter'] == 'tasa de generación de biomasa')
                gen_agr_comp = df3_agr_data_scen.loc[mask_agr_biomass]
                gen_agro_type = gen_agr_comp['Type'].tolist()
                gen_arg_comp_proj = gen_agr_comp['Projection'].tolist()
                gen_agr_comp_by = gen_agr_comp[time_vector[0]].tolist()
                
                
                agr_comp_list_dict = {}
    
                for t in range(len(gen_agro_type)):
                    this_agr = gen_agro_type[t]
                    this_proj = gen_arg_comp_proj[t]
                    this_val = gen_agr_comp_by[t]
                    gen_agr_comp_list = []
                    if this_proj == 'flat':
                        gen_agr_comp_list = [this_val]* len(time_vector)
                        agr_comp_list_dict.update({this_agr:gen_agr_comp_list})
                    elif this_proj == 'user_defined':                 
                        for y in range(len(time_vector)):
                            gen_agr_comp_list.append(gen_agr_comp[time_vector[y]][l])
                        agr_comp_list_dict.update({this_agr: gen_agr_comp_list})
                    
             
                mask_agr_comp_ef = (df4_ef_agro_res_scen['Type'] == 'Compostaje')  # unit: %
                agr_comp_ef_df = df4_ef_agro_res_scen.loc[mask_agr_comp_ef]
                agr_comp_ef_proj = agr_comp_ef_df['Projection'].iloc[0]
                agr_comp_ef_by = agr_comp_ef_df[time_vector[0]].iloc[0]
                if agr_comp_ef_proj == 'flat':
                    agr_comp_ef_list = [agr_comp_ef_by] * len(time_vector)
                    
                
                # Calculating Emissions:[kg CH4/ ton][ha* adim][ton/ha] -> [kton CH4]
                agr_comp_emis = [(ef * act * share * ts)/1e6 for ef, act, share, ts \
                                 in zip(agr_comp_ef_list, total_burns_list, \
                                        agr_comp_list, agr_comp_list_dict['Sabanas'])]

                
                # Calculating Costs:
                list_agr_comp_capex = []
                list_agr_comp_olife = []
                list_agr_comp_opex = []
                list_agr_comp_vopex = []
                list_agr_comp_fopex = []
                # Grabbing CAPEX:
                d5_res_comp_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Compostaje') & \
                    (d5_res['Parameter'] == 'CAPEX')
                d5_res_comp_capex = d5_res.loc[d5_res_comp_capex_mask]
                d5_res_comp_capex_by = d5_res_comp_capex[time_vector[0]].iloc[0]
                d5_res_comp_capex_proj = d5_res_comp_capex['Projection'].iloc[0]
                if d5_res_comp_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_agr_comp_capex.append(d5_res_comp_capex_by)
                # Grabbing operational life:
                d5_res_comp_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Compostaje') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_res_comp_ol = d5_res.loc[d5_res_comp_ol_mask]
                d5_res_comp_ol_by = d5_res_comp_ol[time_vector[0]].iloc[0]
                d5_res_comp_ol_proj = d5_res_comp_ol['Projection'].iloc[0]
                if d5_res_comp_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_agr_comp_olife.append(d5_res_comp_ol_by)
                # Grabbing VOPEX:
                d5_res_comp_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Compostaje') & \
                    (d5_res['Parameter'] == 'Variable FOM')
                d5_res_comp_vopex = d5_res.loc[d5_res_comp_vopex_mask]
                d5_res_comp_vopex_by = d5_res_comp_vopex[time_vector[0]].iloc[0]
                d5_res_comp_vopex_proj = d5_res_comp_vopex['Projection'].iloc[0]
                if d5_res_comp_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_agr_comp_vopex.append(d5_res_comp_vopex_by)
                # Grabbing FOPEX:
                d5_res_comp_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Compostaje') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_res_comp_fopex = d5_res.loc[d5_res_comp_fopex_mask]
                d5_res_comp_fopex_by = d5_res_comp_fopex[time_vector[0]].iloc[0]
                d5_res_comp_fopex_proj = d5_res_comp_fopex['Projection'].iloc[0]
                if d5_res_comp_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_agr_comp_fopex.append(d5_res_comp_fopex_by)
                
                #Nueva área de quema de sabana
                total_sabana_comp = [(sabana*(comp)) for sabana, comp \
                                         in zip(total_burns_list, agr_comp_list)]
                # Calculate investment requirements (ha):
                agr_comp_list_delta = [0]
                for y in range(1, len(time_vector)):
                    if total_sabana_comp[y] - total_sabana_comp[y-1] > 0:
                        agr_comp_list_delta.append(total_sabana_comp[y] - total_sabana_comp[y-1])
                    else:
                        agr_comp_list_delta.append(0)
                for y in range(int(list_agr_comp_olife[0]), len(time_vector)):
                    agr_comp_list_delta[y] += agr_comp_list_delta[y - int(list_agr_comp_olife[y])]
            

                list_agr_comp_opex = [varcost + fcost for varcost, fcost in zip(list_agr_comp_vopex, list_agr_comp_fopex)]
                agr_comp_opex = [(ucost * act* share* ts)/1e6 for ucost, act, share, ts in zip(list_agr_comp_opex, total_burns_list, agr_comp_list, agr_comp_list_dict['Sabanas'])]
                agr_comp_capex = [(ucost * act * ts)/1e6 for ucost, act, ts in zip(list_agr_comp_capex, agr_comp_list_delta, agr_comp_list_dict['Sabanas'])]
                
                
                agr_comp_capex_disc, agr_comp_opex_disc = deepcopy(agr_comp_capex), deepcopy(agr_comp_opex)

                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    agr_comp_capex_disc[y] *= disc_constant
                    agr_comp_opex_disc[y] *= disc_constant
                
            
                #-----
               
                
                mask_fe = (df4_ef_agro_res_scen['Group'] == 'Quema de sabanas') &\
                    (df4_ef_agro_res_scen['Apply'] == 'CH4')
                df4_ef_burns_fe = df4_ef_agro_res_scen.loc[mask_fe]
                types_burns = df4_ef_burns_fe['Type'].tolist()
                types_projection_burns_ef = df4_ef_burns_fe['Projection'].iloc[0]
                types_by_vals_fe_burns = df4_ef_burns_fe[time_vector[0]].iloc[0]
                
                fe_burns_list = []
                if types_projection_burns_ef == 'flat':
                    fe_burns_list = [types_by_vals_fe_burns] * len(time_vector)
                elif types_projection_burns_ef == 'user_defined': 
                    for y in range(len(time_vector)):
                        fe_burns_list.append(df4_ef_burns_fe[time_vector[y]].iloc[0])
                        
                #Emission estimation (kton CH4)
                burns_emis = [(ef * act)/1e6 for ef, act in zip(fe_burns_list, total_sabana_burns)]
                
                #Grabbing the BC emission factor
                mask_bc = (df4_ef_agro_res_scen['Group'] == 'Quema de sabanas')&\
                    (df4_ef_agro_res_scen['Apply'] == 'BC')
                df4_ef_burns_bc = df4_ef_agro_res_scen.loc[mask_bc]
                types_projection_burns_bc = df4_ef_burns_bc['Projection'].iloc[0]
                types_by_vals_bc_burns = df4_ef_burns_bc[time_vector[0]].iloc[0]
                #Black carbon factor emission (kg BC/ha)
                bc_burns_list = []
                if types_projection_burns_bc == 'flat':
                    bc_burns_list = [types_by_vals_bc_burns] * len(time_vector)
                elif types_projection_burns_bc == 'user_defined': 
                    for y in range(len(time_vector)):
                        bc_burns_list.append(df4_ef_burns_bc[time_vector[y]].iloc[0])
                        
                #Black carbon estimation (ton BC)
                burns_bc_emis = [(ef * act)/1e3 for ef, act in zip(bc_burns_list, total_sabana_burns)]
                
                
                # Calculating Costs:
                list_bunrs_capex = []
                list_burns_olife = []
                list_burns_opex = []

                # Grabbing CAPEX:
                d5_burns_capex_mask = (d5_agr['Tech'] == 'Quema_Sabana') & \
                    (d5_agr['Parameter'] == 'CAPEX') &\
                        (d5_agr['Scenario'] == this_scen)
                d5_agr_br_capex = d5_agr.loc[d5_burns_capex_mask]
                d5_agr_br_capex_by = d5_agr_br_capex[time_vector[0]].iloc[0]
                d5_agr_br_capex_proj = d5_agr_br_capex['Projection'].iloc[0]
                if d5_agr_br_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_bunrs_capex.append(d5_agr_br_capex_by)
                # Grabbing operational life:
                d5_agr_br_ol_mask = (d5_agr['Tech'] == 'Quema_Sabana') & \
                    (d5_agr['Parameter'] == 'Operational life') &\
                        (d5_agr['Scenario'] == this_scen)
                d5_agr_br_ol = d5_agr.loc[d5_agr_br_ol_mask]
                d5_agr_br_ol_by = d5_agr_br_ol[time_vector[0]].iloc[0]
                d5_agr_br_ol_proj = d5_agr_br_ol['Projection'].iloc[0]
                if d5_agr_br_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_burns_olife.append(d5_agr_br_ol_by)
                # Grabbing OPEX:
                d5_agr_br_opex_mask = (d5_agr['Tech'] == 'Quema_Sabana') & \
                    (d5_agr['Parameter'] == 'OPEX') &\
                        (d5_agr['Scenario'] == this_scen)
                d5_agr_br_opex = d5_agr.loc[d5_agr_br_opex_mask]
                d5_agr_br_opex_by = d5_agr_br_opex[time_vector[0]].iloc[0]
                d5_agr_br_opex_proj = d5_agr_br_opex['Projection'].iloc[0]
                if d5_agr_br_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_burns_opex.append(d5_agr_br_opex_by)
                elif d5_agr_br_opex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_burns_opex.append(d5_agr_br_opex[time_vector[y]].iloc[0])
                        
               
                # Calculate investment requirements:
                total_burns_list_delta = [0]
                for y in range(1, len(time_vector)):
                    if total_sabana_burns[y] - total_sabana_burns[y-1] > 0:
                        total_burns_list_delta.append(total_sabana_burns[y] - total_sabana_burns[y-1])
                    else:
                        total_burns_list_delta.append(0)
                for y in range(int(list_burns_olife[0]), len(time_vector)):
                    total_burns_list_delta[y] += total_burns_list_delta[y - int(list_burns_olife[y])]


                
                br_opex = [(ucost * act)/1e6 for ucost, act in zip(list_burns_opex, total_sabana_burns)]
                br_capex = [(ucost * act)/1e6 for ucost, act in zip(list_bunrs_capex, total_burns_list_delta)]
                
                
                br_capex_disc, br_opex_disc = deepcopy(br_capex), deepcopy(br_opex)

                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    br_capex_disc[y] *= disc_constant
                    br_opex_disc[y] *= disc_constant

                #Storing burnt sabanas costs and emissions results 
                dict_local_country[this_country].update({'Emisiones de quema de sabanas [kton CH4]': deepcopy(burns_emis)})
                dict_local_country[this_country].update({'OPEX para quema de saban [MUSD]': deepcopy(br_opex)})
                dict_local_country[this_country].update({'CAPEX para quema de saban [MUSD]': deepcopy(br_capex)})
                dict_local_country[this_country].update({'OPEX para quema de saban [MUSD](disc)': deepcopy(br_opex_disc)})
                dict_local_country[this_country].update({'CAPEX para quema de saban [MUSD](disc)': deepcopy(br_capex_disc)}) 
                dict_local_country[this_country].update({'Emisiones carbono negro de quema de sabanas [ton]': deepcopy(burns_bc_emis)})
                dict_local_country[this_country].update({'Emisiones de compostaje residuos en sabanas [kton CH4]': deepcopy(agr_comp_emis)})
                dict_local_country[this_country].update({'OPEX para compostaje de sabanas [MUSD]': deepcopy(agr_comp_opex)})
                dict_local_country[this_country].update({'CAPEX para compostaje de sabanas [MUSD]': deepcopy(agr_comp_capex)})
                dict_local_country[this_country].update({'OPEX para compostaje de sabanas [MUSD](disc)': deepcopy(agr_comp_opex_disc)})
                dict_local_country[this_country].update({'CAPEX para compostaje de sabanas [MUSD](disc)': deepcopy(agr_comp_capex_disc)}) 
                
               
                # Quema de residuos agricolas
                mask_burns_ag = (df3_agr_data_scen['Parameter'] == 'Quema de residuos')  # unit: ha 
                df3_agr_data_scen_burns_ag = df3_agr_data_scen.loc[mask_burns_ag]
                types_burns_ag = df3_agr_data_scen_burns_ag['Type'].tolist()
                types_projection_burns_ag = df3_agr_data_scen_burns_ag['Projection'].iloc[0]
                types_by_vals_burns_ag = df3_agr_data_scen_burns_ag[time_vector[0]].iloc[0]
                
                #data for cereals
                mask_burns_cereal = (df3_agr_data_scen['Type'] == 'Cereales')  # unit: ha
                df3_br_ce = df3_agr_data_scen.loc[mask_burns_cereal]
                types_projection_burn_ce = df3_br_ce['Projection'].iloc[0]
                grs_br_ce_proj = df3_br_ce['Projection'].iloc[0]
                grs_br_ce_by = df3_br_ce[time_vector[0]].iloc[0]
                
            
                # Projection growth type:
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df3_agr_data_scen_burns_ag[time_vector[y]].iloc[0]})
            
                if  types_projection_burn_ce == 'grow_gdp_pc':
                    total_burns_ce_list = []
                    gen_burns_ce_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_burns_ce_pc.append(grs_br_ce_by/this_pop_vector[0]) #quemas per cápita 
                            total_burns_ce_list.append(grs_br_ce_by)
                        else:
                            next_val_gen_pc = gen_burns_ce_pc[-1] * (1 + gdp_pc_gr)
                            gen_burns_ce_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_burns_ce_list.append(next_val_total)
                elif  types_projection_burn_ce == 'user_defined':
                    total_burns_ce_list = []
                    gen_burns_ce_pc = []
                    for y in range(len(time_vector)):
                        #gen_burns_ce_pc.append(all_vals_gen_pc_dict[time_vector[y]])
                        total_burns_ce_list.append(all_vals_gen_pc_dict[time_vector[y]])
                elif types_projection_burn_ce == 'flat':
                    total_burns_ce_list = [grs_br_ce_by] * len(time_vector)
                
                
                mask_fe = (df4_ef_agro_res_scen['Group'] == 'Quema de residuos') & \
                    (df4_ef_agro_res_scen['Type'] == 'Cereales')&\
                       (df4_ef_agro_res_scen['Apply'] == 'CH4')  
                df4_ef_burns_ce = df4_ef_agro_res_scen.loc[mask_fe]
                types_projection_burns_ef_ce = df4_ef_burns_ce['Projection'].iloc[0]
                types_by_vals_burns_ef_ce = df4_ef_burns_ce[time_vector[0]].iloc[0]
                
                fe_burns_ce_list = []
                if types_projection_burns_ef_ce == 'flat':
                    fe_burns_ce_list = [types_by_vals_burns_ef_ce] * len(time_vector)
                elif types_projection_burns_ef_ce == 'user_defined':
                    for y in range(len(time_vector)):
                        fe_burns_ce_list.append(df4_ef_burns_ce[time_vector[y]].iloc[0])
                
                #Nueva área de quema de otros residuos
                total_ce_burns = [(ce*(1- comp)) for ce, comp \
                                 in zip(total_burns_ce_list, agr_comp_list)]
                
        
                # Calculating Emissions:[kg CH4/ ton][ha* adim][ton/ha] -> [kton CH4]
                agr_ce_comp_emis = [(ef * act * share * ts)/1e6 for ef, act, share, ts \
                                 in zip(agr_comp_ef_list, total_burns_ce_list, \
                                        agr_comp_list, agr_comp_list_dict['Otros cultivos'])]

                
                #Área de compostaje de otros residuos
                total_ce_comp = [(ce*(comp)) for ce, comp \
                                         in zip(total_burns_ce_list, agr_comp_list)]
                # Calculate investment requirements (ha):
                ce_comp_list_delta = [0]
                for y in range(1, len(time_vector)):
                    if total_ce_comp[y] - total_ce_comp[y-1] > 0:
                        ce_comp_list_delta.append(total_ce_comp[y] - total_ce_comp[y-1])
                    else:
                        ce_comp_list_delta.append(0)
                for y in range(int(list_agr_comp_olife[0]), len(time_vector)):
                    ce_comp_list_delta[y] += ce_comp_list_delta[y - int(list_agr_comp_olife[y])]
                    
                
                
                agr_ce_comp_opex = [(ucost * act* share* ts)/1e6 for ucost, act, share, ts in zip(list_agr_comp_opex, total_burns_ce_list, agr_comp_list, agr_comp_list_dict['Otros cultivos'])]
                agr_ce_comp_capex = [(ucost * act * ts)/1e6 for ucost, act, ts in zip(list_agr_comp_capex, ce_comp_list_delta, agr_comp_list_dict['Otros cultivos'])]
                
                
                agr_ce_comp_capex_disc, agr_ce_comp_opex_disc = deepcopy(agr_ce_comp_capex), deepcopy(agr_ce_comp_opex)

                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    agr_ce_comp_capex_disc[y] *= disc_constant
                    agr_ce_comp_opex_disc[y] *= disc_constant
                
                
                #------
                #Emission estimation (kton of CH4)
                ce_burns_emis = [(ef * act)/1e6 for ef, act in zip(fe_burns_ce_list, total_ce_burns)]
                
                  
                mask_bc = (df4_ef_agro_res_scen['Group'] == 'Quema de residuos') & \
                    (df4_ef_agro_res_scen['Type'] == 'Otros cultivos') &\
                       (df4_ef_agro_res_scen['Apply'] == 'BC')  
                df4_bc_burns_ce = df4_ef_agro_res_scen.loc[mask_bc]
                types_projection_burns_bc_ce = df4_bc_burns_ce['Projection'].iloc[0]
                types_by_vals_burns_bc_ce = df4_bc_burns_ce[time_vector[0]].iloc[0]
                
                bc_burns_ce_list = []
                if types_projection_burns_bc_ce == 'flat':
                    bc_burns_ce_list = [types_by_vals_burns_bc_ce] * len(time_vector)
                elif types_projection_burns_bc_ce == 'user_defined':
                    for y in range(len(time_vector)):
                        bc_burns_ce_list.append(df4_bc_burns_ce[time_vector[y]].iloc[0])
                
                
                #Black carbon estimation (ton of BC)
                ce_bc_burns_emis = [(ef * act)/1e3 for ef, act in zip(bc_burns_ce_list, total_ce_burns)]
                #data for sugarcane
                mask_burns_sugarcane = (df3_agr_data_scen['Type'] == 'Caña de Azúcar')  # unit: ha
                df3_br_sc = df3_agr_data_scen.loc[mask_burns_sugarcane]
                types_projection_burn_sc = df3_br_sc['Projection'].iloc[0]
                grs_br_sc_proj = df3_br_sc['Projection'].iloc[0]
                grs_br_sc_by = df3_br_sc[time_vector[0]].iloc[0]
                
                # Projection growth type:
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df3_br_sc[time_vector[y]].iloc[0]})
                if  types_projection_burn_sc == 'grow_gdp_pc':
                    total_burns_sc_list = []
                    gen_burns_sc_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_burns_sc_pc.append(grs_br_sc_by/this_pop_vector[0]) #quemas per cápita 
                            total_burns_sc_list.append(grs_br_sc_by)
                        else:
                            next_val_gen_pc = gen_burns_sc_pc[-1] * (1 + gdp_pc_gr)
                            gen_burns_sc_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_burns_sc_list.append(next_val_total)
                elif  types_projection_burn_sc == 'user_defined':
                    total_burns_sc_list = []
                    gen_burns_sc_pc = []
                    for y in range(len(time_vector)):
                        #gen_burns_sc_pc.append(all_vals_gen_pc_dict[time_vector[y]])
                        total_burns_sc_list.append(all_vals_gen_pc_dict[time_vector[y]])
                elif types_projection_burn_sc == 'flat':
                    total_burns_sc_list = [grs_br_sc_by] * len(time_vector)
                    
                #Nueva área de quema de otros residuos  (ha)
                total_sc_burns = [(ce*(1- comp)) for ce, comp \
                                 in zip(total_burns_sc_list, agr_comp_list)]
                
        
                # Calculating Emissions:[kg CH4/ ton][ha* adim][ton/ha] -> [kton CH4]
                agr_sc_comp_emis = [(ef * act * share * ts)/1e6 for ef, act, share, ts \
                                 in zip(agr_comp_ef_list, total_burns_sc_list, \
                                        agr_comp_list, agr_comp_list_dict['Caña de Azúcar'])]

                
                #Nueva área de quema de otros residuos
                total_sc_comp = [(sc*(comp)) for sc, comp \
                                         in zip(total_burns_sc_list, agr_comp_list)]
                # Calculate investment requirements (ha):
                sc_comp_list_delta = [0]
                for y in range(1, len(time_vector)):
                    if total_sc_comp[y] - total_sc_comp[y-1] > 0:
                        sc_comp_list_delta.append(total_sc_comp[y] - total_sc_comp[y-1])
                    else:
                        sc_comp_list_delta.append(0)
                for y in range(int(list_agr_comp_olife[0]), len(time_vector)):
                    sc_comp_list_delta[y] += sc_comp_list_delta[y - int(list_agr_comp_olife[y])]
                    
                
                
                agr_sc_comp_opex = [(ucost * act* share* ts)/1e6 for ucost, act, share, ts in zip(list_agr_comp_opex, total_burns_sc_list, agr_comp_list, agr_comp_list_dict['Caña de Azúcar'])]
                agr_sc_comp_capex = [(ucost * act * ts)/1e6 for ucost, act, ts in zip(list_agr_comp_capex, sc_comp_list_delta, agr_comp_list_dict['Caña de Azúcar'])]
                
                
                agr_sc_comp_capex_disc, agr_sc_comp_opex_disc = deepcopy(agr_sc_comp_capex), deepcopy(agr_sc_comp_opex)

                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    agr_sc_comp_capex_disc[y] *= disc_constant
                    agr_sc_comp_opex_disc[y] *= disc_constant
                
                
                #------                
                mask_fe_sc = (df4_ef_agro_res_scen['Group'] == 'Quema de residuos') & \
                     (df4_ef_agro_res_scen['Type'] == 'Caña de azúcar') & \
                          (df4_ef_agro_res_scen['Apply'] == 'CH4')
                df4_ef_burns_sc = df4_ef_agro_res_scen.loc[mask_fe_sc]
                types_projection_burns_ef_sc = df4_ef_burns_sc['Projection'].iloc[0]
                types_by_vals_burns_ef_sc = df4_ef_burns_sc[time_vector[0]].iloc[0]
                 
                fe_burns_sc_list = [] 
                if types_projection_burns_ef_sc == 'flat':
                     fe_burns_sc_list = [types_by_vals_burns_ef_sc] * len(time_vector)
                elif types_projection_burns_ef_sc == 'user_defined':
                    for y in range(len(time_vector)):
                        fe_burns_sc_list.append(df4_ef_burns_sc[time_vector[y]].iloc[0])
                 
                #Emission estimation (kton of CH4)
                sc_burns_emis = [(ef * act)/1e6 for ef, act in zip(fe_burns_sc_list, total_sc_burns)]
                
                mask_bc_sc = (df4_ef_agro_res_scen['Group'] == 'Quema de residuos') & \
                     (df4_ef_agro_res_scen['Type'] == 'Caña de azúcar')& \
                          (df4_ef_agro_res_scen['Apply'] == 'BC')
                df4_bc_burns_sc = df4_ef_agro_res_scen.loc[mask_bc_sc]
                types_projection_burns_bc_sc = df4_bc_burns_sc['Projection'].iloc[0]
                types_by_vals_burns_bc_sc = df4_bc_burns_sc[time_vector[0]].iloc[0]
                 
                bc_burns_sc_list = [] 
                if types_projection_burns_bc_sc == 'flat':
                     bc_burns_sc_list = [types_by_vals_burns_bc_sc] * len(time_vector)
                elif types_projection_burns_bc_sc == 'user_defined':
                    for y in range(len(time_vector)):
                        bc_burns_sc_list.append(df4_bc_burns_sc[time_vector[y]].iloc[0])
                 
                 #Emission estimation (ton of BC)
                sc_burns_emis_bc = [(ef * act)/1e3 for ef, act in zip(bc_burns_sc_list, total_sc_burns)]
                # Calculating Costs:
                list_bunrs_re_capex = []
                list_burns_re_olife = []
                list_burns_re_opex = []

                # Grabbing CAPEX:
                d5_burns_re_capex_mask = (d5_agr['Tech'] == 'Quema_Residuos') & \
                    (d5_agr['Parameter'] == 'CAPEX') &\
                        (d5_agr['Scenario'] == this_scen)
                d5_agr_brre_capex = d5_agr.loc[d5_burns_re_capex_mask]
                d5_agr_brre_capex_by = d5_agr_brre_capex[time_vector[0]].iloc[0]
                d5_agr_brre_capex_proj = d5_agr_brre_capex['Projection'].iloc[0]
                if d5_agr_brre_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_bunrs_re_capex.append(d5_agr_brre_capex_by)
                # Grabbing operational life:
                d5_agr_brre_ol_mask = (d5_agr['Tech'] == 'Quema_Residuos') & \
                    (d5_agr['Parameter'] == 'Operational life') &\
                        (d5_agr['Scenario'] == this_scen)
                d5_agr_brre_ol = d5_agr.loc[d5_agr_brre_ol_mask]
                d5_agr_brre_ol_by = d5_agr_brre_ol[time_vector[0]].iloc[0]
                d5_agr_brre_ol_proj = d5_agr_brre_ol['Projection'].iloc[0]
                if d5_agr_brre_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_burns_re_olife.append(d5_agr_brre_ol_by)
                # Grabbing OPEX:
                d5_agr_brre_opex_mask = (d5_agr['Tech'] == 'Quema_Residuos') & \
                    (d5_agr['Parameter'] == 'OPEX') &\
                        (d5_agr['Scenario'] == this_scen)
                d5_agr_brre_opex = d5_agr.loc[d5_agr_brre_opex_mask]
                d5_agr_brre_opex_by = d5_agr_brre_opex[time_vector[0]].iloc[0]
                d5_agr_brre_opex_proj = d5_agr_brre_opex['Projection'].iloc[0]
                if d5_agr_brre_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_burns_re_opex.append(d5_agr_brre_opex_by)
                elif d5_agr_brre_opex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_burns_re_opex.append(d5_agr_brre_opex[time_vector[y]].iloc[0])       
                    
                # Calculate investment requirements:
                total_burns_sc_list_delta = [0]
                for y in range(1, len(time_vector)):
                    if total_sc_burns[y] - total_sc_burns[y-1] > 0:
                        total_burns_sc_list_delta.append(total_sc_burns[y] - total_sc_burns[y-1])
                    else:
                        total_burns_sc_list_delta.append(0)
                for y in range(int(list_burns_re_olife[0]), len(time_vector)):
                    total_burns_sc_list_delta[y] += total_burns_sc_list_delta[y - int(list_burns_re_olife[y])]
                    
                total_burns_ce_list_delta = [0]
                for y in range(1, len(time_vector)):
                    if total_ce_burns[y] - total_ce_burns[y-1] > 0:
                        total_burns_ce_list_delta.append(total_ce_burns[y] - total_ce_burns[y-1])
                    else:
                        total_burns_ce_list_delta.append(0)
                for y in range(int(list_burns_re_olife[0]), len(time_vector)):
                    total_burns_ce_list_delta[y] += total_burns_ce_list_delta[y - int(list_burns_re_olife[y])]


                br_sc_opex = [(ucost * act)/1e6 for ucost, act in zip(list_burns_re_opex, total_sc_burns)]
                br_ce_opex = [(ucost * act)/1e6 for ucost, act in zip(list_burns_re_opex, total_ce_burns)]
                br_sc_capex = [(ucost * act)/1e6 for ucost, act in zip(list_bunrs_re_capex, total_burns_sc_list_delta)]   
                br_ce_capex = [(ucost * act)/1e6 for ucost, act in zip(list_bunrs_re_capex, total_burns_ce_list_delta)]  
                
                br_sc_opex_disc, br_ce_opex_disc, br_sc_capex_disc, br_ce_capex_disc = \
                    deepcopy(br_sc_opex), deepcopy(br_ce_opex), deepcopy(br_sc_capex), \
                    deepcopy(br_ce_capex)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    br_sc_opex_disc[y] *= disc_constant
                    br_ce_opex_disc[y] *= disc_constant
                    br_sc_capex_disc[y] *= disc_constant
                    br_ce_capex_disc[y] *= disc_constant
                   
                    
                #Storing agricultural burns emissions and costs results 
                dict_local_country[this_country].update({'Quema de residuos agrícolas de caña de azúcar [kt CH4]': deepcopy(sc_burns_emis)})
                dict_local_country[this_country].update({'Quema de residuos agrícolas de cereales [kt CH4]': deepcopy(ce_burns_emis)})
                dict_local_country[this_country].update({'Carbono negro de quema de otros residuos agrícolas [ton]': deepcopy(ce_bc_burns_emis)})
                dict_local_country[this_country].update({'Carbono negro de quema de residuos agrícolas de caña de azúcar [ton]': deepcopy(sc_burns_emis_bc)})                                                                                                                                                                                                                                                                                  
                dict_local_country[this_country].update({'CAPEX de quema de residuos agrícolas de caña de azúcar [MUSD]': deepcopy(br_sc_capex)})
                dict_local_country[this_country].update({'CAPEX de quema de residuos agrícolas de cereales [MUSD]': deepcopy(br_ce_capex)})
                dict_local_country[this_country].update({'OPEX de quema de residuos agrícolas de caña de azúcar [MUSD]': deepcopy(br_sc_opex)})
                dict_local_country[this_country].update({'OPEX de quema de residuos agrícolas de cereales [MUSD]': deepcopy(br_ce_opex)})
                dict_local_country[this_country].update({'CAPEX de quema de residuos agrícolas de caña de azúcar [MUSD] (disc)': deepcopy(br_sc_capex_disc)})
                dict_local_country[this_country].update({'CAPEX de quema de residuos agrícolas de cereales [MUSD] (disc)': deepcopy(br_ce_capex_disc)})
                dict_local_country[this_country].update({'OPEX de quema de residuos agrícolas de caña de azúcar [MUSD] (disc)': deepcopy(br_sc_opex_disc)})
                dict_local_country[this_country].update({'OPEX de quema de residuos agrícolas de cereales [MUSD] (disc)': deepcopy(br_ce_opex_disc)})
                dict_local_country[this_country].update({'Compostaje de residuos agrícolas de caña de azúcar [kt CH4]': deepcopy(agr_sc_comp_emis)})
                dict_local_country[this_country].update({'Compostaje de residuos agrícolas de otros cultivos [kt CH4]': deepcopy(agr_ce_comp_emis)})
                dict_local_country[this_country].update({'CAPEX de compostaje de residuos agrícolas de caña de azúcar [MUSD]': deepcopy(agr_sc_comp_capex)})
                dict_local_country[this_country].update({'CAPEX de compostaje de residuos agrícolas de otros cultivos [MUSD]': deepcopy(agr_ce_comp_capex)})
                dict_local_country[this_country].update({'OPEX de compostaje de residuos agrícolas de caña de azúcar [MUSD]': deepcopy(agr_sc_comp_opex)})
                dict_local_country[this_country].update({'OPEX de compostaje de residuos agrícolas de otros cultivos [MUSD]': deepcopy(agr_ce_comp_opex)})
                dict_local_country[this_country].update({'CAPEX de compostaje de residuos agrícolas de caña de azúcar [MUSD] (disc)': deepcopy(agr_sc_comp_capex_disc)})
                dict_local_country[this_country].update({'CAPEX de compostaje de residuos agrícolas de otros cultivos [MUSD] (disc)': deepcopy(agr_ce_comp_capex_disc)})
                dict_local_country[this_country].update({'OPEX de quema de residuos agrícolas de caña de azúcar [MUSD] (disc)': deepcopy(agr_sc_comp_opex_disc)})
                dict_local_country[this_country].update({'OPEX de quema de residuos agrícolas de otros cultivos [MUSD] (disc)': deepcopy(agr_ce_comp_opex_disc)})
                # STORING EMISSIONS
                dict_local_country[this_country].update({'Emisiones de fermentación entérica [kt CH4]': deepcopy(out_emis_fe_proj_dict)})
                dict_local_country[this_country].update({'Sistema de gestión de estiércol [kt CH4]': deepcopy(out_emis_sge_proj_dict)})


                print('Agricultural emissions have been computed!')
                
            
                # Calculate future activity data:
                mask_scen = (df3_res_data['Scenario'] == this_scen)
                df3_res_data_scen = df3_res_data.loc[mask_scen]

                mask_scen = (df4_ef_agro_res['Scenario'] == this_scen)
                df4_ef_agro_res_scen = df4_ef_agro_res.loc[mask_scen]

                mask_scen = (df4_ar_emi['Scenario'] == this_scen)
                df4_ar_emi_scen = df4_ar_emi.loc[mask_scen]

                # General data used:
                # Population: this_pop_vector
                # GDP per capita: this_gdp_per_cap_vals

                # Solid waste:
                mask_gen_res_sol_pc = (df3_res_data_scen['Type'] == 'Generación diaria de residuos')  # unit: kg/persona/día
                gen_res_sol_pc_df = df3_res_data_scen.loc[mask_gen_res_sol_pc]
                gen_res_sol_pc_proj = gen_res_sol_pc_df['Projection'].iloc[0]
                gen_res_sol_pc_by = gen_res_sol_pc_df[time_vector[0]].iloc[0]

                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:gen_res_sol_pc_df[time_vector[y]].iloc[0]})
                # generación de residuos en [kg]
                if gen_res_sol_pc_proj == 'grow_gdp_pc':
                    total_sw_list = []
                    gen_res_sol_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_res_sol_pc.append(gen_res_sol_pc_by)
                            total_sw_list.append(gen_res_sol_pc_by*this_pop_vector[0]*1e6*365)
                        else:
                            next_val_gen_pc = gen_res_sol_pc[-1] * (1 + gdp_pc_gr)
                            gen_res_sol_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_sw_list.append(next_val_total*1e6*365)
                elif gen_res_sol_pc_proj == 'user_defined':
                    total_sw_list = []
                    gen_res_sol_pc = []
                    for y in range(len(time_vector)):
                        gen_res_sol_pc.append(all_vals_gen_pc_dict[time_vector[y]])
                        total_sw_list.append(gen_res_sol_pc[-1]*this_pop_vector[y]*1e6*365)
                elif gen_res_sol_pc_proj == 'flat':
                    total_sw_list = []
                    gen_res_sol_pc = []
                    for y in range(len(time_vector)):
                        gen_res_sol_pc.append(gen_res_sol_pc_by)
                        total_sw_list.append(gen_res_sol_pc[-1]*this_pop_vector[y]*1e6*365)
                
                
                
                # > Emissions and costs for relleno saniario:
                mask_grs_rell_san = (df3_res_data_scen['Type'] == 'Relleno sanitario')  # unit: %
                grs_rs_df = df3_res_data_scen.loc[mask_grs_rell_san]
                grs_rs_proj = grs_rs_df['Projection'].iloc[0]
                grs_rs_by = grs_rs_df[time_vector[0]].iloc[0]
                grs_rs_list_delta = [0] * len(time_vector)
                if grs_rs_proj == 'user_defined':
                    grs_rs_list = [total_sw_list[0]*grs_rs_by/100]
                    for y in range(1, len(time_vector)):
                        grs_rs_list.append(total_sw_list[y] * \
                            grs_rs_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_rs_list_delta[y] = \
                                grs_rs_list[y] - grs_rs_list[y-1]
                elif grs_rs_proj == 'flat':
                    grs_rs_list = [total_sw_list[0]*grs_rs_by/100]
                    for y in range(1, len(time_vector)):
                        grs_rs_list.append(total_sw_list[y] * \
                            grs_rs_df[time_vector[0]].iloc[0]/100)
                        if y > 0:
                            grs_rs_list_delta[y] = \
                                grs_rs_list[y] - grs_rs_list[y-1]
                
                mask_grs_rs_ef = (df4_ef_agro_res_scen['Type'] == 'Relleno sanitario')  # unit: %
                grs_rs_ef_df = df4_ef_agro_res_scen.loc[mask_grs_rs_ef]
                grs_rs_ef_proj = grs_rs_ef_df['Projection'].iloc[0]
                grs_rs_ef_by = grs_rs_ef_df[time_vector[0]].iloc[0]
                if grs_rs_ef_proj == 'flat':
                    grs_rs_ef_list = [grs_rs_ef_by] * len(time_vector)

                
                # Calculating Emissions:
                grs_rs_emis = [(ef * act)/1e9 for ef, act in zip(grs_rs_ef_list, grs_rs_list)]
                
                # Substract methane emissions from methane capture
                '''
                This code subtracts emissions from methane capture
                First, the Landfill_CH4_capture row under Parameter is read for 16_res_data.
                Then, the "grs_rs_emis" is changed.
                '''
                mask_grs_capmet = (df3_res_data_scen['Parameter'] == 'Landfill_CH4_capture')  # unit: %
                grs_capmet_df = df3_res_data_scen.loc[mask_grs_capmet]
                grs_capmet_proj = grs_capmet_df['Projection'].iloc[0]
                grs_capmet_by = grs_capmet_df[time_vector[0]].iloc[0]
                if grs_capmet_proj == 'flat':
                    grs_capmet_list = [grs_capmet_by] * len(time_vector)
                elif grs_capmet_proj == 'user_defined':
                    grs_capmet_list = []
                    for y in range(len(time_vector)):
                        grs_capmet_list.append(grs_capmet_df[time_vector[y]].iloc[0])
                else:
                    print('No projection type defined for: Landfill_CH4_capture')

                # Update the emissions here:
                grs_rs_emis = [a*(1-(b/100)) for a, b in zip(grs_rs_emis, grs_capmet_list)]
                grs_rs_emis_captured = [a*(b/100) for a, b in zip(grs_rs_emis, grs_capmet_list)]

                                                                 
                # Calculating Costs:
                list_res_rs_capex = []
                list_res_rs_olife = []
                list_res_rs_opex = []
                list_res_rs_vopex = []
                list_res_rs_fopex = []
                # Grabbing CAPEX:
                d5_res_rs_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Relleno sanitario') & \
                    (d5_res['Parameter'] == 'CAPEX')
                d5_res_rs_capex = d5_res.loc[d5_res_rs_capex_mask]
                d5_res_rs_capex_by = d5_res_rs_capex[time_vector[0]].iloc[0]
                d5_res_rs_capex_proj = d5_res_rs_capex['Projection'].iloc[0]
                if d5_res_rs_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rs_capex.append(d5_res_rs_capex_by)
                elif d5_res_rs_capex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_res_rs_capex.append(d5_res_rs_capex[time_vector[y]].iloc[0])   
                # Grabbing operational life:
                d5_res_rs_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Relleno sanitario') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_res_rs_ol = d5_res.loc[d5_res_rs_ol_mask]
                d5_res_rs_ol_by = d5_res_rs_ol[time_vector[0]].iloc[0]
                d5_res_rs_ol_proj = d5_res_rs_ol['Projection'].iloc[0]
                if d5_res_rs_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rs_olife.append(d5_res_rs_ol_by)
                # Grabbing VOPEX:
                d5_res_rs_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Relleno sanitario') & \
                    (d5_res['Parameter'] == 'Variable FOM')
                d5_res_rs_vopex = d5_res.loc[d5_res_rs_vopex_mask]
                d5_res_rs_vopex_by = d5_res_rs_vopex[time_vector[0]].iloc[0]
                d5_res_rs_vopex_proj = d5_res_rs_vopex['Projection'].iloc[0]
                if d5_res_rs_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rs_vopex.append(d5_res_rs_vopex_by)
                elif d5_res_rs_vopex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_res_rs_vopex.append(d5_res_rs_vopex[time_vector[y]].iloc[0])
                # Grabbing FOPEX:
                d5_res_rs_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Relleno sanitario') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_res_rs_fopex = d5_res.loc[d5_res_rs_fopex_mask]
                d5_res_rs_fopex_by = d5_res_rs_fopex[time_vector[0]].iloc[0]
                d5_res_rs_fopex_proj = d5_res_rs_fopex['Projection'].iloc[0]
                if d5_res_rs_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rs_fopex.append(d5_res_rs_fopex_by)
                elif d5_res_rs_fopex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_res_rs_fopex.append(d5_res_rs_fopex[time_vector[y]].iloc[0])
                        
                # Calculate investment requirements:
                grs_rs_list_delta_inv = [0] * len(time_vector)
                for y in range(len(time_vector)):
                    if grs_rs_list_delta[y] > 0:
                        grs_rs_list_delta_inv[y] = grs_rs_list_delta[y]
                    olife = int(list_res_rs_olife[y])
                    if y < olife:
                        pass
                    else:
                        # print('happens?')
                        if grs_rs_list_delta[y-olife] > 0:
                            grs_rs_list_delta_inv[y] += \
                                grs_rs_list_delta[y-olife]

                list_res_rs_opex = [varcost + fcost for varcost, fcost in zip(list_res_rs_vopex, list_res_rs_fopex)]
                grs_rs_opex = [(ucost * act)/1e9 for ucost, act in zip(list_res_rs_opex, grs_rs_list)]
                grs_rs_capex = [(ucost * act)/1e9 for ucost, act in zip(list_res_rs_capex, grs_rs_list_delta_inv)]

                '''
                This code increases the cost of the landfills because of methane capture
                First, we need to define a cost for the methane capture in sheet: 27_res_cost
                Second, we need add CAPEX and OPEX costs to the landfill costs (simple sum)
                '''
                # Calculating Costs:
                list_res_capmet_capex = []
                list_res_capmet_olife = []
                list_res_capmet_opex = []
                list_res_capmet_vopex = []
                list_res_capmet_fopex = []
                # Grabbing CAPEX:
                d5_res_capmet_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Relleno sanitario') & \
                    (d5_res['Parameter'] == 'CAPEX')
                d5_res_capmet_capex = d5_res.loc[d5_res_capmet_capex_mask]
                d5_res_capmet_capex_by = d5_res_capmet_capex[time_vector[0]].iloc[0]
                d5_res_capmet_capex_proj = d5_res_capmet_capex['Projection'].iloc[0]
                if d5_res_capmet_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_capmet_capex.append(d5_res_capmet_capex_by)
                elif d5_res_capmet_capex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_res_capmet_capex.append(d5_res_capmet_capex[time_vector[y]].iloc[0])
                # Grabbing operational life:
                d5_res_capmet_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Relleno sanitario') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_res_capmet_ol = d5_res.loc[d5_res_capmet_ol_mask]
                d5_res_capmet_ol_by = d5_res_capmet_ol[time_vector[0]].iloc[0]
                d5_res_capmet_ol_proj = d5_res_capmet_ol['Projection'].iloc[0]
                if d5_res_capmet_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_capmet_olife.append(d5_res_capmet_ol_by)
                # Grabbing VOPEX:
                d5_res_capmet_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Relleno sanitario') & \
                    (d5_res['Parameter'] == 'Variable FOM')
                d5_res_capmet_vopex = d5_res.loc[d5_res_capmet_vopex_mask]
                d5_res_capmet_vopex_by = d5_res_capmet_vopex[time_vector[0]].iloc[0]
                d5_res_capmet_vopex_proj = d5_res_capmet_vopex['Projection'].iloc[0]
                if d5_res_capmet_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_capmet_vopex.append(d5_res_capmet_vopex_by)
                elif d5_res_capmet_vopex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_res_capmet_vopex.append(d5_res_capmet_vopex[time_vector[y]].iloc[0])
                # Grabbing FOPEX:
                d5_res_capmet_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Relleno sanitario') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_res_capmet_fopex = d5_res.loc[d5_res_capmet_fopex_mask]
                d5_res_capmet_fopex_by = d5_res_capmet_fopex[time_vector[0]].iloc[0]
                d5_res_capmet_fopex_proj = d5_res_capmet_fopex['Projection'].iloc[0]
                if d5_res_capmet_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_capmet_fopex.append(d5_res_capmet_fopex_by)
                elif d5_res_capmet_fopex_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_res_capmet_fopex.append(d5_res_capmet_fopex[time_vector[y]].iloc[0])
                 
               
                # Calculate investment requirements:
                grs_capmet_list_delta = [0] + [grs_rs_emis_captured[n]-grs_rs_emis_captured[n-1] for n in range(len(time_vector))]
                grs_capmet_list_delta_inv = [0] * len(time_vector)
                for y in range(len(time_vector)):
                    if grs_capmet_list_delta[y] > 0:
                        grs_capmet_list_delta_inv[y] = grs_capmet_list_delta[y]
                    olife = int(list_res_capmet_olife[y])
                    if y < olife:
                        pass
                    else:
                        # print('happens?')
                        if grs_capmet_list_delta[y-olife] > 0:
                            grs_capmet_list_delta_inv[y] += \
                                grs_capmet_list_delta[y-olife]

                # [kton CH4]*[MUSD/MtonCH4] -> MUSD
                list_res_capmet_opex = [varcost + fcost for varcost, fcost in zip(list_res_capmet_vopex, list_res_capmet_fopex)]
                grs_capmet_opex = [(ucost * act)/1e3 for ucost, act in zip(list_res_capmet_opex, grs_rs_emis_captured)]
                grs_capmet_capex = [(ucost * act)/1e3 for ucost, act in zip(list_res_capmet_capex, grs_capmet_list_delta_inv)]
                
                # Now we must create a resulting summation:
                list_res_rs_opex = [a + b for a, b in zip(list_res_rs_opex, list_res_capmet_opex)]
                grs_rs_opex = [a + b for a, b in zip(grs_rs_opex, grs_capmet_opex)]
                grs_rs_capex = [a + b for a, b in zip(grs_rs_capex, grs_capmet_capex)]

                   
                # > Emissions and costs for cielo abierto:
                mask_grs_cielabi = (df3_res_data_scen['Type'] == 'Cielo abierto')
                grs_ca_df = df3_res_data_scen.loc[mask_grs_cielabi]
                grs_ca_proj = grs_ca_df['Projection'].iloc[0]
                grs_ca_by = grs_ca_df[time_vector[0]].iloc[0]
                grs_ca_list_delta = [0] * len(time_vector)
                if grs_ca_proj == 'user_defined':
                    grs_ca_list = [total_sw_list[0]*grs_ca_by/100]
                    for y in range(1, len(time_vector)):
                        grs_ca_list.append(total_sw_list[y] * \
                            grs_ca_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_ca_list_delta[y] = \
                                grs_ca_list[y] - grs_ca_list[y-1]
                elif grs_ca_proj == 'flat':
                    grs_ca_list = [total_sw_list[0]*grs_ca_by/100]
                    for y in range(1, len(time_vector)):
                        grs_ca_list.append(total_sw_list[y] * \
                            grs_ca_df[time_vector[0]].iloc[0]/100)
                        if y > 0:
                            grs_ca_list_delta[y] = \
                                grs_ca_list[y] - grs_ca_list[y-1]

                mask_grs_ca_ef = (df4_ef_agro_res_scen['Type'] == 'Cielo abierto')  # unit: kg Ch4/Gg waste
                grs_ca_ef_df = df4_ef_agro_res_scen.loc[mask_grs_ca_ef]
                grs_ca_ef_proj = grs_ca_ef_df['Projection'].iloc[0]
                grs_ca_ef_by = grs_ca_ef_df[time_vector[0]].iloc[0]
                if grs_ca_ef_proj == 'flat':
                    grs_ca_ef_list = [grs_ca_ef_by] * len(time_vector)

                # Calculating Emissions:  [kg Ch4/Gg waste]*[kg waste]
                grs_ca_emis = [(ef * act)/1e12 for ef, act in zip(grs_ca_ef_list, grs_ca_list)]
                
                # Calculating Costs:
                list_res_ca_capex = []
                list_res_ca_olife = []
                list_res_ca_opex = []
                list_res_ca_vopex = []
                list_res_ca_fopex = []
                # Grabbing CAPEX:
                d5_res_ca_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Cielo abierto') & \
                    (d5_res['Parameter'] == 'CAPEX')
                d5_res_ca_capex = d5_res.loc[d5_res_ca_capex_mask]
                d5_res_ca_capex_by = d5_res_ca_capex[time_vector[0]].iloc[0]
                d5_res_ca_capex_proj = d5_res_ca_capex['Projection'].iloc[0]
                if d5_res_ca_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ca_capex.append(d5_res_ca_capex_by)
                # Grabbing operational life:
                d5_res_ca_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Cielo abierto') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_res_ca_ol = d5_res.loc[d5_res_ca_ol_mask]
                d5_res_ca_ol_by = d5_res_ca_ol[time_vector[0]].iloc[0]
                d5_res_ca_ol_proj = d5_res_ca_ol['Projection'].iloc[0]
                if d5_res_ca_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ca_olife.append(d5_res_ca_ol_by)
                # Grabbing VOPEX:
                d5_res_ca_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Cielo abierto') & \
                    (d5_res['Parameter'] == 'Variable FOM')
                d5_res_ca_vopex = d5_res.loc[d5_res_ca_vopex_mask]
                d5_res_ca_vopex_by = d5_res_ca_vopex[time_vector[0]].iloc[0]
                d5_res_ca_vopex_proj = d5_res_ca_vopex['Projection'].iloc[0]
                if d5_res_ca_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ca_vopex.append(d5_res_ca_vopex_by)
                # Grabbing FOPEX:
                d5_res_ca_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Cielo abierto') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_res_ca_fopex = d5_res.loc[d5_res_ca_fopex_mask]
                d5_res_ca_fopex_by = d5_res_ca_fopex[time_vector[0]].iloc[0]
                d5_res_ca_fopex_proj = d5_res_ca_fopex['Projection'].iloc[0]
                if d5_res_ca_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ca_fopex.append(d5_res_ca_fopex_by)
                # Calculate investment requirements:
                grs_ca_list_delta_inv = [0] * len(time_vector)
                for y in range(len(time_vector)):
                    if grs_ca_list_delta[y] > 0:
                        grs_ca_list_delta_inv[y] = grs_ca_list_delta[y]
                    olife = int(list_res_ca_olife[y])
                    if y < olife:
                        pass
                    else:
                        # print('happens?')
                        if grs_ca_list_delta[y-olife] > 0:
                            grs_ca_list_delta_inv[y] += \
                                grs_ca_list_delta[y-olife]

                list_res_ca_opex = [varcost + fcost for varcost, fcost in zip(list_res_ca_vopex, list_res_ca_fopex)]
                grs_ca_opex = [(ucost * act)/1e9 for ucost, act in zip(list_res_ca_opex, grs_ca_list)]
                grs_ca_capex = [(ucost * act)/1e9 for ucost, act in zip(list_res_ca_capex, grs_ca_list_delta_inv)]

                # > Emissions and costs for reciclaje:
                mask_grs_recic = (df3_res_data_scen['Type'] == 'Reciclaje')
                grs_re_df = df3_res_data_scen.loc[mask_grs_recic]
                grs_re_proj = grs_re_df['Projection'].iloc[0]
                grs_re_by = grs_re_df[time_vector[0]].iloc[0]
                grs_re_list_delta = [0] * len(time_vector)
                if grs_re_proj == 'user_defined':
                    grs_re_list = [total_sw_list[0]*grs_re_by/100]
                    for y in range(1, len(time_vector)):
                        grs_re_list.append(total_sw_list[y] * \
                            grs_re_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_re_list_delta[y] = \
                                grs_re_list[y] - grs_re_list[y-1]
                elif grs_re_proj == 'flat':
                    grs_re_list = [total_sw_list[0]*grs_re_by/100]
                    for y in range(1, len(time_vector)):
                        grs_re_list.append(total_sw_list[y] * \
                            grs_re_df[time_vector[0]].iloc[0]/100)
                        if y > 0:
                            grs_re_list_delta[y] = \
                                grs_re_list[y] - grs_re_list[y-1]

                mask_grs_re_ef = (df4_ef_agro_res_scen['Type'] == 'Reciclaje')  # unit: %
                grs_re_ef_df = df4_ef_agro_res_scen.loc[mask_grs_re_ef]
                grs_re_ef_proj = grs_re_ef_df['Projection'].iloc[0]
                grs_re_ef_by = grs_re_ef_df[time_vector[0]].iloc[0]
                if grs_re_ef_proj == 'flat':
                    grs_re_ef_list = [grs_re_ef_by] * len(time_vector)

                # Calculating Emissions:
                grs_re_emis = [(ef * act)/1e9 for ef, act in zip(grs_re_ef_list, grs_re_list)]

                # Calculating Costs:
                list_res_re_capex = []
                list_res_re_olife = []
                list_res_re_opex = []
                list_res_re_vopex = []
                list_res_re_fopex = []
                list_res_re_sp = []
                # Grabbing CAPEX:
                d5_res_re_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Reciclaje') & \
                    (d5_res['Parameter'] == 'CAPEX')
                d5_res_re_capex = d5_res.loc[d5_res_re_capex_mask]
                d5_res_re_capex_by = d5_res_re_capex[time_vector[0]].iloc[0]
                d5_res_re_capex_proj = d5_res_re_capex['Projection'].iloc[0]
                if d5_res_re_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_re_capex.append(d5_res_re_capex_by)
                # Grabbing operational life:
                d5_res_re_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Reciclaje') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_res_re_ol = d5_res.loc[d5_res_re_ol_mask]
                d5_res_re_ol_by = d5_res_re_ol[time_vector[0]].iloc[0]
                d5_res_re_ol_proj = d5_res_re_ol['Projection'].iloc[0]
                if d5_res_re_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_re_olife.append(d5_res_re_ol_by)
                # Grabbing VOPEX:
                d5_res_re_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Reciclaje') & \
                    (d5_res['Parameter'] == 'Variable FOM')
                d5_res_re_vopex = d5_res.loc[d5_res_re_vopex_mask]
                d5_res_re_vopex_by = d5_res_re_vopex[time_vector[0]].iloc[0]
                d5_res_re_vopex_proj = d5_res_re_vopex['Projection'].iloc[0]
                if d5_res_re_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_re_vopex.append(d5_res_re_vopex_by)
                # Grabbing FOPEX:
                d5_res_re_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Reciclaje') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_res_re_fopex = d5_res.loc[d5_res_re_fopex_mask]
                d5_res_re_fopex_by = d5_res_re_fopex[time_vector[0]].iloc[0]
                d5_res_re_fopex_proj = d5_res_re_fopex['Projection'].iloc[0]
                if d5_res_re_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_re_fopex.append(d5_res_re_fopex_by)
                # Grabbing sales price:
                d5_res_re_sp_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Venta de material') & \
                    (d5_res['Parameter'] == 'Reciclaje')
                d5_res_re_sp = d5_res.loc[d5_res_re_sp_mask]
                d5_res_re_sp_by = d5_res_re_sp[time_vector[0]].iloc[0]
                d5_res_re_sp_proj = d5_res_re_sp['Projection'].iloc[0]
                if d5_res_re_sp_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_re_sp.append(d5_res_re_sp_by)
                # Calculate investment requirements:
                grs_re_list_delta_inv = [0] * len(time_vector)
                for y in range(len(time_vector)):
                    if grs_re_list_delta[y] > 0:
                        grs_re_list_delta_inv[y] = grs_re_list_delta[y]
                    olife = int(list_res_re_olife[y])
                    if y < olife:
                        pass
                    else:
                        # print('happens?')}
                        if grs_re_list_delta[y-olife] > 0:
                            grs_re_list_delta_inv[y] += \
                                grs_re_list_delta[y-olife]

                list_res_re_opex = [varcost + fcost for varcost, fcost in zip(list_res_re_vopex, list_res_re_fopex)]
                grs_re_opex = [(ucost * act)/1e9 for ucost, act in zip(list_res_re_opex, grs_re_list)]
                grs_re_capex = [(ucost * act)/1e9 for ucost, act in zip(list_res_re_capex, grs_re_list_delta_inv)]
                grs_re_sale = [(ucost * act)/1e9 for ucost, act in zip(list_res_re_sp, grs_re_list)]

                
                # > Emissions and costs for compostaje:
                mask_grs_comp = (df3_res_data_scen['Type'] == 'Compostaje')
                grs_comp_df = df3_res_data_scen.loc[mask_grs_comp]
                grs_comp_proj = grs_comp_df['Projection'].iloc[0]
                grs_comp_by = grs_comp_df[time_vector[0]].iloc[0]
                grs_comp_list_delta = [0] * len(time_vector)
                if grs_comp_proj == 'user_defined':
                    grs_comp_list = [total_sw_list[0]*grs_comp_by/100]
                    for y in range(1, len(time_vector)):
                        grs_comp_list.append(total_sw_list[y] * \
                            grs_comp_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_comp_list_delta[y] = \
                                grs_comp_list[y] - grs_comp_list[y-1]
                elif grs_comp_proj == 'flat':
                    grs_comp_list = [total_sw_list[0]*grs_comp_by/100]
                    for y in range(1, len(time_vector)):
                        grs_comp_list.append(total_sw_list[y] * \
                            grs_comp_df[time_vector[0]].iloc[0]/100)
                        if y > 0:
                            grs_comp_list_delta[y] = \
                                grs_comp_list[y] - grs_comp_list[y-1]

                mask_grs_comp_ef = (df4_ef_agro_res_scen['Type'] == 'Compostaje')  # unit: %
                grs_comp_ef_df = df4_ef_agro_res_scen.loc[mask_grs_comp_ef]
                grs_comp_ef_proj = grs_comp_ef_df['Projection'].iloc[0]
                grs_comp_ef_by = grs_comp_ef_df[time_vector[0]].iloc[0]
                if grs_comp_ef_proj == 'flat':
                    grs_comp_ef_list = [grs_comp_ef_by] * len(time_vector)

                # Calculating Emissions:[kg CH4/t waste]*[kg waste] -> [kton CH4]
                grs_comp_emis = [(ef * act)/1e9 for ef, act in zip(grs_comp_ef_list, grs_comp_list)]

                # Calculating Costs:
                list_res_comp_capex = []
                list_res_comp_olife = []
                list_res_comp_opex = []
                list_res_comp_vopex = []
                list_res_comp_fopex = []
                list_res_comp_sp = []
                # Grabbing CAPEX:
                d5_res_comp_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Compostaje') & \
                    (d5_res['Parameter'] == 'CAPEX')
                d5_res_comp_capex = d5_res.loc[d5_res_comp_capex_mask]
                d5_res_comp_capex_by = d5_res_comp_capex[time_vector[0]].iloc[0]
                d5_res_comp_capex_proj = d5_res_comp_capex['Projection'].iloc[0]
                if d5_res_comp_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_comp_capex.append(d5_res_comp_capex_by)
                # Grabbing operational life:
                d5_res_comp_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Compostaje') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_res_comp_ol = d5_res.loc[d5_res_comp_ol_mask]
                d5_res_comp_ol_by = d5_res_comp_ol[time_vector[0]].iloc[0]
                d5_res_comp_ol_proj = d5_res_comp_ol['Projection'].iloc[0]
                if d5_res_comp_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_comp_olife.append(d5_res_comp_ol_by)
                # Grabbing VOPEX:
                d5_res_comp_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Compostaje') & \
                    (d5_res['Parameter'] == 'Variable FOM')
                d5_res_comp_vopex = d5_res.loc[d5_res_comp_vopex_mask]
                d5_res_comp_vopex_by = d5_res_comp_vopex[time_vector[0]].iloc[0]
                d5_res_comp_vopex_proj = d5_res_comp_vopex['Projection'].iloc[0]
                if d5_res_comp_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_comp_vopex.append(d5_res_comp_vopex_by)
                # Grabbing FOPEX:
                d5_res_comp_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Compostaje') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_res_comp_fopex = d5_res.loc[d5_res_comp_fopex_mask]
                d5_res_comp_fopex_by = d5_res_comp_fopex[time_vector[0]].iloc[0]
                d5_res_comp_fopex_proj = d5_res_comp_fopex['Projection'].iloc[0]
                if d5_res_comp_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_comp_fopex.append(d5_res_comp_fopex_by)
                # Grabbing compost sale price:
                d5_res_comp_sp_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Venta de material') & \
                    (d5_res['Parameter'] == 'Compostaje' )
                d5_res_comp_sp = d5_res.loc[d5_res_comp_sp_mask]
                d5_res_comp_sp_by = d5_res_comp_sp[time_vector[0]].iloc[0]
                d5_res_comp_sp_proj = d5_res_comp_sp['Projection'].iloc[0]
                if d5_res_comp_sp_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_comp_sp.append(d5_res_comp_sp_by)
                # Calculate investment requirements:
                grs_comp_list_delta_inv = [0] * len(time_vector)
                for y in range(len(time_vector)):
                    if grs_comp_list_delta[y] > 0:
                        grs_comp_list_delta_inv[y] = grs_comp_list_delta[y]
                    olife = int(list_res_comp_olife[y])
                    if y < olife:
                        pass
                    else:
                        # print('happens?')
                        if grs_comp_list_delta[y-olife] > 0:
                            grs_comp_list_delta_inv[y] += \
                                grs_comp_list_delta[y-olife]

                list_res_comp_opex = [varcost + fcost for varcost, fcost in zip(list_res_comp_vopex, list_res_comp_fopex)]
                grs_comp_opex = [(ucost * act)/1e9 for ucost, act in zip(list_res_comp_opex, grs_comp_list)]
                grs_comp_capex = [(ucost * act)/1e9 for ucost, act in zip(list_res_comp_capex, grs_comp_list_delta_inv)]
                grs_comp_sale = [(ucost * act)/1e9 for ucost, act in zip(list_res_comp_sp, grs_comp_list)] # $/tonne*kg*1e-9 -> MUSD
                # > Emissions and costs for dig. anaeróbica:
                mask_grs_digana = (df3_res_data_scen['Type'] == 'Digestión anaeróbica para biogas')
                grs_da_df = df3_res_data_scen.loc[mask_grs_digana]
                grs_da_proj = grs_da_df['Projection'].iloc[0]
                grs_da_by = grs_da_df[time_vector[0]].iloc[0]
                grs_da_list_delta = [0] * len(time_vector)
                if grs_da_proj == 'user_defined':
                    grs_da_list = [total_sw_list[0]*grs_da_by/100]
                    for y in range(1, len(time_vector)):
                        grs_da_list.append(total_sw_list[y] * \
                            grs_da_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_da_list_delta[y] = \
                                grs_da_list[y] - grs_da_list[y-1]          
                elif grs_da_proj == 'flat':
                    grs_da_list = [total_sw_list[0]*grs_comp_by/100]
                    for y in range(1, len(time_vector)):
                        grs_da_list.append(total_sw_list[y] * \
                            grs_da_df[time_vector[0]].iloc[0]/100)
                        if y > 0:
                            grs_da_list_delta[y] = \
                                grs_da_list[y] - grs_da_list[y-1]

                mask_grs_da_ef = (df4_ef_agro_res_scen['Type'] == 'Digestión anaeróbica para biogas')  # unit: %
                grs_da_ef_df = df4_ef_agro_res_scen.loc[mask_grs_da_ef]
                grs_da_ef_proj = grs_da_ef_df['Projection'].iloc[0]
                grs_da_ef_by = grs_da_ef_df[time_vector[0]].iloc[0]
                if grs_da_ef_proj == 'flat':
                    grs_da_ef_list = [grs_da_ef_by] * len(time_vector)

                # Calculating Emissions:[kg CH4/t waste]*[kg waste] -> [kton CH4]
                grs_da_emis = [(ef * act)/1e9 for ef, act in zip(grs_da_ef_list, grs_da_list)]
                
                # Calculating Costs:
                list_res_da_capex = []
                list_res_da_olife = []
                list_res_da_opex = []
                list_res_da_vopex = []
                list_res_da_fopex = []
                # Grabbing CAPEX:
                d5_res_da_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Digestión anaeróbica para biogas') & \
                    (d5_res['Parameter'] == 'CAPEX')
                d5_res_da_capex = d5_res.loc[d5_res_da_capex_mask]
                d5_res_da_capex_by = d5_res_da_capex[time_vector[0]].iloc[0]
                d5_res_da_capex_proj = d5_res_da_capex['Projection'].iloc[0]
                if d5_res_da_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_da_capex.append(d5_res_da_capex_by)
                # Grabbing operational life:
                d5_res_da_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Digestión anaeróbica para biogas') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_res_da_ol = d5_res.loc[d5_res_da_ol_mask]
                d5_res_da_ol_by = d5_res_da_ol[time_vector[0]].iloc[0]
                d5_res_da_ol_proj = d5_res_da_ol['Projection'].iloc[0]
                if d5_res_da_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_da_olife.append(d5_res_da_ol_by)
                # Grabbing VOPEX:
                d5_res_da_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Digestión anaeróbica para biogas') & \
                    (d5_res['Parameter'] == 'Variable FOM')
                d5_res_da_vopex = d5_res.loc[d5_res_da_vopex_mask]
                d5_res_da_vopex_by = d5_res_da_vopex[time_vector[0]].iloc[0]
                d5_res_da_vopex_proj = d5_res_da_vopex['Projection'].iloc[0]
                if d5_res_da_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_da_vopex.append(d5_res_da_vopex_by)
                # Grabbing FOPEX:
                d5_res_da_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Digestión anaeróbica para biogas') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_res_da_fopex = d5_res.loc[d5_res_da_fopex_mask]
                d5_res_da_fopex_by = d5_res_da_fopex[time_vector[0]].iloc[0]
                d5_res_da_fopex_proj = d5_res_da_fopex['Projection'].iloc[0]
                if d5_res_da_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_da_fopex.append(d5_res_da_fopex_by)
                # Calculate investment requirements:
                grs_da_list_delta_inv = [0] * len(time_vector)
                for y in range(len(time_vector)):
                    if grs_da_list_delta[y] > 0:
                        grs_da_list_delta_inv[y] = grs_da_list_delta[y]
                    olife = int(list_res_da_olife[y])
                    if y < olife:
                        pass
                    else:
                        # print('happens?')
                        if grs_da_list_delta[y-olife] > 0:
                            grs_da_list_delta_inv[y] += \
                                grs_da_list_delta[y-olife]

                list_res_da_opex = [varcost + fcost for varcost, fcost in zip(list_res_da_vopex, list_res_da_fopex)]
                grs_da_opex = [(ucost * act)/1e9 for ucost, act in zip(list_res_da_opex, grs_da_list)]
                grs_da_capex = [(ucost * act)/1e9 for ucost, act in zip(list_res_da_capex, grs_da_list_delta_inv)]

                # > Emissions and costs for incineración de residuos:
                mask_grs_inci = (df3_res_data_scen['Type'] == 'Incineración de residuos')
                grs_ir_df = df3_res_data_scen.loc[mask_grs_inci]
                grs_ir_proj = grs_ir_df['Projection'].iloc[0]
                grs_ir_by = grs_ir_df[time_vector[0]].iloc[0]
                grs_ir_list_delta = [0] * len(time_vector)
                if grs_ir_proj == 'user_defined':
                    grs_ir_list = [total_sw_list[0]*grs_ir_by/100]
                    for y in range(1, len(time_vector)):
                        grs_ir_list.append(total_sw_list[y] * \
                            grs_ir_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_ir_list_delta[y] = \
                                grs_ir_list[y] - grs_ir_list[y-1]
                elif grs_ir_proj == 'flat':
                    grs_ir_list = [total_sw_list[0]*grs_ir_by/100]
                    for y in range(1, len(time_vector)):
                        grs_ir_list.append(total_sw_list[y] * \
                            grs_ir_df[time_vector[0]].iloc[0]/100)
                        if y > 0:
                            grs_ir_list_delta[y] = \
                                grs_ir_list[y] - grs_ir_list[y-1]
                                
                mask_grs_ir_ef = (df4_ef_agro_res_scen['Type'] == 'Incineración de residuos') &\
                (df4_ef_agro_res_scen['Apply'] == 'CH4')# unit: %
                grs_ir_ef_df = df4_ef_agro_res_scen.loc[mask_grs_ir_ef]
                grs_ir_ef_proj = grs_ir_ef_df['Projection'].iloc[0]
                grs_ir_ef_by = grs_ir_ef_df[time_vector[0]].iloc[0]
                if grs_ir_ef_proj == 'flat':
                    grs_ir_ef_list = [grs_ir_ef_by] * len(time_vector)
                      
                mask_grs_ir_ef_bc = (df4_ef_agro_res_scen['Type'] == 'Incineración de residuos') &\
                (df4_ef_agro_res_scen['Apply'] == 'BC')# unit: %
                grs_ir_ef_df_bc = df4_ef_agro_res_scen.loc[mask_grs_ir_ef_bc]
                grs_ir_ef_bc_proj = grs_ir_ef_df_bc['Projection'].iloc[0]
                grs_ir_ef_bc_by = grs_ir_ef_df_bc[time_vector[0]].iloc[0]
                if grs_ir_ef_bc_proj == 'flat':
                    grs_ir_ef_bc_list = [grs_ir_ef_bc_by] * len(time_vector)


                # Calculating Emissions:[kg CH4/t waste]*[kg waste] -> [kton CH4]
                grs_ir_emis = [(ef * act)/1e9 for ef, act in zip(grs_ir_ef_list, grs_ir_list)]
                # Calculating Emissions:[kg BC/t waste]*[kg waste] -> [ton BC]
                grs_ir_bc_emis = [(ef * act)/1e6 for ef, act in zip(grs_ir_ef_bc_list, grs_ir_list)]
                                                                                             
                # Calculating Costs:
                list_res_ir_capex = []
                list_res_ir_olife = []
                list_res_ir_opex = []
                list_res_ir_vopex = []
                list_res_ir_fopex = []
                # Grabbing CAPEX:
                d5_res_ir_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Incineración de residuos') & \
                    (d5_res['Parameter'] == 'CAPEX')
                d5_res_ir_capex = d5_res.loc[d5_res_ir_capex_mask]
                d5_res_ir_capex_by = d5_res_ir_capex[time_vector[0]].iloc[0]
                d5_res_ir_capex_proj = d5_res_ir_capex['Projection'].iloc[0]
                if d5_res_ir_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ir_capex.append(d5_res_ir_capex_by)
                # Grabbing operational life:
                d5_res_ir_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Incineración de residuos') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_res_ir_ol = d5_res.loc[d5_res_ir_ol_mask]
                d5_res_ir_ol_by = d5_res_ir_ol[time_vector[0]].iloc[0]
                d5_res_ir_ol_proj = d5_res_ir_ol['Projection'].iloc[0]
                if d5_res_ir_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ir_olife.append(d5_res_ir_ol_by)
                # Grabbing VOPEX:
                d5_res_ir_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Incineración de residuos') & \
                    (d5_res['Parameter'] == 'Variable FOM')
                d5_res_ir_vopex = d5_res.loc[d5_res_ir_vopex_mask]
                d5_res_ir_vopex_by = d5_res_ir_vopex[time_vector[0]].iloc[0]
                d5_res_ir_vopex_proj = d5_res_ir_vopex['Projection'].iloc[0]
                if d5_res_ir_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ir_vopex.append(d5_res_ir_vopex_by)
                # Grabbing FOPEX:
                d5_res_ir_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Incineración de residuos') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_res_ir_fopex = d5_res.loc[d5_res_ir_fopex_mask]
                d5_res_ir_fopex_by = d5_res_ir_fopex[time_vector[0]].iloc[0]
                d5_res_ir_fopex_proj = d5_res_ir_fopex['Projection'].iloc[0]
                if d5_res_ir_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_ir_fopex.append(d5_res_ir_fopex_by)
                # Calculate investment requirements:
                grs_ir_list_delta_inv = [0] * len(time_vector)
                for y in range(len(time_vector)):
                    if grs_ir_list_delta[y] > 0:
                        grs_ir_list_delta_inv[y] = grs_ir_list_delta[y]
                    olife = int(list_res_ir_olife[y])
                    if y < olife:
                        pass
                    else:
                        # print('happens?')
                        if grs_ir_list_delta[y-olife] > 0:
                            grs_ir_list_delta_inv[y] += \
                                grs_ir_list_delta[y-olife]

                list_res_ir_opex = [varcost + fcost for varcost, fcost in zip(list_res_ir_vopex, list_res_ir_fopex)]
                grs_ir_opex = [(ucost * act)/1e9 for ucost, act in zip(list_res_ir_opex, grs_ir_list)]
                grs_ir_capex = [(ucost * act)/1e9 for ucost, act in zip(list_res_ir_capex, grs_ir_list_delta_inv)]

                # > Emissions and costs for incinaración abierta de residuos:
                mask_grs_inciabi = (df3_res_data_scen['Type'] == 'Incineración abierta de residuos')
                grs_iar_df = df3_res_data_scen.loc[mask_grs_inciabi]
                grs_iar_proj = grs_iar_df['Projection'].iloc[0]
                grs_iar_by = grs_iar_df[time_vector[0]].iloc[0]
                grs_iar_list_delta = [0] * len(time_vector)
                if grs_iar_proj == 'user_defined':
                    grs_iar_list = [total_sw_list[0]*grs_iar_by/100]
                    for y in range(1, len(time_vector)):
                        grs_iar_list.append(total_sw_list[y] * \
                            grs_iar_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_iar_list_delta[y] = \
                                grs_iar_list[y] - grs_iar_list[y-1]
                elif grs_iar_proj == 'flat':
                    grs_iar_list = [total_sw_list[0]*grs_iar_by/100]
                    for y in range(1, len(time_vector)):
                        grs_iar_list.append(total_sw_list[y] * \
                            grs_iar_df[time_vector[0]].iloc[0]/100)
                        if y > 0:
                            grs_iar_list_delta[y] = \
                                grs_iar_list[y] - grs_iar_list[y-1]
                mask_grs_iar_ef = (df4_ef_agro_res_scen['Type'] == 'Incineración abierta de residuos')  # unit: %
                grs_iar_ef_df = df4_ef_agro_res_scen.loc[mask_grs_iar_ef]
                grs_iar_ef_proj = grs_iar_ef_df['Projection'].iloc[0]
                grs_iar_ef_by = grs_iar_ef_df[time_vector[0]].iloc[0]
                if grs_iar_ef_proj == 'flat':
                    grs_iar_ef_list = [grs_iar_ef_by] * len(time_vector)

                # Calculating Emissions:[kg CH4/t waste]*[kg waste] -> [kton CH4]
                grs_iar_emis = [(ef * act)/1e9 for ef, act in zip(grs_iar_ef_list, grs_iar_list)]
                
                # Calculating Costs:
                list_res_iar_capex = []
                list_res_iar_olife = []
                list_res_iar_opex = []
                list_res_iar_vopex = []
                list_res_iar_fopex = []
                # Grabbing CAPEX:
                d5_res_iar_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Incineración abierta de residuos') & \
                    (d5_res['Parameter'] == 'CAPEX')
                d5_res_iar_capex = d5_res.loc[d5_res_iar_capex_mask]
                d5_res_iar_capex_by = d5_res_iar_capex[time_vector[0]].iloc[0]
                d5_res_iar_capex_proj = d5_res_iar_capex['Projection'].iloc[0]
                if d5_res_iar_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_iar_capex.append(d5_res_iar_capex_by)
                # Grabbing operational life:
                d5_res_iar_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Incineración abierta de residuos') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_res_iar_ol = d5_res.loc[d5_res_iar_ol_mask]
                d5_res_iar_ol_by = d5_res_iar_ol[time_vector[0]].iloc[0]
                d5_res_iar_ol_proj = d5_res_iar_ol['Projection'].iloc[0]
                if d5_res_iar_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_iar_olife.append(d5_res_iar_ol_by)
                # Grabbing VOPEX:
                d5_res_iar_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Incineración abierta de residuos') & \
                    (d5_res['Parameter'] == 'Variable FOM')
                d5_res_iar_vopex = d5_res.loc[d5_res_iar_vopex_mask]
                d5_res_iar_vopex_by = d5_res_iar_vopex[time_vector[0]].iloc[0]
                d5_res_iar_vopex_proj = d5_res_iar_vopex['Projection'].iloc[0]
                if d5_res_iar_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_iar_vopex.append(d5_res_iar_vopex_by)
                # Grabbing FOPEX:
                d5_res_iar_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Incineración abierta de residuos') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_res_iar_fopex = d5_res.loc[d5_res_iar_fopex_mask]
                d5_res_iar_fopex_by = d5_res_iar_fopex[time_vector[0]].iloc[0]
                d5_res_iar_fopex_proj = d5_res_iar_fopex['Projection'].iloc[0]
                if d5_res_iar_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_iar_fopex.append(d5_res_iar_fopex_by)
                # Calculate investment requirements:
                grs_iar_list_delta_inv = [0] * len(time_vector)
                for y in range(len(time_vector)):
                    if grs_iar_list_delta[y] > 0:
                        grs_iar_list_delta_inv[y] = grs_iar_list_delta[y]
                    olife = int(list_res_iar_olife[y])
                    if y < olife:
                        pass
                    else:
                        # print('happens?')
                        if grs_iar_list_delta[y-olife] > 0:
                            grs_iar_list_delta_inv[y] += \
                                grs_iar_list_delta[y-olife]

                list_res_iar_opex = [varcost + fcost for varcost, fcost in zip(list_res_iar_vopex, list_res_iar_fopex)]
                grs_iar_opex = [(ucost * act)/1e9 for ucost, act in zip(list_res_iar_opex, grs_iar_list)]
                grs_iar_capex = [(ucost * act)/1e9 for ucost, act in zip(list_res_iar_capex, grs_iar_list_delta_inv)]

                # > Emissions and costs for residuos sólidos no gestionados:
                mask_grs_nogest = (df3_res_data_scen['Type'] == 'Residuos sólidos no gestionados')
                grs_rsng_df = df3_res_data_scen.loc[mask_grs_nogest]
                grs_rsng_proj = grs_rsng_df['Projection'].iloc[0]
                grs_rsng_by = grs_rsng_df[time_vector[0]].iloc[0]
                grs_rsng_list_delta = [0] * len(time_vector)
                if grs_rsng_proj == 'user_defined':
                    grs_rsng_list = [total_sw_list[0]*grs_rsng_by/100] #ton
                    for y in range(1, len(time_vector)):
                        grs_rsng_list.append(total_sw_list[y] * \
                            grs_rsng_df[time_vector[y]].iloc[0]/100)
                        if y > 0:
                            grs_rsng_list_delta[y] = \
                                grs_rsng_list[y] - grs_rsng_list[y-1]
                elif grs_rsng_proj == 'flat':
                    grs_rsng_list = [total_sw_list[0]*grs_rsng_by/100]
                    for y in range(1, len(time_vector)):
                        grs_rsng_list.append(total_sw_list[y] * \
                            grs_rsng_df[time_vector[0]].iloc[0]/100)
                        if y > 0:
                            grs_rsng_list_delta[y] = \
                                grs_rsng_list[y] - grs_rsng_list[y-1]
                                
                mask_grs_rsng_ef = (df4_ef_agro_res_scen['Type'] == 'Residuos sólidos no gestionados')  # unit: %
                grs_rsng_ef_df = df4_ef_agro_res_scen.loc[mask_grs_rsng_ef]
                grs_rsng_ef_proj = grs_rsng_ef_df['Projection'].iloc[0] 
                grs_rsng_ef_by = grs_rsng_ef_df[time_vector[0]].iloc[0] # kg CH4/Gg waste-año
                if grs_rsng_ef_proj == 'flat':
                    grs_rsng_ef_list = [grs_rsng_ef_by] * len(time_vector)

                # Calculating Emissions:
                grs_rsng_emis = [(ef * act)/1e9 for ef, act in zip(grs_rsng_ef_list, grs_rsng_list)]

                # Calculating Costs:
                list_res_rsng_capex = []
                list_res_rsng_olife = []
                list_res_rsng_opex = []
                list_res_rsng_vopex = []
                list_res_rsng_fopex = []
                # Grabbing CAPEX:
                d5_res_rsng_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Residuos sólidos no gestionados') & \
                    (d5_res['Parameter'] == 'CAPEX')
                d5_res_rsng_capex = d5_res.loc[d5_res_rsng_capex_mask]
                d5_res_rsng_capex_by = d5_res_rsng_capex[time_vector[0]].iloc[0]
                d5_res_rsng_capex_proj = d5_res_rsng_capex['Projection'].iloc[0]
                if d5_res_rsng_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rsng_capex.append(d5_res_rsng_capex_by)
                # Grabbing operational life:
                d5_res_rsng_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Residuos sólidos no gestionados') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_res_rsng_ol = d5_res.loc[d5_res_rsng_ol_mask]
                d5_res_rsng_ol_by = d5_res_rsng_ol[time_vector[0]].iloc[0]
                d5_res_rsng_ol_proj = d5_res_rsng_ol['Projection'].iloc[0]
                if d5_res_rsng_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rsng_olife.append(d5_res_rsng_ol_by)
                # Grabbing VOPEX:
                d5_res_rsng_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Residuos sólidos no gestionados') & \
                    (d5_res['Parameter'] == 'Variable FOM')
                d5_res_rsng_vopex = d5_res.loc[d5_res_rsng_vopex_mask]
                d5_res_rsng_vopex_by = d5_res_rsng_vopex[time_vector[0]].iloc[0]
                d5_res_rsng_vopex_proj = d5_res_rsng_vopex['Projection'].iloc[0]
                if d5_res_rsng_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rsng_vopex.append(d5_res_rsng_vopex_by)
                # Grabbing FOPEX:
                d5_res_rsng_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Residuos sólidos no gestionados') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_res_rsng_fopex = d5_res.loc[d5_res_rsng_fopex_mask]
                d5_res_rsng_fopex_by = d5_res_rsng_fopex[time_vector[0]].iloc[0]
                d5_res_rsng_fopex_proj = d5_res_rsng_fopex['Projection'].iloc[0]
                if d5_res_rsng_fopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_res_rsng_fopex.append(d5_res_rsng_fopex_by)
                # Calculate investment requirements:
                grs_rsng_list_delta_inv = [0] * len(time_vector)
                for y in range(len(time_vector)):
                    if grs_rsng_list_delta[y] > 0:
                        grs_rsng_list_delta_inv[y] = grs_rsng_list_delta[y]
                    olife = int(list_res_rsng_olife[y])
                    if y < olife:
                        pass
                    else:
                        if grs_rsng_list_delta[y-olife] > 0:
                            grs_rsng_list_delta_inv[y] += \
                                grs_rsng_list_delta[y-olife]

                list_res_rsng_opex = [varcost + fcost for varcost, fcost in zip(list_res_rsng_vopex, list_res_rsng_fopex)]
                grs_rsng_opex = [(ucost * act)/1e9 for ucost, act in zip(list_res_rsng_opex, grs_rsng_list)]
                grs_rsng_capex = [(ucost * act)/1e9 for ucost, act in zip(list_res_rsng_capex, grs_rsng_list_delta_inv)]

                # > Emissions and costs for recuperación del metano del relleno
                mask_ch4_landfill_rec = (df3_res_data_scen['Type'] == 'Metano del relleno sanitario extraído')

                #-
                # Wastewater treatment:
                mask_gen_dbo_pc = (df3_res_data_scen['Type'] == 'DBO per cápita')  # unit: g/persona/año
                gen_dbo_pc_df = df3_res_data_scen.loc[mask_gen_dbo_pc]
                gen_dbo_pc_proj = gen_dbo_pc_df['Projection'].iloc[0]
                gen_dbo_pc_by = gen_dbo_pc_df[time_vector[0]].iloc[0]
                if gen_dbo_pc_proj == 'flat':
                    gen_dbo_pc_list = [gen_dbo_pc_by] * len(time_vector)
                elif gen_dbo_pc_proj == 'user_defined':
                    gen_dbo_pc_list = gen_dbo_pc_df[time_vector].iloc[0].tolist()

                # Population:
                '''
                Just call:
                this_pop_vector
                '''

                # Get treated waters
                mask_tre_wat = (df3_res_data_scen['Type'] == 'Tratamiento de aguas residuales domésticas')  # unit: %
                tre_wat_df = df3_res_data_scen.loc[mask_tre_wat]
                tre_wat_proj = tre_wat_df['Projection'].iloc[0]
                tre_wat_by = tre_wat_df[time_vector[0]].iloc[0]
                if tre_wat_proj == 'flat':
                    tre_wat_list = [tre_wat_by] * len(time_vector)
                elif tre_wat_proj == 'user_defined':
                    tre_wat_list = tre_wat_df[time_vector].iloc[0].tolist()

                # Multiply Pop, DBO, and share of treated water [kg DBO/año]
                tre_wat_kg = [(a * 365 * b * 1e6 * c)/(1000 * 100) for a, b, c in zip(gen_dbo_pc_list, this_pop_vector, tre_wat_list)]

                unit_capex_tre_wat = unpack_values_df_2(
                    d5_res, "Tech", "Parameter",
                    "Tratamiento de aguas residuales domésticas",
                    "CAPEX", time_vector, this_scen)  # $/persona
                unit_fopex_tre_wat = unpack_values_df_2(
                    d5_res, "Tech", "Parameter",
                    "Tratamiento de aguas residuales domésticas",
                    "Fixed FOM", time_vector, this_scen)  # % del CAPEX
                unit_vopex_tre_wat = unpack_values_df_2(
                    d5_res, "Tech", "Parameter",
                    "Tratamiento de aguas residuales domésticas",
                    "Variable FOM", time_vector, this_scen)  # $/persona
                ol_tre_wat = unpack_values_df_2(
                    d5_res, "Tech", "Parameter",
                    "Tratamiento de aguas residuales domésticas",
                    "Operational life", time_vector, this_scen)  # years

                this_pop_vector_delta = [0]
                for y in range(1, len(time_vector)):
                    this_pop_vector_delta.append(this_pop_vector[y] - this_pop_vector[y-1])
                total_capex_tre_wat = [unit_capex * pop * share/100 for unit_capex, pop, share in zip(unit_capex_tre_wat, this_pop_vector_delta, tre_wat_list)]
                for y in range(int(ol_tre_wat[0]), len(time_vector)):
                    total_capex_tre_wat[y] += total_capex_tre_wat[y - int(ol_tre_wat[y])]

                total_fopex_tre_wat = [a*b/100 for a, b in zip(unit_fopex_tre_wat, total_capex_tre_wat)]
                total_vopex_tre_wat = [unit_vopex * pop * share/100 for unit_vopex, pop, share in zip(unit_vopex_tre_wat, this_pop_vector, tre_wat_list)]
                
                total_capex_tre_wat_disc, total_fopex_tre_wat_disc, total_vopex_tre_wat_disc = \
                    deepcopy(total_capex_tre_wat), deepcopy(total_fopex_tre_wat), deepcopy(total_vopex_tre_wat)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    total_capex_tre_wat_disc[y] *= disc_constant
                    total_fopex_tre_wat_disc[y] *= disc_constant
                    total_vopex_tre_wat_disc[y] *= disc_constant
            

                # Get untreated waters
                mask_unt_wat = (df3_res_data_scen['Type'] == 'Aguas no tratadas')  # unit: %
                unt_wat_df = df3_res_data_scen.loc[mask_unt_wat]
                unt_wat_proj = unt_wat_df['Projection'].iloc[0]
                unt_wat_by = unt_wat_df[time_vector[0]].iloc[0]
                if unt_wat_proj == 'flat':
                    unt_wat_list = [unt_wat_by] * len(time_vector)
                elif unt_wat_proj == 'user_defined':
                    unt_wat_list = unt_wat_df[time_vector].iloc[0].tolist()

                # Multiply Pop, DBO, and share of untreated water [kg DBO/ año]
                unt_wat_kg = [(a * 365 * b*1e6 * c)/(1000 * 100) for a, b, c in zip(gen_dbo_pc_list, this_pop_vector, unt_wat_list)]

                # Read urbanization, emission factors, and tech utilization
                mask_ar_urb = (df4_ar_emi_scen['Parameter'] == 'Grado de urbanización')
                ar_urb_df = df4_ar_emi_scen.loc[mask_ar_urb]
                ar_urb_tech = ar_urb_df['Urbanization'].tolist()
                dict_ar_urb_tech = {}
                for aut in ar_urb_tech:
                    ar_urb_df_t = ar_urb_df.loc[(ar_urb_df['Urbanization'] == aut)]
                    ar_urb_proj = ar_urb_df_t['Projection'].iloc[0]
                    ar_urb_by = ar_urb_df_t[time_vector[0]].iloc[0]
                    if ar_urb_proj == 'flat':
                        ar_urb_list = [ar_urb_by] * len(time_vector)
                    elif ar_urb_proj == 'user_defined':
                        ar_urb_list = ar_urb_df_t[time_vector].iloc[0].tolist()
                    dict_ar_urb_tech.update({aut: ar_urb_list})
                mask_ar_ef = (df4_ar_emi_scen['Parameter'] == 'Factor de emisión')
                ar_ef_df = df4_ar_emi_scen.loc[mask_ar_ef]
                ar_ef_tech = ar_ef_df['Technology'].tolist()
                dict_ar_ef_tech = {}
                for aut in ar_ef_tech:
                    ar_ef_df_t = ar_ef_df.loc[(ar_ef_df['Technology'] == aut)]
                    ar_ef_proj = ar_ef_df_t['Projection'].iloc[0]
                    ar_ef_by = ar_ef_df_t[time_vector[0]].iloc[0]
                    if ar_ef_proj == 'flat':
                        ar_ef_list = [ar_ef_by] * len(time_vector)
                    elif ar_ef_proj == 'user_defined':
                        ar_ef_list = ar_ef_df_t[time_vector].iloc[0].tolist()
                    dict_ar_ef_tech.update({aut: ar_ef_list})

                # Calcualte the wieghted emissionf actor
                ef_weighted_list = [0] * len(time_vector)

                mask_artechuti = (df4_ar_emi_scen['Parameter'] == 'Grado de utilización de tecnología')
                artechuti_df = df4_ar_emi_scen.loc[mask_artechuti]
                artechuti_df.reset_index(drop=True, inplace=True)
                for auti in range(len(list(artechuti_df.index))):
                    artechuti_df_tech = artechuti_df['Technology'].iloc[auti]
                    artechuti_df_urb = artechuti_df['Urbanization'].iloc[auti]
                    artechuti_df_proj = artechuti_df['Projection'].iloc[auti]
                    artechuti_df_by = artechuti_df[time_vector[0]].iloc[auti]
                    if artechuti_df_proj == 'flat':
                        artechuti_list = [artechuti_df_by] * len(time_vector)
                    elif artechuti_df_proj == 'user_defined':
                        artechuti_list = artechuti_df[time_vector].iloc[auti].tolist()
                        
                    
                    ar_urb_list_call = dict_ar_urb_tech[artechuti_df_urb]
                    ar_ef_list_call = dict_ar_ef_tech[artechuti_df_tech]
                    
                    ar_factor_ef = [a*b*c for a, b, c in zip(ar_urb_list_call, artechuti_list, ar_ef_list_call)]
                    ef_weighted_list = [a + b for a, b in zip(ef_weighted_list, ar_factor_ef)]
                
                # Calculate emissions:
                all_wat_kg = [a + b for a, b in zip(tre_wat_kg, unt_wat_kg)]
                waste_water_emissions = [(a * b)/1e6 for a, b in zip(ef_weighted_list, all_wat_kg)]  # in kt CH4
                
                #Industrial waterwaste (cabezas o toneladas)
                mask_ind_ww = df3_res_data_scen['Parameter'] == 'Wastewater_Industrial_demand'
                df3_res_data_scen_dem = df3_res_data_scen.loc[mask_ind_ww]
                types_industry = df3_res_data_scen_dem['Type'].tolist()
                types_projection = df3_res_data_scen_dem['Projection'].tolist()
                types_by_vals_dem = df3_res_data_scen_dem[time_vector[0]].tolist()
                

                dem_ind_proj_dict = {}
                all_vals_dict  = {}
                for y in range(len(time_vector)):
                    all_vals_dict.update({time_vector[y]:df3_res_data_scen_dem[time_vector[y]].tolist()})
                for l in range(len(types_industry)):
                    this_ind = types_industry[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_dem[l]
                    this_val_list = []

                    if this_proj == 'grow_gdp_pc':
                        for y in range(len(time_vector)):
                            gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                            if y == 0:
                                this_val_list.append(this_by_val)
                            else:
                                next_val = this_val_list[-1] * (1 + gdp_pc_gr)
                                this_val_list.append(next_val)
                        dem_ind_proj_dict.update({this_ind: this_val_list})
                    elif this_proj == 'user_defined':
                        for y in range(len(time_vector)):
                            for y in range(len(time_vector)):
                                this_val_list.append(all_vals_dict[time_vector[y]][l])
                        dem_ind_proj_dict.update({this_ind: this_val_list})   
                
                #Industrial DQO charge (kg DQO)
                mask_ind_share = df3_res_data_scen['Parameter'] == 'Industial_wastewater_disposal_share'
                df3_res_data_scen_share = df3_res_data_scen.loc[mask_ind_share]
                share_proj = df3_res_data_scen_share['Projection'].iloc[0]
                types_by_vals_share = df3_res_data_scen_share[time_vector[0]].iloc[0]
                
                
                if share_proj == 'flat':
                    ind_treated = [types_by_vals_share] * len(time_vector)
                elif share_proj == 'user_defined':
                    ind_treated = []
                    for y in range(len(time_vector)):
                        ind_treated.append(df3_res_data_scen_share[time_vector[y]].iloc[0])
                    
                
                #Industrial DQO charge (kg DQO)
                mask_ind_dqo = df3_res_data_scen['Parameter'] == 'Industrial_DQO'
                df3_res_data_scen_dqo = df3_res_data_scen.loc[mask_ind_dqo]
                types_industry_dqo = df3_res_data_scen_dqo['Type'].tolist()
                types_projection = df3_res_data_scen_dqo['Projection'].tolist()
                types_by_vals_dem = df3_res_data_scen_dqo[time_vector[0]].tolist()
                
                all_vals_dqo_dict = {}
                for y in range(len(time_vector)):
                    all_vals_dqo_dict.update({time_vector[y]:df3_res_data_scen_dqo[time_vector[y]].tolist()})
                dqo_dict_ind_ww = {}
                for l in range(len(types_industry_dqo)):
                    this_ind = types_industry_dqo[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_dem[l]
                    this_val_list = []

                    if this_proj == 'flat':
                        this_val_list = [this_by_val] * len(time_vector)
                        dqo_dict_ind_ww.update({this_ind: this_val_list})
                    elif this_proj == 'user_defined':
                        this_val_list = []
                        for y in range(len(time_vector)):
                            this_val_list.append(all_vals_dqo_dict[time_vector[y]][l])
                        dqo_dict_ind_ww.update({this_ind: this_val_list})
                        
                # Emissions factor of industrial water waste
                mask_fe_ind_ww = (df4_ef_agro_res_scen['Group'] == 'Aguas residuales industriales')
                df4_ef_ww_ind = df4_ef_agro_res_scen.loc[mask_fe_ind_ww]

                types_ind = df4_ef_ww_ind['Type'].tolist()
                types_projection = df4_ef_ww_ind['Projection'].tolist()
                types_by_vals_fe = df4_ef_ww_ind[time_vector[0]].tolist()

                all_vals_fe_dict = {}
                for y in range(len(time_vector)):
                    all_vals_fe_dict.update({time_vector[y]:df4_ef_ww_ind[time_vector[y]].tolist()})

                emisfac_fe_proj_dict_ind_ww = {}
                for l in range(len(types_ind)):
                    this_ind = types_ind[l]
                    this_proj = types_projection[l]
                    this_by_val = types_by_vals_fe[l]
                    this_val_list = []

                    if this_proj == 'flat':
                        this_val_list = [this_by_val] * len(time_vector)
                        emisfac_fe_proj_dict_ind_ww.update({this_ind: this_val_list})
                    elif this_proj == 'user_defined':
                        this_val_list = []
                        for y in range(len(time_vector)):
                            this_val_list.append(all_vals_fe_dict[time_vector[y]][l])
                        emisfac_fe_proj_dict_ind_ww.update({this_ind: this_val_list})
                    
                # Estimate the industrial water waste emissions:
                out_emis_ind_ww_proj_dict = {}
                for l in range(len(types_ind)):
                    this_ind = types_ind[l]
                    list_out_emis_ind_ww_proj = []
                    for y in range(len(time_vector)):
                        local_out_emis_ind_ww_proj = \
                            emisfac_fe_proj_dict_ind_ww[this_ind][y] * \
                            dem_ind_proj_dict[this_ind][y] / 1e6
                        list_out_emis_ind_ww_proj.append(local_out_emis_ind_ww_proj)
                    out_emis_ind_ww_proj_dict.update({this_ind:
                        list_out_emis_ind_ww_proj})
                        
                # Calculating Costs:
                list_ind_ww_capex = []
                list_ind_ww_olife = []
                list_ind_ww_fopex = []
                list_ind_ww_vopex = []

                # Grabbing CAPEX:(MUSD/kg DQO)
                d5_res_ind_ww_capex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Tratamiento de aguas industriales') &\
                    (d5_res['Parameter'] == 'CAPEX')
                d5_ind_ww_capex = d5_res.loc[d5_res_ind_ww_capex_mask]
                d5_ind_ww_capex_by = d5_ind_ww_capex[time_vector[0]].iloc[0]
                d5_ind_ww_capex_proj = d5_ind_ww_capex['Projection'].iloc[0]
                if d5_ind_ww_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_ind_ww_capex.append(d5_ind_ww_capex_by)
                # Grabbing operational life:
                d5_ind_ww_ol_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Tratamiento de aguas industriales') & \
                    (d5_res['Parameter'] == 'Operational life')
                d5_ind_ww_ol = d5_res.loc[d5_ind_ww_ol_mask]
                d5_ind_ww_ol_by = d5_ind_ww_ol[time_vector[0]].iloc[0]
                d5_ind_ww_ol_proj = d5_ind_ww_ol['Projection'].iloc[0]
                if d5_ind_ww_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_ind_ww_olife.append(d5_ind_ww_ol_by)
                # Grabbing FOPEX:(MUSD/kg DQO)
                d5_ind_ww_fopex_mask = (d5_res['Scenario'] == this_scen) &\
                    (d5_res['Tech'] == 'Tratamiento de aguas industriales') & \
                    (d5_res['Parameter'] == 'Fixed FOM')
                d5_ind_ww_fopex = d5_res.loc[d5_ind_ww_fopex_mask]
                d5_ind_ww_opex_by = d5_ind_ww_fopex[time_vector[0]].iloc[0]
                d5_ind_ww_opex_proj = d5_ind_ww_fopex['Projection'].iloc[0]
                if d5_ind_ww_opex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_ind_ww_fopex.append(d5_ind_ww_opex_by)
                # Grabbing VOPEX: (MUSD/kg DQO)
                d5_ind_ww_vopex_mask = (d5_res['Scenario'] == this_scen) &\
                   (d5_res['Tech'] == 'Tratamiento de aguas industriales') & \
                   (d5_res['Parameter'] == 'Variable FOM')
                d5_ind_ww_vopex = d5_res.loc[d5_ind_ww_vopex_mask]
                d5_ind_ww_vopex_by = d5_ind_ww_vopex[time_vector[0]].iloc[0]
                d5_ind_ww_vopex_proj = d5_ind_ww_vopex['Projection'].iloc[0]
                if d5_ind_ww_vopex_proj == 'flat':
                    for y in range(len(time_vector)):
                        list_ind_ww_vopex.append(d5_ind_ww_vopex_by)
                
                
                #Calculate total dqo 
                dqo_proj_dict = {}
                for l in range(len(types_industry)):
                    this_ind = types_industry[l]
                    this_val_list = []
                    for y in range(len(time_vector)):
                        local_dqo = dqo_dict_ind_ww[this_ind][y]*dem_ind_proj_dict[this_ind][y]
                        this_val_list.append(local_dqo)
                    dqo_proj_dict.update({this_ind: this_val_list}) #kq DQO
                
                # Calculate investment requirements (kg DQO)
            
                ind_ww_total = [sum(values) for values in \
                                         zip(*dqo_proj_dict.values())]
            
                total_ind_delta = [0]
                for y in range(1, len(time_vector)):
                    total_ind_delta.append(ind_ww_total[y] - ind_ww_total[y-1])
                for y in range(int(list_ind_ww_olife[0]), len(time_vector)):
                    total_ind_delta[y] += total_ind_delta[y - int(list_ind_ww_olife[y])]
                
                #Calculate Investment (MUSD)
                ind_ww_capex = [(ucost * act * share) for ucost, act, share in \
                              zip(list_ind_ww_capex, ind_ww_total, ind_treated)]
                
                ind_ww_capex_disc = deepcopy(ind_ww_capex)
                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    ind_ww_capex_disc[y] *= disc_constant
                    

                # Estimate the O&M cost (MUSD)
                out_opex_ind_ww_proj_dict = {}
                out_opex_ind_ww_proj_dict_disc = {}
                for l in range(len(types_industry)):
                    this_ind = types_industry[l]
                    list_opex_ind_proj = []
                    list_opex_ind_proj_disc = []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        local_out_opex_ind_proj = \
                            (list_ind_ww_fopex[y]+list_ind_ww_vopex[y]) * \
                                dqo_proj_dict[this_ind][y]* ind_treated[y]
                        local_out_opex_ind_proj_disc = \
                            (list_ind_ww_fopex[y]+list_ind_ww_vopex[y]) * \
                            dqo_proj_dict[this_ind][y] * \
                              disc_constant * ind_treated[y]
                        list_opex_ind_proj.append(local_out_opex_ind_proj)
                        list_opex_ind_proj_disc.append(local_out_opex_ind_proj_disc)
                    out_opex_ind_ww_proj_dict.update({this_ind:
                        list_opex_ind_proj})
                    out_opex_ind_ww_proj_dict_disc.update({this_ind:
                        list_opex_ind_proj_disc})
                        
                
                
                #List for externalities 
                list_ext_salud = []
                list_water_cont = []
                list_ext_tur = []
                #morbidity externalities for solid waste and water waste
                d5_salud_ext_mask = (d5_res['Scenario'] == this_scen) &\
                   (d5_res['Tech'] == 'Externalities') & \
                   (d5_res['Parameter'] == 'Salud(Morbilidad)')
                d5_ext_salud = d5_res.loc[d5_salud_ext_mask]
                d5_ext_salud_by = d5_ext_salud[time_vector[0]].iloc[0]
                d5_ext_salud_proj = d5_ext_salud['Projection'].iloc[0]
                if d5_ext_salud_proj == 'flat':
                    list_ext_salud = [d5_ext_salud_by]* len(time_vector)
                elif  d5_ext_salud_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_ext_salud.append(d5_ext_salud[time_vector[y]].iloc[0])
               # Water contamination for waster 
                d5_water_cont_mask = (d5_res['Scenario'] == this_scen) &\
                   (d5_res['Tech'] == 'Externalities') & \
                   (d5_res['Parameter'] == 'Contaminación de aguas')
                d5_water_cont = d5_res.loc[d5_water_cont_mask]
                d5_water_cont_by = d5_water_cont[time_vector[0]].iloc[0]
                d5_water_cont_proj = d5_water_cont['Projection'].iloc[0]
                if d5_water_cont_proj == 'flat':
                    list_water_cont = [d5_water_cont_by]* len(time_vector)
                elif  d5_water_cont_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_water_cont.append(d5_water_cont[time_vector[y]].iloc[0])
                # Externalities for turisim loss 
                d5_ext_tur_mask = (d5_res['Scenario'] == this_scen) &\
                   (d5_res['Tech'] == 'Externalities') & \
                   (d5_res['Parameter'] == 'Pérdidas por actividades turísticas')
                d5_ext_tur = d5_res.loc[d5_ext_tur_mask]
                d5_ext_tur_by = d5_ext_tur[time_vector[0]].iloc[0]
                d5_ext_tur_proj = d5_ext_tur['Projection'].iloc[0]
                if d5_ext_tur_proj == 'flat':
                    list_ext_tur = [d5_ext_tur_by]* len(time_vector)
                elif  d5_ext_tur_proj == 'user_defined':
                    for y in range(len(time_vector)):
                        list_ext_tur.append(d5_ext_tur[time_vector[y]].iloc[0])     
                        
                        
                #Adding the mass of waste for the techs that generate costs due to externalities [kg]
                waste_ext_tonne = [(ir + iar + rsng + ca) for ir, iar, rsng, ca in zip(grs_ir_list, grs_iar_list, grs_rsng_list, grs_ca_list)]
                # Calculating externalities: [$/kg]*[kg]/1e6 = [MUSD]
                ext_salud = [(ef * act)/1e6 for ef, act in zip(list_ext_salud, waste_ext_tonne)]
                ext_cont = [(ef * act)/1e6 for ef, act in zip(list_water_cont, waste_ext_tonne)]
                ext_tur = [(ef * act)/1e6 for ef, act in zip(list_ext_tur, waste_ext_tonne)]
                                                                                                                                           
                #Storing industrial waterwaste costs (MUSD) and emissiones (kton CH4) 
                dict_local_country[this_country].update({'CAPEX de aguas residuales industriales [MUSD]': deepcopy(ind_ww_capex)})
                dict_local_country[this_country].update({'CAPEX de aguas residuales industriales [MUSD] (disc)': deepcopy(ind_ww_capex_disc)})
                dict_local_country[this_country].update({'OPEX de aguas residuales industriales [MUSD]': deepcopy(out_opex_ind_ww_proj_dict)})
                dict_local_country[this_country].update({'OPEX de aguas residuales industriales [MUSD] (disc)': deepcopy(out_opex_ind_ww_proj_dict_disc)})
                dict_local_country[this_country].update({'Emisiones de aguas residuales industriales [kton CH4]': deepcopy(out_emis_ind_ww_proj_dict)})
                print('Wastewater emissions have been computed!')
                print('Wastewater costs have been computed!')
                
                
                # DISCOUNTING ALL COSTS
                techs_sw = [
                    'Relleno sanitario',
                    'Cielo abierto',
                    'Reciclaje',
                    'Compostaje',
                    'Digestión anaeróbica para biogas',
                    'Incineración de residuos',
                    'Incineración abierta de residuos',
                    'Residuos sólidos no gestionados']

                techs_sales_sw = [
                    'Reciclaje',
                    'Compostaje'
                    ]
                tech_ext_sw = [
                    'Salud(Morbilidad)',
                    'Contaminación de aguas',
                    'Pérdidas por actividades turísticas'
                    ]
                
                opex_sw, capex_sw, sale_sw, ext_sw = {}, {}, {}, {}


                opex_sw.update({'Relleno sanitario': grs_rs_opex})
                capex_sw.update({'Relleno sanitario': grs_rs_capex})
                opex_sw.update({'Cielo abierto': grs_ca_opex})
                capex_sw.update({'Cielo abierto': grs_ca_capex})
                opex_sw.update({'Reciclaje': grs_re_opex})
                capex_sw.update({'Reciclaje': grs_re_capex})
                sale_sw.update({'Reciclaje':grs_re_sale})
                opex_sw.update({'Compostaje': grs_comp_opex})
                capex_sw.update({'Compostaje': grs_comp_capex})
                sale_sw.update({'Compostaje':grs_comp_sale})
                opex_sw.update({'Digestión anaeróbica para biogas': grs_da_opex})
                capex_sw.update({'Digestión anaeróbica para biogas': grs_da_capex})
                opex_sw.update({'Incineración de residuos': grs_ir_opex})
                capex_sw.update({'Incineración de residuos': grs_ir_capex})
                opex_sw.update({'Incineración abierta de residuos': grs_iar_opex})
                capex_sw.update({'Incineración abierta de residuos': grs_iar_capex})
                opex_sw.update({'Residuos sólidos no gestionados': grs_rsng_opex})
                capex_sw.update({'Residuos sólidos no gestionados': grs_rsng_capex})
                ext_sw.update({'Salud(Morbilidad)': ext_salud})
                ext_sw.update({'Contaminación de aguas': ext_cont})
                ext_sw.update({'Pérdidas por actividades turísticas': ext_tur})
                opex_sw_disc, capex_sw_disc, sale_sw_disc, ext_sw_disc =\
                    deepcopy(opex_sw), deepcopy(capex_sw), deepcopy(sale_sw), \
                        deepcopy(ext_sw)

                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    for tech in techs_sw:
                        opex_sw_disc[tech][y] *= disc_constant
                        capex_sw_disc[tech][y] *= disc_constant
                    for tech in techs_sales_sw:
                        sale_sw_disc[tech][y] *= disc_constant
                    for tech in tech_ext_sw:
                        ext_sw_disc[tech][y] *= disc_constant
                        
                
                # STORING ALL COSTS
                dict_local_country[this_country].update({'Solid waste CAPEX [MUSD]': deepcopy(capex_sw)})
                dict_local_country[this_country].update({'Solid waste CAPEX [MUSD] (disc)': deepcopy(capex_sw_disc)})
                dict_local_country[this_country].update({'Solid waste OPEX [MUSD]': deepcopy(opex_sw)})
                dict_local_country[this_country].update({'Solid waste OPEX [MUSD] (disc)': deepcopy(opex_sw_disc)})
                dict_local_country[this_country].update({'Solid waste sales [MUSD]': deepcopy(sale_sw)})
                dict_local_country[this_country].update({'Solid waste sales [MUSD] (disc)': deepcopy(opex_sw_disc)})
                dict_local_country[this_country].update({'Solid waste externalities [MUSD]': deepcopy(ext_sw)})
                dict_local_country[this_country].update({'Solid waste externalities [MUSD] (disc)': deepcopy(ext_sw_disc)})
                dict_local_country[this_country].update({'CAPEX para aguas residuales tratadas [MUSD]': deepcopy(total_capex_tre_wat)})
                dict_local_country[this_country].update({'OPEX fijo para aguas residuales tratadas [MUSD]': deepcopy(total_fopex_tre_wat)})
                dict_local_country[this_country].update({'OPEX variable para aguas residuales tratadas [MUSD]': deepcopy(total_vopex_tre_wat)})
                dict_local_country[this_country].update({'CAPEX para aguas residuales tratadas [MUSD] (disc)': deepcopy(total_capex_tre_wat_disc)})
                dict_local_country[this_country].update({'OPEX fijo para aguas residuales tratadas [MUSD] (disc)': deepcopy(total_fopex_tre_wat_disc)})
                dict_local_country[this_country].update({'OPEX variable para aguas residuales tratadas [MUSD] (disc)': deepcopy(total_vopex_tre_wat_disc)})
                
                
                # STORING EMISSIONS
                emis_sw = {}
                emis_sw.update({'Relleno sanitario': grs_rs_emis})
                emis_sw.update({'Cielo abierto': grs_ca_emis})
                emis_sw.update({'Reciclaje': grs_re_emis})
                emis_sw.update({'Compostaje': grs_comp_emis})
                emis_sw.update({'Digestión anaeróbica para biogas': grs_da_emis})
                emis_sw.update({'Incineración de residuos': grs_ir_emis})
                emis_sw.update({'Incineración abierta de residuos': grs_iar_emis})
                emis_sw.update({'Residuos sólidos no gestionados': grs_rsng_emis})

                dict_local_country[this_country].update({'Carbono negro por incineración de residuos [t BC]': deepcopy(grs_ir_bc_emis)})
                dict_local_country[this_country].update({'Solid waste emissions [kt CH4]': deepcopy(emis_sw)})
                dict_local_country[this_country].update({'Emisiones de aguas residuales tratadas [kt CH4]': deepcopy(waste_water_emissions)})
                dict_local_country[this_country].update({'Emisiones capturadas en rellenos [kt CH4]': deepcopy(grs_rs_emis_captured)})

                print('Waste emissions have been computed!')
                print('Waste costs have been computed!')
                
            """
            INSTRUCTIONS:
            1) Load inputs for RAC sector
            2) Create projections for RAC sector
            3) Estimate costs and emissions 
            4) Store the variables to print
            """
            if model_rac:
                #Load demanad for all sector
                mask_scen = (df4_rac_data['Scenario'] == this_scen)
                df4_rac_data_scen = df4_rac_data.loc[mask_scen]
                # Project the demand:
                mask_dem = (df4_rac_data['Scenario'] == this_scen) & \
                    (df4_rac_data['Parameter'] == 'Demand')
                df4_rac_data_scen_dem = df4_rac_data.loc[mask_dem]
                types_projection_dem_rac = df4_rac_data_scen_dem['Projection'].iloc[0]
                types_by_vals_dem_rac = df4_rac_data_scen_dem[time_vector[0]].iloc[0]
                
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df4_rac_data_scen_dem[time_vector[y]].iloc[0]})
                if  types_projection_dem_rac == 'grow_gdp_pc':
                    total_dem_rac_list = []
                    gen_dem_rac_pc = []
                    for y in range(len(time_vector)):
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        if y == 0:
                            gen_dem_rac_pc.append(types_by_vals_dem_rac/this_pop_vector[0]) #quemas per cápita 
                            total_dem_rac_list.append(types_by_vals_dem_rac)
                        else:
                            next_val_gen_pc = gen_dem_rac_pc[-1] * (1 + gdp_pc_gr)
                            gen_dem_rac_pc.append(next_val_gen_pc)
                            next_val_total = next_val_gen_pc*this_pop_vector[y]
                            total_dem_rac_list.append(next_val_total)
                elif  types_projection_dem_rac == 'user_defined':
                    total_dem_rac_list = []
                    gen_dem_rac_pc = []
                    for y in range(len(time_vector)):
                        total_dem_rac_list.append(all_vals_gen_pc_dict[time_vector[y]])
                        #total_dem_rac_list.append(gen_dem_rac_pc[-1]*this_pop_vector[y])
                elif types_projection_dem_rac == 'flat':
                    total_dem_rac_list = [types_by_vals_dem_rac] * len(time_vector)
                
                
                #Load shares by type of refrigerant for AC subsector
                mask_shares_ac = (df4_rac_data['Scenario'] == this_scen) & \
                    (df4_rac_data['Parameter'] == 'Shares') & \
                        (df4_rac_data['Subsector'] == 'AC')
                df4_rac_data_scen_shares_ac = df4_rac_data.loc[mask_shares_ac]
                types_projection_shares_ac = df4_rac_data_scen_shares_ac['Projection'].tolist()
                types_shares_ac = df4_rac_data_scen_shares_ac['Refrigerante'].tolist()
                types_by_vals_shares_ac = df4_rac_data_scen_shares_ac[time_vector[0]].tolist()
                
                ac_shares_ac_out_dict = {}
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df4_rac_data_scen_shares_ac[time_vector[y]].tolist()})
                for l in range(len(types_shares_ac)):
                    this_rac = types_shares_ac[l]
                    this_proj = types_projection_shares_ac[l]
                    this_by_val = types_by_vals_shares_ac[l]
                    this_val_list = []
                    if this_proj == 'flat':
                        this_val_list = [this_by_val]*len(time_vector)
                        ac_shares_ac_out_dict.update({this_rac: this_val_list})
                    elif this_proj == 'user_defined':
                        total_shares_list = []
                        for y in range(len(time_vector)):
                            total_shares_list.append(all_vals_gen_pc_dict[time_vector[y]][l])
                        ac_shares_ac_out_dict.update({this_rac: total_shares_list})
                
                
                # Calculate total refrigerant (ton):
                ac_dem_by_type_dict = {}
                for l in range(len(types_shares_ac)):
                    ac_dem_by_type = [dem*share for dem, share in \
                                        zip(total_dem_rac_list, ac_shares_ac_out_dict[types_shares_ac[l]])]
                    ac_dem_by_type_dict.update({types_shares_ac[l]:ac_dem_by_type})
                
                
                #Leakage factor (adim)
                mask_leakage_factor = (df4_rac_data['Scenario'] == this_scen) & \
                    (df4_rac_data['Parameter'] == 'Factor_fugas') 
                df4_leakage_factor = df4_rac_data.loc[mask_leakage_factor]
                types_projection_leakage_factor = df4_leakage_factor['Projection'].iloc[0]
                types_by_vals_leakage_factor = df4_leakage_factor[time_vector[0]].iloc[0]
            
                #projection mode for leakage factor
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df4_leakage_factor[time_vector[y]].iloc[0]})
                if  types_projection_leakage_factor == 'user_defined':
                    total_lf_list = []
                    gen_lf_pc = []
                    for y in range(len(time_vector)):
                        total_lf_list.append(df4_leakage_factor[time_vector[y]].iloc[0])
                        #total_lf_list.append(gen_lf_pc[-1]*this_pop_vector[y])
                elif types_projection_leakage_factor == 'flat':
                    total_lf_list = [types_by_vals_leakage_factor] * len(time_vector)
                
                
                #Load Global Warming Potencial (adim)
                types_projection_emi_GWP = df4_rac_emi['Projection'].tolist()
                types_emis_GWP = df4_rac_emi['Nombre común'].tolist()
                types_by_vals_semis_GWP = df4_rac_emi[time_vector[0]].tolist()
                
            
                #Projection for GWP
                rac_GWP_out_dict = {}
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df4_rac_emi[time_vector[y]].tolist()})
                for l in range(len(types_emis_GWP)):
                    this_rac = types_emis_GWP[l]
                    this_proj = types_projection_emi_GWP[l]
                    this_by_val = types_by_vals_semis_GWP[l]
                    this_val_list = []
                    if this_proj == 'flat':
                        this_val_list = [this_by_val]*len(time_vector)
                        rac_GWP_out_dict.update({this_rac: this_val_list})    
                    elif this_proj == 'user_defined':
                        total_shares_list = []
                        for y in range(len(time_vector)):
                            total_shares_list.append(all_vals_gen_pc_dict[time_vector[y]][l])
                        rac_GWP_out_dict.update({this_rac: total_shares_list})
                
                
                ac_dem_by_type_dict = {}
                for l in range(len(types_shares_ac)):
                    ac_dem_by_type = [dem*share for dem, share in \
                                        zip(total_dem_rac_list, ac_shares_ac_out_dict[types_shares_ac[l]])]
                    ac_dem_by_type_dict.update({types_shares_ac[l]:ac_dem_by_type})
                  
                    
                #Estimating emisions (kton de CO2 eq)
                ac_emi_out_dict = {}
                for l in range(len(types_shares_ac)):
                    rac_emis = [(ef*act*lf)/1e3 for ef, act, lf in \
                                zip(rac_GWP_out_dict[types_shares_ac[l]], \
                                    ac_dem_by_type_dict[types_shares_ac[l]],\
                                        total_lf_list)]
                    ac_emi_out_dict.update({types_shares_ac[l]:rac_emis})

                #Load price (USD/kg)
                types_projection_rac_price = d5_rac['Projection'].tolist()
                types_rac = d5_rac['Refrigerante'].tolist()
                types_by_vals_rac_price = d5_rac[time_vector[0]].tolist()
                
                
                #Projection for price 
                rac_price_out_dict = {}
                for l in range(len(types_rac)):
                    this_rac = types_rac[l]
                    this_proj_prices = types_projection_rac_price[l]
                    this_by_val = types_by_vals_rac_price[l]
                    this_val_list = []
                    if this_proj_prices == 'flat':
                        this_val_list = [this_by_val]*len(time_vector)
                        rac_price_out_dict.update({this_rac:this_val_list})
                
                
                #Estimating prices (MUSD)
                ac_prices_ac_out_dict = {}
                ac_prices_ac_out_dict_disc = {}
                for l in range(len(types_shares_ac)):
                    ac_prices_ac = [(price*act)/1e3 for price, act in \
                                zip(rac_price_out_dict[types_shares_ac[l]], \
                                    ac_dem_by_type_dict[types_shares_ac[l]])]
                    list_price_ac_disc = []
                    for y in range(len(time_vector)):
                        this_year = int(time_vector[y])
                        disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                        local_disc = ac_prices_ac[y]*disc_constant
                        list_price_ac_disc.append(local_disc)
                    ac_prices_ac_out_dict.update({types_shares_ac[l]:ac_prices_ac})    
                    ac_prices_ac_out_dict_disc.update({types_shares_ac[l]:list_price_ac_disc})
                 
                ac_prices_ac_out_dict = {}
                for l in range(len(types_shares_ac)):
                    ac_prices_ac = [(price*act)/1e3 for price, act in \
                                 zip(rac_price_out_dict[types_shares_ac[l]], \
                                     ac_dem_by_type_dict[types_shares_ac[l]])]
                    ac_prices_ac_out_dict.update({types_shares_ac[l]:ac_prices_ac})    
                #Estimating discounted values 
                ac_prices_ac_out_dict_disc = {}
                for key, values in ac_prices_ac_out_dict.items():
                    discounted_prices = [value / ((1 + r_rate/100) ** t) for t, value in enumerate(values, 0)]
                    ac_prices_ac_out_dict_disc[key] = discounted_prices
                    
                
                #Storing costs and emissions for AC subsector 
                dict_local_country[this_country].update({'Emisiones para subsector de AC [kt CO2 eq]': deepcopy(ac_emi_out_dict)})
                dict_local_country[this_country].update({'Costos para subsector de AC [MUSD]': deepcopy(ac_prices_ac_out_dict)})
                dict_local_country[this_country].update({'Costos para subsector de AC [MUSD] (disc)': deepcopy(ac_prices_ac_out_dict_disc)})
                
                #Shares for Refrigeration
                mask_shares_ref = (df4_rac_data['Scenario'] == this_scen) & \
                    (df4_rac_data['Parameter'] == 'Shares') & \
                        (df4_rac_data['Subsector'] == 'Refrigeración')
                df4_rac_data_scen_shares_ref = df4_rac_data.loc[mask_shares_ref]
                types_projection_shares_ref = df4_rac_data_scen_shares_ref['Projection'].tolist()
                types_shares_ref = df4_rac_data_scen_shares_ref['Refrigerante'].tolist()
                types_by_vals_shares_ref = df4_rac_data_scen_shares_ref[time_vector[0]].tolist()
                
                
                rac_shares_ref_out_dict = {}
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df4_rac_data_scen_shares_ref[time_vector[y]].tolist()})        
                for l in range(len(types_shares_ref)):
                    this_rac = types_shares_ref[l]
                    this_proj = types_projection_shares_ref[l]
                    this_by_val = types_by_vals_shares_ref[l]
                    this_val_list = []
                    if this_proj == 'flat':
                        this_val_list = [this_by_val]*len(time_vector)
                        rac_shares_ref_out_dict.update({this_rac: this_val_list})
                    elif this_proj == 'user_defined':
                        total_shares_list = []
                        for y in range(len(time_vector)):
                            total_shares_list.append(all_vals_gen_pc_dict[time_vector[y]][l])
                        rac_shares_ref_out_dict.update({this_rac: total_shares_list})
                 
                        
                # Calculate total refrigerant (ton):
                ref_dem_by_type_dict = {}
                for l in range(len(types_shares_ref)):
                    ref_dem_by_type = [dem*share for dem, share in \
                                        zip(total_dem_rac_list, rac_shares_ref_out_dict[types_shares_ref[l]])]
                    ref_dem_by_type_dict.update({types_shares_ref[l]:ref_dem_by_type})
                
                    
                #Estimating emisions (kton de CO2 eq)
                ref_emi_out_dict = {}
                for l in range(len(types_shares_ref)):
                    ref_emis = [(ef*act*lf)/1e3 for ef, act, lf in \
                                zip(rac_GWP_out_dict[types_shares_ref[l]], \
                                    ref_dem_by_type_dict[types_shares_ref[l]],\
                                        total_lf_list)]
                    ref_emi_out_dict.update({types_shares_ref[l]:ref_emis})    
                
                   
                #Estimating prices (MUSD)
                ref_prices_out_dict = {}
                for l in range(len(types_shares_ref)):
                    ref_prices_rac = [(price*act)/1e3 for price, act in \
                                zip(rac_price_out_dict[types_shares_ref[l]], \
                                    ref_dem_by_type_dict[types_shares_ref[l]])]
                    ref_prices_out_dict.update({types_shares_ref[l]:ref_prices_rac})
                #Estimating discounted values 
                ref_prices_out_dict_disc = {}
                for key, values in ref_prices_out_dict.items():
                    discounted_prices = [value / ((1 + r_rate/100) ** t) for t, value in enumerate(values, 0)]
                    ref_prices_out_dict_disc[key] = discounted_prices
                    
                #Storing costs and emissions for refrigerants subsector 
                dict_local_country[this_country].update({'Emisiones para subsector de refrigeración [kt CO2 eq]': deepcopy(ref_emi_out_dict)})
                dict_local_country[this_country].update({'Costos para subsector de refrigeración [MUSD]': deepcopy(ref_prices_out_dict)})    
                dict_local_country[this_country].update({'Costos para subsector de refrigeración [MUSD] (disc)': deepcopy(ref_prices_out_dict_disc)})
                
                #Shares for Extintores
                mask_shares_ext = (df4_rac_data['Scenario'] == this_scen) & \
                    (df4_rac_data['Parameter'] == 'Shares') & \
                        (df4_rac_data['Subsector'] == 'Extintores')
                df4_rac_data_scen_shares_ext = df4_rac_data.loc[mask_shares_ext]
                types_projection_shares_ext = df4_rac_data_scen_shares_ext['Projection'].tolist()
                types_shares_ext = df4_rac_data_scen_shares_ext['Refrigerante'].tolist()
                types_by_vals_shares_ext = df4_rac_data_scen_shares_ext[time_vector[0]].tolist()
                
                rac_shares_ext_out_dict = {}
                all_vals_gen_pc_dict = {}
                for y in range(len(time_vector)):
                    all_vals_gen_pc_dict.update({time_vector[y]:df4_rac_data_scen_shares_ext[time_vector[y]].tolist()})
                for l in range(len(types_shares_ext)):
                    this_rac = types_shares_ext[l]
                    this_proj = types_projection_shares_ext[l]
                    this_by_val = types_by_vals_shares_ext[l]
                    this_val_list = []
                    if this_proj == 'flat':
                        this_val_list = [this_by_val]*len(time_vector)
                        rac_shares_ext_out_dict.update({this_rac: this_val_list})
                    elif this_proj == 'user_defined':
                        total_shares_list = []
                        for y in range(len(time_vector)):
                            total_shares_list.append(all_vals_gen_pc_dict[time_vector[y]][l])
                        rac_shares_ext_out_dict.update({this_rac: total_shares_list})
                        
                        
                # Calculate total refrigerant (ton):
                ext_dem_by_type_dict = {}
                for l in range(len(types_shares_ext)):
                    ext_dem_by_type = [dem*share for dem, share in \
                                        zip(total_dem_rac_list, rac_shares_ext_out_dict[types_shares_ext[l]])]
                    ext_dem_by_type_dict.update({types_shares_ext[l]:ext_dem_by_type})
                
                    
                #Estimating emisions (kton de CO2 eq)
                ext_emi_out_dict = {}
                for l in range(len(types_shares_ext)):
                    ext_emis = [(ef*act*lf)/1e3 for ef, act, lf in \
                                zip(rac_GWP_out_dict[types_shares_ext[l]], \
                                    ext_dem_by_type_dict[types_shares_ext[l]],\
                                        total_lf_list)]
                    ext_emi_out_dict.update({types_shares_ext[l]:ext_emis})    
                
                    
                #Estimating prices (MUSD)
                ext_prices_out_dict = {}
                for l in range(len(types_shares_ext)):
                    ext_prices_rac = [(price*act)/1e3 for price, act in \
                                zip(rac_price_out_dict[types_shares_ext[l]], \
                                    ext_dem_by_type_dict[types_shares_ext[l]])]
                    ext_prices_out_dict.update({types_shares_ext[l]:ext_prices_rac})
                #Estimating discounted values 
                ext_prices_out_dict_disc = {}
                for key, values in ext_prices_out_dict.items():
                    discounted_prices = [value / ((1 + r_rate/100) ** t) for t, value in enumerate(values, 0)]
                    ext_prices_out_dict_disc[key] = discounted_prices
                
                #Storing costs and emissions for AC subsector 
                dict_local_country[this_country].update({'Emisiones para subsector de extintores [kt CO2 eq]': deepcopy(ext_emi_out_dict)})
                dict_local_country[this_country].update({'Costos para subsector de extintores [MUSD]': deepcopy(ext_prices_out_dict)})
                dict_local_country[this_country].update({'Costos para subsector de extintores [MUSD] (disc)': deepcopy(ext_prices_out_dict_disc)})
                print('RAC costs and emissions have been computed!')
               
            
            """
            INSTRUCTIONS:
            1) Perform the transport demand calculations
            2) Check the demand component: rewrite the transport energy demand projection
            3) Store the demand and print the energy demand difference
            """
            #print('Rewrite the demand component here')
            types_all = []  # this is defined here to avoid an undefined variable

            # Define the dictionary that stores emissions here to keep
            # emissions from transport technologies.
            # 'emission factors':
            mask_emission_factors = (df4_ef['Type'] == 'Standard') & (df4_ef['Parameter'] == 'Emission factor')
            this_df4_ef = df4_ef.loc[mask_emission_factors]
            this_df4_ef.reset_index(drop=True, inplace=True)
            this_df4_ef_fuels = \
                this_df4_ef['Fuel'].tolist()
    
            emissions_fuels_list = []
            emissions_fuels_dict = {}
            for f in range(len(this_df4_ef_fuels)):
                this_fuel = this_df4_ef_fuels[f]
                this_proj = this_df4_ef.iloc[f]['Projection']
                base_emission_val = this_df4_ef.iloc[f][time_vector[0]]
                if this_proj == 'flat':
                    list_emission_year = [base_emission_val for y in range(len(time_vector))]
                emissions_fuels_list.append(this_fuel)
                emissions_fuels_dict.update({this_fuel:list_emission_year})

            # Include the emission factor for black carbon:
            mask_emission_factors = (df4_ef['Parameter'] == 'Emission factor_black carbon')
            this_df4_ef_2 = df4_ef.loc[mask_emission_factors]
            this_df4_ef_2.reset_index(drop=True, inplace=True)
            this_df4_ef_2_fuels = \
                this_df4_ef_2['Fuel'].tolist()
            this_df4_ef_2_techs = \
                this_df4_ef_2['Type'].tolist()
    
            emissions_2_fuels_list = []
            emissions_2_techs_list = []
            emissions_2_fuels_dict = {}
            for f in range(len(this_df4_ef_2_fuels)):
                this_fuel = this_df4_ef_2_fuels[f]
                this_tech = this_df4_ef_2_techs[f]
                this_proj = this_df4_ef_2.iloc[f]['Projection']
                base_emission_val = this_df4_ef_2.iloc[f][time_vector[0]]
                if this_proj == 'flat':
                    list_emission_year = [base_emission_val for y in range(len(time_vector))]

                if this_tech not in emissions_2_techs_list:
                    emissions_2_techs_list.append(this_tech)
                    emissions_2_fuels_dict.update({this_tech:{}})

                if this_fuel not in emissions_2_fuels_list:
                    emissions_2_fuels_list.append(this_fuel)
                    emissions_2_fuels_dict[this_tech].update({this_fuel:0})

                emissions_2_fuels_dict[this_tech][this_fuel] = list_emission_year
            
            
            # Include emission factor for methane
            mask_emission_factors = (df4_ef['Parameter'] == 'Emission factor_methane')
            this_df4_ef_3 = df4_ef.loc[mask_emission_factors]
            this_df4_ef_3.reset_index(drop=True, inplace=True)
            this_df4_ef_3_fuels = \
                this_df4_ef_3['Fuel'].tolist()
            this_df4_ef_3_techs = \
                this_df4_ef_3['Type'].tolist()
            # this_df4_ef_3_unique = [af + ' AND ' + at for af, at in zip(this_df4_ef_3_fuels, this_df4_ef_3_techs)]

            emissions_3_fuels_list = []
            emissions_3_techs_list = []
            emissions_3_fuels_dict = {}
            for f in range(len(this_df4_ef_3_fuels)):
                this_fuel = this_df4_ef_3_fuels[f]
                this_tech = this_df4_ef_3_techs[f]
                this_proj = this_df4_ef_3.iloc[f]['Projection']
                base_emission_val = this_df4_ef_3.iloc[f][time_vector[0]]
                if this_proj == 'flat':
                    list_emission_year = [base_emission_val for y in range(len(time_vector))]

                if this_tech not in emissions_3_techs_list:
                    emissions_3_techs_list.append(this_tech)
                    emissions_3_fuels_dict.update({this_tech:{}})

                if this_fuel not in emissions_3_fuels_list:
                    emissions_3_fuels_list.append(this_fuel)
                    emissions_3_fuels_dict[this_tech].update({this_fuel:0})

                emissions_3_fuels_dict[this_tech][this_fuel] = list_emission_year

            # print('review this so far')
            # sys.exit()

            # Define emission output
            emissions_demand = {}  # crucial output
            emissions_demand_black_carbon = {}
            emissions_demand_methane = {}
            if overwrite_transport_model:
                # We must edit the "dict_energy_demand" content with the transport modeling
                dict_energy_demand_trn = deepcopy(dict_energy_demand['Transport'])
                transport_fuel_sets = \
                    df2_trans_sets_eq['Transport Fuel'].tolist()
                transport_scenario_sets = \
                    df2_trans_sets_eq['Energy Fuel'].tolist()
                dict_eq_transport_fuels = {}
                for te in range(len(transport_fuel_sets)):
                    t_fuel_set = transport_fuel_sets[te]
                    t_scen_set = transport_scenario_sets[te]
                    dict_eq_transport_fuels.update({t_fuel_set:t_scen_set})
    
                # NOTE: we must edit the projection from "dict_energy_demand" first;
                # once that is complete, we must edit the "dict_energy_demand_by_fuel" accordingly.
    
                # Now we must follow standard transport equations and estimations:
                # TM 1) Estimate the demand projection for the country
                """
                Demand_Passenger = sum_i (km_passenger_i * fleet_passenger_i * load_factor_i)
                """
                # Store load factor and kilometers:
                dict_lf, dict_km = {}, {}
                #
                # 1.a) estimate the demand in the base year
    
                set_pass_trn = \
                    [list_trn_type[n]
                    for n, v in enumerate(list_trn_lvl1_u_raw)
                    if v == 'Passenger']
    
                set_pass_trn_dem_by = {}
                set_pass_trn_fleet_by = {}
                set_pass_trn_dem_sh = {}
    
                for n in range(len(set_pass_trn)):
                    set_pass_trn_dem_by.update({set_pass_trn[n]:0})
                    set_pass_trn_fleet_by.update({set_pass_trn[n]:0})
                    set_pass_trn_dem_sh.update({set_pass_trn[n]:0})
    
                # Select the scenario:
                mask_select_trn_scen = \
                    (df3_tpt_data['Scenario'] == this_scen) | (df3_tpt_data['Scenario'] == 'ALL')
                df_trn_data = df3_tpt_data.loc[mask_select_trn_scen]
                df_trn_data.reset_index(drop=True, inplace=True)
    
                sum_pass_trn_dem_by = 0  # Gpkm
                for spt in set_pass_trn:
                    mask_fby_spt = (df_trn_data['Type'] == spt) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Residual fleet')
                    fby_spt = df_trn_data.loc[mask_fby_spt][per_first_yr].sum()
    
                    mask_km_spt = (df_trn_data['Type'] == spt) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Distance')
                    km_spt = df_trn_data.loc[mask_km_spt][per_first_yr].sum()
    
                    mask_lf_spt = (df_trn_data['Type'] == spt) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Load Factor')
                    lf_spt = df_trn_data.loc[mask_lf_spt][per_first_yr].sum()
    
                    dict_lf.update({spt: deepcopy(lf_spt)})
                    dict_km.update({spt: deepcopy(km_spt)})
    
                    set_pass_trn_fleet_by[spt] = fby_spt
                    set_pass_trn_dem_by[spt] = fby_spt*km_spt*lf_spt/1e9
                    sum_pass_trn_dem_by += fby_spt*km_spt*lf_spt/1e9
    
                for spt in set_pass_trn:
                    set_pass_trn_dem_sh[spt] = 100*set_pass_trn_dem_by[spt]/sum_pass_trn_dem_by
    
                set_fre_trn = \
                    [list_trn_type[n]
                    for n, v in enumerate(list_trn_lvl1_u_raw)
                    if v == 'Freight']
    
                set_fre_trn_dem_by = {}
                set_fre_trn_fleet_by = {}
                set_fre_trn_dem_sh = {}
    
                for n in range(len(set_fre_trn)):
                    set_fre_trn_dem_by.update({set_fre_trn[n]:0})
                    set_fre_trn_fleet_by.update({set_fre_trn[n]:0})
                    set_fre_trn_dem_sh.update({set_fre_trn[n]:0})
    
                sum_fre_trn_dem_by = 0  # Gtkm
                for spt in set_fre_trn:
                    mask_fby_spt = (df_trn_data['Type'] == spt) & \
                                    (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                    (df_trn_data['Parameter'] == 'Residual fleet')
                    fby_spt = df_trn_data.loc[mask_fby_spt][per_first_yr].sum()
    
                    mask_km_spt = (df_trn_data['Type'] == spt) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Distance')
                    km_spt = df_trn_data.loc[mask_km_spt][per_first_yr].sum()
    
                    mask_lf_spt = (df_trn_data['Type'] == spt) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Load Factor')
                    lf_spt = df_trn_data.loc[mask_lf_spt][per_first_yr].sum()
    
                    dict_lf.update({spt: deepcopy(lf_spt)})
                    dict_km.update({spt: deepcopy(km_spt)})
    
                    set_fre_trn_fleet_by[spt] = fby_spt
                    set_fre_trn_dem_by[spt] = fby_spt*km_spt*lf_spt/1e9
                    sum_fre_trn_dem_by += fby_spt*km_spt*lf_spt/1e9
    
                for spt in set_fre_trn:
                    set_fre_trn_dem_sh[spt] = 100*set_fre_trn_dem_by[spt]/sum_fre_trn_dem_by
    
                # 1.b) estimate the demand growth
                # for this we need to extract a couple of variables from
                # the "df_trn_data":
                # Elasticities:
                projtype_ela_pas, mask_ela_pas = \
                    fun_dem_model_projtype('Passenger', this_country, 'Elasticity',
                                        'projection', df_trn_data)
                list_ela_pas = \
                    fun_dem_proj(time_vector, projtype_ela_pas,
                                mask_ela_pas, df_trn_data)
    
                projtype_ela_fre, mask_ela_fre = \
                    fun_dem_model_projtype('Freight', this_country, 'Elasticity',
                                        'projection', df_trn_data)
                list_ela_fre = \
                    fun_dem_proj(time_vector, projtype_ela_fre,
                                mask_ela_fre, df_trn_data)
    
                projtype_ela_oth, mask_ela_oth = \
                    fun_dem_model_projtype('Other', this_country, 'Elasticity',
                                        'projection', df_trn_data)
                list_ela_oth = \
                    fun_dem_proj(time_vector, projtype_ela_oth,
                                mask_ela_oth, df_trn_data)
    
                # Demands:
                projtype_dem_pas, mask_dem_pas = \
                    fun_dem_model_projtype('Passenger', this_country, 'Demand',
                                        'projection', df_trn_data)
                if 'endogenous' not in projtype_dem_pas:
                    pass_trn_dem = \
                        fun_dem_proj(time_vector, projtype_dem_pas,
                                    mask_dem_pas, df_trn_data)
    
                projtype_dem_fre, mask_dem_fre = \
                    fun_dem_model_projtype('Freight', this_country, 'Demand',
                                        'projection', df_trn_data)
                if 'endogenous' not in projtype_dem_fre:
                    fre_trn_dem = \
                        fun_dem_proj(time_vector, projtype_dem_fre,
                                    mask_dem_fre, df_trn_data)

                projtype_dem_oth, mask_dem_oth = \
                    fun_dem_model_projtype('Other', this_country, 'Demand',
                                        'projection', df_trn_data)
                '''
                Note: "Other" [transport demands] is a category currently
                unused
                '''
    
                if 'endogenous' in projtype_dem_pas:
                    pass_trn_dem = [0 for y in range(len(time_vector))]
                if 'endogenous' in projtype_dem_fre:
                    fre_trn_dem = [0 for y in range(len(time_vector))]
                # We must project transport demand here.
                for y in range(len(time_vector)):
                    if y == 0:
                        pass_trn_dem[y] = sum_pass_trn_dem_by
                        fre_trn_dem[y] = sum_fre_trn_dem_by
                    else:
                        gdp_gr = this_gdp_growth_vals[y]/100
                        gdp_pc_gr = this_gdp_pc_growth_vals[y]/100
                        pop_gr = this_pop_growth_vals[y]/100
    
                        if projtype_dem_pas == 'endogenous_gdp':
                            trn_gr_pas = 1 + (gdp_gr*list_ela_pas[y])
                        
                        if projtype_dem_fre == 'endogenous_gdp':
                            trn_gr_fre = 1 + (gdp_gr*list_ela_fre[y])
    
                        if projtype_dem_pas == 'endogenous_gdp_pc':
                            trn_gr_pas = 1 + (gdp_pc_gr*list_ela_pas[y])
                        
                        if projtype_dem_fre == 'endogenous_gdp_pc':
                            trn_gr_fre = 1 + (gdp_pc_gr*list_ela_fre[y])
    
                        if projtype_dem_pas == 'endogenous_pop':
                            trn_gr_pas = 1 + (pop_gr*list_ela_pas[y])
                        
                        if projtype_dem_fre == 'endogenous_pop':
                            trn_gr_fre = 1 + (pop_gr*list_ela_fre[y])
    
                        if 'endogenous' in projtype_dem_pas:
                            pass_trn_dem[y] = trn_gr_pas*pass_trn_dem[y-1]
                        if 'endogenous' in projtype_dem_fre:
                            fre_trn_dem[y] = trn_gr_fre*fre_trn_dem[y-1]
    
                # 1.c) apply the mode shift and non-motorized parameters:
                set_pass_trn_priv = \
                    [list_trn_type[n]
                    for n, v in enumerate(list_trn_lvl2_u_raw)
                    if v == 'Private']
                set_pass_trn_pub = \
                    [list_trn_type[n]
                    for n, v in enumerate(list_trn_lvl2_u_raw)
                    if v == 'Public']
    
                pass_trn_dem_sh_private = 0  # this should be an integer
                for n in range(len(set_pass_trn_priv)):
                    pass_trn_dem_sh_private += \
                        set_pass_trn_dem_sh[set_pass_trn_priv[n]]
    
                pass_trn_dem_sh_public = 0
                for n in range(len(set_pass_trn_pub)):
                    pass_trn_dem_sh_public += \
                        set_pass_trn_dem_sh[set_pass_trn_pub[n]]
    
                pass_trn_dem_sh_private_k = {}
                for n in range(len(set_pass_trn_priv)):
                    this_sh_k = set_pass_trn_dem_sh[set_pass_trn_priv[n]]
                    this_sh_k_adj = \
                        100*this_sh_k/pass_trn_dem_sh_private
                    add_sh_k = {set_pass_trn_priv[n]:this_sh_k_adj}
                    pass_trn_dem_sh_private_k.update(deepcopy(add_sh_k))
    
                pass_trn_dem_sh_public_k = {}
                for n in range(len(set_pass_trn_pub)):
                    this_sh_k = set_pass_trn_dem_sh[set_pass_trn_pub[n]]
                    this_sh_k_adj = \
                        100*this_sh_k/pass_trn_dem_sh_public
                    add_sh_k = {set_pass_trn_pub[n]:this_sh_k_adj}
                    pass_trn_dem_sh_public_k.update(deepcopy(add_sh_k))
    
                # ...the goal is to have a list of participation per type:
                gpkm_pri_k = {}
                for n in range(len(set_pass_trn_priv)):
                    gpkm_pri_k.update({set_pass_trn_priv[n]:[]})
                gpkm_pub_k = {}
                for n in range(len(set_pass_trn_pub)):
                    gpkm_pub_k.update({set_pass_trn_pub[n]:[]})
                gpkm_nonmot = []
    
                list_mode_shift = []
                mask_mode_shift = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Mode shift')
                list_non_motorized = []
                mask_non_motorized = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                    (df_trn_data['Parameter'] == 'Non-motorized transport')
                for y in range(len(time_vector)):
                    list_mode_shift.append(df_trn_data.loc[mask_mode_shift][time_vector[y]].iloc[0])
                    list_non_motorized.append(df_trn_data.loc[mask_non_motorized][time_vector[y]].iloc[0])
    
                    # Private types:
                    this_gpkm_priv = \
                        pass_trn_dem[y]*(pass_trn_dem_sh_private-list_mode_shift[-1]-list_non_motorized[-1])/100
                    for n in range(len(set_pass_trn_priv)):
                        this_sh_k_adj = \
                            pass_trn_dem_sh_private_k[set_pass_trn_priv[n]]
                        this_gpkm_k = \
                            this_gpkm_priv*this_sh_k_adj/100
                        gpkm_pri_k[set_pass_trn_priv[n]].append(this_gpkm_k)
                        
                    """
                    NOTE: the share of autos and motos relative to private stays constant (units: pkm)
                    """
                    # Public types:
                    this_gpkm_pub = pass_trn_dem[y]*(pass_trn_dem_sh_public+list_mode_shift[-1])/100
                    for n in range(len(set_pass_trn_pub)):
                        this_sh_k_adj = \
                            pass_trn_dem_sh_public_k[set_pass_trn_pub[n]]
                        this_gpkm_k = \
                            this_gpkm_pub*this_sh_k_adj/100
                        gpkm_pub_k[set_pass_trn_pub[n]].append(this_gpkm_k)
    
                    # Non-mot types:
                    this_gpkm_nonmot = pass_trn_dem[y]*(list_non_motorized[-1])
                    gpkm_nonmot.append(this_gpkm_nonmot)
    
                # 1.d) apply the logistics parameters:
                gtkm_freight_k = {}
                for n in range(len(set_fre_trn)):
                    gtkm_freight_k.update({set_fre_trn[n]: []})
    
                list_logistics = []
                mask_logistics = \
                    (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                    (df_trn_data['Parameter'] == 'Logistics')
                for y in range(len(time_vector)):
                    list_logistics.append(df_trn_data.loc[mask_logistics][time_vector[y]].iloc[0])
    
                    for n in range(len(set_fre_trn)):                    
                        this_fre_sh_k = set_fre_trn_dem_sh[set_fre_trn[n]]
                        this_fre_k = fre_trn_dem[y]*this_fre_sh_k/100
                        gtkm_freight_k[set_fre_trn[n]].append(this_fre_k)
    
                # TM 2) Estimate the required energy for transport
                """
                Paso 1: obtener el % de flota por fuel de cada carrocería
                """
                # A dictionary with the residual fleet will come in handy:
                dict_resi_cap_trn = {}
    
                # Continue distributing the fleet:
                types_pass = set_pass_trn
                types_fre = set_fre_trn
                fuels = transport_fuel_sets
                fuels_nonelectric = [i for i in transport_fuel_sets if
                                    i not in ['ELECTRICIDAD', 'HIDROGENO']]
                set_pass_trn_fleet_by_sh = {}
                for t in types_pass:
                    dict_resi_cap_trn.update({t:{}})
                    total_type = set_pass_trn_fleet_by[t]
                    set_pass_trn_fleet_by_sh.update({t:{}})
                    for f in fuels:
                        dict_resi_cap_trn[t].update({f:[]})
                        mask_tf = (df_trn_data['Type'] == t) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Residual fleet') & \
                                (df_trn_data['Fuel'] == f)
                        fby_tf = df_trn_data.loc[mask_tf][per_first_yr].iloc[0]
                        set_pass_trn_fleet_by_sh[t].update({f:100*fby_tf/total_type})
                        for y in range(len(time_vector)):
                            a_fleet = \
                                df_trn_data.loc[mask_tf][time_vector[y]].iloc[0]
                            dict_resi_cap_trn[t][f].append(a_fleet)
    
                set_fre_trn_fleet_by_sh = {}
                fuels_fre = []
                for t in types_fre:
                    dict_resi_cap_trn.update({t:{}})
                    total_type = set_fre_trn_fleet_by[t]
                    set_fre_trn_fleet_by_sh.update({t:{}})
                    for f in fuels:
                        dict_resi_cap_trn[t].update({f:[]})
                        mask_tf = (df_trn_data['Type'] == t) & \
                                (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                (df_trn_data['Parameter'] == 'Residual fleet') & \
                                (df_trn_data['Fuel'] == f)
                        try:
                            fby_tf = \
                                df_trn_data.loc[mask_tf][per_first_yr].iloc[0]
                            fuels_fre.append(f)
                        except Exception:
                            fby_tf = 0
    
                        if total_type > 0:
                            set_fre_trn_fleet_by_sh[t].update({f:100*fby_tf/total_type})
                        else:
                            set_fre_trn_fleet_by_sh[t].update({f:0})
    
                        for y in range(len(time_vector)):
                            a_fleet = \
                                df_trn_data.loc[mask_tf][time_vector[y]].iloc[0]
                            dict_resi_cap_trn[t][f].append(a_fleet)
    
                """
                Paso 2: Proyectar la participación de cada fuel en cada carrocería usando el parámetro "Electrification"
                """
                dict_fuel_economy = {}
                dict_shares_fleet = {}
                types_all = types_pass + types_fre
    
                for t in types_all:
                    # ...calculating non-electric fuel distribution
                    if t in types_pass:
                        set_trn_fleet_by_sh = set_pass_trn_fleet_by_sh
                    else:
                        set_trn_fleet_by_sh = set_fre_trn_fleet_by_sh
                    sh_non_electric = 0
                    for fne in fuels_nonelectric:
                        sh_non_electric += \
                            set_trn_fleet_by_sh[t][fne]
    
                    sh_non_electric_k = {}
                    for fne in fuels_nonelectric:
                        this_sh_k_f = set_trn_fleet_by_sh[t][fne]
                        if sh_non_electric > 0:
                            this_sh_non_ele_k = \
                                100*this_sh_k_f/sh_non_electric
                        else:
                            this_sh_non_ele_k = 0
                        sh_non_electric_k.update({fne:this_sh_non_ele_k})
    
                    # ...opening electrification percentages:
                    list_electrification = []
                    list_hydrogen = []
                    list_non_electric = []
                    mask_ele = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                            (df_trn_data['Parameter'] == 'Electrification') & \
                            (df_trn_data['Type'] == t)
                    mask_h2 = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                            (df_trn_data['Parameter'] == 'Hydrogen Penetration') & \
                            (df_trn_data['Type'] == t)
    
                    # ...opening fuel economies
                    list_fe_k = {}
                    list_fe_k_masks = {}
                    list_nonele_fleet_k = {}  # open non-electrical fleet
                    for af in fuels:
                        list_fe_k.update({af: []})
                        if af in fuels_nonelectric:
                            list_nonele_fleet_k.update({af: []})
                        mask_fe_af = (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                                    (df_trn_data['Parameter'] == 'Fuel economy') & \
                                    (df_trn_data['Type'] == t) & \
                                    (df_trn_data['Fuel'] == af)
                        list_fe_k_masks.update({af: deepcopy(mask_fe_af)})
                
                    # ...iterating across years:
                    for y in range(len(time_vector)):
                        this_yea = time_vector[y]
    
                        this_ele = df_trn_data.loc[mask_ele][this_yea].iloc[0]
                        if math.isnan(this_ele):
                            this_ele = 0
                        list_electrification.append(this_ele)
    
                        this_h2 = df_trn_data.loc[mask_h2][this_yea].iloc[0]
                        if math.isnan(this_h2):
                            this_h2 = 0
                        list_hydrogen.append(this_h2)
                        list_non_electric.append(100-this_ele-this_h2)
    
                        for af in fuels_nonelectric:
                            this_sh_ne_k = sh_non_electric_k[af]
                            this_fleet_ne_k = \
                                this_sh_ne_k*list_non_electric[-1]/100
                            if math.isnan(this_fleet_ne_k):
                                this_fleet_ne_k = 0
                            list_nonele_fleet_k[af].append(this_fleet_ne_k)
    
                        for af in fuels:
                            this_mask_fe = list_fe_k_masks[af]
                            this_fe_k = \
                                df_trn_data.loc[this_mask_fe
                                                    ][this_yea].iloc[0]
                            if math.isnan(this_fe_k):
                                this_fe_k = 0
                            list_fe_k[af].append(this_fe_k)
    
                    '''
                    # Disturb 10: make the list electrification and hydrogen different
                    list_electrification_raw = deepcopy(list_electrification)
                    list_electrification = \
                        interpolation_non_linear_final(
                            time_vector, list_electrification_raw, 0.5, 2025)
                    print('review')
                    sys.exit()
                    '''
    
                    # ...store the data for this "type":
                    dict_fuel_economy.update({t:{}})
                    dict_shares_fleet.update({t:{}})
                    for af in fuels:
                        if af in fuels_nonelectric:
                            this_fleet_sh_k = \
                                deepcopy(list_nonele_fleet_k[af])
                        elif af == 'ELECTRICIDAD':
                            this_fleet_sh_k = \
                                deepcopy(list_electrification)
                        elif af == 'HIDROGENO':
                            this_fleet_sh_k = \
                                deepcopy(list_hydrogen)
                        else:
                            print('Undefined set (1). Please check.')
                            sys.exit()
    
                        dict_shares_fleet[t].update({af:this_fleet_sh_k})
                        add_fe_k = deepcopy(list_fe_k[af])
                        dict_fuel_economy[t].update({af:add_fe_k})
    
                """
                Paso 3: calcular la energía requerida para el sector transporte
                """
                dict_trn_pj = {}
                for af in fuels:
                    dict_trn_pj.update({af: []})
    
                dict_gpkm_gtkm = {}
                for t in types_all:
                    if t in list(gpkm_pri_k.keys()):
                        this_gpkm_add = gpkm_pri_k[t]
                    if t in list(gpkm_pub_k.keys()):
                        this_gpkm_add = gpkm_pub_k[t]
                    if t in list(gtkm_freight_k.keys()):
                        this_gpkm_add = gtkm_freight_k[t]
                    dict_gpkm_gtkm.update({t:this_gpkm_add})
    
                dict_trn_pj_2 = {}
                for af in fuels:
                    dict_trn_pj_2.update({af: {}})
                    for t in types_all:
                        dict_trn_pj_2[af].update({t: [0]*len(time_vector)})
    
                # To calculate the fleet vectors, we need the gpkm per
                # vehicle type, which can be obtained in the loop below.
                dict_gpkm_gtkm_k, dict_fleet_k = {}, {}
                dict_fuel_con = {}
                zero_list = [0 for y in range(len(time_vector))]
                for t in types_all:
                    dict_gpkm_gtkm_k.update({t:{}})
                    dict_fleet_k.update({t:{}})
                    dict_fuel_con.update({t:{}})
                    for f in fuels:
                        dict_gpkm_gtkm_k[t].update({f:deepcopy(zero_list)})
                        dict_fleet_k[t].update({f:deepcopy(zero_list)})
                        dict_fuel_con[t].update({f:deepcopy(zero_list)})
    
                # For all fuels and techs, find if there is a valid projected fleet
                # to overwrite the demand modeling.
                dict_proj_fleet = {}
                for this_f in fuels:
                    dict_proj_fleet.update({this_f:{}})
                    for t in types_all:
                        dict_proj_fleet[this_f].update({t:{}})
                        mask_fy_tf = \
                            (df_trn_data['Scenario'] == this_scen) & \
                            (df_trn_data['Fuel'] == this_f) & \
                            (df_trn_data['Type'] == t) & \
                            (df_trn_data['Application_Countries'].isin(['All', this_country])) & \
                            (df_trn_data['Parameter'] == 'Projected fleet')
                        fy_tf_list = []
                    
                        for y in range(len(time_vector)):
                            fy_tf = \
                                df_trn_data.loc[mask_fy_tf][time_vector[y]].iloc[0]
                            fy_tf_list.append(fy_tf)
    
                        # Decide what to do with the inputted data:
                        fy_tf_proj = \
                            df_trn_data.loc[mask_fy_tf]['projection'].iloc[0]
                        if fy_tf_proj == 'ignore':
                            dict_proj_fleet[this_f][t].update({'indicate':'ignore'})
                        elif fy_tf_proj == 'user_defined':
                            dict_proj_fleet[this_f][t].update({'indicate':'apply'})
                            dict_proj_fleet[this_f][t].update({'vals':fy_tf_list})
                        else:
                            print('There is an undefined projection type specified for the Projected fleet (12_trans_data).')
                            print("The code will stop at line 1417 for review.")
                            sys.exit()
    
                # For all fuels, find the energy consumption:
                    
                dict_diffs_f_rf = {}
                
                # emissions_demand = {}  # crucial output
                
                for this_f in fuels:
                    
                    dict_diffs_f_rf.update({this_f:{}})
                    
                    this_list = []
                    
                    emis_transport_dict = {}
                    emis_transport_black_carbon_dict = {}
                    emis_transport_methane_dict = {}                                
                    
                    for y in range(len(time_vector)):
                        this_fuel_con = 0
                        for t in types_all:
                            if y == 0:
                                emis_transport_dict.update({t:[]})
                                emis_transport_black_carbon_dict.update({t:[]})
                                emis_transport_methane_dict.update({t:[]})
                                if t not in list(emissions_demand.keys()):
                                    emissions_demand.update({t:{}})
                                if t not in list(emissions_demand_black_carbon.keys()):
                                    emissions_demand_black_carbon.update({t:{}})
                                if t not in list(emissions_demand_methane.keys()):
                                    emissions_demand_methane.update({t:{}})
                                    
                            this_gpkm_gtkm = dict_gpkm_gtkm[t][y]
                            this_sh_fl = dict_shares_fleet[t][this_f][y]/100
                            this_fe = dict_fuel_economy[t][this_f][y]
    
                            # Extract the fuel consumption:
                            add_fuel_con = \
                                this_gpkm_gtkm*this_sh_fl*this_fe/dict_lf[t]
    
                            dict_fuel_con[t][this_f][y] = deepcopy(add_fuel_con)
                            '''
                            Units analysis: Gpkm*MJ/km = PJ
                            '''
                            this_fuel_con += deepcopy(add_fuel_con)
                            
                            dict_trn_pj_2[this_f][t][y] = deepcopy(add_fuel_con)
                            
                            # Calculate the distribution of gpkm or gtkm:
                            this_gpkm_gtkm_k = this_gpkm_gtkm*this_sh_fl
                            dict_gpkm_gtkm_k[t][this_f][y] = \
                                this_gpkm_gtkm_k
                            # Calculate the fleet:
                            this_fleet_k = 1e9*\
                                (this_gpkm_gtkm_k/dict_lf[t])/dict_km[t]
                            dict_fleet_k[t][this_f][y] = this_fleet_k
    
                            if dict_km[t] == 0:
                                print('review division by zero')
                                sys.exit()
    
                            resi_fleet = dict_resi_cap_trn[t][this_f][0]
                            if y == 0 and resi_fleet != 0:
                                dict_diffs_f_rf[this_f].update({t:this_fleet_k/resi_fleet})
    
                            # This code is added to overwrite the fleet projections:
                            if dict_proj_fleet[this_f][t]['indicate'] == 'apply':
                                proj_fleet_list = dict_proj_fleet[this_f][t]['vals']
                                proj_fleet_y = proj_fleet_list[y]
    
                                # Overwriting fleet:
                                dict_fleet_k[t][this_f][y] = deepcopy(proj_fleet_y)
    
                                # Overwriting gpkm_gtkm (gotta sum):
                                delta_km = (proj_fleet_y-this_fleet_k)*dict_km[t]
                                delta_gpkm_gtkm = delta_km*dict_lf[t]/1e9
                                dict_gpkm_gtkm[t][y] += deepcopy(delta_gpkm_gtkm)
    
                                # Overwriting fuel (gotta sum):
                                delta_fuel_con = delta_km*this_fe/1e9  # PJ
                                dict_fuel_con[t][this_f][y] += delta_fuel_con
                                this_fuel_con += delta_fuel_con
    
                                dict_trn_pj_2[this_f][t][y] += deepcopy(delta_fuel_con)
    
                            # Estimate emissions:
                            fuel_energy_model = dict_eq_transport_fuels[this_f]
                            if fuel_energy_model in list(emissions_fuels_dict.keys()):
                                emis_fact = emissions_fuels_dict[fuel_energy_model][y]
                            else:
                                emis_fact = 0
                            if fuel_energy_model in list(emissions_2_fuels_dict['Standard'].keys()):
                                emis_fact_black_carbon = emissions_2_fuels_dict['Standard'][fuel_energy_model][y]
                            else:
                                emis_fact_black_carbon = 0
                            if fuel_energy_model in list(emissions_3_fuels_dict['Transport_Land'].keys()):
                                emis_fact_methane = emissions_3_fuels_dict['Transport_Land'][fuel_energy_model][y]
                            else:
                                emis_fact_methane = 0

                            emis_transport = dict_fuel_con[t][this_f][y]*emis_fact
                            emis_transport_black_carbon = dict_fuel_con[t][this_f][y]*emis_fact_black_carbon
                            emis_transport_methane = dict_fuel_con[t][this_f][y]*emis_fact_methane
                            
                                                                                                  
    
                            emis_transport_dict[t].append(emis_transport)
                            emis_transport_black_carbon_dict[t].append(emis_transport_black_carbon)
                            emis_transport_methane_dict[t].append(emis_transport_methane)
                        this_list.append(this_fuel_con)
    
                    #
                    for t in types_all:
                        emissions_demand[t].update({
                            this_f:deepcopy(emis_transport_dict[t])})
                        emissions_demand_black_carbon[t].update({
                            this_f:deepcopy(emis_transport_black_carbon_dict[t])})
                        emissions_demand_methane[t].update({
                            this_f:deepcopy(emis_transport_methane_dict[t])})
                    dict_trn_pj[this_f] = deepcopy(this_list)
 
                # if this_scen != 'BAU':
                #    print('review transport demand projections up until here')
                #    sys.exit()

                # *********************************************************
                # We can calculate the required new fleets to satisfy the demand:
                dict_new_fleet_k, dict_accum_new_fleet_k = {}, {}
    
                # We will take advantage to estimate the costs related to
                # fleet and energy; we can check the cost and tax params:
                cost_params = list(dict.fromkeys(d5_tpt['Parameter'].tolist()))
                # ['CapitalCost', 'FixedCost', 'VariableCost', 'OpLife']
                cost_units = list(dict.fromkeys(d5_tpt['Unit'].tolist()))
                # 
                tax_params = list(dict.fromkeys(d5_tax['Parameter'].tolist()))
                # ['Imports', 'IMESI_Venta', 'IVA_Venta', 'Patente',
                # 'IMESI_Combust', 'IVA_Gasoil', 'IVA_Elec', 'Impuesto_Carbono',
                # 'Otros_Gasoil']
    
                # Define the cost outputs:
                dict_capex_out = {}
                dict_fopex_out = {}
                dict_vopex_out = {}
                dict_capex_out_disc = {}
                dict_fopex_out_disc = {}
                dict_vopex_out_disc = {}
                # Define the tax outputs:
                dict_tax_out = {}
                for atax in tax_params:
                    dict_tax_out.update({atax:{}})
                    for t in types_all:
                        dict_tax_out[atax].update({t:{}})
                        for f in fuels:
                            dict_tax_out[atax][t].update({f:{}})
    
                # Let's now start the loop:
                times_neg_new_fleet, times_neg_new_fleet_sto = 0, []
                for t in types_all:
                    dict_new_fleet_k.update({t:{}})
                    dict_capex_out.update({t:{}})
                    dict_fopex_out.update({t:{}})
                    dict_vopex_out.update({t:{}})
                    dict_capex_out_disc.update({t:{}})
                    dict_fopex_out_disc.update({t:{}})
                    dict_vopex_out_disc.update({t:{}})
                    for f in fuels:
                        # Unpack the costs:
                        list_cap_cost, unit_cap_cost = \
                            fun_unpack_costs('CapitalCost', t, f,
                                            d5_tpt,
                                            time_vector)
                        list_fix_cost, unit_fix_cost = \
                            fun_unpack_costs('FixedCost', t, f,
                                            d5_tpt,
                                            time_vector)
                        list_var_cost, unit_var_cost = \
                            fun_unpack_costs('VariableCost', t, f,
                                            d5_tpt,
                                            time_vector)
                        list_op_life, unit_op_life = \
                            fun_unpack_costs('OpLife', t, f,
                                            d5_tpt,
                                            time_vector)
                        apply_costs = \
                            {'CapitalCost':deepcopy(list_cap_cost),
                            'VariableCost':deepcopy(list_var_cost)}
    
                        # Now we are ready to estimate the "new fleet":
                        tot_fleet_lst = dict_fleet_k[t][f]
                        res_fleet_lst = dict_resi_cap_trn[t][f]
                        fuel_con_lst = dict_fuel_con[t][f]
    
                        # We need to calculate new_fleet and accum_fleet:
                        accum_fleet_lst = [0 for y in range(len(time_vector))]
                        new_fleet_lst = [0 for y in range(len(time_vector))]
                        for y in range(len(time_vector)):
                            if y == 0:
                                this_new_fleet = tot_fleet_lst[y] - \
                                    res_fleet_lst[y]
                            else:
                                this_new_fleet = tot_fleet_lst[y] - \
                                    res_fleet_lst[y] - accum_fleet_lst[y]
                            # We can store the fleet below:
                            if this_new_fleet >= 0:
                                new_fleet_lst[y] = this_new_fleet
                                # The "this_new_fleet" existis during the vehicles lifetime
                                for y2 in range(y, y+int(list_op_life[y])):
                                    if y2 < len(time_vector):
                                        accum_fleet_lst[y2] += this_new_fleet
                            else:
                                times_neg_new_fleet += 1
                                times_neg_new_fleet_sto.append(this_new_fleet)
                                # if this_new_fleet < -10000:
                                #    print('This is surprising')
                                #    sys.exit()
    
                        '''
                        Remember to apply conversions depending on unit
                        USD/veh
                        USD/pkm
                        USD/1000 km
                        USD/liter
                        USD/kWh
                        USD/kg
                        Ref: # http://w.astro.berkeley.edu/~wright/fuel_energy.html
                        '''
                        if unit_var_cost == 'USD/liter' and 'DIESEL' in f:
                            conv_cons = 38.6*(1e-9)  # from liter to PJ
                        if unit_var_cost == 'USD/liter' and 'GASOLINA' in f:
                            conv_cons = 34.2*(1e-9)  # from liter to PJ
                        if unit_var_cost == 'USD/kWh':
                            conv_cons = 3.6e-9  # from kWh to PJ
                        if unit_var_cost == 'USD/kg':
                            conv_cons = (3.6*33.33)*1e-9  # from kg to PJ
    
                        # Proceed with cost calculations:
                        usd_capex_lst = []
                        usd_fopex_lst = []
                        usd_vopex_lst = []
                        usd_capex_lst_disc = []
                        usd_fopex_lst_disc = []
                        usd_vopex_lst_disc = []
                        for y in range(len(time_vector)):
                            this_year = int(time_vector[y])
                            disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                            add_usd_capex = \
                                new_fleet_lst[y]*list_cap_cost[y]
                            add_usd_fopex = \
                                tot_fleet_lst[y]*list_fix_cost[y]*dict_km[t]/1000
                            add_usd_vopex = \
                                fuel_con_lst[y]*list_var_cost[y]/conv_cons
                            add_usd_capex_disc = add_usd_capex * disc_constant
                            add_usd_fopex_disc = add_usd_fopex * disc_constant
                            add_usd_vopex_disc = add_usd_vopex * disc_constant                                                                                                
                            usd_capex_lst.append(add_usd_capex)
                            usd_fopex_lst.append(add_usd_fopex)
                            usd_vopex_lst.append(add_usd_vopex)
                            usd_capex_lst_disc.append(add_usd_capex_disc)
                            usd_fopex_lst_disc.append(add_usd_fopex_disc)
                            usd_vopex_lst_disc.append(add_usd_vopex_disc)
                        # Bring cost store variable:
                        dict_new_fleet_k[t].update({f:new_fleet_lst})
                        dict_capex_out[t].update({f:usd_capex_lst})
                        dict_fopex_out[t].update({f:usd_fopex_lst})
                        dict_vopex_out[t].update({f:usd_vopex_lst})
                        dict_capex_out_disc[t].update({f:usd_capex_lst_disc})
                        dict_fopex_out_disc[t].update({f:usd_fopex_lst_disc})
                        dict_vopex_out_disc[t].update({f:usd_vopex_lst_disc})
                        # Unpack taxes:
                        apply_taxes = {'list_rate':{},
                                    'list_unit':{}, 'ref_param':{}}
                        for atax in tax_params:
                            list_atax, ref_atax, mult_depr = \
                                fun_unpack_taxes(atax, t, f,
                                                d5_tax,
                                                time_vector)
                            list_atax_unit = []  # tax ready for activity data
    
                            '''
                            "mult_depr" is a constant to multiply the depreciation
                            factor; if it varies across years, the implementation
                            must change.
                            '''
    
                            for y in range(len(time_vector)):
                                try:
                                    if ref_atax == 'CapitalCost*':
                                        ref_atax_call = 'CapitalCost'
                                    else:
                                        ref_atax_call = ref_atax
                                    apply_costs_atax = apply_costs[ref_atax_call][y]
                                except Exception:
                                    apply_costs_atax = 0
                                add_atax_unit = \
                                    apply_costs_atax*list_atax[y]/100
                                list_atax_unit.append(add_atax_unit*mult_depr)
    
                            apply_taxes['list_rate'].update({atax:deepcopy(list_atax)})
                            apply_taxes['list_unit'].update({atax:deepcopy(list_atax_unit)})
                            apply_taxes['ref_param'].update({atax:deepcopy(ref_atax)})
    
                            add_atax_val_lst = []
                            for y in range(len(time_vector)):
                                if ref_atax == 'CapitalCost':
                                    add_atax_val = list_atax_unit[y]*new_fleet_lst[y]
                                elif ref_atax == 'CapitalCost*':
                                    add_atax_val = list_atax_unit[y]*tot_fleet_lst[y]
                                else:  # variable cost
                                    add_atax_val = list_atax_unit[y]*fuel_con_lst[y]/conv_cons
                                add_atax_val_lst.append(add_atax_val)
    
                            dict_tax_out[atax][t][f] = \
                                deepcopy(add_atax_val_lst)
    
                # TM 3) Update the transport vector
                dict_eq_trn_fuels_rev = {}
                for this_f in fuels:
                    this_equivalence = dict_eq_transport_fuels[this_f]
                    if 'HIBRIDO' not in this_f:
                        dict_energy_demand_trn[this_equivalence] = \
                            deepcopy(dict_trn_pj[this_f])
                        dict_eq_trn_fuels_rev.update({this_equivalence:this_f})
                    else:
                        for y in range(len(time_vector)):
                            dict_energy_demand_trn[this_equivalence][y] += \
                                deepcopy(dict_trn_pj[this_f][y])
    
                dict_test_transport_model[this_scen][this_country].update({'original':deepcopy(dict_energy_demand),
                                                                        'adjusted':deepcopy(dict_energy_demand_trn)})
                # We must code a test that reviews the error of transport
                # and simple energy modeling:
                fuel_sets_all = list(dict_energy_demand['Transport'].keys())
                fuel_sets_trn = []
                diff_list_all = [0 for y in range(len(time_vector))]
                diff_dict_all = {}
                sum_list_all = [0 for y in range(len(time_vector))]
                error_list_all = [0 for y in range(len(time_vector))]
                for fs in fuel_sets_all:
                    diff_list_fs = []
                    for y in range(len(time_vector)):
                        this_orig_dem = \
                            dict_energy_demand['Transport'][fs][y]
                        this_orig_adj = \
                            dict_energy_demand_trn[fs][y]
                        this_diff = this_orig_dem-this_orig_adj
                        diff_list_fs.append(this_diff)
                        if round(this_diff, 2) != 0:
                            fuel_sets_trn.append(fs)
                            diff_list_all[y] += this_diff
                            sum_list_all[y] += this_orig_adj
                    diff_dict_all.update({fs: diff_list_fs})
    
                error_list_all = \
                    [100*diff_list_all[n]/v for n, v in enumerate(sum_list_all)]
    
                dict_ed_trn_ref = dict_energy_demand['Transport']
                dict_ed_trn_trn = dict_energy_demand_trn

                # Here we must adjust the dictionaries for compatibility:
                dict_energy_demand['Transport'] = \
                    deepcopy(dict_energy_demand_trn)

                # Add fuel consumption per transport tech
                for t in types_all:
                    dict_energy_demand.update({t:{}})
                    for af in fuels:
                        add_dict_trn_pj_2_list = \
                            dict_trn_pj_2[af][t]
                        dict_energy_demand[t].update({af:add_dict_trn_pj_2_list})

            # if this_scen != 'BAU':
            #    print(this_scen, 'compare the estimated energies')
            #    sys.exit()

            ###########################################################

            # ... here we already have a country's demands, now...
            # 3f) open the *externality* and *emission factors*
            # 'externality':
            mask_cost_ext = ((df5_ext['Country'] == this_country) &
                             (df5_ext['Use_row'] == 'Yes'))
            this_df_cost_externalities = df5_ext.loc[mask_cost_ext]
            this_df_cost_externalities.reset_index(drop=True, inplace=True)
            this_df_cost_externalities_fuels = \
                this_df_cost_externalities['Fuel'].tolist()

            externality_fuels_list = []
            externality_fuels_dict = {}

            for f in range(len(this_df_cost_externalities_fuels)):
                this_fuel = this_df_cost_externalities_fuels[f]
                factor_global_warming = this_df_cost_externalities.iloc[f]['Global warming']
                factor_local_pollution = this_df_cost_externalities.iloc[f]['Local pollution']
                unit_multiplier = this_df_cost_externalities.iloc[f]['Unit multiplier']

                externality_fuels_list.append(this_fuel)
                this_externality_dict = {'Global warming':factor_global_warming,
                                         'Local pollution':factor_local_pollution}
                externality_fuels_dict.update({this_fuel: this_externality_dict})

            # ...this is a good space to store externality and emission data of demand (by fuel) and fuels:
            externalities_globalwarming_demand = {}  # crucial output
            externalities_localpollution_demand = {}  # crucial output

            #### ERASE AFTER TEST
            #### Add black carbon projections:
            #### emissions_demand_black_carbon = deepcopy(emissions_demand)

            demand_tech_list = [
                i for i in list(dict_energy_demand.keys())
                if i not in types_all]
            for tech in demand_tech_list:
                demand_fuel_list = list(dict_energy_demand[tech].keys())
                emissions_demand.update({tech:{}})
                emissions_demand_black_carbon.update({tech:{}})
                emissions_demand_methane.update({tech:{}})
                externalities_globalwarming_demand.update({tech:{}})
                externalities_localpollution_demand.update({tech:{}})
                for fuel in demand_fuel_list:
                    if fuel in emissions_fuels_list:  # store emissions
                        list_emissions_demand = []
                        for y in range(len(time_vector)):
                            add_value = \
                                dict_energy_demand[tech][fuel][y]*emissions_fuels_dict[fuel][y]
                            list_emissions_demand.append(add_value)
                        emissions_demand[tech].update({fuel:list_emissions_demand})
                    if fuel in emissions_2_fuels_list:
                        list_emissions_demand_2 = []
                        for y in range(len(time_vector)):
                            add_value = \
                                dict_energy_demand[tech][fuel][y]*emissions_2_fuels_dict['Standard'][fuel][y]
                            list_emissions_demand_2.append(add_value)
                        emissions_demand_black_carbon[tech].update({fuel:list_emissions_demand_2})
                    if fuel in emissions_3_fuels_list:
                        list_emissions_demand_3 = []
                        for y in range(len(time_vector)):
                            if tech == 'Industry':
                                tech_grab = 'Industry'
                            elif tech in ['Residential', 'Commercial, services, and public', 'Agriculture, fisheries, and mining']:
                                tech_grab = 'Comercial/residencial/agricultura'
                            elif tech == 'Transport':
                                tech_grab = 'Transport_Land'
                            else:
                                tech_grab = 'pass_tech'
                            
                            if tech_grab != 'pass_tech':
                                if fuel in emissions_3_fuels_dict[tech_grab]:
                                    add_value = \
                                        dict_energy_demand[tech][fuel][y]*emissions_3_fuels_dict[tech_grab][fuel][y]/1000
                                else:
                                    add_value = 0
                            else:
                                add_value = 0
                            list_emissions_demand_3.append(add_value)
                        emissions_demand_methane[tech].update({fuel:list_emissions_demand_3})

                    if fuel in externality_fuels_list:  # store externalities
                        list_globalwarming_demand = []
                        list_localpollution_demand = []
                        for y in range(len(time_vector)):
                            try:
                                add_value_globalwarming = \
                                    dict_energy_demand[tech][fuel][y]*externality_fuels_dict[fuel]['Global warming']
                            except Exception:
                                add_value_globalwarming = 0
                            list_globalwarming_demand.append(add_value_globalwarming)

                            try:
                                add_value_localpollution = \
                                    dict_energy_demand[tech][fuel][y]*externality_fuels_dict[fuel]['Local pollution']
                            except Exception:
                                add_value_localpollution = 0
                            list_localpollution_demand.append(add_value_localpollution)

                        externalities_globalwarming_demand[tech].update({fuel:list_globalwarming_demand})
                        externalities_localpollution_demand[tech].update({fuel:list_localpollution_demand})

            ext_by_country.update({this_country:deepcopy(externality_fuels_dict)})

            dict_local_country[this_country].update({'Global warming externalities by demand': externalities_globalwarming_demand})
            dict_local_country[this_country].update({'Local pollution externalities by demand': externalities_localpollution_demand})
            dict_local_country[this_country].update({'Emissions by demand': emissions_demand})
            dict_local_country[this_country].update({'Black carbon emissions by demand [ton]': emissions_demand_black_carbon})
            dict_local_country[this_country].update({'Methane emissions by demand [kt CH4]': emissions_demand_methane})

            # print('Review emissions calculation')
            # sys.exit()

            # Select the existing capacity from the base cap dictionary:
            """
            if this_reg == '2_Central America':
                this_reg_alt = '2_CA'
            else:
                this_reg_alt = this_reg
            """
            dict_base_caps = \
                dict_database['Cap'][this_reg][this_country]  # by pp type

            # Select the "param_related_4"
            param_related_4 = 'Distribution of new electrical energy generation'
            mask_param_related_4 = (df_scen_rc['Parameter'] == param_related_4)
            df_param_related_4 = df_scen_rc.loc[mask_param_related_4]
            df_param_related_4.reset_index(drop=True, inplace=True)

            list_electric_sets = \
                df_param_related_4['Tech'].tolist()

            mask_filt_techs = \
                ((d5_power_techs['Projection'] != 'none') &
                 (d5_power_techs['Parameter'] == 'Net capacity factor'))
            list_electric_sets_2 = \
                d5_power_techs.loc[mask_filt_techs]['Tech'].tolist()

            list_electric_sets_3 = \
                list(set(list(dict_base_caps.keys())) &
                     set(list_electric_sets_2))
            list_electric_sets_3.sort()

            # > Call other auxiliary variables:

            # Select the "param_related_5"
            param_related_5 = '% Imports for consumption'
            mask_param_related_5 = (df_scen_rc['Parameter'] == param_related_5)
            df_param_related_5 = df_scen_rc.loc[mask_param_related_5]
            df_param_related_5.reset_index(drop=True, inplace=True)

            # Select the "param_related_6"
            param_related_6 = '% Exports for production'
            mask_param_related_6 = (df_scen_rc['Parameter'] == param_related_6)
            df_param_related_6 = df_scen_rc.loc[mask_param_related_6]
            df_param_related_6.reset_index(drop=True, inplace=True)

            # Select the "param_related_7"
            param_related_7 = 'Fuel prices'
            mask_param_related_7 = (df_scen_rc['Parameter'] == param_related_7)
            df_param_related_7 = df_scen_rc.loc[mask_param_related_7]
            df_param_related_7.reset_index(drop=True, inplace=True)

            # ...proceed with the interpolation of fuel prices:
            fuel_list_local = df_param_related_7['Fuel'].tolist()

            for this_fuel in fuel_list_local:
                fuel_idx_7 = df_param_related_7['Fuel'].tolist().index(this_fuel)
                this_fuel_price_projection = df_param_related_7.loc[fuel_idx_7, 'projection']
                this_fuel_price_value_type = df_param_related_7.loc[fuel_idx_7, 'value']

                if (this_fuel_price_projection == 'flat' and this_fuel_price_value_type == 'constant'):
                    for y in range(len(time_vector)):
                        df_param_related_7.loc[fuel_idx_7, time_vector[y]] = \
                            float(df_param_related_7.loc[fuel_idx_7, time_vector[0]])

                elif this_fuel_price_projection == 'Percent growth of incomplete years':
                    growth_param = df_param_related_7.loc[ fuel_idx_7, 'value' ]
                    for y in range(len(time_vector)):
                        value_field = df_param_related_7.loc[ fuel_idx_7, time_vector[y] ]
                        if math.isnan(value_field) == True:
                            df_param_related_7.loc[ fuel_idx_7, time_vector[y] ] = \
                                round(df_param_related_7.loc[ fuel_idx_7, time_vector[y-1] ]*(1 + growth_param/100), 4)

            # Select the "param_related_8"
            param_related_8 = 'Planned new capacity'
            mask_param_related_8 = (df_scen_rc['Parameter'] == param_related_8)
            df_param_related_8 = df_scen_rc.loc[mask_param_related_8]
            df_param_related_8.reset_index(drop=True, inplace=True)

            # Select the "param_related_9"
            param_related_9 = 'Phase-out capacity'
            mask_param_related_9 = (df_scen_rc['Parameter'] == param_related_9)
            df_param_related_9 = df_scen_rc.loc[mask_param_related_9]
            df_param_related_9.reset_index(drop=True, inplace=True)

            # Select the "param_related_10"
            param_related_10 = 'Capacity factor change'
            mask_param_related_10 = (df_scen_rc['Parameter'] == param_related_10)
            df_param_related_10 = df_scen_rc.loc[mask_param_related_10]
            df_param_related_10.reset_index(drop=True, inplace=True)

            # Select the existing transformation inputs information (receive negative values only):
            dict_base_transformation = \
                dict_database['EB'][this_reg][this_country_2]['Total transformation']['Power plants']  # by fuel (input)
            dict_base_transformation_2 = \
                dict_database['EB'][this_reg][this_country_2]['Total transformation']['Self-producers']  # by fuel (input)

            base_transformation_fuels = \
                list(dict_base_transformation.keys()) + \
                list(dict_base_transformation_2.keys())
            base_transformation_fuels = list(set(base_transformation_fuels))
            base_transformation_fuels.sort()
            base_electric_fuels_use = []
            base_electric_fuel_use = {}
            base_electric_production = {}
            base_electric_production_pps = {}  # power plants
            base_electric_production_sps = {}  # self-producers

            # ...search here if the fuels have negative values, which indicates whether we have a reasonable match
            for btf in base_transformation_fuels:
                try:
                    btf_value_1 = dict_base_transformation[btf][base_year]
                except Exception:
                    btf_value_1 = 0
                try:
                    btf_value_2 = dict_base_transformation_2[btf][base_year]
                except Exception:
                    btf_value_2 = 0

                btf_value = btf_value_1 + btf_value_2  # // ignore self-producers
                if btf_value < 0:
                    base_electric_fuels_use.append(btf)
                    base_electric_fuel_use.update({btf:-1*btf_value})
                if btf_value > 0:
                    base_electric_production.update({btf:btf_value})
                    base_electric_production_pps.update({btf:btf_value_1/btf_value})
                    base_electric_production_sps.update({btf:btf_value_2/btf_value})

            # ...extract losses and self-consumption
            electricity_losses = \
                dict_database['EB'][this_reg][this_country_2]['Losses']['none']['Electricity'][base_year]
            electricity_self_consumption = \
                dict_database['EB'][this_reg][this_country_2]['Self-consumption']['none']['Electricity'][base_year]
            electricity_imports = \
                dict_database['EB'][this_reg][this_country_2]['Total supply']['Imports']['Electricity'][base_year]
            electricity_exports = \
                dict_database['EB'][this_reg][this_country_2]['Total supply']['Exports']['Electricity'][base_year]

            # ...create imports and exports list:
            electricity_losses_list = []
            electricity_self_consumption_list = []
            electricity_imports_list = []
            electricity_exports_list = []

            losses_share = \
                electricity_losses/dict_energy_demand_by_fuel['Electricity'][0]
            self_consumption_share = \
                electricity_self_consumption/dict_energy_demand_by_fuel['Electricity'][0]
            imports_share = \
                electricity_imports/dict_energy_demand_by_fuel['Electricity'][0]
            exports_share = \
                electricity_exports/dict_energy_demand_by_fuel['Electricity'][0]  # this will be negative

            # ...here we must manipulate the limit to the losses!
            # Select the "param_related_11"
            param_related_11 = 'Max losses'
            mask_param_related_11 = (df_scen_rc['Parameter'] == param_related_11)
            df_param_related_11 = df_scen_rc.loc[mask_param_related_11]
            df_param_related_11.reset_index(drop=True, inplace=True)

            maxloss_projection = df_param_related_11.iloc[0]['projection']
            maxloss_baseyear_str = df_param_related_11.loc[0, time_vector[0]]
            loss_vector = []

            if maxloss_projection == 'flat' and maxloss_baseyear_str == 'endogenous':
                for y in range(len(time_vector)):
                    loss_vector.append(losses_share)

            if maxloss_projection == 'interpolate' and maxloss_baseyear_str == 'endogenous':
                this_known_loss_vals = []
                for y in range(len(time_vector)):
                    if y == 0:
                        this_known_loss_vals.append(losses_share)
                    elif type(df_param_related_11.loc[0, time_vector[y]]) is int:
                        suggested_maxloss = df_param_related_11.loc[0, time_vector[y]]/100
                        if suggested_maxloss < losses_share:
                            this_known_loss_vals.append(suggested_maxloss)
                        else:
                            this_known_loss_vals.append(losses_share)
                    else:
                        this_known_loss_vals.append('')

                loss_vector = \
                    interpolation_to_end(time_vector, ini_simu_yr,
                                         this_known_loss_vals, 'ini',
                                         this_scen, '')

            # ... now apply the appropriate loss vector:
            for y in range(len(time_vector)):
                electricity_losses_list.append(loss_vector[y]*dict_energy_demand_by_fuel['Electricity'][y])
                electricity_self_consumption_list.append(self_consumption_share*dict_energy_demand_by_fuel['Electricity'][y])
                electricity_imports_list.append(imports_share*dict_energy_demand_by_fuel['Electricity'][y])
                electricity_exports_list.append(exports_share*dict_energy_demand_by_fuel['Electricity'][y])

            # 3g) Here we must call some inputs to make the model adjust to the desired electrical demands
            param_related_12 = 'RE TAG'
            mask_12 = (df_scen_rc['Parameter'] == param_related_12)
            df_param_related_12 = df_scen_rc.loc[mask_12]
            df_param_related_12.reset_index(drop=True, inplace=True)

            reno_targets_exist = False
            if len(df_param_related_12.index.tolist()) > 0:
                reno_targets_exist = True
                reno_target_list = df_param_related_12[time_vector].iloc[0].tolist()

            # 3h) obtain the required *new electrical capacity by pp*, *electricity production by pp*, *fuel consumption by pp*
            #     to supply the electricity demand:
            # NOTE: we have to subtract the elements
            electrical_demand_to_supply = [0 for y in range(len(time_vector))]
            for y in range(len(time_vector)):
                electrical_demand_to_supply[y] = \
                    dict_energy_demand_by_fuel['Electricity'][y] + \
                    electricity_losses_list[y] + \
                    electricity_self_consumption_list[y] - \
                    electricity_imports_list[y] - \
                    electricity_exports_list[y]

            # ...here, 'Total' is the net energy loss in transformation
            base_electric_prod = base_electric_production['Electricity']
            base_electric_use_fuels = \
                deepcopy(base_electric_fuels_use)
            #base_electric_use_fuels.remove('Total')  # not needed anymore
            #base_electric_use_fuels.remove('Total primary sources')

            # ...we can extract the fuels we use in our technological options:
            used_fuel_list = []
            for tech in list_electric_sets:
                used_fuel = tech.split('_')[-1]
                if used_fuel not in used_fuel_list:
                    used_fuel_list.append(used_fuel)

            # ...here we need to iterate across the list of technologies and find the base distribution of electricity production:
            res_energy_shares = {}
            res_energy_sum_1 = 0
            res_energy_sum_2 = 0
            res_energy_sum_3 = 0
            store_percent = {}
            store_percent_rem = {}
            store_use_cap = {}
            store_res_energy = {}
            store_res_energy_all = [0 for y in range(len(time_vector))]

            # Blank capacity factors:
            cf_by_tech = {}
            forced_newcap_energy_by_tech = {}
            forced_newcap_by_tech = {}
            forced_newcap_energy_all = [0 for y in range(len(time_vector))]

            accum_forced_newcap_by_tech = {}
            # ...this is the first previous loop:
            # 1st power sector loop
            for tech in list_electric_sets_3:

                store_use_cap.update({tech:[0 for y in range(len(time_vector))]})
                store_res_energy.update({tech:[0 for y in range(len(time_vector))]})
                forced_newcap_energy_by_tech.update({tech:[0 for y in range(len(time_vector))]})
                forced_newcap_by_tech.update({tech:[0 for y in range(len(time_vector))]})

                accum_forced_newcap_by_tech.update({tech:[0 for y in range(len(time_vector))]})

                tech_idx = df_param_related_4['Tech'].tolist().index(tech)

                # Extract planned and phase out capacities if they exist:
                try:
                    tech_idx_8 = df_param_related_8['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_8 = ''
                try:
                    tech_idx_9 = df_param_related_9['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_9 = ''
                tech_counter += 1

                # ...here extract the technical characteristics of the power techs
                mask_this_tech = \
                    (d5_power_techs['Tech'] == tech)
                this_tech_df_cost_power_techs = \
                    deepcopy(d5_power_techs.loc[mask_this_tech])

                # CAU
                mask_cau = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'CAU')
                df_mask_cau = \
                    deepcopy(this_tech_df_cost_power_techs.loc[mask_cau])
                df_mask_cau_unitmult = df_mask_cau.iloc[0]['Unit multiplier']
                df_mask_cau_proj = df_mask_cau.iloc[0]['Projection']
                list_tech_cau = []
                if df_mask_cau_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_cau.iloc[0][y]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)
                if df_mask_cau_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_cau.iloc[0][time_vector[0]]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)

                # Net capacity factor
                mask_cf = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Net capacity factor')
                df_mask_cf = \
                    deepcopy(this_tech_df_cost_power_techs.loc[mask_cf])
                df_mask_cf_unitmult = df_mask_cf.iloc[0]['Unit multiplier']
                df_mask_cf_proj = df_mask_cf.iloc[0]['Projection']
                list_tech_cf = []
                if df_mask_cf_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_cf.iloc[0][y]*df_mask_cf_unitmult
                        list_tech_cf.append(add_value)
                if df_mask_cf_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_cf.iloc[0][time_vector[0]]*df_mask_cf_unitmult
                        list_tech_cf.append(add_value)

                # ...here calculate the capacity factor if existent:
                this_tech_base_cap = dict_base_caps[tech][base_year]

                # Here we call the capacity factors from an external analysis:
                if df_mask_cf_proj in ['normalize_by_endogenous', 'flat_endogenous']:
                    ###########################################################
                    #2021
                    mask_cf_tech_2021 = \
                        ((df4_cfs['Capacity Set'] == tech) & \
                         (df4_cfs['Year'] == 2021) & \
                         (df4_cfs['Country'] == this_country))
                    this_df4_cfs_2021 = \
                        deepcopy(df4_cfs.loc[mask_cf_tech_2021])
                    this_df4_cfs_2021.reset_index(inplace=True, drop=True)

                    ###########################################################
                    #2021 avg
                    if len(this_df4_cfs_2021.index.tolist()) != 0:
                        this_cf_yrl_2021 = this_df4_cfs_2021['Capacity factor'].iloc[0]
                        if this_cf_yrl_2021 < 1:
                            this_cf_yrl_2021_add = this_cf_yrl_2021
                            this_cf_cnt_2021 = 1
                        else:
                            this_cf_yrl_2021_add, this_cf_cnt_2021 = 0, 0
                    else:
                        this_cf_yrl_2021_add, this_cf_cnt_2021 = 0, 0

                    this_cf_list = []
                    for y in time_vector:
                        mask_cf_tech = \
                            ((df4_cfs['Capacity Set'] == tech) & \
                             (df4_cfs['Year'] == time_vector[0]) & \
                             (df4_cfs['Country'] == this_country)
                            )

                        this_df4_cfs = \
                            deepcopy(df4_cfs.loc[mask_cf_tech])
                        this_df4_cfs.reset_index(inplace=True, drop=True)

                        if len(this_df4_cfs.index.tolist()) != 0 or len(this_df4_cfs_2021.index.tolist()) != 0:  # proceed

                            if len(this_df4_cfs.index.tolist()) != 0:
                                this_cf_yrl = this_df4_cfs['Capacity factor'].iloc[0]
                            else:
                                this_cf_yrl = 99
                                print('This capacity factor condition is not possible for this version. Review inputs and structure.')
                                sys.exit()

                            # Select the appropiate historic capacity factors:
                            this_cf = this_cf_yrl

                        else:
                            this_cf = 0  # this means we must use a default CF // to be selected

                        this_cf_list.append(this_cf)
                        #
                    #
                #
                # Here we define the capacity factors endogenously:
                if df_mask_cf_proj == 'normalize_by_endogenous':
                    y_counter_i = 0
                    for y in time_vector:
                        this_cf = this_cf_list[y_counter_i]
                        if this_cf != 0:  # this applies only if energy production is existent
                            mult_value = \
                                df_mask_cf.iloc[0][y]/df_mask_cf.iloc[0][time_vector[0]]*df_mask_cf_unitmult
                            add_value = mult_value*this_cf
                        else:  # this applies when the technology is non-existent
                            add_value = \
                                df_mask_cf.iloc[0][y]*df_mask_cf_unitmult
                        list_tech_cf.append(add_value)
                        y_counter_i += 1

                if df_mask_cf_proj == 'flat_endogenous':
                    y_counter_i = 0
                    for y in time_vector:
                        this_cf = this_cf_list[y_counter_i]
                        if this_cf != 0:  # this applies only if energy production is existent
                            add_value = \
                                this_cf*df_mask_cf_unitmult
                        else:
                            add_value = \
                                df_mask_cf.iloc[0][time_vector[0]]*df_mask_cf_unitmult
                        list_tech_cf.append(add_value)
                        y_counter_i += 1

                # ...calculate the base capacity/energy relationships for base year:
                # This is equivalent to parameter 9:
                this_tech_phase_out_cap = [0 for y in range(len(time_vector))]  # should be an input
                if tech_idx_9 != '':
                    for y in range(len(time_vector)):
                        this_tech_phase_out_cap[y] = \
                            df_param_related_9.loc[tech_idx_9, time_vector[y]]/1000  # in GW (MW to GW)

                res_energy_base_1 = this_tech_base_cap*8760*list_tech_cf[0]/1000  # MW to GW // CF
                res_energy_sum_1 += res_energy_base_1  # just for the base year

                cf_by_tech.update({tech:deepcopy(list_tech_cf)})

                # ...here store the potential "res_energy" and "use_cap":
                residual_cap_intermediate = \
                    [0 for y in range(len(time_vector))]

                accum_forced_cap_vector = [0 for y in range(len(time_vector))]
                forced_cap_vector = [0 for y in range(len(time_vector))]
                for y in range(len(time_vector)):

                    # ...here we need to store the energy production from forced new capacity
                    if tech_idx_8 != '':
                        forced_cap = \
                            df_param_related_8.loc[tech_idx_8, time_vector[y]]/1000  # in GW (MW to GW)
                    else:
                        forced_cap = 0

                    forced_cap_vector[y] += forced_cap

                    if y == 0:
                        residual_cap_intermediate[y] = this_tech_base_cap/1000
                    else:
                        residual_cap_intermediate[y] = \
                            residual_cap_intermediate[y-1] - this_tech_phase_out_cap[y]

                    # ...here we must add the accumulated planned new capacity:
                    for y_past in range(y+1):
                        accum_forced_cap_vector[y] += forced_cap_vector[y_past]
                    accum_forced_cap = accum_forced_cap_vector[y]

                    use_cap = residual_cap_intermediate[y]  # cap with calibrated values
                    res_energy = \
                        (use_cap)*list_tech_cau[y]*list_tech_cf[y]  # energy with calibrated values

                    store_use_cap[tech][y] += deepcopy(use_cap)
                    store_res_energy[tech][y] += deepcopy(res_energy)
                    store_res_energy_all[y] += deepcopy(res_energy)

                    forced_newcap_energy_by_tech[tech][y] += \
                        deepcopy(accum_forced_cap*list_tech_cau[y]*list_tech_cf[y])
                    forced_newcap_by_tech[tech][y] += \
                        deepcopy(forced_cap)
                    forced_newcap_energy_all[y] += \
                        deepcopy(accum_forced_cap*list_tech_cau[y]*list_tech_cf[y])

                    accum_forced_newcap_by_tech[tech][y] += \
                        deepcopy(accum_forced_cap)

            # Store the energy of the base year:
            store_res_energy_orig = deepcopy(store_res_energy)

            # ...this is the second previous loop:
            # 2nd power sector loop
            for tech in list_electric_sets_3:
                tech_idx = df_param_related_4['Tech'].tolist().index(tech)

                # ...let's bring to the front the base shares:
                this_tech_base_cap = dict_base_caps[tech][base_year]
                list_tech_cf_loc = cf_by_tech[tech]

                res_energy_base_1 = this_tech_base_cap*8760*list_tech_cf_loc[0]/1000  # MW to GW
                energy_dist_1 = res_energy_base_1/res_energy_sum_1

                # ...here we must take advantage of the loop to define the shares we will use:
                check_percent = False
                if list(set(df_param_related_4['value']))[0] == 'percent':
                    check_percent = True
                this_tech_dneeg_df_param_related = df_param_related_4.iloc[tech_idx]
                this_tech_dneeg_value_type = this_tech_dneeg_df_param_related['value']
                this_tech_dneeg_known_vals_raw = []
                this_tech_dneeg_known_vals = []
                this_tech_dneeg_known_vals_count = 0
                if check_percent is True:  # is compatible with "interpolate"
                    for y in time_vector:
                        add_value = \
                            this_tech_dneeg_df_param_related[y]

                        this_tech_dneeg_known_vals_raw.append(add_value)
                        # if str(y) == str(base_year):
                            # this_tech_dneeg_known_vals.append(energy_dist_1)  # this had been zero before
                        # elif type(add_value) is int or type(add_value) is float:
                        if type(add_value) is int or isinstance(add_value, (float, np.floating, int)):
                            if math.isnan(add_value) is False:
                                this_tech_dneeg_known_vals.append(add_value/100)
                                this_tech_dneeg_known_vals_count += 1
                            elif np.isnan(add_value):
                                this_tech_dneeg_known_vals.append('')
                            else:
                                pass
                        else:
                            this_tech_dneeg_known_vals.append('')

                    if add_value != 'rem':
                        this_tech_dneeg_vals = \
                            interpolation_to_end(time_vector, ini_simu_yr,
                                                 this_tech_dneeg_known_vals,
                                                 'last', this_scen, 'power')

                        if this_tech_dneeg_known_vals_count == len(this_tech_dneeg_vals):
                            this_tech_dneeg_vals = deepcopy(this_tech_dneeg_known_vals)

                    # if this_scen == 'NDCPLUS' and tech == 'PP_PV Utility_Solar':
                    #    print('review dneeg')
                    #    sys.exit()

                    else:  # we need to fill later
                        this_tech_dneeg_vals = \
                            [0 for y in range(len(time_vector))]
                        store_percent_rem.update({tech:this_tech_dneeg_vals})
                    store_percent.update({tech:this_tech_dneeg_vals})


                    # if tech == 'PP_PV Utility_Solar':
                    #    print('review this')
                    #    sys.exit()


            # ...here we need to run the remainder if necessary:
            if check_percent is True:
                tech_rem = list(store_percent_rem.keys())[0]
                oneminus_rem_list = store_percent_rem[tech_rem]
                for tech in list_electric_sets_3:
                    if tech != tech_rem:
                        for y in range(len(time_vector)):
                            oneminus_rem_list[y] += store_percent[tech][y]

                for y in range(len(time_vector)):
                    store_percent[tech_rem][y] = 1-oneminus_rem_list[y]

                # if this_scen == 'NDCPLUS':
                #    print('review this please')
                #    sys.exit()

            # ...we should store the BAU's "store percent" approach:
            if 'BAU' in this_scen:
                store_percent_BAU.update({this_country:deepcopy(store_percent)})

            # ...below, we need to apply an adjustment factor to match supply and demand:
            adjustment_factor = base_electric_prod/store_res_energy_all[0]
            for y in range(len(time_vector)):
                store_res_energy_all[y] *= adjustment_factor
                for tech in list_electric_sets_3:
                    store_res_energy[tech][y] *= adjustment_factor
                    cf_by_tech[tech][y] *= adjustment_factor

            # ...here we need to iterate across the list of technologies:
            fuel_use_electricity = {}  # crucial outputs
            externalities_globalwarming_electricity = {}  # crucial output
            externalities_localpollution_electricity = {}  # crucial output
            emissions_electricity = {}  # crucial output
            total_capacity = {}  # crucial output
            residual_capacity = {}  # crucial output
            new_capacity = {}  # crucial output

            # ...capacity disaggregation:
            cap_new_unplanned = {}
            cap_new_planned = {}
            cap_phase_out = {}

            # ...debugging dictionaries:
            ele_prod_share = {}
            ele_endogenous = {}
            cap_accum = {}

            total_production = {}  # crucial output
            new_production = {}
            capex = {}  # crucial output
            fopex = {}  # crucial output
            vopex = {}  # crucial output
            gcc = {}  # crucial output

            # Create dictionaries to store data from printing
            idict_u_capex = {}
            idict_u_fopex = {}
            idict_u_vopex = {}
            idict_u_gcc = {}
            idict_cau = {}
            idict_net_cap_factor = {}
            idict_hr = {}
            idict_oplife = {}

            # ...create a variable to represent lock-in decisions
            accum_cap_energy_vector = [0 for y in range(len(time_vector))]

            # 3rd power sector loop spotting conditions of surpassing capacity potential
            # ...we copy some of the elements of the 4th power sector loop
            # ...the unique sets with restriction:
            restriction_sets = list(set(df4_caps_rest['Set'].tolist()))

            # ...define the "adjustment_fraction" to recalculate the production shares
            adjustment_fraction = {}

            unit_gen_cost_dict = {}

            for tech in list_electric_sets_3:
                adjustment_fraction.update({tech:1})

                # ...extract the indices of the technology to define capacity:
                tech_idx = df_param_related_4['Tech'].tolist().index(tech)

                this_tech_dneeg_df_param_related = df_param_related_4.iloc[tech_idx]
                this_tech_dneeg_apply_type = this_tech_dneeg_df_param_related['apply_type']
                this_tech_dneeg_projection = this_tech_dneeg_df_param_related['projection']
                this_tech_dneeg_value_type = this_tech_dneeg_df_param_related['value']

                try:
                    tech_idx_8 = df_param_related_8['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_8 = ''
                try:
                    tech_idx_9 = df_param_related_9['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_9 = ''
                tech_counter += 1

                # This is equivalent to parameter 8:
                this_tech_forced_new_cap = [0 for y in range(len(time_vector))]  # should be an input
                if tech_idx_8 != '':
                    for y in range(len(time_vector)):
                        this_tech_forced_new_cap[y] = \
                            df_param_related_8.loc[tech_idx_8, time_vector[y]]/1000

                # This is equivalent to parameter 9:
                this_tech_phase_out_cap = [0 for y in range(len(time_vector))]  # should be an input
                if tech_idx_9 != '':
                    for y in range(len(time_vector)):
                        this_tech_phase_out_cap[y] = \
                            df_param_related_9.loc[tech_idx_9, time_vector[y]]/1000

                # ...extract the capacity restriction (by 2050)
                mask_restriction = ((df4_caps_rest['Set'] == tech) &\
                                    (df4_caps_rest['Country'] == this_country))
                restriction_value_df = df4_caps_rest.loc[mask_restriction]
                restriction_value_df.reset_index(drop=True, inplace=True)
                if len(restriction_value_df.index.tolist()) != 0:
                    restriction_value = restriction_value_df['Restriction (MW)'].iloc[0]/1000
                else:
                    restriction_value = 999999999

                # ...extract the "Net capacity factor"
                list_tech_cf = cf_by_tech[tech]

                # ...extract the CAU
                mask_cau = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'CAU')
                df_mask_cau = \
                    deepcopy(this_tech_df_cost_power_techs.loc[mask_cau])
                df_mask_cau_unitmult = df_mask_cau.iloc[0]['Unit multiplier']
                df_mask_cau_proj = df_mask_cau.iloc[0]['Projection']
                list_tech_cau = []
                if df_mask_cau_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_cau.iloc[0][y]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)
                if df_mask_cau_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_cau.iloc[0][time_vector[0]]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)

                # Variable FOM
                mask_vfom = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Variable FOM')
                df_mask_vfom = \
                    this_tech_df_cost_power_techs.loc[mask_vfom]
                df_mask_vfom_unitmult = df_mask_vfom.iloc[0]['Unit multiplier']
                df_mask_vfom_proj = df_mask_vfom.iloc[0]['Projection']
                list_tech_vfom = []
                if df_mask_vfom_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_vfom.iloc[0][y]*df_mask_vfom_unitmult
                        list_tech_vfom.append(add_value)
                if df_mask_vfom_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_vfom.iloc[0][time_vector[0]]*df_mask_vfom_unitmult
                        list_tech_vfom.append(add_value)

                # Heat Rate
                mask_hr = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Heat Rate')
                df_mask_hr = \
                    this_tech_df_cost_power_techs.loc[mask_hr]
                if len(df_mask_hr.index.tolist()) != 0:
                    df_mask_hr_unitmult = df_mask_hr.iloc[0]['Unit multiplier']
                    df_mask_hr_proj = df_mask_hr.iloc[0]['Projection']
                    list_tech_hr = []
                    if df_mask_hr_proj == 'user_defined':
                        for y in time_vector:
                            add_value = \
                                df_mask_hr.iloc[0][y]*df_mask_hr_unitmult
                            list_tech_hr.append(add_value)
                    if df_mask_hr_proj == 'flat':
                        for y in range(len(time_vector)):
                            add_value = \
                                df_mask_hr.iloc[0][time_vector[0]]*df_mask_hr_unitmult
                            list_tech_hr.append(add_value)
                else:
                    list_tech_hr = [0 for y in range(len(time_vector))]

                # Store the unit generation cost:               
                unit_gen_cost = [a * b for a, b in zip(list_tech_hr, list_tech_vfom)]  # most likely in $/PJ
                unit_gen_cost_dict.update({tech: deepcopy(unit_gen_cost)})

                # ...define some intermediate variables; some are redefined later
                this_tech_accum_new_cap_unplanned = [0 for y in range(len(time_vector))]
                accum_cap_energy_vector = [0 for y in range(len(time_vector))]

                this_tech_accum_new_cap = [0 for y in range(len(time_vector))]
                this_tech_new_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_new_cap_unplanned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)
                this_tech_new_cap_planned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)

                this_tech_residual_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_prod = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_new_prod = [0 for y in range(len(time_vector))]

                # ...apply the capacity estimation algorithm:
                if (this_tech_dneeg_projection == 'interpolate' and
                        this_tech_dneeg_value_type == 'percent' and
                        this_tech_dneeg_apply_type == 'all') or (
                        this_tech_dneeg_projection == 'keep_proportions'):

                    for y in range(len(time_vector)):
                        # calculate the energy that the accumulated unplanned capacity supplies (this is actually unused)
                        if y != 0:
                            this_tech_accum_cap_energy = \
                                list_tech_cau[y]*list_tech_cf[y]*this_tech_accum_new_cap_unplanned[y-1]
                            accum_cap_energy_vector[y] += \
                                this_tech_accum_cap_energy

                        # ...estimate the energy requirement
                        new_req_energy = \
                            electrical_demand_to_supply[y] \
                            - forced_newcap_energy_all[y] \
                            - store_res_energy_all[y] \

                        if new_req_energy < 0:
                            count_under_zero += 1
                            new_req_energy = 0

                        use_cap = store_use_cap[tech][y]  # cap with calibrated values
                        res_energy = store_res_energy[tech][y]  # energy that calibrated value produces (not considering new capacity)

                        planned_energy = forced_newcap_energy_by_tech[tech][y]  # energy from planned plants

                        if this_tech_dneeg_projection == 'keep_proportions':  # do not mix things up
                            if y == 0:
                                res_energy_base = this_tech_base_cap*8760*list_tech_cf[0]/1000  # MW to GW
                                res_energy_sum = res_energy_sum_1
                            else:
                                res_energy_base = this_tech_base_cap*8760*list_tech_cf[y]/1000  # MW to GW
                                res_energy_sum = res_energy_sum_1
                            energy_dist = res_energy_base/res_energy_sum  # distribution of energy for "keep_proportions"
                            new_energy_assign = new_req_energy*energy_dist
                        else:
                            new_energy_assign = new_req_energy*store_percent[tech][y]

                        this_tech_total_prod[y] = deepcopy(res_energy + planned_energy + new_energy_assign)
                        if y != 0:
                            this_tech_new_prod[y] = \
                                this_tech_total_prod[y] \
                                - this_tech_total_prod[y-1]
                            if this_tech_new_prod[y] < 0:
                                this_tech_new_prod[y] = 0

                        # Remembering how much should be subtracted
                        if y == 0:
                            subtract_new_cap = 0
                        else:
                            subtract_new_cap = this_tech_accum_new_cap_unplanned[y-1]

                        # Estimating unplanned capacity
                        if list_tech_cau[y]*list_tech_cf[y] != 0: #try:
                            new_cap_unplanned = \
                                new_energy_assign/(list_tech_cau[y]*list_tech_cf[y]) - \
                                subtract_new_cap
                        else:
                            print('division by zero', 'interpolate')
                            sys.exit()

                        # This is a filter to avoid inconsistencies:
                        if new_cap_unplanned < 0:
                            new_cap_unplanned = 0

                        new_cap = new_cap_unplanned + forced_newcap_by_tech[tech][y]

                        # Update the residual capacity
                        if y == 0:
                            residual_cap = use_cap
                            this_tech_total_cap[y] = use_cap
                            this_tech_residual_cap[y] = use_cap
                        else:
                            residual_cap = this_tech_residual_cap[y-1] - this_tech_phase_out_cap[y]
                            this_tech_total_cap[y] += residual_cap + this_tech_accum_new_cap[y-1]
                            this_tech_residual_cap[y] = residual_cap

                        # Adjust accumulated new capacities
                        if y == 0:
                            this_tech_accum_new_cap[y] = new_cap
                            this_tech_accum_new_cap_unplanned[y] = new_cap_unplanned
                        else:
                            this_tech_accum_new_cap[y] = \
                                new_cap + this_tech_accum_new_cap[y-1]
                            this_tech_accum_new_cap_unplanned[y] = \
                                new_cap_unplanned + this_tech_accum_new_cap_unplanned[y-1]

                        this_tech_new_cap[y] += new_cap
                        this_tech_total_cap[y] += new_cap

                        this_tech_new_cap_unplanned[y] = deepcopy(new_cap_unplanned)
                        this_tech_new_cap_planned[y] = deepcopy(forced_newcap_by_tech[tech][y])

                    # ...below we must assess if we need to recalculate the shares:
                    if this_tech_total_cap[-1] < restriction_value:
                        pass  # there is no need to do anything else
                    elif this_tech_total_cap[-1] <= 0:
                        pass  # there is no need to specify anything
                    else:  # we must re-estimate the shares
                        this_adjustment_fraction = \
                            (restriction_value-sum(this_tech_new_cap_planned)) / (this_tech_total_cap[-1] - this_tech_total_cap[0])
                        adjustment_fraction[tech] = this_adjustment_fraction

            # With the adjustment factor complete, we proceed to update:
            sum_adjustment = [0 for y in range(len(time_vector))]
            cum_adjusted = [0 for y in range(len(time_vector))]
            old_share = {}
            new_share = {}
            for tech in list_electric_sets_3:
                old_share.update({tech: [0 for y in range(len(time_vector))]})
                new_share.update({tech: [0 for y in range(len(time_vector))]})
                for y in range(len(time_vector)):
                    if adjustment_fraction[tech] < 1:
                        new_share[tech][y] = adjustment_fraction[tech]*store_percent[tech][y]
                        cum_adjusted[y] += adjustment_fraction[tech]*store_percent[tech][y]
                    else:
                        sum_adjustment[y] += store_percent[tech][y]
                    old_share[tech][y] = store_percent[tech][y]
            for tech in list_electric_sets_3:
                for y in range(len(time_vector)):
                    if adjustment_fraction[tech] < 1:
                        pass  # there is nothing to do here; previous loop was done
                    else:
                        new_share[tech][y] = store_percent[tech][y]*((1-cum_adjusted[y])/(sum_adjustment[y]))

            old_share_sum = [0 for y in range(len(time_vector))]
            new_share_sum = [0 for y in range(len(time_vector))]
            for tech in list_electric_sets_3:
                for y in range(len(time_vector)):
                    old_share_sum[y] += old_share[tech][y]
                    new_share_sum[y] += new_share[tech][y]

            store_percent_freeze = deepcopy(store_percent)
            for tech in list_electric_sets_3:
                for y in range(len(time_vector)):
                    if time_vector[y] <= ini_simu_yr:  # this is to control the first few years
                        pass
                    elif old_share[tech][y] == 0:
                        pass
                    else:
                        store_percent[tech][y] *= \
                            new_share[tech][y]/old_share[tech][y]

            # Now let us sort across technologies to have the sorted cost
            # Create the dictionaries per year with sorted techs
            yearly_sorted_tech_costs = {}
            for year in range(len(next(iter(unit_gen_cost_dict.values())))):  # Using the length of one of the tech lists to determine the number of years
                sorted_techs = sorted(unit_gen_cost_dict.keys(), key=lambda tech: unit_gen_cost_dict[tech][year], reverse=True)
                yearly_sorted_tech_costs[year] = sorted_techs

            # 4th power sector loop
            list_electric_sets_3.sort()
            thermal_filter_out = [
                'PP_Thermal_Diesel',
                'PP_Thermal_Fuel oil',
                'PP_Thermal_Coal',
                'PP_Thermal_Crude',
                'PP_Thermal_Natural Gas']
            list_electric_sets_3_shuffle = [item for item in list_electric_sets_3 if item not in thermal_filter_out]
            list_electric_sets_3_shuffle += thermal_filter_out

            # Remove the specified item
            item_to_move = 'PP_PV Utility+Battery_Solar'
            list_electric_sets_3_shuffle.remove(item_to_move)

            # Append it to the end
            list_electric_sets_3_shuffle.append(item_to_move)
            
            thermal_reductions_store = {}
            thermal_reductions_order = {}
            
            tech_counter = 0
            for tech in list_electric_sets_3_shuffle:

                tech_idx = df_param_related_4['Tech'].tolist().index(tech)

                # Extract planned and phase out capacities if they exist:
                try:
                    tech_idx_8 = df_param_related_8['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_8 = ''
                try:
                    tech_idx_9 = df_param_related_9['Tech'].tolist().index(tech)
                except Exception:
                    tech_idx_9 = ''
                tech_counter += 1

                # ...here extract the technical characteristics of the power techs
                mask_this_tech = \
                    (d5_power_techs['Tech'] == tech)
                this_tech_df_cost_power_techs = \
                    d5_power_techs.loc[mask_this_tech]

                # ...we can extract one parameter at a time:
                # CAPEX
                mask_capex = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'CAPEX')
                df_mask_capex = \
                    this_tech_df_cost_power_techs.loc[mask_capex]
                df_mask_capex_unitmult = df_mask_capex.iloc[0]['Unit multiplier']
                df_mask_capex_proj = df_mask_capex.iloc[0]['Projection']
                list_tech_capex = []
                if df_mask_capex_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_capex.iloc[0][y]*df_mask_capex_unitmult
                        list_tech_capex.append(add_value)
                if df_mask_capex_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_capex.iloc[0][time_vector[0]]*df_mask_capex_unitmult
                        list_tech_capex.append(add_value)

                # CAU
                mask_cau = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'CAU')
                df_mask_cau = \
                    this_tech_df_cost_power_techs.loc[mask_cau]
                df_mask_cau_unitmult = df_mask_cau.iloc[0]['Unit multiplier']
                df_mask_cau_proj = df_mask_cau.iloc[0]['Projection']
                list_tech_cau = []
                if df_mask_cau_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_cau.iloc[0][y]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)
                if df_mask_cau_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_cau.iloc[0][time_vector[0]]*df_mask_cau_unitmult
                        list_tech_cau.append(add_value)

                # Fixed FOM
                mask_ffom = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Fixed FOM')
                df_mask_ffom = \
                    this_tech_df_cost_power_techs.loc[mask_ffom]
                df_mask_ffom_unitmult = df_mask_ffom.iloc[0]['Unit multiplier']
                df_mask_ffom_proj = df_mask_ffom.iloc[0]['Projection']
                list_tech_ffom = []
                if df_mask_ffom_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_ffom.iloc[0][y]*df_mask_ffom_unitmult
                        list_tech_ffom.append(add_value)
                if df_mask_ffom_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_ffom.iloc[0][time_vector[0]]*df_mask_ffom_unitmult
                        list_tech_ffom.append(add_value)

                # Grid connection cost
                mask_gcc = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Grid connection cost')
                df_mask_gcc = \
                    this_tech_df_cost_power_techs.loc[mask_gcc]
                df_mask_gcc_unitmult = df_mask_gcc.iloc[0]['Unit multiplier']
                df_mask_gcc_proj = df_mask_gcc.iloc[0]['Projection']
                list_tech_gcc = []
                if df_mask_gcc_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_gcc.iloc[0][y]*df_mask_gcc_unitmult
                        list_tech_gcc.append(add_value)
                if df_mask_gcc_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_gcc.iloc[0][time_vector[0]]*df_mask_gcc_unitmult
                        list_tech_gcc.append(add_value)

                # Net capacity factor
                list_tech_cf = cf_by_tech[tech]

                # Operational life
                mask_ol = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Operational life')
                df_mask_ol = \
                    this_tech_df_cost_power_techs.loc[mask_ol]
                df_mask_ol_unitmult = df_mask_ol.iloc[0]['Unit multiplier']
                df_mask_ol_proj = df_mask_ol.iloc[0]['Projection']
                list_tech_ol = []
                if df_mask_ol_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_ol.iloc[0][y]*df_mask_ol_unitmult
                        list_tech_ol.append(add_value)
                if df_mask_ol_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_ol.iloc[0][time_vector[0]]*df_mask_ol_unitmult
                        list_tech_ol.append(add_value)

                # Variable FOM
                mask_vfom = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Variable FOM')
                df_mask_vfom = \
                    this_tech_df_cost_power_techs.loc[mask_vfom]
                df_mask_vfom_unitmult = df_mask_vfom.iloc[0]['Unit multiplier']
                df_mask_vfom_proj = df_mask_vfom.iloc[0]['Projection']
                list_tech_vfom = []
                if df_mask_vfom_proj == 'user_defined':
                    for y in time_vector:
                        add_value = \
                            df_mask_vfom.iloc[0][y]*df_mask_vfom_unitmult
                        list_tech_vfom.append(add_value)
                if df_mask_vfom_proj == 'flat':
                    for y in range(len(time_vector)):
                        add_value = \
                            df_mask_vfom.iloc[0][time_vector[0]]*df_mask_vfom_unitmult
                        list_tech_vfom.append(add_value)

                # Heat Rate
                mask_hr = \
                    (this_tech_df_cost_power_techs['Parameter'] == 'Heat Rate')
                df_mask_hr = \
                    this_tech_df_cost_power_techs.loc[mask_hr]
                if len(df_mask_hr.index.tolist()) != 0:
                    df_mask_hr_unitmult = df_mask_hr.iloc[0]['Unit multiplier']
                    df_mask_hr_proj = df_mask_hr.iloc[0]['Projection']
                    list_tech_hr = []
                    if df_mask_hr_proj == 'user_defined':
                        for y in time_vector:
                            add_value = \
                                df_mask_hr.iloc[0][y]*df_mask_hr_unitmult
                            list_tech_hr.append(add_value)
                    if df_mask_hr_proj == 'flat':
                        for y in range(len(time_vector)):
                            add_value = \
                                df_mask_hr.iloc[0][time_vector[0]]*df_mask_hr_unitmult
                            list_tech_hr.append(add_value)
                else:
                    list_tech_hr = [0 for y in range(len(time_vector))]

                # ...we next need to incorporate the heat rate data into the variable opex //
                # hence, link the commodity to the technology fuel consumption

                # ...storing the power plant information for printing
                idict_u_capex.update({tech:deepcopy(list_tech_capex)})
                idict_u_fopex.update({tech:deepcopy(list_tech_ffom)})
                idict_u_vopex.update({tech:deepcopy(list_tech_vfom)})
                idict_u_gcc.update({tech:deepcopy(list_tech_gcc)})
                idict_cau.update({tech:deepcopy(list_tech_cau)})
                # idict_net_cap_factor.update({tech:deepcopy(list_tech_cf)})
                idict_hr.update({tech:deepcopy(list_tech_hr)})
                idict_oplife.update({tech:deepcopy(list_tech_ol)})

                # idict_net_cap_factor_by_scen_by_country[this_scen][this_country] = deepcopy(idict_net_cap_factor)

                # ...acting for "Distribution of new electrical energy generation" (_dneeg)
                this_tech_dneeg_df_param_related = df_param_related_4.iloc[tech_idx]
                this_tech_dneeg_apply_type = this_tech_dneeg_df_param_related['apply_type']
                this_tech_dneeg_projection = this_tech_dneeg_df_param_related['projection']
                this_tech_dneeg_value_type = this_tech_dneeg_df_param_related['value']
                this_tech_dneeg_known_vals_raw = []
                this_tech_dneeg_known_vals = []
                use_fuel = this_tech_dneeg_df_param_related['Fuel']

                this_tech_base_cap = dict_base_caps[tech][base_year]
                this_tech_accum_new_cap = [0 for y in range(len(time_vector))]
                this_tech_accum_new_cap_unplanned = [0 for y in range(len(time_vector))]
                this_tech_new_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)

                this_tech_new_cap_unplanned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)
                this_tech_new_cap_planned = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)
                this_tech_energy_dist = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)

                this_tech_total_endo = [0 for y in range(len(time_vector))]  # disaggregated output (list, dict above)

                this_tech_residual_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_cap = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_total_prod = [0 for y in range(len(time_vector))]  # crucial output (list, dict above)
                this_tech_new_prod = [0 for y in range(len(time_vector))]

                reno_ene_delta_add = [0] * len(time_vector)

                # This is equivalent to parameter 8:
                this_tech_forced_new_cap = [0 for y in range(len(time_vector))]  # should be an input
                if tech_idx_8 != '':
                    for y in range(len(time_vector)):
                        this_tech_forced_new_cap[y] = \
                            df_param_related_8.loc[tech_idx_8, time_vector[y]]/1000

                # This is equivalent to parameter 9:
                this_tech_phase_out_cap = [0 for y in range(len(time_vector))]  # should be an input
                if tech_idx_9 != '':
                    for y in range(len(time_vector)):
                        this_tech_phase_out_cap[y] = \
                            df_param_related_9.loc[tech_idx_9, time_vector[y]]/1000

                if (this_tech_dneeg_projection == 'interpolate' and
                        this_tech_dneeg_value_type == 'percent' and
                        this_tech_dneeg_apply_type == 'all') or (
                        this_tech_dneeg_projection == 'keep_proportions') or (
                        this_tech_dneeg_projection == 'user_defined'):

                    # REVIEW THIS
                    new_req_energy_list = []
                    new_req_energy_list_2 = []
                    new_ene_assign_list = []


                    for y in range(len(time_vector)):
                        # calculate the energy that the accumulated unplanned capacity supplies (this is actually unused)
                        if y != 0:
                            this_tech_accum_cap_energy = \
                                list_tech_cau[y]*list_tech_cf[y]*this_tech_accum_new_cap_unplanned[y-1]
                            accum_cap_energy_vector[y] += \
                                this_tech_accum_cap_energy
                    
                        # ...estimate the energy requirement
                        new_req_energy = \
                            electrical_demand_to_supply[y] \
                            - forced_newcap_energy_all[y] \
                            - store_res_energy_all[y]
                        #     - store_unplanned_energy_all[y]

                        new_req_energy_list.append(new_req_energy)
                    
                        # It is convenient to call the capacity here
                        if y > 0:
                            ref_current_cap = this_tech_total_cap[y-1]
                        else:
                            ref_current_cap = store_use_cap[tech][y]
                    
                        # Here we must add the capacity factor adjustment for thermal, such that the renewable target is met and no unnecessary capacity is further planned
                        # Also, considerations about renewable targets must be established
                        '''
                        There are basically 2 options to increase renewability:
                        - reduce thermal capacity factors
                        - increase renewable generation
                        
                        Furthermore, if more production than needed occurs, the thermal capacity factors can be reduced
                        '''
                    
                        # Establish technical capacity factor minimums to cope with reductions:
                        # Take the elements from the power plants:
                        max_cf_dict = {
                            'PP_Thermal_Diesel': 0.0001,
                            'PP_Thermal_Fuel oil': 0.000001,
                            'PP_Thermal_Coal': 0.0001,
                            'PP_Thermal_Crude': 0.0001,
                            'PP_Thermal_Natural Gas': 0.0001}
                    
                        # The sorted thermal technologies will come in handy
                        # Worst-to-best variable cost technologies:
                        wtb_tech_list = yearly_sorted_tech_costs[y][:5]
                        # We need to reshuffle this list so we remove nuclear and introduce Crude;
                        # this may have to change in the future when we update the model with costs.
                        wtb_tech_list = ['PP_Thermal_Diesel',
                            'PP_Thermal_Fuel oil', 'PP_Thermal_Coal',
                            'PP_Thermal_Crude', 'PP_Thermal_Natural Gas']
                        
                        if tech_counter == 1:
                            thermal_reductions_order.update({y:1})

                        # if this_country_2 == 'Costa Rica' and y == 0:
                        #     print('find out the original capacity factor *')
                        #     print(y,tech,cf_by_tech['PP_Thermal_Natural Gas'])
                        #     sys.exit()


                        if new_req_energy < 0:
                            count_under_zero += 1
                            if tech_counter == 1:
                                thermal_reductions = abs(deepcopy(new_req_energy))
                                thermal_reductions_store.update({y:deepcopy(thermal_reductions)})
                            else:
                                try:
                                    thermal_reductions = deepcopy(thermal_reductions_store[y])
                                except Exception:
                                    thermal_reductions = 0
                            new_req_energy = 0
                            
                            if this_scen == 'BAU' or time_vector[y] < 2024:
                                thermal_reductions = 0
                                
                            if tech in wtb_tech_list:
                                max_cf_for_reductions = max_cf_dict[tech]
                                list_th_tech_cf = deepcopy(cf_by_tech[tech])
                                if list_th_tech_cf[y] < max_cf_for_reductions:
                                    thermal_reductions_order[y] += 1


                            # This indicates that thermal generation has an opportunity for reduction, i.e., reduce until technical minimums are met
                            enter_reduction_conditionals = False
                            if tech == wtb_tech_list[0] and thermal_reductions_order[y] == 1:
                                enter_reduction_conditionals = True
                            if tech == wtb_tech_list[1] and thermal_reductions_order[y] == 2:
                                enter_reduction_conditionals = True
                            if tech == wtb_tech_list[2] and thermal_reductions_order[y] == 3:
                                enter_reduction_conditionals = True
                            if tech == wtb_tech_list[3] and thermal_reductions_order[y] == 4:
                                enter_reduction_conditionals = True
                            if tech == wtb_tech_list[4] and thermal_reductions_order[y] == 5:
                                enter_reduction_conditionals = True
                            if enter_reduction_conditionals and list_tech_cf[y] > max_cf_for_reductions and tech in wtb_tech_list and thermal_reductions > 0:
                                
                                
                                # print('THIS HAPPENED? 1', time_vector[y], thermal_reductions)
                                
                                curr_energy_tech = list_tech_cau[y]*list_tech_cf[y]*ref_current_cap
                                min_energy_tech = list_tech_cau[y]*max_cf_for_reductions*ref_current_cap
                                max_red_tech = curr_energy_tech - min_energy_tech
                                if max_red_tech >= thermal_reductions:
                                    new_cf = (curr_energy_tech - thermal_reductions)/(list_tech_cau[y]*ref_current_cap)
                                    thermal_reductions = 0
                                else:
                                    new_cf = max_cf_for_reductions
                                    thermal_reductions -= max_red_tech
                                    thermal_reductions_order[y] += 1
                                # for y_aux_cf in range(y, len(time_vector)):
                                # for y_aux_cf in range(y, y+1):
                                for y_aux_cf in range(y, len(time_vector)):
                                    list_tech_cf[y_aux_cf] = deepcopy(new_cf)
                                    store_res_energy[tech][y_aux_cf] *= new_cf/cf_by_tech[tech][y_aux_cf]
                                    forced_newcap_energy_by_tech[tech][y_aux_cf] *= new_cf/cf_by_tech[tech][y_aux_cf]

                                cf_by_tech[tech] = deepcopy(list_tech_cf)
                    
                            if y in list(thermal_reductions_store.keys()):
                                thermal_reductions_store[y] = deepcopy(thermal_reductions)
                    
                        else:
                            thermal_reductions_store.update({y:0})
                    
                        new_req_energy_list_2.append(new_req_energy)
                    
                        '''
                        NOTE:
                        This means that the excess energy must be covered.
                        We design for an excess considering the additional planned capacity out of the excess.
                        '''
                    
                        # NOTE: "store_res_energy_all" has the production of the residual capacity
                    
                        # Instead of distribution to keep proportion, we need to proceed with the interpolation
                        # of the shares that this source will produce:
                        use_cap = store_use_cap[tech][y]  # cap with calibrated values
                        res_energy = store_res_energy[tech][y]  # energy that calibrated value produces (not considering new capacity)
                        res_energy_change = 0
                        if y > 0:
                            res_energy_change = store_res_energy[tech][y] - store_res_energy[tech][y-1]
                    
                        planned_energy = forced_newcap_energy_by_tech[tech][y]  # energy from planned plants
                    
                        if this_tech_dneeg_projection == 'keep_proportions':  # do not mix things up
                            if y == 0:
                                res_energy_base = this_tech_base_cap*8760*list_tech_cf[0]/1000  # MW to GW
                                res_energy_sum = res_energy_sum_1
                            else:
                                res_energy_base = this_tech_base_cap*8760*list_tech_cf[y]/1000  # MW to GW
                                res_energy_sum = res_energy_sum_1
                            energy_dist = res_energy_base/res_energy_sum  # distribution of energy for "keep_proportions"
                            new_energy_assign = new_req_energy*energy_dist
                        else:
                            new_energy_assign = new_req_energy*store_percent[tech][y]
                    
                        new_ene_assign_list.append(new_energy_assign)
                        
                        if tech_counter < len(list_electric_sets_3_shuffle):
                            this_tech_total_prod[y] = deepcopy(res_energy + planned_energy + new_energy_assign)
                        
                        if this_scen == 'BAU':
                            cf_ngas_max = 0.8
                        else:
                            cf_ngas_max = 0.8  # more backup operation
                        # stop_check = False
                        # stop_check = True
                        # if this_country_2 == 'Costa Rica' and y == 2 and stop_check is False:
                        #     print('what happened  here?')
                        #     sys.exit()
                        
                        # if this_country_2 == 'Costa Rica' and y == 0:
                        #     print('find out the original capacity factor')
                        #     print(y,tech,cf_by_tech['PP_Thermal_Natural Gas'])
                        #     sys.exit()
                        # if this_country_2 == 'Costa Rica':
                        #     print(y,tech,cf_by_tech['PP_Thermal_Natural Gas'])
                        
                        if tech == 'PP_Thermal_Natural Gas' and new_energy_assign > 0 and cf_by_tech[tech][y] < cf_ngas_max and y > 0:
                            cf_original = deepcopy(cf_by_tech[tech][y])
                            cf_new_changed = deepcopy(cf_by_tech[tech][y])
                            if (res_energy + planned_energy) > 0:
                                cf_new_changed *= (res_energy + planned_energy + new_energy_assign)/(res_energy + planned_energy)
                            else:
                                cf_new_changed = deepcopy(cf_ngas_max)
                            cf_new_changed_original = deepcopy(cf_new_changed)
                            if cf_new_changed > cf_ngas_max:
                                cf_new_changed = deepcopy(cf_ngas_max)
                            # for y_aux_cf in range(y, len(time_vector)):
                            # for y_aux_cf in range(y, y+1):
                            for y_aux_cf in range(y, len(time_vector)):
                                list_tech_cf[y_aux_cf] = deepcopy(cf_new_changed)
                                store_res_energy[tech][y_aux_cf] *= cf_new_changed/cf_by_tech[tech][y_aux_cf]
                                forced_newcap_energy_by_tech[tech][y_aux_cf] *= cf_new_changed/cf_by_tech[tech][y_aux_cf]
                            cf_by_tech[tech] = deepcopy(list_tech_cf)

                            # if this_country_2 == 'Costa Rica':
                            #    print('check natural gas')
                            #    sys.exit()

                        # if tech == 'PP_Hydro':
                        #   print(time_vector[y], res_energy, planned_energy, new_energy_assign, new_req_energy)
                        
                        # Here the new energy assign of renewables can increase to meet the renewable targets:
                        if tech_counter == len(list_electric_sets_3_shuffle) and reno_targets_exist:
                            
                            # print('count y until here')
                            
                            # sys.exit()
                            
                            list_electric_sets_3_shuffle_rest = list_electric_sets_3_shuffle[:-1]
                            # Here we must find out about the renewable generation
                            reno_gen = [0] * len(time_vector)
                            all_gen_reno_target = [0] * len(time_vector)
                            for suptech in list_electric_sets_3_shuffle_rest:
                                if 'Solar' in suptech or 'Geo' in suptech or 'Sugar' in suptech or 'Wind' in suptech or 'Hydro' in suptech:
                                    reno_gen = [a + b for a, b in zip(reno_gen, total_production[suptech])]
                                all_gen_reno_target = [a + b for a, b in zip(all_gen_reno_target, total_production[suptech])]

                            reno_gen = [a + b for a, b in zip(reno_gen, this_tech_total_prod)]
                            
                            reno_est_demand_to_supply = [100*a / b for a, b in zip(reno_gen, electrical_demand_to_supply)]
                            reno_est = [100*a / b for a, b in zip(reno_gen, all_gen_reno_target)]

                            # We can compare the percentage of renewables
                            if isinstance(reno_target_list[y], (float, np.floating, int)):
                                if reno_est[y] < reno_target_list[y] and not np.isnan(reno_target_list[y]):
                                    print('We need to increase renewability! Basically replace thermal generation with renewable generation for THIS tech.')
                                    # First let's calculate the energy swap required, which is similar to "thermal_reductions":
                                    reno_ene_delta_demand_based = ((reno_target_list[y] - reno_est[y])/100) * electrical_demand_to_supply[y]
                                    
                                    # The 'reno_ene_delta_demand_based' assumes a perfect match between generation and demand, but there are some mismatches.
                                    # To override the mismatches, let's estimate the difference based on total production                                  
                                    reno_ene_delta = ((reno_target_list[y] - reno_est[y])/100) * all_gen_reno_target[y]
                                    
                                    # print('THIS HAPPENED? 2', time_vector[y], thermal_reductions)
                                    
                                    # Then, let us update the important energy variables that control capacity expansion:
                                    # for y_aux_cf in range(y, len(time_vector)):
                                    # for y_aux_cf in range(y, y+1):
                                    for y_aux_cf in range(y, len(time_vector)):
                                        this_tech_total_prod[y_aux_cf] += deepcopy(reno_ene_delta)
                                        reno_ene_delta_add[y_aux_cf] += deepcopy(reno_ene_delta)
                                    new_energy_assign += deepcopy(reno_ene_delta)
                                    new_ene_assign_list[-1] = deepcopy(new_energy_assign)
                                    
                                    thermal_reductions_2 = deepcopy(reno_ene_delta)
                                    
                                    for th_tech in wtb_tech_list:
                                        max_cf_for_reductions = max_cf_dict[th_tech]
                                        list_th_tech_cf = deepcopy(cf_by_tech[th_tech])
                                        if list_th_tech_cf[y] <= max_cf_for_reductions:
                                            thermal_reductions_order[y] += 1
                                        enter_reduction_conditionals = False
                                        
                                        print(th_tech, thermal_reductions_order[y], list_th_tech_cf[y], max_cf_for_reductions)
                                        
                                        if th_tech == wtb_tech_list[0] and thermal_reductions_order[y] == 1:
                                            enter_reduction_conditionals = True
                                        if th_tech == wtb_tech_list[1] and thermal_reductions_order[y] == 2:
                                            enter_reduction_conditionals = True
                                        if th_tech == wtb_tech_list[2] and thermal_reductions_order[y] == 3:
                                            enter_reduction_conditionals = True
                                        if th_tech == wtb_tech_list[3] and thermal_reductions_order[y] == 4:
                                            enter_reduction_conditionals = True
                                        if th_tech == wtb_tech_list[4] and thermal_reductions_order[y] == 5:
                                            enter_reduction_conditionals = True
                                        if enter_reduction_conditionals and list_th_tech_cf[y] > max_cf_for_reductions and thermal_reductions_2 >= 0:
                                            # print('got in')
                                            curr_energy_tech = list_tech_cau[y]*list_th_tech_cf[y]*total_capacity[th_tech][y]
                                            min_energy_tech = list_tech_cau[y]*max_cf_for_reductions*total_capacity[th_tech][y]
                                            max_red_tech = curr_energy_tech - min_energy_tech
                                            if max_red_tech >= thermal_reductions_2:
                                                new_cf = (curr_energy_tech - thermal_reductions_2)/(list_tech_cau[y]*total_capacity[th_tech][y])
                                                thermal_reductions_2 = 0
                                                enter_reduction_conditionals = False
                                            else:
                                                new_cf = max_cf_for_reductions
                                                thermal_reductions_2 -= max_red_tech
                                                thermal_reductions_order[y] += 1
                                            # for y_aux_cf in range(y, len(time_vector)):
                                            # for y_aux_cf in range(y, y+1):
                                            for y_aux_cf in range(y, len(time_vector)):
                                                list_th_tech_cf[y_aux_cf] = deepcopy(new_cf)
                                                total_production[th_tech][y_aux_cf] *= new_cf/cf_by_tech[th_tech][y_aux_cf]
                                            cf_by_tech[th_tech] = deepcopy(list_th_tech_cf)
                    
                                            # # if 'PP_Nuclear' == th_tech:
                                            # if 'PP_Thermal_Natural Gas' == th_tech:
                                            #     print('check what is happening with the thermal production')
                                            #     sys.exit()
                    
                                    # print('REVIEW IMPACT')
                                    # sys.exit()

                                    # print_reno_test = True
                                    print_reno_test = False
                                    if print_reno_test:
                                        print('Writing a test verifying the the renewability has been reached according to the RE Target parameter')
                                        # Here we must write a test to check the renewability of the system
                                        reno_gen_verify = [0] * len(time_vector)
                                        all_gen_verify = [0] * len(time_vector)
                                        for suptech in list_electric_sets_3_shuffle_rest:
                                            if 'Solar' in suptech or 'Geo' in suptech or 'Sugar' in suptech or 'Wind' in suptech or 'Hydro' in suptech:
                                                print(suptech)
                                                reno_gen_verify = [a + b for a, b in zip(reno_gen_verify, total_production[suptech])]
                                            all_gen_verify = [a + b for a, b in zip(all_gen_verify, total_production[suptech])]
                                        reno_gen_verify = [a + b for a, b in zip(reno_gen_verify, this_tech_total_prod)]
                                        all_gen_verify = [a + b for a, b in zip(all_gen_verify, this_tech_total_prod)]

                                        ratio_all_gen_verify = [a / b for a, b in zip(reno_gen_verify, all_gen_verify)]
                                        ratio_electrical_demand_to_supply = [a / b for a, b in zip(reno_gen_verify, electrical_demand_to_supply)]

                                        index_2030 = time_vector.index(2030)
                                        ratio_all_gen_verify_2030 = ratio_all_gen_verify[index_2030]
                                        ratio_electrical_demand_to_supply_2030 = ratio_electrical_demand_to_supply[index_2030]

                                        # Take advantage of this area to calculate the difference between the electrical demand to supply and teh generation, in case there is an error
                                        diff_tot_sup_dem = [a - b for a, b in zip(all_gen_verify, electrical_demand_to_supply)]
                                        diff_tot_sup_err = [100*(a - b)/a for a, b in zip(all_gen_verify, electrical_demand_to_supply)]

                                        for suptech2 in list_electric_sets_3_shuffle_rest:
                                            print('>', suptech2, total_production[suptech2][y])

                                        print('Review elements that can be wrong')
                                        sys.exit()

                        if tech_counter == len(list_electric_sets_3_shuffle):
                            this_tech_total_prod[y] = deepcopy(res_energy + planned_energy + new_energy_assign)

                        # if this_scen == 'NDCPLUS' and y == len(time_vector)-1 and tech_counter == len(list_electric_sets_3_shuffle):
                        #    print('check demand balance')
                        #    sys.exit()

                        if y != 0 and tech not in thermal_filter_out:  # This is a debugging section
                            this_tech_new_prod[y] = \
                                this_tech_total_prod[y] \
                                - this_tech_total_prod[y-1]
                            # Some tolerance can be added to allow "negative CAPEX" in a small proportion:
                            tol_min_neg_capex_pj_ele = 0.1
                            if abs(this_tech_new_prod[y]) < tol_min_neg_capex_pj_ele and this_tech_new_prod[y] < 0 and time_vector[y] <= 2023:
                                print('An expected negative in generaton CAPEX occured!')
                                print(this_scen, tech, this_country)
                            elif this_tech_new_prod[y] < -tol_min_neg_capex_pj_ele and time_vector[y] > 2023 and res_energy_change < 0:
                                pass  # this is normal
                            elif this_tech_new_prod[y] < -tol_min_neg_capex_pj_ele and time_vector[y] > 2023 and res_energy_change >= 0:
                                # This means that more capacity probably exists than necessary, so for the punctual year, we adjust the capacity factor
                                ref_list_tech_cf = deepcopy(list_tech_cf[y])
                                if 'Solar' not in tech and 'Wind' not in tech:
                                    list_tech_cf[y] = this_tech_total_prod[y] / (this_tech_total_cap[y-1]*list_tech_cau[y-1])
                                    cf_by_tech[tech][y] = deepcopy(list_tech_cf[y])
                                mult_factor_cf_reno = list_tech_cf[y]/ref_list_tech_cf
                                # print(mult_factor_cf_reno)
                                # print('Does this inconsistency happen?', this_scen, tech, this_country)
                                # print(store_percent[tech])
                                # sys.exit()
                                # this_tech_new_prod[y] = 0
                    
                        # Remembering how much should be subtracted
                        if y == 0:
                            subtract_new_cap = 0
                        else:
                            subtract_new_cap = this_tech_accum_new_cap_unplanned[y-1]
                    
                        # Estimating unplanned capacity
                        if list_tech_cau[y]*list_tech_cf[y] != 0: #try:
                            new_cap_unplanned = \
                                new_energy_assign/(list_tech_cau[y]*list_tech_cf[y]) - \
                                subtract_new_cap
                        else:
                            print('division by zero', 'interpolate', 2)
                            sys.exit()

                        # if tech == 'PP_Hydro' and time_vector[y] == 2024:
                        #    print('Check this')
                        #    sys.exit()

                        # This is a filter to avoid inconsistencies:
                        if new_cap_unplanned < 0:
                            new_cap_unplanned = 0
                    
                        new_cap = new_cap_unplanned + forced_newcap_by_tech[tech][y]
                    
                        # Update the residual capacity
                        if y == 0:
                            residual_cap = use_cap
                            this_tech_total_cap[y] = use_cap
                            this_tech_residual_cap[y] = use_cap
                        else:
                            residual_cap = this_tech_residual_cap[y-1] - this_tech_phase_out_cap[y]
                            this_tech_total_cap[y] += residual_cap + this_tech_accum_new_cap[y-1]
                            this_tech_residual_cap[y] = residual_cap
                    
                        # Adjust accumulated new capacities
                        if y == 0:
                            this_tech_accum_new_cap[y] = new_cap
                            this_tech_accum_new_cap_unplanned[y] = new_cap_unplanned
                        else:
                            this_tech_accum_new_cap[y] = \
                                new_cap + this_tech_accum_new_cap[y-1]
                            this_tech_accum_new_cap_unplanned[y] = \
                                new_cap_unplanned + this_tech_accum_new_cap_unplanned[y-1]
                    
                        this_tech_new_cap[y] += new_cap
                        this_tech_total_cap[y] += new_cap
                    
                        this_tech_new_cap_unplanned[y] = deepcopy(new_cap_unplanned)
                        this_tech_new_cap_planned[y] = deepcopy(forced_newcap_by_tech[tech][y])
                    
                        # ...these are further debugging energy variables
                        this_tech_energy_dist[y] = deepcopy(store_percent[tech][y])
                    
                        # ...these are further debugging capacity/energy variables
                        for aux_y in range(y, len(time_vector)):
                            this_tech_total_endo[aux_y] += deepcopy(new_energy_assign)

                    """
                    print(tech)
                    print(new_req_energy_list)
                    print(new_req_energy_list_2)
                    print(new_ene_assign_list)
                    print('\n')
                    # sys.exit()
                    """

                # ...we must now see the additional energy requirements of primary or secondary carriers because of total capacity
                if sum(list_tech_hr) != 0:  # means there is fuel a requirement:
                    list_use_fuel = []
                    for y in range(len(time_vector)):
                        add_value = \
                            this_tech_total_cap[y]*list_tech_cau[y]*list_tech_cf[y]*list_tech_hr[y]
                        list_use_fuel.append(add_value)
                    fuel_use_electricity.update({tech:{use_fuel:list_use_fuel}})
                else:
                    fuel_use_electricity.update({tech:'none'})

                # ...here we store the correspoding physical variables
                total_capacity.update({tech:this_tech_total_cap})
                residual_capacity.update({tech:this_tech_residual_cap})
                new_capacity.update({tech:this_tech_new_cap})

                cap_new_unplanned.update({tech:this_tech_new_cap_unplanned})
                cap_new_planned.update({tech:this_tech_new_cap_planned})
                cap_phase_out.update({tech:this_tech_phase_out_cap})

                total_production.update({tech:this_tech_total_prod})
                new_production.update({tech:this_tech_new_prod})

                # ...here we compute debugging variables:
                ele_prod_share.update({tech:this_tech_energy_dist})
                ele_endogenous.update({tech:this_tech_total_endo})
                cap_accum.update({tech:this_tech_accum_new_cap})

                # ...here we compute the costs by multiplying capacities times unit costs:
                this_tech_capex = [0 for y in range(len(time_vector))]
                this_tech_fopex = [0 for y in range(len(time_vector))]
                this_tech_vopex = [0 for y in range(len(time_vector))]
                this_tech_gcc = [0 for y in range(len(time_vector))]
                for y in range(len(time_vector)):
                    this_tech_capex[y] = this_tech_new_cap[y]*list_tech_capex[y]
                    this_tech_fopex[y] = this_tech_total_cap[y]*list_tech_ffom[y]
                    this_tech_vopex[y] = \
                        this_tech_total_prod[y]*list_tech_vfom[y]
                    this_tech_gcc[y] = this_tech_new_cap[y]*list_tech_gcc[y]

                # if tech == 'PP_Thermal_Crude':
                #     print('please stop here to check what is going on')
                #     sys.exit()

                capex.update({tech:this_tech_capex})
                fopex.update({tech:this_tech_fopex})
                vopex.update({tech:this_tech_vopex})
                gcc.update({tech:this_tech_gcc})

                # ...here we compute the externalities and emissions by multiplying fuel use times unit values:
                if sum(list_tech_hr) != 0:  # means there is fuel a requirement:
                    list_emissions = []
                    list_externalities_globalwarming = []
                    list_externalities_localpollution = []
                    for y in range(len(time_vector)):
                        if use_fuel in emissions_fuels_list:  # ...store emissions here
                            add_value_emissions = \
                                list_use_fuel[y]*emissions_fuels_dict[use_fuel][y]
                            list_emissions.append(add_value_emissions)

                            # ...besides, we must add variable costs from fuel consumption
                            fuel_idx_7 = df_param_related_7['Fuel'].tolist().index(use_fuel)
                            this_tech_vopex[y] += \
                                deepcopy(list_use_fuel[y]*df_param_related_7.loc[fuel_idx_7, time_vector[y]])

                            idict_u_vopex[tech][y] += \
                                deepcopy(df_param_related_7.loc[fuel_idx_7, time_vector[y]])

                        if use_fuel in externality_fuels_list:  # ...store externalities
                            add_value_globalwarming = \
                                list_use_fuel[y]*externality_fuels_dict[use_fuel]['Global warming']
                            add_value_localpollution = \
                                list_use_fuel[y]*externality_fuels_dict[use_fuel]['Local pollution']
                            list_externalities_globalwarming.append(add_value_globalwarming)
                            list_externalities_localpollution.append(add_value_localpollution)

                    emissions_electricity.update({tech:{use_fuel:list_emissions}})
                    externalities_globalwarming_electricity.update({tech:{use_fuel:list_externalities_globalwarming}})
                    externalities_localpollution_electricity.update({tech:{use_fuel:list_externalities_localpollution}})


                # if tech == 'PP_Thermal_Crude':
                #if this_scen == 'BAU' and tech == 'PP_Thermal_Fuel oil':
                   #print('please stop here to check what is going on')
                   #sys.exit()
            
            # Here we must calculate the emissions of methane:
            power_techs_with_3_emissions = [i for i in emissions_3_fuels_dict.keys() if 'PP_' in i]
            emissions_demand_methane_pp = {}
                       
            for tech in power_techs_with_3_emissions:
                activity_list = total_production[tech]
                emissions_demand_methane_pp.update({tech:{}})
                if tech == 'PP_Hydro':
                    ef_3_methane_list = emissions_3_fuels_dict[tech]['Hydro']
                    resulting_emissions = [a*b/1e6 for a, b in zip(activity_list, ef_3_methane_list)]
                    emissions_demand_methane_pp[tech].update({'Hydro':resulting_emissions})
                elif tech == 'PP_Thermal_Diesel':
                    ef_3_methane_list = emissions_3_fuels_dict[tech]['Diesel']
                    resulting_emissions = [a*b/1e6 for a, b in zip(activity_list, ef_3_methane_list)]
                    emissions_demand_methane_pp[tech].update({'Diesel':resulting_emissions})
                elif tech == 'PP_Thermal_Fuel oil':
                    ef_3_methane_list = emissions_3_fuels_dict[tech]['Fuel Oil']
                    resulting_emissions = [a*b/1e6 for a, b in zip(activity_list, ef_3_methane_list)]
                    emissions_demand_methane_pp[tech].update({'Fuel Oil':resulting_emissions})
                elif tech == 'PP_Thermal.re_Sugar cane and derivatives':
                    ef_3_methane_list = emissions_3_fuels_dict[tech]['Sugar cane and derivatives']
                    resulting_emissions = [a*b/1e6 for a, b in zip(activity_list, ef_3_methane_list)]
                    emissions_demand_methane_pp[tech].update({'Sugar cane and derivatives':resulting_emissions})
                elif tech == 'PP_Thermal_Coal':
                    ef_3_methane_list = emissions_3_fuels_dict[tech]['Coal']
                    resulting_emissions = [a*b/1e6 for a, b in zip(activity_list, ef_3_methane_list)]
                    emissions_demand_methane_pp[tech].update({'Coal':resulting_emissions})
                else:
                    print('Methane emission type not recognized in system!')
                    sys.exit()
 
                dict_local_country[this_country]['Methane emissions by demand [kt CH4]'].update({tech: emissions_demand_methane_pp[tech]})
            print('Methane emission type not recognized in system!')
            

            # Here we must calculate the emissions of black carbon:
            power_techs_with_2_emissions = [i for i in emissions_2_fuels_dict.keys() if 'PP_' in i]
            emissions_demand_black_carbon_pp = {}
            for tech in power_techs_with_2_emissions:
                activity_list = total_production[tech]
                emissions_demand_black_carbon_pp.update({tech:{}})
                if tech == 'PP_Thermal_Diesel':
                    ef_3_black_carbon_list = emissions_2_fuels_dict[tech]['Diesel']
                    resulting_emissions = [a*b/1e3 for a, b in zip(activity_list, ef_3_black_carbon_list)]
                    emissions_demand_black_carbon_pp[tech].update({'Diesel':resulting_emissions})
                elif tech == 'PP_Thermal_Fuel oil':
                    ef_3_black_carbon_list = emissions_2_fuels_dict[tech]['Fuel Oil']
                    resulting_emissions = [a*b/1e3 for a, b in zip(activity_list, ef_3_black_carbon_list)]
                    emissions_demand_black_carbon_pp[tech].update({'Fuel Oil':resulting_emissions})
                elif tech == 'PP_Thermal.re_Sugar cane and derivatives':
                    ef_3_black_carbon_list = emissions_2_fuels_dict[tech]['Sugar cane and derivatives']
                    resulting_emissions = [a*b/1e3 for a, b in zip(activity_list, ef_3_black_carbon_list)]
                    emissions_demand_black_carbon_pp[tech].update({'Sugar cane and derivatives':resulting_emissions})
                else:
                    print('Black carbon emission type not recognized in system!')
                    sys.exit()
 
                dict_local_country[this_country]['Black carbon emissions by demand [ton]'].update({tech: emissions_demand_black_carbon_pp[tech]})
            
            # 3i) Store the transport calculations:
            '''
            *Use these variables:*
            dict_fleet_k
            dict_new_fleet_k
            dict_capex_out
            dict_fopex_out
            dict_vopex_out
            
            Remember to apply this: dict_eq_transport_fuels
            '''
            if overwrite_transport_model:
                dict_tax_out_t1 = dict_tax_out['Imports']
                dict_tax_out_t2 = dict_tax_out['IMESI_Venta']
                dict_tax_out_t3 = dict_tax_out['IVA_Venta']
                dict_tax_out_t4 = dict_tax_out['Patente']
                dict_tax_out_t5 = dict_tax_out['IMESI_Combust']
                dict_tax_out_t6 = dict_tax_out['IVA_Gasoil']
                dict_tax_out_t7 = dict_tax_out['IVA_Elec']
                dict_tax_out_t8 = dict_tax_out['Impuesto_Carbono']
                dict_tax_out_t9 = dict_tax_out['Otros_Gasoil']
                dict_tax_out_t10 = dict_tax_out['Tasa_Consular']
                #Discount values here 
                dict_local_country[this_country].update({'Fleet': deepcopy(dict_fleet_k)})
                dict_local_country[this_country].update({'New Fleet': deepcopy(dict_new_fleet_k)})
                dict_local_country[this_country].update({'Transport CAPEX [$]': deepcopy(dict_capex_out)})
                dict_local_country[this_country].update({'Transport Fixed OPEX [$]': deepcopy(dict_fopex_out)})
                dict_local_country[this_country].update({'Transport Variable OPEX [$]': deepcopy(dict_vopex_out)})
                dict_local_country[this_country].update({'Transport Tax Imports [$]': deepcopy(dict_tax_out_t1)})
                dict_local_country[this_country].update({'Transport Tax IMESI_Venta [$]': deepcopy(dict_tax_out_t2)})
                dict_local_country[this_country].update({'Transport Tax IVA_Venta [$]': deepcopy(dict_tax_out_t3)})
                dict_local_country[this_country].update({'Transport Tax Patente [$]': deepcopy(dict_tax_out_t4)})
                dict_local_country[this_country].update({'Transport Tax IMESI_Combust [$]': deepcopy(dict_tax_out_t5)})
                dict_local_country[this_country].update({'Transport Tax IVA_Gasoil [$]': deepcopy(dict_tax_out_t6)})
                dict_local_country[this_country].update({'Transport Tax IVA_Elec [$]': deepcopy(dict_tax_out_t7)})
                dict_local_country[this_country].update({'Transport Tax IC [$]': deepcopy(dict_tax_out_t8)})
                dict_local_country[this_country].update({'Transport Tax Otros_Gasoil [$]': deepcopy(dict_tax_out_t9)})
                dict_local_country[this_country].update({'Transport Tax Tasa_Consular [$]': deepcopy(dict_tax_out_t10)})
                dict_local_country[this_country].update({'Transport CAPEX [$] (disc)': deepcopy(dict_capex_out_disc)})
                dict_local_country[this_country].update({'Transport Fixed OPEX [$] (disc)': deepcopy(dict_fopex_out_disc)})
                dict_local_country[this_country].update({'Transport Variable OPEX [$] (disc)': deepcopy(dict_vopex_out_disc)})

            # 3j) Store the data for printing:
            dict_local_country[this_country].update({'Electricity fuel use': deepcopy(fuel_use_electricity)})
            dict_local_country[this_country].update({'Global warming externalities in electricity': deepcopy(externalities_globalwarming_electricity)})
            dict_local_country[this_country].update({'Local pollution externalities in electricity': deepcopy(externalities_localpollution_electricity)})
            dict_local_country[this_country].update({'Emissions in electricity': deepcopy(emissions_electricity)})
            dict_local_country[this_country].update({'Electricity total capacity': deepcopy(total_capacity)})
            dict_local_country[this_country].update({'Electricity residual capacity': deepcopy(residual_capacity)})
            dict_local_country[this_country].update({'Electricity new capacity': deepcopy(new_capacity)})
            dict_local_country[this_country].update({'Electricity total production': deepcopy(total_production)})
            dict_local_country[this_country].update({'Electricity CAPEX': deepcopy(capex)})
            dict_local_country[this_country].update({'Electricity Fixed OPEX': deepcopy(fopex)})
            dict_local_country[this_country].update({'Electricity Variable OPEX': deepcopy(vopex)})
            dict_local_country[this_country].update({'Electricity Grid Connection Cost': deepcopy(gcc)})

            # ...disaggregate the new capacity:
            dict_local_country[this_country].update({'Electricity new capacity unplanned': deepcopy(cap_new_unplanned)})
            dict_local_country[this_country].update({'Electricity new capacity planned': deepcopy(cap_new_planned)})
            dict_local_country[this_country].update({'Electricity phase out capacity': deepcopy(cap_phase_out)})

            # ...let's store additional debugging variables per power plant (1):
            dict_local_country[this_country].update({'Electricity production share (unplanned)': deepcopy(ele_prod_share)})
            dict_local_country[this_country].update({'New energy assign': deepcopy(ele_endogenous)})
            dict_local_country[this_country].update({'Accumulated new capacity': deepcopy(cap_accum)})

            # ...let's store the "required energy" components:
            dict_local_country[this_country].update({'Electricity demand to supply': deepcopy(electrical_demand_to_supply)})
            dict_local_country[this_country].update({'Electricity planned supply': deepcopy(forced_newcap_energy_all)})

            # ...let's store additional debugging variables per power plant (2):
            dict_local_country[this_country].update({'Accumulated forced new capacity': deepcopy(accum_forced_newcap_by_tech)})
            dict_local_country[this_country].update({'Electricity planned supply per technology': deepcopy(forced_newcap_energy_by_tech)})
            dict_local_country[this_country].update({'Electricity residual supply': deepcopy(store_res_energy_all)})
            dict_local_country[this_country].update({'Electricity residual supply per tech': deepcopy(store_res_energy)})

            # *...here we need a supporting variable*:
            dict_local_country[this_country].update({'Electricity new production per tech': deepcopy(new_production)})

            # ...here we can execute the discount rate to 5 variables:
            '''
            'Global warming externalities in electricity'
            'Local pollution externalities in electricity'
            'Electricity CAPEX'
            'Electricity Fixed OPEX'
            'Electricity Variable OPEX'
            'Electricity Grid Connection Cost'
            '''
            disc_capex = deepcopy(capex)
            disc_fopex = deepcopy(fopex)
            disc_vopex = deepcopy(vopex)
            disc_gcc = deepcopy(gcc)
            disc_externalities_globalwarming_electricity = deepcopy(externalities_globalwarming_electricity)
            disc_externalities_localpollution_electricity = deepcopy(externalities_localpollution_electricity)

            '''
            # This is the generic equation you must apply:
            this_val_disc = this_value / ((1 + r_rate/100)**(float(this_year) - r_year))
            '''

            for y in range(len(time_vector)):
                this_year = int(time_vector[y])
                disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                for tech in list_electric_sets_3:  # extract the references
                    disc_capex[tech][y] *= disc_constant
                    disc_fopex[tech][y] *= disc_constant
                    disc_vopex[tech][y] *= disc_constant
                    disc_gcc[tech][y] *= disc_constant
                    for use_fuel in externality_fuels_list:
                        try:
                            disc_externalities_globalwarming_electricity[tech][use_fuel][y] *= disc_constant
                        except Exception:
                            pass  # in case the technology does not have an externality
                        try:
                            disc_externalities_localpollution_electricity[tech][use_fuel][y] *= disc_constant
                        except Exception:
                            pass  # in case the technology does not have an externality

            dict_local_country[this_country].update({'Electricity CAPEX (disc)': deepcopy(disc_capex)})
            dict_local_country[this_country].update({'Electricity Fixed OPEX (disc)': deepcopy(disc_fopex)})
            dict_local_country[this_country].update({'Electricity Variable OPEX (disc)': deepcopy(disc_vopex)})
            dict_local_country[this_country].update({'Electricity Grid Connection Cost (disc)': deepcopy(disc_gcc)})
            dict_local_country[this_country].update({'Global warming externalities in electricity (disc)': deepcopy(disc_externalities_globalwarming_electricity)})
            dict_local_country[this_country].update({'Local pollution externalities in electricity (disc)': deepcopy(disc_externalities_localpollution_electricity)})

        dict_local_reg.update({this_reg: deepcopy(dict_local_country)})

  
    ###########################################################################
    # *Here it is crucial to implement the exports as share of total LAC demand:

    # ...calculate the total natural gas demand
    lac_ng_dem = [0 for y in range(len(time_vector))]
    keys_2list_regions = list(dict_local_reg.keys())
    country_2_region = {}
    for ng_reg in keys_2list_regions:
        keys_2list_countries = list(dict_local_reg[ng_reg].keys())
        for ng_cntry in keys_2list_countries:
            query_dict = dict_local_reg[ng_reg][ng_cntry]
            local_ng_dem = []
            for y in range(len(time_vector)):
                add_val = \
                    query_dict['Energy demand by fuel']['Natural Gas'][y] + \
                    query_dict['Electricity fuel use']['PP_Thermal_Natural Gas']['Natural Gas'][y]  # this is a list
                local_ng_dem.append(add_val)

            for y in range(len(time_vector)):
                lac_ng_dem[y] += deepcopy(local_ng_dem[y])

            # ...store the dictionary below to quickly store the export values
            country_2_region.update({ng_cntry:ng_reg})

    # ...extract the exporting countries to LAC // assume *df_scen* is correct from previous loops
    mask_exports = \
        (df_scen['Parameter'] == '% Exports for production')
    mask_exports_pipeline = \
        (df_scen['Parameter'] == '% Exports for production through pipeline')

    df_scen_exports = df_scen.loc[mask_exports]
    df_scen_exports_countries = \
        df_scen_exports['Application_Countries'].tolist()
    # Add a filter to include countries with transport data only:
    df_scen_exports_countries = \
        [c for c in df_scen_exports_countries if c in tr_list_app_countries_u]

    df_scen_exports_pipeline = df_scen.loc[mask_exports_pipeline]
    df_scen_exports_countries_pipeline = \
        df_scen_exports_pipeline['Application_Countries'].tolist()
    # Add a filter to include countries with transport data only:
    df_scen_exports_countries_pipeline = \
        [c for c in df_scen_exports_countries_pipeline if c in tr_list_app_countries_u]

    # ...now we must extract all natural gas prices:
    mask_ngas_prices = \
        ((df_scen['Parameter'].isin(['Fuel prices sales through pipeline',
                                     'Fuel prices sales liquified'])) &
         (df_scen['Fuel'] == 'Natural Gas'))
    df_ngas_prices = df_scen.loc[mask_ngas_prices]
    df_ngas_prices.reset_index(drop=True, inplace=True)

    # ...now we must extract the quantitiy of natural gas exports to LAC!
    # In a loop, iterate across countries:
    for this_con in df_scen_exports_countries:
        mask_con = (df_scen_exports['Application_Countries'] == this_con)
        df_scen_exports_select = df_scen_exports.loc[mask_con]
        df_scen_exports_select.reset_index(drop=True, inplace=True)

        mask_con_pipe = (df_scen_exports_pipeline['Application_Countries'] == this_con)
        df_scen_exports_select_pipe = df_scen_exports_pipeline.loc[mask_con_pipe]
        df_scen_exports_select_pipe.reset_index(drop=True, inplace=True)

        exports_country = [0 for y in range(len(time_vector))]  # in PJ
        exports_country_pipe = [0 for y in range(len(time_vector))]
        exports_country_liq = [0 for y in range(len(time_vector))]
        exports_income = [0 for y in range(len(time_vector))]

        for y in range(len(time_vector)):
            this_year = int(time_vector[y])

            disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))

            export_price_pipe = df_ngas_prices.loc[0, int(time_vector[0])]  # FOR NOW ASSUME THE PRICE IS CONSTANT
            export_price_liq = df_ngas_prices.loc[1, int(time_vector[0])]

            # ...here we must calculate the natural gas exports for the country:
            exports_country[y] = \
                lac_ng_dem[y]*df_scen_exports_select.loc[0, this_year]/100

            if len(df_scen_exports_select_pipe.index.tolist()) != 0:
                # here we need to discriminate pipeline and non-pipeline elements
                q_ngas_pipe = \
                    exports_country[y]*df_scen_exports_select_pipe.loc[0, this_year]/100
            else:
                q_ngas_pipe = 0

            exports_country_pipe[y] = q_ngas_pipe
            exports_country_liq[y] = exports_country[y] - q_ngas_pipe

            exports_income[y] = \
                exports_country_pipe[y]*export_price_pipe + \
                exports_country_liq[y]*export_price_liq
            exports_income[y] *= disc_constant

        # ...now we must store the result and intermediary info to the dictionary
        this_reg = country_2_region[this_con]
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports (PJ)':deepcopy(exports_country)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports via Pipeline (PJ)':deepcopy(exports_country_pipe)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports Liquified (PJ)':deepcopy(exports_country_liq)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Exports Income (M USD)':deepcopy(exports_income)})  # only print the disocunted value

    ###########################################################################
    # *For fugitive emissions, we must use the "imports" sheet, with a similar approach as above

    # ...extract fugitive emissions of natural gas (exclusively):
    mask_fugitive_emissions = \
        ((df4_ef['Apply'] == 'Production') &
         (df4_ef['Fuel'] == 'Natural Gas'))
    this_df_fugef_ngas = df4_ef.loc[mask_fugitive_emissions]
    this_df_fugef_ngas.reset_index(drop=True, inplace=True)
    fugef_ngas = this_df_fugef_ngas.iloc[0][per_first_yr]  # assume this is a constant

    mask_fugitive_emissions_2 = \
        ((df4_ef['Apply'] == 'Imports') &
         (df4_ef['Fuel'] == 'Natural Gas'))
    this_df_fugef_ngas_2 = df4_ef.loc[mask_fugitive_emissions_2]
    this_df_fugef_ngas_2.reset_index(drop=True, inplace=True)
    fugef_ngas_2 = this_df_fugef_ngas_2.iloc[0][per_first_yr]  # assume this is a constant

    # ...extract the dataframe with imports information:
    mask_ngas_imports = \
        (df_scen['Parameter'] == '% Imports for consumption')
    df_ngas_imports = df_scen.loc[mask_ngas_imports]
    df_ngas_imports.reset_index(drop=True, inplace=True)
    df_ngas_imports_countries = \
        df_ngas_imports['Application_Countries'].tolist()
    # Add a filter to include countries with transport data only:
    df_ngas_imports_countries = \
        [c for c in df_ngas_imports_countries if c in tr_list_app_countries_u]

    # ...iterate across country-wide consumption and find the imports, local production, and add the exports from above
    for acon in range(len(df_ngas_imports_countries)):
        this_con = df_ngas_imports_countries[acon]
        this_reg = country_2_region[this_con]

        query_dict = dict_local_reg[this_reg][this_con]
        local_ng_dem = []
        for y in range(len(time_vector)):
            add_val = \
                query_dict['Energy demand by fuel']['Natural Gas'][y] + \
                query_dict['Electricity fuel use']['PP_Thermal_Natural Gas']['Natural Gas'][y]  # this is a list
            local_ng_dem.append(add_val)

        try:
            local_ng_exp = \
                query_dict['Natural Gas Exports (PJ)']
        except Exception:
            local_ng_exp = [0 for y in range(len(time_vector))]

        local_ng_production = []
        local_ng_fugitive_emissions = []

        imps_ng = []
        imps_ng_fugitive_emissions = []

        for y in range(len(time_vector)):
            this_year = int(time_vector[y])

            imports_share = df_ngas_imports.loc[acon, this_year]
            imports_PJ = local_ng_dem[y]*imports_share/100
            local_prod_PJ = local_ng_dem[y] - imports_PJ
            local_prod_PJ += local_ng_exp[y]

            local_ng_production.append(local_prod_PJ)
            local_ng_fugitive_emissions.append(local_prod_PJ*fugef_ngas)

            imps_ng.append(imports_PJ)
            imps_ng_fugitive_emissions.append(imports_PJ*fugef_ngas_2)

        dict_local_reg[this_reg][this_con].update({'Natural Gas Production (PJ)':deepcopy(local_ng_production)})  # aggregate
        dict_local_reg[this_reg][this_con].update({'Natural Gas Production Fugitive Emissions (MTon)':deepcopy(local_ng_fugitive_emissions)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Imports (PJ)':deepcopy(imps_ng)})
        dict_local_reg[this_reg][this_con].update({'Natural Gas Imports Fugitive Emissions (MTon)':deepcopy(imps_ng_fugitive_emissions)})

    ###########################################################################
    # *For job estimates, we must multiply times the installed capacity:
    # *For T&D estimates, we must check the electricity supply:

    # ...iterate across all countries and estimate the jobs
    for ng_reg in keys_2list_regions:
        keys_2list_countries = list(dict_local_reg[ng_reg].keys())
        this_reg = ng_reg
        for ng_cntry in keys_2list_countries:
            this_con = ng_cntry

            # ...now we must iterate across technologies with the technological capacity
            list_electric_sets_3.sort()  # this must work
            tech_counter = 0
            for tech in list_electric_sets_3:
                # -------------------------------------------------------------
                # >>> this section is for JOBS:
                try:
                    list_cap = \
                        dict_local_reg[this_reg][this_con]['Electricity total capacity'][tech]
                except Exception:
                    list_cap = [0 for y in range(len(time_vector))]

                try:
                    list_new_cap = \
                        dict_local_reg[this_reg][this_con]['Electricity new capacity'][tech]
                except Exception:
                    list_new_cap = [0 for y in range(len(time_vector))]

                try:
                    list_new_prod = \
                        dict_local_reg[this_reg][this_con]['Electricity new production per tech'][tech]
                except Exception:
                    list_new_prod = [0 for y in range(len(time_vector))]

                try:
                    list_demand_2_supply = \
                        dict_local_reg[this_reg][this_con]['Electricity demand to supply'][tech]
                except Exception:
                    list_demand_2_supply = [0 for y in range(len(time_vector))]

                # ...we must also extract the jobs per unit of installed capacity
                mask_df4_job_fac = \
                    (df4_job_fac['Tech'] == tech)
                this_df4_job_fac = \
                    df4_job_fac.loc[mask_df4_job_fac]

                if len(this_df4_job_fac.index.tolist()) != 0:
                    jobs_factor_constru = this_df4_job_fac['Construction/installation (Job years/ MW)'].iloc[0]
                    jobs_factor_manufac = this_df4_job_fac['Manufacturing (Job years/ MW)'].iloc[0]
                    jobs_factor_opeyman = this_df4_job_fac['Operations & maintenance (Jobs/MW)'].iloc[0]
                    jobs_factor_decom = this_df4_job_fac['Decommissioning (Jobs/MW)'].iloc[0]
                else:
                    jobs_factor_constru = 0
                    jobs_factor_manufac = 0
                    jobs_factor_opeyman = 0
                    jobs_factor_decom = 0

                # ...we must create a LAC multiplier (based on the paper
                # https://link.springer.com/content/pdf/10.1007%2F978-3-030-05843-2_10.pdf)
                jobs_LAC_mult_vector_raw = ['' for x in range(len(time_vector))]
                for x in range(len(time_vector)):
                    if int(time_vector[x]) <= 2030:
                        jobs_LAC_mult_vector_raw[x] = 3.4
                    elif int(time_vector[x]) == 2040:
                        jobs_LAC_mult_vector_raw[x] = 3.1
                    elif int(time_vector[x]) == 2050:
                        jobs_LAC_mult_vector_raw[x] = 2.9
                    else:
                        pass
                jobs_LAC_mult_vector = \
                    interpolation_to_end(time_vector, ini_simu_yr,
                                         jobs_LAC_mult_vector_raw, 'ini',
                                         this_scen, '')

                # ...we must estimate the jobs
                jobs_con_list_per_tech = \
                    [jobs_factor_constru*(1000*list_new_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]
                jobs_man_list_per_tech = \
                    [jobs_factor_manufac*(1000*list_new_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]
                jobs_oym_list_per_tech = \
                    [jobs_factor_opeyman*(1000*list_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]
                jobs_dec_list_per_tech = \
                    [jobs_factor_decom*(1000*list_cap[y])*jobs_LAC_mult_vector[y] for y in range(len(time_vector))]

                # ...and store the results
                if tech_counter == 0:
                    dict_local_reg[this_reg][this_con].update({'Related construction jobs':{}})
                    dict_local_reg[this_reg][this_con].update({'Related manufacturing jobs':{}})
                    dict_local_reg[this_reg][this_con].update({'Related O&M jobs':{}})
                    dict_local_reg[this_reg][this_con].update({'Related decommissioning jobs':{}})
                dict_local_reg[this_reg][this_con]['Related construction jobs'].update({tech: jobs_con_list_per_tech})  # per tech
                dict_local_reg[this_reg][this_con]['Related manufacturing jobs'].update({tech: jobs_man_list_per_tech})
                dict_local_reg[this_reg][this_con]['Related O&M jobs'].update({tech: jobs_oym_list_per_tech})
                dict_local_reg[this_reg][this_con]['Related decommissioning jobs'].update({tech: jobs_dec_list_per_tech})

                # -------------------------------------------------------------
                # >>> this section is for T&D:
                try:
                    list_generation = \
                        dict_local_reg[this_reg][this_con]['Electricity total production'][tech]
                except Exception:
                    list_generation = [0 for y in range(len(time_vector))]

                # ...we must also extract the costs per unit of generated electricity
                mask_df4_tran_dist_fac = \
                    (df4_tran_dist_fac['Tech'] == tech)
                this_df4_tran_dist_fac = \
                    df4_tran_dist_fac.loc[mask_df4_tran_dist_fac]

                if len(this_df4_tran_dist_fac.index.tolist()) != 0:
                    transmi_capex = this_df4_tran_dist_fac['Transmission Capital Cost (M US$/PJ produced)'].iloc[0]
                    transmi_fopex = this_df4_tran_dist_fac['Transmission 2% Fixed Cost (M US$/PJ produced)'].iloc[0]
                    distri_capex = this_df4_tran_dist_fac['Distribution Capital Cost (M US$/PJ produced)'].iloc[0]
                    distri_fopex = this_df4_tran_dist_fac['Distribution 2% Fixed Cost (M US$/PJ produced)'].iloc[0]
                else:
                    transmi_capex = 0
                    transmi_fopex = 0
                    distri_capex = 0
                    distri_fopex = 0

                # ...we must estimate the t&d costs
                transmi_capex_list_per_tech = \
                    [transmi_capex*(list_new_prod[y]) for y in range(len(time_vector))]
                transmi_fopex_list_per_tech = \
                    [transmi_fopex*(list_generation[y]) for y in range(len(time_vector))]
                distri_capex_list_per_tech = \
                    [distri_capex*(list_new_prod[y]) for y in range(len(time_vector))]
                distri_fopex_list_per_tech = \
                    [distri_fopex*(list_generation[y]) for y in range(len(time_vector))]

                transmi_capex_list_per_tech_disc = deepcopy(transmi_capex_list_per_tech)
                transmi_fopex_list_per_tech_disc = deepcopy(transmi_fopex_list_per_tech)
                distri_capex_list_per_tech_disc = deepcopy(distri_capex_list_per_tech)
                distri_fopex_list_per_tech_disc = deepcopy(distri_fopex_list_per_tech)

                for y in range(len(time_vector)):
                    this_year = int(time_vector[y])
                    disc_constant = 1 / ((1 + r_rate/100)**(float(this_year) - r_year))
                    transmi_capex_list_per_tech_disc[y] *= disc_constant
                    transmi_fopex_list_per_tech_disc[y] *= disc_constant
                    distri_capex_list_per_tech_disc[y] *= disc_constant
                    distri_fopex_list_per_tech_disc[y] *= disc_constant

                # ...and store the results
                if tech_counter == 0:
                    dict_local_reg[this_reg][this_con].update({'Transmission CAPEX':{}})
                    dict_local_reg[this_reg][this_con].update({'Transmission Fixed OPEX':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution CAPEX':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution Fixed OPEX':{}})

                    dict_local_reg[this_reg][this_con].update({'Transmission CAPEX (disc)':{}})
                    dict_local_reg[this_reg][this_con].update({'Transmission Fixed OPEX (disc)':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution CAPEX (disc)':{}})
                    dict_local_reg[this_reg][this_con].update({'Distribution Fixed OPEX (disc)':{}})

                dict_local_reg[this_reg][this_con]['Transmission CAPEX'].update({tech: transmi_capex_list_per_tech})  # per tech
                dict_local_reg[this_reg][this_con]['Transmission Fixed OPEX'].update({tech: transmi_fopex_list_per_tech})
                dict_local_reg[this_reg][this_con]['Distribution CAPEX'].update({tech: distri_capex_list_per_tech})
                dict_local_reg[this_reg][this_con]['Distribution Fixed OPEX'].update({tech: distri_fopex_list_per_tech})
                dict_local_reg[this_reg][this_con]['Transmission CAPEX (disc)'].update({tech: transmi_capex_list_per_tech_disc})  # per tech
                dict_local_reg[this_reg][this_con]['Transmission Fixed OPEX (disc)'].update({tech: transmi_fopex_list_per_tech_disc})
                dict_local_reg[this_reg][this_con]['Distribution CAPEX (disc)'].update({tech: distri_capex_list_per_tech_disc})
                dict_local_reg[this_reg][this_con]['Distribution Fixed OPEX (disc)'].update({tech: distri_fopex_list_per_tech_disc})

                ###############################################################
                # ...increase the tech count:
                tech_counter += 1

    ###########################################################################
    # Store the elements
    dict_scen.update({this_scen: deepcopy(dict_local_reg)})

print('End the simulation here. The printing is done below.')
#sys.exit()

# ...our scenarios have run here
# 4) Now we can print the results file

# ...but first, review everything we have:
# Enlist the names of your output and the contained keys:
'''
OUTPUT:
'Global warming externalities by demand': tech (demand) / fuel / list with values as long as year
'Local pollution externalities by demand': tech (demand) / fuel / list with values as long as year
'Emissions by demand': tech (demand) / fuel / list with values as long as year
'Electricity fuel use': tech / fuel / list with values as long as year
'Global warming externalities in electricity': tech / fuel / list with values as long as year
'Local pollution externalities in electricity': tech / fuel / list with values as long as year
'Emissions in electricity': tech / fuel / list with values as long as year
'Electricity total capacity': tech / list with values as long as year
'Electricity residual capacity': tech / list with values as long as year
'Electricity new capacity': tech / list with values as long as year
'Electricity total production': tech / list with values as long as year || implicitly, electricity
'Electricity CAPEX': tech / list with values as long as year
'Electricity Fixed OPEX': tech / list with values as long as year
'Electricity Variable OPEX': tech / list with values as long as year
'Electricity Grid Connection Cost': tech / list with values as long as year
'''


# Enlist the names of your input and the contained keys:
'''
PARAMETERS:
this_externality_dict[fuel]['Global warming']  # constant
this_externality_dict[fuel]['Local pollution']  # constant
emissions_fuels_dict[fuel]  # list with values as long as year
idict_u_capex[tech]  # list with values as long as year
idict_u_fopex[tech]  # list with values as long as year
idict_u_vopex[tech]  # list with values as long as year
idict_u_gcc[tech]  # list with values as long as year
idict_cau[tech]  # list with values as long as year
idict_net_cap_factor[tech]  # list with values as long as year
idict_hr[tech]  # list with values as long as year
idict_oplife[tech]  # list with values as long as year
'''

### THIS IS ALL THE DATA THAT WE HAVE AVAILABLE:
# ...now, iterate and create a list of things you want to store:
list_dimensions = ['Strategy', 'Region', 'Country', 'Technology', 'Technology type', 'Fuel', 'Year']
list_inputs = ['Unit ext. global warming', #1
               'Unit ext. local pollution', #2
               'Emission factor', #3
               'Unit CAPEX', #4
               'Unit fixed OPEX', #5
               'Unit variable OPEX', #6
               'Unit grid connection cost', #7
               'Net capacity factor', #8
               'Heat rate', #9
               'Operational life'] #10
'''
list_outputs = ['Global warming externalities by demand', #1
                'Local pollution externalities by demand', #2
                'Emissions by demand', #3
                'Electricity fuel use', #4
                'Global warming externalities in electricity', #5
                'Local pollution externalities in electricity', #6
                'Emissions in electricity', #7
                'Electricity total capacity', #8
                'Electricity new capacity', #9
                'Electricity total production', #10
                'Electricity CAPEX', #11
                'Electricity Fixed OPEX', #12
                'Electricity Variable OPEX', #13
                'Electricity Grid Connection Cost'] #14
'''
list_outputs = ['Electricity CAPEX', #1
                'Electricity Fixed OPEX', #2
                'Electricity fuel use', #3
                'Electricity Grid Connection Cost', #4
                'Electricity new capacity', #5
                'Electricity residual capacity', #6
                'Electricity total capacity', #7
                'Electricity total production', #8
                'Electricity Variable OPEX', #9
                'Emissions by demand', #10
                'Emissions in electricity', #11
                'Energy demand by sector', #12
                'Energy demand by fuel', #13
                'Energy intensity by sector', #14
                'Global warming externalities by demand', #15
                'Global warming externalities in electricity', #16
                'Local pollution externalities by demand', #17
                'Local pollution externalities in electricity', #18
                'Electricity CAPEX (disc)', # 19 // discounted costs
                'Electricity Fixed OPEX (disc)', # 20
                'Electricity Variable OPEX (disc)', # 21
                'Electricity Grid Connection Cost (disc)', # 22
                'Global warming externalities in electricity (disc)', # 23
                'Local pollution externalities in electricity (disc)', # 24
                'Natural Gas Exports (PJ)' , # 25 // natural gas exports
                'Natural Gas Exports via Pipeline (PJ)' , # 26
                'Natural Gas Exports Liquified (PJ)' , # 27
                'Natural Gas Exports Income (M USD)' , # 28
                'Natural Gas Production (PJ)' , # 29 // production and emission factors
                'Natural Gas Production Fugitive Emissions (MTon)' , # 30
                'Natural Gas Imports (PJ)' , # 31 // production and emission factors
                'Natural Gas Imports Fugitive Emissions (MTon)' , # 32
                'Related construction jobs' , # 33
                'Related manufacturing jobs' , # 34
                'Related O&M jobs' , # 35
                'Related decommissioning jobs' , # 36
                'Transmission CAPEX' , # 37
                'Transmission Fixed OPEX' , # 38
                'Distribution CAPEX' , # 39
                'Distribution Fixed OPEX' , # 40
                'Transmission CAPEX (disc)' , # 41
                'Transmission Fixed OPEX (disc)' , # 42
                'Distribution CAPEX (disc)' , # 43
                'Distribution Fixed OPEX (disc)', # 44
                'Electricity new capacity unplanned', # 45
                'Electricity new capacity planned', # 46
                'Electricity phase out capacity', # 47
                'Electricity production share (unplanned)', # 48
                'New energy assign', # 49
                'Accumulated new capacity', # 50
                'Electricity demand to supply', # 51
                'Electricity planned supply', # 52
                'Accumulated forced new capacity', # 53
                'Electricity planned supply per technology', # 54
                'Electricity residual supply', # 55
                'Electricity residual supply per tech', # 56
                'Fleet',  # 57
                'New Fleet',  # 58
                'Transport CAPEX [$]',  # 59
                'Transport Fixed OPEX [$]',  # 60
                'Transport Variable OPEX [$]',  # 61
                'Transport Tax Imports [$]',  # 62
                'Transport Tax IMESI_Venta [$]',  # 63
                'Transport Tax IVA_Venta [$]',  # 64
                'Transport Tax Patente [$]',  # 65
                'Transport Tax IMESI_Combust [$]',  # 66
                'Transport Tax IVA_Gasoil [$]',  # 67
                'Transport Tax IVA_Elec [$]',  # 68
                'Transport Tax IC [$]',  # 69
                'Transport Tax Otros_Gasoil [$]',  # 70
                'Transport Tax Tasa_Consular [$]', # 71
                'Emisiones de fermentación entérica [kt CH4]', # 72
                'Sistema de gestión de estiércol [kt CH4]', # 73
                'Solid waste CAPEX [MUSD]', # 74
                'Solid waste CAPEX [MUSD] (disc)', # 75
                'Solid waste OPEX [MUSD]', # 76
                'Solid waste OPEX [MUSD] (disc)', # 77
                'Solid waste emissions [kt CH4]', # 78
                'Black carbon emissions by demand [ton]', # 79
                'OPEX de importación ganadera [MUSD]', #80
                'OPEX de importación ganadera [MUSD] (disc)', # 81
                'OPEX de exportación ganadera [MUSD]', # 82
                'OPEX de exportación ganadera [MUSD] (disc)', # 83
                'CAPEX de ganadera [MUSD]', #84
                'CAPEX de ganadera [MUSD] (disc)', #85
                'OPEX de ganadera [MUSD]', # 86
                'OPEX de ganadera [MUSD] (disc)', #87
                'CAPEX por Sistema de tratemiento de estiércol de ganado [MUSD]', # 88
                'OPEX fijo por Sistema de tratemiento de estiércol de ganado [MUSD]', #89
                'OPEX variable por Sistema de tratemiento de estiércol de ganado [MUSD]', #90
                'CAPEX por Sistema de tratemiento de estiércol de ganado [MUSD] (disc)', #91
                'OPEX fijo por Sistema de tratemiento de estiércol de ganado [MUSD](disc)', #92
                'OPEX variable por Sistema de tratemiento de estiércol de ganado [MUSD](disc)', #93
                'OPEX para importación de arroz [MUSD]', # 94
                'OPEX para exportación de arroz [MUSD]', # 95
                'OPEX para importación de arroz [MUSD](disc)', #96
                'OPEX para exportación de arroz [MUSD](disc)', #97
                'Emisiones de cultivo de arroz por inundación [kton CH4]', # 98
                'OPEX para cultivo de arroz por inundación [MUSD]', # 99
                'CAPEX para cultivo de arroz por inundación [MUSD]', # 100
                'OPEX para cultivo de arroz por inundación [MUSD](disc)', # 101
                'CAPEX para cultivo de arroz por inundación [MUSD](disc)', # 102
                'Emisiones de cultivo de arroz por irrigación [kton CH4]', # 103
                'OPEX para cultivo de arroz por irrigación [MUSD]', # 104
                'CAPEX para cultivo de arroz por irrigación [MUSD]', # 105
                'OPEX para cultivo de arroz por irrigación [MUSD](disc)', # 106
                'CAPEX para cultivo de arroz por irrigación [MUSD](disc)', # 107
                'Emisiones de cultivo de arroz por aireado [kton CH4]', # 108
                'OPEX para cultivo de arroz por aireado [MUSD]', # 109
                'CAPEX para cultivo de arroz por aireado [MUSD]', # 110
                'OPEX para cultivo de arroz por aireado [MUSD](disc)', # 111
                'CAPEX para cultivo de arroz por aireado [MUSD](disc)', # 112
                'Emisiones de quema de sabanas [kton CH4]', # 113
                'OPEX para quema de saban [MUSD]', # 114
                'CAPEX para quema de saban [MUSD]', # 115
                'OPEX para quema de saban [MUSD](disc)', # 116
                'CAPEX para quema de saban [MUSD](disc)', # 117
                'Quema de residuos agrícolas de caña de azúcar [kt CH4]', # 118
                'Quema de residuos agrícolas de cereales [kt CH4]', # 119
                'CAPEX de quema de residuos agrícolas de caña de azúcar [MUSD]', # 120
                'CAPEX de quema de residuos agrícolas de cereales [MUSD]', # 121
                'OPEX de quema de residuos agrícolas de caña de azúcar [MUSD]', # 122
                'OPEX de quema de residuos agrícolas de cereales [MUSD]', # 123
                'CAPEX de quema de residuos agrícolas de caña de azúcar [MUSD] (disc)', # 124
                'CAPEX de quema de residuos agrícolas de cereales [MUSD] (disc)', # 125
                'OPEX de quema de residuos agrícolas de caña de azúcar [MUSD] (disc)', # 126
                'OPEX de quema de residuos agrícolas de cereales [MUSD] (disc)', # 127
                'CAPEX para aguas residuales tratadas [MUSD]', # 128
                'OPEX fijo para aguas residuales tratadas [MUSD]', #129
                'OPEX variable para aguas residuales tratadas [MUSD]', # 130
                'CAPEX para aguas residuales tratadas [MUSD] (disc)', # 131
                'OPEX fijo para aguas residuales tratadas [MUSD] (disc)', # 132
                'OPEX variable para aguas residuales tratadas [MUSD] (disc)', # 133
                'Emisiones de aguas residuales tratadas [kt CH4]', # 134
                'Emisiones para subsector de AC [kt CO2 eq]', # 135
                'Costos para subsector de AC [MUSD]', # 136
                'Costos para subsector de AC [MUSD] (disc)', # 137
                'Emisiones para subsector de refrigeración [kt CO2 eq]', # 138
                'Costos para subsector de refrigeración [MUSD]', # 139
                'Costos para subsector de refrigeración [MUSD] (disc)', # 140
                'Emisiones para subsector de extintores [kt CO2 eq]', # 141
                'Costos para subsector de extintores [MUSD]', # 142
                'Costos para subsector de extintores [MUSD] (disc)', # 143
                'CAPEX de aguas residuales industriales [MUSD]', #144
                'CAPEX de aguas residuales industriales [MUSD] (disc)', #145
                'OPEX de aguas residuales industriales [MUSD]', #146
                'OPEX de aguas residuales industriales [MUSD] (disc)', #147
                'Emisiones de aguas residuales industriales [kton CH4]', #148
                'Emisiones carbono negro de quema de sabanas [ton]', #149
                'Carbono negro de quema de otros residuos agrícolas [ton]', #150
                'Carbono negro de quema de residuos agrícolas de caña de azúcar [ton]',  #151
                'Carbono negro por incineración de residuos [t BC]', #152
                'Transport CAPEX [$] (disc)', #153
                'Transport Fixed OPEX [$] (disc)', #154
                'Transport Variable OPEX [$] (disc)', #155
                'Compostaje de residuos agrícolas de caña de azúcar [kt CH4]', #156
                'Compostaje de residuos agrícolas de otros cultivos [kt CH4]', #157
                'CAPEX de compostaje de residuos agrícolas de caña de azúcar [MUSD]', #158
                'CAPEX de compostaje de residuos agrícolas de otros cultivos [MUSD]', #159
                'OPEX de compostaje de residuos agrícolas de caña de azúcar [MUSD]', #160
                'OPEX de compostaje de residuos agrícolas de otros cultivos [MUSD]', #161
                'CAPEX de compostaje de residuos agrícolas de caña de azúcar [MUSD] (disc)', #162
                'CAPEX de compostaje de residuos agrícolas de otros cultivos [MUSD] (disc)', #163
                'OPEX de quema de residuos agrícolas de caña de azúcar [MUSD] (disc)', #164
                'OPEX de quema de residuos agrícolas de otros cultivos [MUSD] (disc)', #165
                'Emisiones de compostaje residuos en sabanas [kton CH4]', #166
                'OPEX para compostaje de sabanas [MUSD]', #167
                'CAPEX para compostaje de sabanas [MUSD]', #168
                'OPEX para compostaje de sabanas [MUSD](disc)', #169
                'CAPEX para compostaje de sabanas [MUSD](disc)', #170
                'Methane emissions by demand [kt CH4]', #171
                'Solid waste sales [MUSD]', #172
                'Solid waste sales [MUSD] (disc)',#173
                'Solid waste externalities [MUSD]', # 174
                'Solid waste externalities [MUSD] (disc)'#175
                ] 

list_inputs_add = [i + ' (input)' for i in list_inputs]
list_outputs_add = [i + ' (output)' for i in list_outputs]

h_strategy, h_region, h_country, h_tech, h_techtype, h_fuel, h_yr = \
    [], [], [], [], [], [], []
h_i1, h_i2, h_i3, h_i4, h_i5, h_i6, h_i7, h_i8, h_i9, h_i10 = \
    [], [], [], [], [], [], [], [], [], []
h_o1, h_o2, h_o3, h_o4, h_o5, h_o6, h_o7, h_o8, h_o9, h_o10 = \
    [], [], [], [], [], [], [], [], [], []
h_o11, h_o12, h_o13, h_o14, h_o15, h_o16, h_o17, h_o18, h_o19, h_o20 = \
    [], [], [], [], [], [], [], [], [], []
h_o21, h_o22, h_o23, h_o24, h_o25, h_o26, h_o27, h_o28, h_o29, h_o30 = \
    [], [], [], [], [], [], [], [], [], []
h_o31, h_o32, h_o33, h_o34, h_o35, h_o36, h_o37, h_o38, h_o39, h_o40 = \
    [], [], [], [], [], [], [], [], [], []
h_o41, h_o42, h_o43, h_o44, h_o45, h_o46, h_o47, h_o48, h_o49, h_o50 = \
    [], [], [], [], [], [], [], [], [], []
h_o51, h_o52, h_o53, h_o54, h_o55, h_o56, h_o57, h_o58, h_o59, h_o60 = \
    [], [], [], [], [], [], [], [], [], []
h_o61, h_o62, h_o63, h_o64, h_o65, h_o66, h_o67, h_o68, h_o69, h_o70, h_o71 = \
    [], [], [], [], [], [], [], [], [], [], []
h_o72, h_o73, h_o74, h_o75, h_o76, h_o77, h_o78, h_o79, h_o80 = \
    [], [], [], [], [], [], [], [], []   
h_o81, h_o82, h_o83, h_o84, h_o85, h_o86, h_o87, h_o88, h_o89, h_o90 = \
    [], [], [], [], [], [], [], [], [], []
h_o91, h_o92, h_o93, h_o94, h_o95, h_o96, h_o97, h_o98, h_o99, h_o100 = \
    [], [], [], [], [], [], [], [], [], []
h_o101, h_o102, h_o103, h_o104, h_o105, h_o106, h_o107, h_o108, h_o109, h_o110 = \
    [], [], [], [], [], [], [], [], [], []
h_o111, h_o112, h_o113, h_o114, h_o115, h_o116, h_o117, h_o118, h_o119, h_o120 = \
    [], [], [], [], [], [], [], [], [], []
h_o121, h_o122, h_o123, h_o124, h_o125, h_o126, h_o127, h_o128, h_o129, h_o130 = \
    [], [], [], [], [], [], [], [], [], []
h_o131, h_o132, h_o133, h_o134, h_o135, h_o136, h_o137, h_o138, h_o139, h_o140 = \
    [], [], [], [], [], [], [], [], [], []
h_o141, h_o142, h_o143, h_o144, h_o145, h_o146, h_o147, h_o148, h_o149, h_o150 = \
    [], [], [], [], [], [], [], [], [], []
h_o151, h_o152, h_o153, h_o154, h_o155, h_o156, h_o157, h_o158, h_o159, h_o160 = \
    [], [], [], [], [], [], [], [], [], []
h_o161, h_o162, h_o163, h_o164, h_o165, h_o166, h_o167, h_o168, h_o169, h_o170 = \
    [], [], [], [], [], [], [], [], [], []
h_o171, h_o172, h_o173, h_o174, h_o175 = \
    [], [], [], [], []
      
# ...here, clean up the fuels:
list_fuel_clean = [i for i in list_fuel if 'Total' not in i]
if overwrite_transport_model:
    list_fuel_clean += list(dict_eq_transport_fuels.keys())
list_fuel_clean += ['']

def determine_tech_type(tech, list_demand_sector_techs, list_electric_sets_3, types_all, 
                        techs_sw, types_livestock, types_liv_opex, types_rice_capex, 
                        types_rice_opex, types_rice_opex_2, types_shares_ac, types_shares_ref, 
                        types_shares_ext, tech_ext_sw, types_industry):
    """
    Determine the technology type based on the given technology and predefined lists.

    Args:
    tech (str): The technology to be classified.
    list_demand_sector_techs, list_electric_sets_3, types_all, techs_sw, types_livestock, 
    types_liv_opex, types_rice_capex, types_rice_opex, types_rice_opex_2, types_shares_ac, 
    types_shares_ref, types_shares_ext (list): Lists of technologies for classification.

    Returns:
    str: The type of technology.
    """
    if tech in list_demand_sector_techs:
        return 'demand'
    elif tech in list_electric_sets_3:
        return 'power_plant'
    elif tech in types_all:
        return 'transport'
    elif tech in techs_sw:
        return 'waste'
    elif tech in tech_ext_sw:
        return 'waste externalities'
    elif tech in types_livestock:
        return 'cattle farming'
    elif tech in types_liv_opex:
        return 'costs for cattle farming'
    elif tech in types_rice_capex + types_rice_opex + types_rice_opex_2:
        return 'costs for rice growth'
    elif tech in types_shares_ac:
        return 'AC subsector'
    elif tech in types_shares_ref:
        return 'refrigeration subsector'
    elif tech in types_shares_ext:
        return 'extinguishers subsector'
    elif tech == types_industry:
        return 'Industrial waterwaste'
    elif tech == '':
        return ''
    else:
        return 'none' 

def handle_processes(count_empties, h_list, data_dict, fuel, tech, case_condition, case='two', fst=None, scd=None, thd=None, fth=None):
    """
    Process and append data to the specified output list based on the given conditions and output ID.

    Args:
    count_empties (int): count of list empties for each loop.
    h_list (list): The output list to append data to.
    data_dict (dict): The dictionary containing the data.
    fuel (str): The fuel type.
    tech (str): The technology type.
    case_condition (str): Indicating the combination of the condition ('01', '10', '00', '11').
        So: 
            if fuel == '' and tech != '': -> 10
            if fuel != '' and tech == '': -> 01
            if fuel != '' and tech != '': -> 00
            if fuel == '' and tech == '': -> 11
    
    case (str): The case type indicating the depth of data access in the dictionary ('two', 'three', 'four').
        Example:
            h_o33.append(this_data_dict[list_outputs[32]][tech][y])
            case = 'three'
            fst = list_outputs[32]
            scd = tech
            thd = y
    
    fst (str or list, optional): The key for the first position in the dictionary. Defaults to None.
    scd (str or int, optional): The key for the second position in the dictionary. Defaults to None.
    thd (str or int, optional): The key for the third position in the dictionary. Defaults to None.
    fth (str or int, optional): The key for the fourth position in the dictionary. Defaults to None.

    Returns:
    int: The count of empty entries appended to the list.
    """
    condition_flag = False

    if case_condition=='10':
        if fuel == '' and tech != '':
            condition_flag = True
    elif case_condition=='01':
        if fuel != '' and tech == '':
            condition_flag = True
    elif case_condition=='00':
        if fuel != '' and tech != '':
            condition_flag = True
    elif case_condition=='11':
        if fuel == '' and tech == '':
            condition_flag = True
            
            
    if condition_flag: 
        try:
            if case == 'two':
                h_list.append(data_dict[fst][scd])
            elif case == 'three':
                h_list.append(data_dict[fst][scd][thd])
            elif case == 'four':
                h_list.append(data_dict[fst][scd][thd][fth])
        except Exception:
            h_list.append(0)
            count_empties += 1                
    else:
        h_list.append(0)
        count_empties += 1  

    return count_empties, h_list

def pop_last_from_outputs(output_lists, output_range):
    """
    Remove the last element from each output list in the specified range.

    Args:
    output_lists (dict): Dictionary containing output lists.
    output_range (range): A range of output IDs for which the last element should be removed.
    """
    for output_id in output_range:
        output_list_name = f'h_o{output_id}'
        if output_list_name in output_lists:
            output_lists[output_list_name].pop()            
    return output_lists

def pop_last_from_inputs(input_lists, input_range):
    """
    Remove the last element from each output list in the specified range.

    Args:
    input_lists (dict): Dictionary containing input lists.
    input_range (range): A range of input IDs for which the last element should be removed.
    """
    for input_id in input_range:
        input_list_name = f'h_i{input_id}'
        if input_list_name in input_lists:
            input_lists[input_list_name].pop()
    return input_lists

output_lists = {f'h_o{i}': [] for i in range(1, 176)}
input_lists = {f'h_i{i}': [] for i in range(1, 11)}

print('\n')
print('PROCESS 2 - PRINTING THE INPUTS AND OUTPUTS')
for s in range(len(scenario_list)):
    this_scen = scenario_list[s]

    regions_list = ['1_Mexico', '2_Central America', '3_Caribbean', '4_The Amazon', '5_Southern Cone']

    for r in range(len(regions_list)):
        this_reg = regions_list[r]

        country_list = dict_regs_and_countries[this_reg]
        country_list.sort()

        # Add a filter to include countries with transport data only:
        country_list = [c for c in country_list if c in tr_list_app_countries_u]

        for c in range(len(country_list)):
            this_country = country_list[c]

            types_all_rac = types_shares_ac + types_shares_ref + types_shares_ext
            types_all_rac = list(set(types_all_rac))
            types_all_rac.sort()

            # First iterable: list_demand_sector_techs
            # Second iterable: list_electric_sets_3
            # inner iterable 1: list_fuel_clean
            # inner iterable 2: time_vector                    
            tech_iterable = list_demand_sector_techs +  list_electric_sets_3 +\
                types_all + techs_sw + types_livestock + types_liv_opex + \
                    types_rice_capex + types_rice_opex + types_rice_opex_2 + \
                    types_all_rac + types_industry + tech_ext_sw + ['']

            for tech in tech_iterable:
                for fuel in list_fuel_clean:
                    for y in range(len(time_vector)):
                        count_empties = 0
                        # Amount of inputs and outputs
                        inputs_real_amount = 10
                        outputs_real_amount = 175

                        # These variables are for loops where it don't need the real amount,
                        # it need a the same quantity but take considerations for lists definitions
                        # in python
                        inputs_len = inputs_real_amount + 1
                        outputs_len = outputs_real_amount + 1

                        tech_type = determine_tech_type(tech, list_demand_sector_techs, list_electric_sets_3, types_all, 
                                                        techs_sw, types_livestock, types_liv_opex, types_rice_capex, 
                                                        types_rice_opex, types_rice_opex_2, types_shares_ac, types_shares_ref,
                                                        types_shares_ext, tech_ext_sw, types_industry)
                        this_data_dict = \
                            dict_scen[this_scen][this_reg][this_country]

                        # Store inputs:                          
                        count_empties, input_lists['h_i1'] = handle_processes(count_empties, input_lists['h_i1'], ext_by_country, fuel, tech, '01','three', this_country, fuel, 'Global warming')
                        count_empties, input_lists['h_i2'] = handle_processes(count_empties, input_lists['h_i2'], ext_by_country, fuel, tech, '01','three', this_country, fuel, 'Local pollution')
                        count_empties, input_lists['h_i3'] = handle_processes(count_empties, input_lists['h_i3'], emissions_fuels_dict, fuel, tech, '01','two', fuel, y)
                        count_empties, input_lists['h_i4'] = handle_processes(count_empties, input_lists['h_i4'], idict_u_capex, fuel, tech, '10', 'two',tech, y)
                        count_empties, input_lists['h_i5'] = handle_processes(count_empties, input_lists['h_i5'], idict_u_fopex, fuel, tech, '10', 'two', tech, y)
                        count_empties, input_lists['h_i6'] = handle_processes(count_empties, input_lists['h_i6'], idict_u_vopex, fuel, tech, '10', 'two', tech, y)
                        count_empties, input_lists['h_i7'] = handle_processes(count_empties, input_lists['h_i7'], idict_u_gcc, fuel, tech, '10', 'two', tech, y)
                        count_empties, input_lists['h_i8'] = handle_processes(count_empties, input_lists['h_i8'], idict_net_cap_factor_by_scen_by_country, fuel, tech, '10', 'four', this_scen, this_country, tech, y)
                        count_empties, input_lists['h_i9'] = handle_processes(count_empties, input_lists['h_i9'], idict_hr, fuel, tech, '10', 'two', tech, y)
                        count_empties, input_lists['h_i10'] = handle_processes(count_empties, input_lists['h_i10'], idict_oplife, fuel, tech, '10', 'two', tech, y)


                        # Store outputs:
                        # Lists of IDs for case conditions
                        list_output_01 = [13]
                        list_output_10 = [1, 2, 4, 5, 6, 7, 8, 9, 14, 19, 20, 21, 22, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 54, 56, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 104, 105, 106, 107, 109, 110, 111, 112, 135, 136, 137, 138, 139, 140, 141, 142, 143, 146, 147, 148, 172, 173, 174, 175]
                        list_output_00 = [3, 10, 11, 12, 15, 16, 17, 18, 23, 24, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 79, 153, 154, 155, 171]
                        list_output_11 = [25, 26, 27, 28, 29, 30, 31, 32, 51, 52, 55, 84, 85, 94, 95, 96, 97, 98, 103, 108, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 144, 145, 149, 150, 151, 152, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]
                        
                        # Lists of cases to select the correct parameters for which one case
                        case_type_two = [25, 26, 27, 28, 29, 30, 31, 32, 51, 52, 55, 84, 85, 94, 95, 96, 97, 98, 103, 108, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 144, 145, 149, 150, 151, 152, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170]
                        case_type_three = [1, 2, 4, 5, 6, 7, 8, 9, 19, 20, 21, 22, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 54, 56, 72, 73, 74, 75, 76, 77, 78, 80, 81, 82, 83, 86, 87, 88, 89, 90, 91, 92, 93, 99, 100, 101, 102, 104, 105, 106, 107, 109, 110, 111, 112, 135, 136, 137, 138, 139, 140, 141, 142, 143, 146, 147, 148, 172, 173, 174, 175]
                        case_type_three_2 = [13]
                        case_type_four = [3, 10, 11, 12, 15, 16, 17, 18, 23, 24, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 79, 153, 154, 155, 171]
                        case_type_four_2 = [14]
                        
                        for output_id in range(1, outputs_len):  # Loop for all outputs

                            # Conditions to select the correct case condition for combination of "fuel" and "tech"                        
                            if output_id in list_output_01:
                                case_condition = '01'
                            elif output_id in list_output_10:
                                case_condition = '10'
                            elif output_id in list_output_00:
                                case_condition = '00'
                            elif output_id in list_output_11:
                                case_condition = '11'   
                                
                            # Conditions to select the correct parameters for which one case
                            if output_id in case_type_two:
                                # Case 'two'
                                case_type = 'two'
                                fst_key = list_outputs[output_id - 1]
                                scd_key = y
                                thd_key = None
                                fth_key = None
                            elif output_id in case_type_three:
                                # Case 'three' 1
                                case_type = 'three'
                                fst_key = list_outputs[output_id - 1]
                                scd_key = tech
                                thd_key = y
                                fth_key = None
                            elif output_id in case_type_three_2:
                                # Case 'three' 2
                                case_type = 'three'
                                fst_key = list_outputs[output_id - 1]
                                scd_key = fuel
                                thd_key = y
                                fth_key = None
                            elif output_id in case_type_four:
                                # Case 'four' 1
                                case_type = 'four'
                                fst_key = list_outputs[output_id - 1]
                                scd_key = tech
                                thd_key = fuel
                                fth_key = y
                            elif output_id in case_type_four_2:
                                # Case 'four' 2
                                case_type = 'four'
                                fst_key = list_outputs[output_id - 1]
                                scd_key = tech
                                thd_key = 'Total'
                                fth_key = y

                            # Process outputs                            
                            count_empties, output_lists[f'h_o{output_id}'] = handle_processes(count_empties, output_lists[f'h_o{output_id}'], this_data_dict, fuel, tech, case_condition, case_type, fst_key, scd_key, thd_key, fth_key)

                        # test: quatities of outputs
                        q_list_ouputs = list_output_01 + list_output_10 + list_output_00 + list_output_11
                        q_list_ouputs.sort()
                        q_case_types = case_type_two + case_type_three + case_type_three_2 + case_type_four + case_type_four_2
                        q_case_types.sort()
                        q_output_lists = list(range(1,len(output_lists)+1))
                        if q_list_ouputs != q_case_types:
                            print('"q_list_ouputs" and "q_case_types" have diferent numbers')
                            print('Make corrections')
                            sys.exit()
                        elif q_list_ouputs != q_output_lists:
                            print('"q_list_ouputs" and "q_output_lists" have diferent numbers')
                            print('Make corrections')
                            sys.exit()
                        elif q_output_lists != q_case_types:
                            print('"q_output_lists" and "q_case_types" have diferent numbers')
                            print('Make corrections')
                            sys.exit()
                                                    
                        if count_empties == (inputs_real_amount + outputs_real_amount):  # gotta pop, because it is an empty row:
                            # Inputs
                            input_lists = pop_last_from_inputs(input_lists, range(1, inputs_len))
                            # Outputs
                            output_lists = pop_last_from_outputs(output_lists, range(1, outputs_len))

                        else:
                            h_strategy.append(this_scen)
                            h_region.append(this_reg)
                            h_country.append(this_country)
                            h_tech.append(tech)
                            h_techtype.append(tech_type)
                            h_fuel.append(fuel)
                            h_yr.append(time_vector[y])

# Review if *zero* elements exist:    
# List of names variables

variable_names = \
    [f'h_i{i}' for i in range(1, inputs_len)] + \
    [f'h_o{i}' for i in range(1, outputs_len)]

# Construct list_variables by accessing dictionaries
list_inputs_variables = [input_lists[var_name] for var_name in variable_names if var_name in input_lists]
list_outputs_variables = [output_lists[var_name] for var_name in variable_names if var_name in output_lists]
list_variables = list_inputs_variables + list_outputs_variables

h_count = 0
h_zeros = []
for h in list_variables:
    if sum(h) == 0.0:
        h_zeros.append(h_count)
    h_count += 1


# Review the lengths:
print(1, list_dimensions[0], len(h_strategy)) #1
print(2, list_dimensions[1], len(h_region)) #2
print(3, list_dimensions[2], len(h_country)) #3
print(4, list_dimensions[3], len(h_tech)) #4
print(5, list_dimensions[4], len(h_techtype)) #5
print(6, list_dimensions[5], len(h_fuel)) #6
print(7, list_dimensions[6], len(h_yr)) #7
# Inputs
for i in range(len(list_inputs_add)):
    print(8 + i, list_inputs_add[i], len(input_lists[f'h_i{i + 1}']))
# Outputs
for i in range(len(list_outputs_add)):
    print(18 + i, list_outputs_add[i], len(output_lists[f'h_o{i + 1}']))

# Convert to output:
print('\n')
print('Convert lists to dataframe for printing:')
dict_output = {list_dimensions[0]: h_strategy, #1
               list_dimensions[1]: h_region, #2
               list_dimensions[2]: h_country, #3
               list_dimensions[3]: h_tech, #4
               list_dimensions[4]: h_techtype, #5
               list_dimensions[5]: h_fuel, #6
               list_dimensions[6]: h_yr, #7                              
               }
# Inputs
for i in range(len(list_inputs_add)):
    dict_output[list_inputs_add[i]] = input_lists[f'h_i{i + 1}']
# Outputs
for i in range(len(list_outputs_add)):
    dict_output[list_outputs_add[i]] = output_lists[f'h_o{i + 1}']

# Let's print variable costs
df_output_name = 'model_BULAC_simulation.csv'
df_output = pd.DataFrame.from_dict(dict_output)
df_output.to_csv(path + '/' + df_output_name, index=None, header=True)

# Recording final time of execution:
end_f = time.time()
te_f = -start_1 + end_f  # te: time_elapsed
print(str(te_f) + ' seconds /', str(te_f/60) + ' minutes')
print('*: This automatic analysis is finished.')
