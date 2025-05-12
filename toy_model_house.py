#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 08:30:10 2024

@author: cghiaus
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import control as ctrl
import time

import dm4bem

"""
Toy model house:
https://cghiaus.github.io/dm4bem_book/tutorials/02_2_0Toy.html

Hierarchy diagram

1. assembled_thermal_circuit_from_folder(folder_bldg)
    - Reads the thermal circuits and wall descriptions from files in
    `folder_bldg`.
    - Returns a dictionary representing the thermal circuit (TC).

2. modify_param_thermal_circuit(TC, controller, indoor_air_capacity,
    glass_capacity, insulation_width_new, folder_bldg)
    - Modifies parameters (conductances and capacities) of the thermal
    circuit `TC`.
    - Returns the modified thermal circuit (TC).

3. time_step(As)
    - Finds the time step from the eigenvalues of the state matrix `As`.
    - Returns the time step (dt).

4. sources_in_time(file_weather, date_start, date_end, folder_bldg)
    - Generates temperature and flow-rate sources in time from `date_start`
    to `date_end`.
    - Returns a DataFrame with the input data set.

5. Euler_explicit(θ0, As, Bs, Cs, Ds, u, dt)
    - Performs numerical integration of the state-space model using
    Euler explicit method.
    - Returns a DataFrame with the output temperatures in time (y).

6. control_sys_response(θ0, As, Bs, Cs, Ds, u, dt)
    - Performs numerical integration of the state-space model using
    the control system's ODE solver.
    - Returns a DataFrame with the output temperatures in time (y).

7. plot_simulation_in_time_pd(dt, input_data_set, y, q_HVAC)
    - Plots the simulation results in time using pandas DataFrames and Series.
    - No return value.

8. plot_simulation_in_time_np(dt, input_data_set, y, q_HVAC)
    - Plots the simulation results in time using numpy arrays.
    - No return value.

9. print_results(dt, input_data_set, y, q_HVAC)
    - Prints the simulation results including mean, min, and
    max temperatures, max load, and energy consumption.
    - No return value.

10. simulation( ... )
    - Simulates in time.
    - Calls:
        - assembled_thermal_circuit_from_folder()
        - modify_param_thermal_circuit()
        - dm4bem.tc2ss(TC)
        - time_step()
        - sources_in_time()
        - input_data_set.resample()
        - dm4bem.inputs_in_time()
        - Euler_explicit()
        - control_sys_response()
        - plot_simulation_in_time_pd()
        - plot_simulation_in_time_np()
        - print_results()
    - Returns None.

11. main()
    - Provides data and calls simulation( ... ).
    - Returns None.
"""


def assembled_thermal_circuit_from_folder(folder_bldg='bldg'):
    """
    Obtain an assembled thermal circuit from a folder containing:
        - TC*.csv             thermal circuits
        - wall_types.csv      description of wall types
        - walls_out.csv       instances of walls based on wall_types
        - assembly_lists.csv  the assembling of walls and TC

    Parameters
    ----------
    folder_bldg : str, optional
        Name of subfolder containing the description of the building.
        The default is 'bldg'.

    Returns
    -------
    TC : dict
        Thermal circuit: A, G, C, b, f, y:
            TC = {'A': DataFrame, Index '*_q*', Columns '*_θ*', values 0, -1, 1
                  'G': Series, Index '*_q*', values float
                  'C': Series, Index '*_θ*', values float
                  'b': Series, Index '*_q*', values str or 0
                  'f': Series, Index '*_θ*', values str or 0
                  'y': Series, Index '*_θ*', values float}
    """
    # Disassembled thermal circuits
    TCd = dm4bem.bldg2TCd(folder_bldg,
                          TC_auto_number=True)

    # Assembled thermal circuit
    ass_lists = pd.read_csv(folder_bldg + '/assembly_lists.csv')
    ass_matrix = dm4bem.assemble_lists2matrix(ass_lists)
    TC = dm4bem.assemble_TCd_matrix(TCd, ass_matrix)
    return TC


def modify_param_thermal_circuit(TC,
                                 controller,
                                 indoor_air_capacity,
                                 glass_capacity,
                                 insulation_width_new,
                                 folder_bldg='bldg'):
    """
    Modify some parameters (i.e., coductances and capacities) of the
    thermal circuit `TC given in `folder_bldg`.

    Parameters
    ----------
    TC : dict
        Thermal circuit: A, G, C, b, f, y:
            TC = {'A': DataFrame, Index '*_q*', Columns '*_θ*', values 0, -1, 1
                  'G': Series, Index '*_q*', values float
                  'C': Series, Index '*_θ*', values float
                  'b': Series, Index '*_q*', values str or 0
                  'f': Series, Index '*_θ*', values str or 0
                  'y': Series, Index '*_θ*', values float}

    controller : bool
        Consider or not the controller.

    indoor_air_capacity : bool
        Consider or not the heat capacity of the indoor air.

    glass_capacity : bool
        Consider or not the heat capacity of the window.

    insulation_width_new : float
        New width for the insulation layer.

    folder_bldg : str, optional
        Name of the folder in which is the description of the building.
        The default is 'bldg'.

    Returns
    -------
    TC : dict
        Thermal circuit: A, G, C, b, f, y:
            TC = {'A': DataFrame, Index '*_q*', Columns '*_θ*', values 0, -1, 1
                  'G': Series, Index '*_q*', values float
                  'C': Series, Index '*_θ*', values float
                  'b': Series, Index '*_q*', values str or 0
                  'f': Series, Index '*_θ*', values str or 0
                  'y': Series, Index '*_θ*', values float}

    """

    if controller:
        TC['G']['c3_q0'] = 1e3  # Kp, controler gain
    if not indoor_air_capacity:
        TC['C']['c2_θ0'] = 0    # indoor air heat capacity
    if not glass_capacity:
        TC['C']['c1_θ0'] = 0    # glass (window) heat capacity

    # get insulation width from wall_types.csv
    df = pd.read_csv(folder_bldg + "/wall_types.csv")
    insulation_width_old = df[
        df['Material'] == 'Insulation']['Width'].values[0]

    # insulation width
    TC['G']['ow0_q3'] *= insulation_width_old / insulation_width_new
    TC['G']['ow0_q4'] = TC['G']['ow0_q3']
    return TC


def time_step(As):
    """
    Find time step from eigenvalues of transfer matrix of the state-space
    representation.

    Parameters
    ----------
    As : DataFrame
        State matrix in state equation. Index '*_θ*', Columns '*_θ*',
        values float.

    Returns
    -------
    dt : float
        Time step in seconds floor rounded to 2 * min(-1. / λ), where λ is the
        list of eigenvalues of state matrix As.

    """
    # Eigen-values analysis
    λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As
    dt_max = 2 * min(-1. / λ)   # max time step for Euler explicit stability
    dt = dm4bem.round_time(dt_max)
    return dt


def sources_in_time(file_weather,
                    date_start,
                    date_end,
                    folder_bldg='bldg'):
    """
    Temperature and flow-rate sources in time from date_start to date_end at
    1 h time step.

    Parameters
    ----------
    file_weather : str
        Name of weather file (EnergyPlus .epw format).

    date_start : str
        Date and time for start in format `yyyy-mm-dd HH:MM`. Note: MM = 00.

    date_end : str
        Date and time for end in format `yyyy-mm-dd HH:MM`. Note: MM = 00.

    folder_bldg : str, optional
        Name of the folder in which is the description of the building.
        The default is 'bldg'.

    Returns
    -------
    input_data_set : DataFrame
        Set of temperature and flow-rate sources in time from `date_start`
        to `date_end` in time step 1 h.
        Index : datetimes
            Time at 1 h time step.
        Columns : names of the temperature and flow sources, e.g., 'To', 'Φo'
        Values: float

    """

    # Weather data
    [data, meta] = dm4bem.read_epw(file_weather, coerce_year=None)
    weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
    del data

    # select weather data from date_start to date_end
    weather.index = weather.index.map(lambda t: t.replace(year=2000))
    weather = weather.loc[date_start:date_end]

    # Temperature sources
    # ===================
    To = weather['temp_air']

    Ti_day, Ti_night = 20, 16
    Ti_sp = pd.Series(20, index=To.index)
    Ti_sp = pd.Series(
        [Ti_day if 6 <= hour <= 22 else Ti_night for hour in To.index.hour],
        index=To.index)

    # Flow-rate sources
    # =================
    # total solar irradiance
    wall_out = pd.read_csv(folder_bldg + '/walls_out.csv')
    w0 = wall_out[wall_out['ID'] == 'w0']

    surface_orientation = {'slope': w0['β'].values[0],
                           'azimuth': w0['γ'].values[0],
                           'latitude': 45}

    rad_surf = dm4bem.sol_rad_tilt_surf(
        weather, surface_orientation, w0['albedo'].values[0])

    Etot = rad_surf.sum(axis=1)

    # window glass properties
    α_gSW = 0.38    # short wave absortivity: reflective blue glass
    τ_gSW = 0.30    # short wave transmitance: reflective blue glass
    S_g = 9         # m2, surface area of glass

    # flow-rate sources:
    # solar radiation
    Φo = w0['α1'].values[0] * w0['Area'].values[0] * Etot
    Φi = τ_gSW * w0['α0'].values[0] * S_g * Etot
    Φa = α_gSW * S_g * Etot
    # auxiliary (internal) sources
    Qa = pd.Series(0, index=To.index)

    # Input data set
    input_data_set = pd.DataFrame({'To': To, 'Ti_sp': Ti_sp,
                                   'Φo': Φo, 'Φi': Φi, 'Qa': Qa, 'Φa': Φa,
                                   'Etot': Etot})
    return input_data_set


def Euler_explicit(θ0, As, Bs, Cs, Ds, u, dt):
    """
    Numerical integration in time of the state-space model using Eulet implicit
    with a fixed time step.

    Parameters
    ----------
    θ0 : float or int
        Initial value for state varaibles (temperatures nodes with capacities).

    As : DataFrame
        State matrix in state equation.
        Index and Columns `*_θ*` state variables.

    Bs : DataFrame
        Input matrix in state equation.
        Index `*_θ*` state variables.
        Columns `*_q* ... *_θ*` branches and nodes with
        temperature and flow-rate sources corresponding to input vector `u`.

    Cs : DataFrame
        Output matrix in observation equation.
        Index `*_θ*` output nodes given by vector `y` in thermal circuit.
        Columns `*_θ*` state variables.

    Ds : DataFrame
        Input matrix in observation equation.
        Index `*_θ*` output nodes given by vector `y` in thermal circuit.
        Columns `*_q* ... *_θ*` branches and nodes with
        temperature and flow-rate sources corresponding to input vector `u`.

    u : DataFrame
        Input vector in time.
        Index datetime from date_start to date_end with step dt.
        Columns `*_q* ... *_θ*` branches and nodes with
        temperature and flow-rate sources.

    dt : float
        Time step in seconds.

    Returns
    -------
    y : DataFrame
        Output temperatures in time.
        Index datetime from date_start to date_end with step dt.
        Columns `*_θ*` temperature nodes that are the Index of Cs and Ds
        indicated by vector `y` in thermal circuit.

    """
    # θ_exp = pd.DataFrame(index=u.index)
    # θ_exp[As.columns] = θ0      # Fill θ with initial valeus θ0
    # θ_exp = θ_exp.astype(float)
    θ_exp = pd.DataFrame(index=u.index, columns=As.columns)
    θ_exp.iloc[0] = θ0

    # time integration
    I = np.eye(As.shape[0])     # identity matrix

    for k in range(u.shape[0] - 1):
        θ_exp.iloc[k + 1] = (I + dt * As)\
            @ θ_exp.iloc[k] + dt * Bs @ u.iloc[k + 1]

    # outputs
    y = (Cs @ θ_exp.T + Ds @  u.T).T
    return y


def control_sys_response(θ0, As, Bs, Cs, Ds, u, dt):
    """
    Numerical integration in time of the state-space model using
    control.input_output_response (which uses ODE solver from SciPy).

    Parameters
    ----------
    θ0 : float or int
        Initial value for state varaibles (temperatures nodes with capacities).

    As : DataFrame
        State matrix in state equation.
        Index and Columns `*_θ*` state variables.

    Bs : DataFrame
        Input matrix in state equation.
        Index `*_θ*` state variables.
        Columns `*_q* ... *_θ*` branches and nodes with
        temperature and flow-rate sources corresponding to input vector `u`.

    Cs : DataFrame
        Output matrix in observation equation.
        Index `*_θ*` output nodes given by vector `y` in thermal circuit.
        Columns `*_θ*` state variables.

    Ds : DataFrame
        Input matrix in observation equation.
        Index `*_θ*` output nodes given by vector `y` in thermal circuit.
        Columns `*_q* ... *_θ*` branches and nodes with
        temperature and flow-rate sources corresponding to input vector `u`.

    u : DataFrame
        Input vector in time.
        Index datetime from date_start to date_end with step dt.
        Columns `*_q* ... *_θ*` branches and nodes with
        temperature and flow-rate sources.

    dt : float
        Time step in seconds.

    Returns
    -------
    y : DataFrame
        Output temperatures in time.
        Index datetime from date_start to date_end with step dt.
        Columns `*_θ*` temperature nodes that are the Index of Cs and Ds
        indicated by vector `y` in thermal circuit.

    """
    sys = ctrl.ss(As, Bs, Cs, Ds)
    θ0 = θ0 * np.ones(As.shape[0])
    u_np = u.values.T               # inputs in time (numpy array).
    t = dt * np.arange(u.shape[0])  # time vector (numpy array)
    # Simulate the system response with Python Control Systems Library
    t, y = ctrl.input_output_response(sys, T=t, U=u_np, X0=θ0)
    y = pd.DataFrame(y.T, index=u.index, columns=Cs.index)
    return y


def plot_simulation_in_time_pd(dt, input_data_set, y, q_HVAC):
    """
    Plots, by using pandas DataFrames and Series,
    the simulation results in time:
        1. Indoor and outdoor temperature in time.
        2. Total solar irradiance and heat load per square meter.

    Advantage: the time axis is set automatically to the time of the year
    (month, day, hour).

    Disadvantage: it is slow for long periods; the may plot take a few minutes
    for 365 days of simumation.

    Parameters
    ----------
    dt : float
        Time step in seconds.

    input_data_set : DataFrame
        Set of temperature and flow-rate sources in time from `date_start`
        to `date_end` in time step 1 h.
        Index : datetimes
            Time at 1 h time step.
        Columns : names of the temperature and flow sources, e.g., 'To', 'Φo'
        Values: float

    y : DataFrame
        Output temperatures in time.
        Index: datetime from date_start to date_end with step dt.
        Columns: `*_θ*` temeprature nodes indicated by `y` in thermal circuit.

    q_HVAC : Series
        Time series of heat load (in W/m²).
        Index datetime from date_start to date_end with step dt.

    Returns
    -------
    None.

    """

    data = pd.DataFrame({'To': input_data_set['To'],
                         'θi': y['c2_θ0'],
                         'Etot': input_data_set['Etot'],
                         'q_HVAC': q_HVAC})

    hours = dt / 3600.
    minutes = dt / 60.
    if hours > 1:
        dt_string = f' = {int(dt)} s = {float(hours):.1f} h'
    elif minutes > 1:
        dt_string = f' = {int(dt)} s = {float(minutes):.1f} min'
    elif dt > 1:
        dt_string = f' = {int(dt)} s'
    else:
        dt_string = f' = {dt:.3f} s'

    fig, axs = plt.subplots(2, 1)
    data[['To', 'θi']].plot(ax=axs[0],
                            xticks=[],
                            ylabel='Temperature, $θ$ / °C')
    axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor}$'],
                  loc='upper right')

    data[['Etot', 'q_HVAC']].plot(ax=axs[1],
                                  ylabel='Heat rate, $q$ / (W·m⁻²)')
    axs[1].set(xlabel='Time')
    axs[1].legend(['$E_{total}$', '$q_{HVAC}$'],
                  loc='upper right')
    axs[0].set_title(f'Time step: $dt$ = {dt:.0f} s')
    axs[0].set_title(f'Time step:{dt_string}')
    plt.show()


def plot_simulation_in_time_np(dt, input_data_set, y, q_HVAC):
    """
    Plots, by using numpy arrays, the simulation results in time:
        1. Indoor and outdoor temperature in time.
        2. Total solar irradiance and heat load per square meter.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    input_data_set : DataFrame
        Set of temperature and flow-rate sources in time from `date_start`
        to `date_end` in time step 1 h.
        Index : datetimes
            Time at 1 h time step.
        Columns : names of the temperature and flow sources, e.g., 'To', 'Φo'
        Values: float
    y : DataFrame
        Output temperatures in time.
        Index datetime from date_start to date_end with step dt.
        Columns `*_θ*` temeprature nodes indicated by `y` in thermal circuit.
    q_HVAC : Series
        Time series of heat load (in W/m²).
        Index datetime from date_start to date_end with step dt.

    Returns
    -------
    None.

    """

    data = pd.DataFrame({'To': input_data_set['To'].values,
                         'θi': y['c2_θ0'].values,
                         'Etot': input_data_set['Etot'].values,
                         'q_HVAC': q_HVAC.values})

    hours = dt / 3600.
    minutes = dt / 60.
    if hours > 1:
        dt_string = f' = {int(dt)} s = {float(hours):.1f} h'
    elif minutes > 1:
        dt_string = f' = {int(dt)} s = {float(minutes):.1f} min'
    elif dt > 1:
        dt_string = f' = {int(dt)} s'
    else:
        dt_string = f' = {dt:.3f} s'

    if data.index[-1] * dt < 6 * 3600:
        time_unit = 's'
    elif data.index[-1] * dt < 5 * 24 * 3600:
        time_unit = 'h'
        new_index = data.index * dt / 3600
    else:
        time_unit = 'day'
        new_index = data.index * dt / (3600 * 24)
    data.index = new_index

    fig, axs = plt.subplots(2, 1)
    data[['To', 'θi']].plot(ax=axs[0],
                            xticks=[],
                            ylabel='Temperature, $θ$ / °C')
    axs[0].legend(['$θ_{outdoor}$', '$θ_{indoor}$'],
                  loc='upper right')

    data[['Etot', 'q_HVAC']].plot(ax=axs[1],
                                  ylabel='Heat rate, $q$ / (W·m⁻²)')
    axs[1].set(xlabel=f'Time, $t$ / {time_unit}')
    axs[1].legend(['$E_{total}$', '$q_{HVAC}$'],
                  loc='upper right')
    axs[0].set_title(f'Time step: $dt$ = {dt:.0f} s')
    axs[0].set_title(f'Time step:{dt_string}')
    plt.show()


def print_results(dt, input_data_set, y, q_HVAC):
    """
    Print the results:
        - Mean, min and max outdoor temperature.
        - Max load.
        - Energy consumption for heating and for cooling.

    Parameters
    ----------
    dt : float
        Time step in seconds.
    input_data_set : DataFrame
        Set of temperature and flow-rate sources in time from `date_start`
        to `date_end` in time step 1 h.
        Index : datetimes
            Time at 1 h time step.
        Columns : names of the temperature and flow sources, e.g., 'To', 'Φo'
        Values: float
    y : DataFrame
        Output temperatures in time.
        Index datetime from date_start to date_end with step dt.
        Columns `*_θ*` temeprature nodes indicated by `y` in thermal circuit.
    q_HVAC : Series
        Time series of heat load (in W/m²).
        Index datetime from date_start to date_end with step dt.

    Returns
    -------
    None.

    """
    data = pd.DataFrame({'To': input_data_set['To'],
                         'θi': y['c2_θ0'],
                         'Etot': input_data_set['Etot'],
                         'q_HVAC': q_HVAC})

    # Outputs
    dm4bem.print_rounded_time("Time step:", dt)
    print(f"Mean outdoor temperature: {data['To'].mean():.1f} °C")
    print(f"Min. indoor temperature: {data['θi'].min():.1f} °C")
    print(f"Max. indoor temperature: {data['θi'].max():.1f} °C")

    max_load = data['q_HVAC'].max()
    max_load_index = data['q_HVAC'].idxmax()
    Q_heat = q_HVAC[q_HVAC > 0].sum() * dt / 3.6e6      # kWh
    Q_cool = q_HVAC[q_HVAC < 0].sum() * dt / 3.6e6      # kWh

    print(f"Max. load: {max_load:.1f} W at {max_load_index}")
    print(f"Energy consumption for heating: {Q_heat:.1f} kWh")
    print(f"Energy consumption for cooling: {Q_cool:.1f} kWh")


def simulation(
        date_start='2000-01-01 12:00',
        date_end='2000-01-02 12:00',
        folder_bldg='bldg',
        file_weather='weather_data/FRA_Lyon.074810_IWEC.epw',
        imposed_time_step=0,
        numerical_integration_Euler=True,
        controller=True,
        indoor_air_capacity=False,
        glass_capacity=False,
        insulation_width_new=0.16,
        plot_with_pandas=False):
    """
    Simulation of toy model house.

    Parameters
    ----------
    date_start : str, optional
        Date and time for start in format `yyyy-mm-dd HH:MM`. Note: MM = 00.
        The default is '2000-01-01 12:00'.
    date_end : str, optional
        Date and time for end in format `yyyy-mm-dd HH:MM`. Note: MM = 00.
        The default is '2000-01-02 12:00'.
    folder_bldg : str, optional
        Name of the folder in which is the description of the building.
        The default is 'bldg'.
    file_weather : str, optional
        Name of weather file (EnergyPlus .epw format).
        The default is 'weather_data/FRA_Lyon.074810_IWEC.epw'.
    imposed_time_step : int or float, optional
        Time step (in seconds). If zero, the time step is selected
        by function `time_step()` so that it is smaller than the double of the
        shortest time constant T = - 1 / λ, 2 * min(-1. / λ),
        where λ is the list of eigenvalues of state matrix As.
        The default is 0 (i.e., selected automatically from
        eigenvalue analysis).
    numerical_integration_Euler : bool, optional
        Numerical integration by using Euler explicit method. If Flase, then
        `control_sys_response()` is used (ODE solver from SciPy).
        The default is True.
    controller : bool, optional
        If the controller is On, the proportional gain of the controller, which
        is the conductance G correspondng to flow q0 from circuit c3, is
        TC['G']['c3_q0'] = 1e3 (see function `modify_param_thermal_circuit()`).
        The default is True.
    indoor_air_capacity : bool, optional
        Consider or not the heat capacity of the indoor air, which
        is the capacity C corresponding to node θ0 from circuit c3,
        TC['C']['c2_θ0']. Can be modified in `modify_param_thermal_circuit()`.
        The default is False.
    glass_capacity : bool, optional
        Consider or not the heat capacity of the glass, which
        is the capacity C corresponding to node θ0 from circuit c1,
        TC['C']['c1_θ0']. Can be modified in `modify_param_thermal_circuit()`.
        The default is False.
    insulation_width_new : float, optional
        The width of the insulation in meters.
        The default is 0.16.
    plot_with_pandas : bool, optional
        Plots the results by using pandas `plot_simulation_in_time_pd()` (slow)
        or by using numpy `plot_simulation_in_time_np()` (fast).
        By using pandas, the time of the year is indicated.
        By using numpy,, the time duration is indicated.
        The default is False.

    Returns
    -------
    None.

    Calls
    -----
    - assembled_thermal_circuit_from_folder()
    - modify_param_thermal_circuit()
    - dm4bem.tc2ss(TC)
    - time_step()
    - sources_in_time()
    - input_data_set.resample()
    - dm4bem.inputs_in_time()
    - Euler_explicit()
    - control_sys_response()
    - plot_simulation_in_time_pd()
    - plot_simulation_in_time_np()
    - print_results()

    """

    # Thermal circuit
    TC = assembled_thermal_circuit_from_folder(folder_bldg)

    TC = modify_param_thermal_circuit(TC,
                                      controller,
                                      indoor_air_capacity,
                                      glass_capacity,
                                      insulation_width_new,
                                      folder_bldg)

    # State-space representation
    [As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)
    if imposed_time_step:
        dt = imposed_time_step
    else:
        dt = time_step(As)

    # Temperature ans flow-rate sources
    input_data_set = sources_in_time(file_weather,
                                     date_start,
                                     date_end,
                                     folder_bldg)

    # Resample hourly data to time step dt
    input_data_set = input_data_set.resample(
        str(dt) + 'S').interpolate(method='linear')

    # Input vector in time from input_data_set
    u = dm4bem.inputs_in_time(us, input_data_set)

    # Time integration with Euler explicit
    θ0 = 20     # °C, initial temperatures

    start_time = time.time()
    if numerical_integration_Euler:
        y = Euler_explicit(θ0, As, Bs, Cs, Ds, u, dt)
    else:
        y = control_sys_response(θ0, As, Bs, Cs, Ds, u, dt)

    end_time = time.time()
    duration_integration = end_time - start_time

    # HVAC load
    Kp = TC['G']['c3_q0']     # W/K, controller gain
    S = 3 * 3                 # m², surface area of the toy house
    q_HVAC = Kp * (u['c3_q0'] - y['c2_θ0']) / S  # W/m²

    # Results
    start_time = time.time()
    if plot_with_pandas:
        plot_simulation_in_time_pd(dt, input_data_set, y, q_HVAC)
    else:
        plot_simulation_in_time_np(dt, input_data_set, y, q_HVAC)
    end_time = time.time()
    duration_plot_simulation = end_time - start_time

    print_results(dt, input_data_set, y, q_HVAC)

    print(f'\nDuration of numerical integration: {duration_integration:.3f} s')
    print(f'Duration of simulation plot: {duration_plot_simulation:.3f} s')


def main():
    date_start = '2000-01-01 12:00'
    date_end = '2000-01-02 12:00'
    folder_bldg = 'bldg'
    file_weather = 'weather_data/FRA_Lyon.074810_IWEC.epw'
    imposed_time_step = 0
    numerical_integration_Euler = True
    controller = True
    indoor_air_capacity = False
    glass_capacity = False
    insulation_width_new = 0.16
    plot_with_pandas = False

    simulation(
        date_start,
        date_end,
        folder_bldg,
        file_weather,
        imposed_time_step,
        numerical_integration_Euler,
        controller,
        indoor_air_capacity,
        glass_capacity,
        insulation_width_new,
        plot_with_pandas)

    simulation(folder_bldg='bldg2',
               numerical_integration_Euler=False)


if __name__ == "__main__":
    main()
