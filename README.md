# Example of application of _dm4bem_: toy model house
blender
# [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/cghiaus/dm4bem_toy_model/HEAD) 
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/dm4bem/DiabBertrouGibertKlos/main?filepath=Code.ipynb)

Short presentation of modeling described in section _Pyhton script_ of [Inputs and simulation](https://cghiaus.github.io/dm4bem_book/tutorials/pd05simulation.html) chapter of Jupyter book on [Dynamic Models for Building Energy Management](https://cghiaus.github.io/dm4bem_book/intro.html).

Detailed information:
- [toy model house](https://cghiaus.github.io/dm4bem_book/tutorials/02_2_0Toy.html);
- [assembling thermal circuits](https://cghiaus.github.io/dm4bem_book/tutorials/pdREADME.html).

## Folders and files
- __bldg__: description of the building ([link](https://cghiaus.github.io/dm4bem_book/tutorials/pd02bldg2TCd.html?highlight=tc0%20csv))
    - _assembly_list.csv_: list of nodes that merge ([link](https://cghiaus.github.io/dm4bem_book/tutorials/pd03assembleTCd.html));
    - _TC0.csv, ... , TC3.csv_: thermal circuits;
    - _wall_types.csv_: composition of walls ([link](https://cghiaus.github.io/dm4bem_book/tutorials/pd01wall2TC.html));
    - _wall_out.csv_: data for specific wall of _wall_type_ ([link](https://cghiaus.github.io/dm4bem_book/tutorials/pd01wall2TC.html#walls-data)).
- __bldg2__: same description of the building as in __bldg__ folder but the space discretization is thinner (10 meshes per layer) in _wall_types.csv_.
- __weather_data__:
    - _FRA_Lyon.074810_IWEC.epw_: EnergyPlus weather file ([link](https://cghiaus.github.io/dm4bem_book/tutorials/01WeatherData.html)).
- __Pyhton scripts__:
    - _dm4bem.py_: Python module;
    - _pd05simulation.py_: example of application ([link](https://cghiaus.github.io/dm4bem_book/tutorials/pd05simulation.html)).
    - _toy_model_house.py_: same example of application ([link](https://cghiaus.github.io/dm4bem_book/tutorials/pd05simulation.html)) with code structured in functions on level of abstractions.
- __Jupyter notebooks:__
    - *run_pd05simulation.ipynb*: run the script _pd05simulation.py_.
    - *toy_model_house.ipynb*: uses _toy_model_house.py_.

## Run `pd05simulation.py` script

Alternatives to run the Python script:
1. Open _Notebook_ `run_pd05simulation.ipynb` and __Restart the kernel and run all cells__.
2. Open _Console_, write the command`run pd05simulation.py` or `exec(open('pd05simulation.py').read())` and press _Shift / Enter_ to run the command.
3. Open _Terminal_ and execute the command `python pd05simulation.py`. Note that only text will be displayed in the terminal.
