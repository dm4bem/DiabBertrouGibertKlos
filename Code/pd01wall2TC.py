

import pandas as pd

import dm4bem



folder = "./pd/bldg_wall2TC"



wall_types = pd.read_csv("bldg/wall_types.csv")
wall_types


walls = pd.read_csv("bldg/wall_out.csv")
walls


# Thermal circuits from wall types and wall data
TCd_generic = dm4bem.wall2TC(wall_types, walls, prefix="g")


TCd_generic['gw0']['A']


dm4bem.print_TC(TCd_generic['gw0'])


print('TCd_generic')
for key in TCd_generic.keys():
    print('Wall:', key)
    dm4bem.print_TC(TCd_generic[key])



walls = pd.read_csv(folder + '/walls_out.csv')
walls

TCd_out = dm4bem.wall2TC(wall_types, walls, prefix="o")

# Uncomment below to print all thermal circuits
print('TCd_out')
for key in TCd_out.keys():
   print('Wall:', key)
   pd_dm4bem.print_TC(TCd_out[key])




walls = pd.read_csv(folder + '/walls_in.csv')
walls


TCd_in = dm4bem.wall2TC(wall_types, walls, prefix="i")

# Uncomment below to print all thermal circuits
print('TCd_in')
for key in TCd_in.keys():
   print('Wall:', key)
   pd_dm4bem.print_TC(TCd_in[key])





