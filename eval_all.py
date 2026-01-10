import os
configs_list = [
    './configs/cfg_openearthmap.py',
    './configs/cfg_vaihingen.py',
    './configs/cfg_floodnet.py',
    './configs/cfg_loveda.py',
    './configs/cfg_potsdam.py',
    './configs/cfg_uavid.py',
    './configs/cfg_udd5.py',
    './configs/cfg_vdd.py',
    './configs/cfg_whu_sat_I.py',
    './configs/cfg_whu_sat_II.py',
    './configs/cfg_chn6-cug.py',
    './configs/cfg_massachusetts_road.py',
]

for config in configs_list:
    print(f"Running {config}")
    os.system(f"bash ./dist_test.sh {config} 2")