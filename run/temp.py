from umbms import get_proj_path, verify_path, get_script_logger
from umbms.loadsave import load_pickle, save_pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

md = np.array([])

data_dict = {
    'n_expt':np.nan,
    'id':np.nan,
    'tum_diam':np.nan,
    'tum_shape':None,
    'tum_x':np.nan,
    'tum_y':np.nan,
    'adi_ref_id':np.nan,
    'adi_ref_id2':np.nan,
    'emp_ref_id':np.nan,
    'date':'20230210',
    'n_session':4,
    'ant_rad':21.0
}

for i in range(51):
    md = np.append(md, data_dict.copy())

start = 2.5
for id in range(8):
    if id % 2 == 0:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['tum_diam'] = 0.3
        md[id]['tum_shape'] = 'rod'
        md[id]['tum_x'] = start
        md[id]['tum_y'] = start
        md[id]['adi_ref_id'] = id + 1
        md[id]['adi_ref_id2'] = id + 1
        md[id]['emp_ref_id'] = 16
        start -= 0.5

    else:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['emp_ref_id'] = 16


start = 2.5
for id in range(8, 12, 1):
    if id % 2 == 0:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['tum_diam'] = 0.3
        md[id]['tum_shape'] = 'rod'
        md[id]['tum_x'] = 0
        md[id]['tum_y'] = start
        md[id]['adi_ref_id'] = id + 1
        md[id]['adi_ref_id2'] = id + 1
        md[id]['emp_ref_id'] = 16
        start -= 1

    else:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['emp_ref_id'] = 16


start = 2.5
for id in range(12, 16, 1):
    if id % 2 == 0:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['tum_diam'] = 0.3
        md[id]['tum_shape'] = 'rod'
        md[id]['tum_x'] = start
        md[id]['tum_y'] = 0
        md[id]['adi_ref_id'] = id + 1
        md[id]['adi_ref_id2'] = id + 1
        md[id]['emp_ref_id'] = 16
        start -= 1

    else:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['emp_ref_id'] = 16


md[16]['n_expt'] = 16
md[16]['id'] = 16

start = -2.5
for id in range(17, 25, 1):
    if id % 2 != 0:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['tum_diam'] = 0.3
        md[id]['tum_shape'] = 'rod'
        md[id]['tum_x'] = start
        md[id]['tum_y'] = start
        md[id]['adi_ref_id'] = id + 1
        md[id]['adi_ref_id2'] = id + 1
        md[id]['emp_ref_id'] = 33
        start += 0.5

    else:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['emp_ref_id'] = 33


start = -2.5
for id in range(25, 29, 1):
    if id % 2 != 0:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['tum_diam'] = 0.3
        md[id]['tum_shape'] = 'rod'
        md[id]['tum_x'] = 0
        md[id]['tum_y'] = start
        md[id]['adi_ref_id'] = id + 1
        md[id]['adi_ref_id2'] = id + 1
        md[id]['emp_ref_id'] = 33
        start += 1

    else:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['emp_ref_id'] = 33


start = -2.5
for id in range(29, 33, 1):
    if id % 2 != 0:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['tum_diam'] = 0.3
        md[id]['tum_shape'] = 'rod'
        md[id]['tum_x'] = start
        md[id]['tum_y'] = 0
        md[id]['adi_ref_id'] = id + 1
        md[id]['adi_ref_id2'] = id + 1
        md[id]['emp_ref_id'] = 33
        start += 1

    else:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['emp_ref_id'] = 33


md[33]['n_expt'] = 33
md[33]['id'] = 33

start = 1.5
for id in range(34, 38, 1):
    if id % 2 == 0:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['tum_diam'] = 0.3
        md[id]['tum_shape'] = 'rod'
        md[id]['tum_x'] = - start
        md[id]['tum_y'] = start
        md[id]['adi_ref_id'] = id + 1
        md[id]['adi_ref_id2'] = id + 1
        md[id]['emp_ref_id'] = 50
        start -= 0.5


    else:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['emp_ref_id'] = 50


start = 1.5
for id in range(38, 42, 1):
    if id % 2 == 0:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['tum_diam'] = 0.3
        md[id]['tum_shape'] = 'rod'
        md[id]['tum_x'] = start
        md[id]['tum_y'] = - start
        md[id]['adi_ref_id'] = id + 1
        md[id]['adi_ref_id2'] = id + 1
        md[id]['emp_ref_id'] = 50
        start -= 0.5

    else:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['emp_ref_id'] = 50


start = - 2.
for id in range(42, 46, 1):
    if id % 2 == 0:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['tum_diam'] = 0.3
        md[id]['tum_shape'] = 'rod'
        md[id]['tum_x'] = start
        md[id]['tum_y'] = 0
        md[id]['adi_ref_id'] = id + 1
        md[id]['adi_ref_id2'] = id + 1
        md[id]['emp_ref_id'] = 50
        start *= -1


    else:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['emp_ref_id'] = 50


start = 2.
for id in range(46, 50, 1):
    if id % 2 == 0:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['tum_diam'] = 0.3
        md[id]['tum_shape'] = 'rod'
        md[id]['tum_x'] = 0
        md[id]['tum_y'] = start
        md[id]['adi_ref_id'] = id + 1
        md[id]['adi_ref_id2'] = id + 1
        md[id]['emp_ref_id'] = 50
        start *= -1


    else:
        
        md[id]['n_expt'] = id
        md[id]['id'] = id
        md[id]['emp_ref_id'] = 50


md[50]['n_expt'] = 50
md[50]['id'] = 50

save_pickle(md, os.path.join(get_proj_path(), 'metadata_new.pickle'))