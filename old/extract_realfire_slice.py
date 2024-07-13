#%%
import h5py
import torch
import numpy as np
#%%
date_dict = {
    'Bear_2020': ['2020-08-20', '2020-08-21', '2020-08-22', '2020-08-23', '2020-08-24', '2020-08-25', '2020-08-26', '2020-08-27', '2020-08-28'],
    'Brattain_2020': ['2020-09-08', '2020-09-09', '2020-09-10', '2020-09-11', '2020-09-12', '2020-09-13', '2020-09-14', '2020-09-15', '2020-09-16'],
    'Buck_2017': ['2017-09-14', '2017-09-15', '2017-09-16', '2017-09-17', '2017-09-18', '2017-09-19', '2017-09-22', '2017-09-23', '2017-09-24', '2017-09-25', '2017-09-26', '2017-09-27', '2017-09-28'],
    'Chimney_2016': ['2016-08-14', '2016-08-15', '2016-08-16', '2016-08-17', '2016-08-18', '2016-08-19', '2016-08-20', '2016-08-21', '2016-08-22', '2016-08-23', '2016-08-24', '2016-08-25'],
    'Ferguson_2018': ['2018-07-14', '2018-07-15', '2018-07-16', '2018-07-17', '2018-07-18', '2018-07-19', '2018-07-20', '2018-07-21', '2018-07-22', '2018-07-23', '2018-07-24', '2018-07-25', '2018-07-26', '2018-07-27', '2018-07-28', '2018-07-29', '2018-07-30', '2018-07-31', '2018-08-01', '2018-08-02', '2018-08-03', '2018-08-04', '2018-08-05', '2018-08-06'],
    'Pier_2017': ['2017-08-30', '2017-08-31', '2017-09-01', '2017-09-02', '2017-09-03', '2017-09-04', '2017-09-05', '2017-09-06', '2017-09-07', '2017-09-08', '2017-09-09', '2017-09-10', '2017-09-11', '2017-09-12', '2017-09-13', '2017-09-14', '2017-09-15', '2017-09-16', '2017-09-17'],
}

initial_fire_date = {
    'Bear_2020': '2020-08-19',
    'Brattain_2020': '2020-09-07',
    'Buck_2017': '2017-09-13',
    'Chimney_2016': '2016-08-13',
    'Ferguson_2018': '2018-07-13',
    'Pier_2017': '2017-08-29',
}

with h5py.File('dataset.hdf5', 'r') as f_in:
    with h5py.File('real_dataset_v4.hdf5', 'w') as f_out:
        for fire_name in date_dict:
            ds_in = f_in[fire_name]
            ds_out = f_out.create_group(fire_name)
            ds_out.attrs['day_count'] = len(date_dict[fire_name])
            
            slope = torch.tensor(ds_in['SLPD2020'][:], dtype=torch.float32) # degrees
            canopy = torch.tensor(ds_in['230CC'][:], dtype=torch.float32) # %
            density = torch.tensor(ds_in['230CBD'][:], dtype=torch.float32) # kg m^{-3} * 100
            initial_fire = torch.tensor(ds_in['fire'][initial_fire_date[fire_name]][:], dtype=torch.bool)
            
            ds_out.create_dataset('slope', data=slope.cpu().detach().numpy(), compression='gzip')
            ds_out.create_dataset('canopy', data=canopy.cpu().detach().numpy(), compression='gzip')
            ds_out.create_dataset('density', data=density.cpu().detach().numpy(), compression='gzip')
            ds_out.create_dataset('initial_fire', data=initial_fire.cpu().detach().numpy(), compression='gzip')
    
            wind_V_list = []
            wind_towards_direction_list = []
            target_list = []
            
            for date in date_dict[fire_name]:
                
                wind_u = torch.tensor(ds_in['u_component_of_wind_10m'][date][:], dtype=torch.float32)
                wind_v = torch.tensor(ds_in['v_component_of_wind_10m'][date][:], dtype=torch.float32)
                # Wind data directs to where air moving towards.
                # Velocity in m/s
                # TODO: process wind
                wind_V_list.append(wind_V.cpu().detach().numpy())
                wind_towards_direction_list.append(wind_towards_direction.cpu().detach().numpy())
                target_list.append(torch.tensor(ds_in['fire'][date][:], dtype=torch.bool).cpu().detach().numpy())
                
            wind_V_array = np.stack(wind_V_list, axis=0)
            wind_towards_direction_array = np.stack(wind_towards_direction_list, axis=0)
            target_array = np.stack(target_list, axis=0)
            
            ds_out.create_dataset('wind_V', data=wind_V_array, compression='gzip')
            ds_out.create_dataset('wind_towards_direction', data=wind_towards_direction_array, compression='gzip')
            ds_out.create_dataset('target', data=target_array, compression='gzip')
                
#%%
