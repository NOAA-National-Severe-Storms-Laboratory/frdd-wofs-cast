import matplotlib.colors as mcolors
# For plotting. 
import os
import numpy as np
import xarray 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from wofscast.plot import WoFSColors, WoFSLevels
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr

display_name_mapper = {'U' : 'U-wind Comp.', 
          'V' : 'V-wind Comp.', 
          'W' : 'Vert. Velocity',
          'T' : 'Pot. Temp.', 
          'GEOPOT' : 'Geopot. Height',
          'QVAPOR' : 'QVAPOR', 
          'T2' : '2-m Temp.', 
          'COMPOSITE_REFL_10CM' : 'Comp. Refl.',
          'UP_HELI_MAX' : '2-5 km UH', 
          'RAIN_AMOUNT' : 'Rain Rate',
          'WMAX' : 'Max Vert. Velocity',
         }

units_mapper = {'T': 'K', 
                'QVAPOR': 'kg/kg', 
                'T2': 'F', 
                'U': 'm/s', 
                'V': 'm/s', 
                'W': 'm/s', 
                'GEOPOT': 'm', 
                'RAIN_AMOUNT': 'in', 
                'COMPOSITE_REFL_10CM': 'dBZ',
                'WMAX' : 'm/s'
               }

VARS_2D = ['COMPOSITE_REFL_10CM', 
           'T2', 
           'RAIN_AMOUNT'
          ]

def get_colormap_and_levels(var):
        if var == 'COMPOSITE_REFL_10CM':
            cmap = WoFSColors.nws_dz_cmap
            levels = WoFSLevels.dz_levels_nws
        elif var == 'RAIN_AMOUNT':
            cmap = WoFSColors.rain_cmap
            levels = WoFSLevels.rain_rate_levels
        elif var == 'UP_HELI_MAX':
            cmap = WoFSColors.wz_cmap_extend
            levels = WoFSLevels.uh_2to5_levels_3000m
        elif var == 'T2':
            cmap = WoFSColors.temp_cmap
            levels = np.arange(40., 90., 2.5)
        elif var == 'QVAPOR': 
            cmap = WoFSColors.temp_cmap
        elif var in ['W', 'WMAX']: 
            cmap = WoFSColors.wz_cmap_extend
            levels = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]#, 25]#, 30, 35, 40]
        else:
            cmap = WoFSColors.wz_cmap_extend
        
        return cmap, levels




class WoFSCastAnimator:
    def __init__(self, 
                 inputs, 
                 predictions, 
                 targets, 
                 mrms_dataset=None, 
                 analysis_dataset=None,
                 domain_size=150, 
                 plot_border=False, 
                 dts=None, 
                 add_rmse=True, 
                 for_randy=False,
                 add_timing_text = False, 
                 title_prefixes=['WoFS', 'WoFSCast', 'Analysis', 'MRMS'],
                 add_mrms_overlay = True
                ):
        
        self.dts = dts  # Placeholder, replace with your datetime conversion function
        self.plot_border = plot_border
        self.domain_size = domain_size
        self.add_rmse = add_rmse 
        self.for_randy = for_randy
        self.add_timing_text = add_timing_text 
        
        self.inputs = inputs
        self.predictions = predictions
        self.targets = targets
        self.mrms_dataset = mrms_dataset
        self.analysis_dataset = analysis_dataset
        self.title_prefixes = title_prefixes
        self.add_mrms_overlay = add_mrms_overlay
    
    
    def set_level(self, var, level):
        if var in VARS_2D:
            level='none'
        
        self.level = level
        
        level_txt = ''
        if level != 'none': 
            level_txt = f', Model level = {level}'
            
        return level_txt
    
    def create_animation(self, var, level=0, ens_idx=0, animation_type='wofs_vs_wofscast'):
        """
        Generalized function to create different types of animations based on the animation_type.
        
        Parameters:
        - var: Variable to animate.
        - level: Vertical level to animate.
        - animation_type: Type of animation ('wofs_vs_wofscast' or 'wofs_wofscast_analysis_mrms').
        
        Returns:
        - FuncAnimation object.
        """
        self.animation_type = animation_type
        self.var = var
        level_txt = self.set_level(var, level)
        
        (self.inputs, 
         self.predictions, 
         self.targets) = self.drop_batch_dim(ens_idx, self.inputs, self.predictions, self.targets)

        if animation_type == 'wofs_vs_wofscast':
            self.titles = [
                f'{self.title_prefixes[0]} {display_name_mapper.get(var, var)}{level_txt}', 
                f'{self.title_prefixes[1]} {display_name_mapper.get(var, var)}{level_txt}'
            ]
            fig, self.axes = plt.subplots(dpi=200, figsize=(12, 6), ncols=2, 
                                          gridspec_kw={'height_ratios': [1], 'bottom': 0.15})
        
        elif animation_type == 'wofs_wofscast_analysis_mrms':
            self.titles = [
                f'{self.title_prefixes[0]} {display_name_mapper.get(var, var)}{level_txt}', 
                f'{self.title_prefixes[1]} {display_name_mapper.get(var, var)}{level_txt}',
                f'{self.title_prefixes[2]} {display_name_mapper.get(var, var)}{level_txt}', 
                f'{self.title_prefixes[3]} Refl'
            ]
            fig, self.axes = plt.subplots(dpi=200, figsize=(12, 12), nrows=2, ncols=2,
                                          gridspec_kw={'height_ratios': [1, 1], 'bottom': 0.1})
        
        plt.tight_layout()
        
        _, levels = self.get_target_and_pred_pair(self.inputs, self.inputs, 
                                                  t=0, level=level, return_rng=True)
    
        self.cmap, self.levels = self.get_colormap_and_levels(var, levels)
    
        # Create a BoundaryNorm instance
        self.norm = mcolors.BoundaryNorm(self.levels, ncolors=len(self.levels)-1, clip=True)
        
        self.cbar_ax = fig.add_axes([0.15, 0.075, 0.7, 0.02])
        self.cbar = None
        
        self.fig = fig
        self.N = len(self.predictions.time)
    
        return FuncAnimation(fig, self.update, frames=self.N, interval=200)

    def drop_batch_dim(self, ens_idx, inputs, predictions, targets):
        dims = ('time', 'level', 'lat', 'lon')
         
        init_ds = inputs.isel(batch=ens_idx)
        preds = predictions.isel(batch=ens_idx)
        tars = targets.isel(batch=ens_idx)
               
        init_ds = init_ds.isel(time=[-1]).transpose(*dims, missing_dims='ignore')
        preds = preds.transpose(*dims, missing_dims='ignore')
        tars = tars.transpose(*dims, missing_dims='ignore')
    
        return init_ds, preds, tars
    
    def get_target_and_pred_pair(self, preds, targets, t, level=0, return_rng=False):
        max_t = len(targets.time) - 1
        target_t = max_t
        if t < max_t:
            target_t = t
        
        if level == 'max':
            zs = [targets[self.var].isel(time=target_t).max(dim='level').values, 
                  preds[self.var].isel(time=t).max(dim='level').values]
        elif level == 'min': 
            zs = [targets[self.var].isel(time=target_t).min(dim='level').values, 
                  preds[self.var].isel(time=t).min(dim='level').values]
        elif level == 'none':
            zs = [targets[self.var].isel(time=target_t).values, 
                  preds[self.var].isel(time=t).values]
        else:
            zs = [targets[self.var].isel(time=target_t, level=level).values, 
                  preds[self.var].isel(time=t, level=level).values]
    
        if self.animation_type == 'wofs_wofscast_analysis_mrms':
            zs.append(self.analysis_dataset[t])
        
        if return_rng:
            global_min = np.percentile(zs, 1)
            global_max = np.percentile(zs, 99)
            rng = np.linspace(global_min, global_max, 10)
            return zs, rng
        
        # Add the MRMS after the range is computed above. 
        if self.animation_type == 'wofs_wofscast_analysis_mrms':
            zs.append(self.mrms_dataset[t])

        return zs 

    def get_colormap_and_levels(self, var, levels):
        self.dz_cmap = WoFSColors.nws_dz_cmap
        self.dz_levels = WoFSLevels.dz_levels_nws
        
        if var == 'COMPOSITE_REFL_10CM':
            cmap = WoFSColors.nws_dz_cmap
            if self.for_randy:
                cmap = 'Spectral_r'
            levels = WoFSLevels.dz_levels_nws            
        elif var == 'RAIN_AMOUNT':
            cmap = WoFSColors.rain_cmap
            levels = WoFSLevels.rain_rate_levels
        elif var == 'UP_HELI_MAX':
            cmap = WoFSColors.wz_cmap_extend
            levels = WoFSLevels.uh_2to5_levels_3000m
        elif var == 'T2':
            cmap = WoFSColors.temp_cmap
            levels = np.arange(40., 90., 2.5)
        elif var == 'QVAPOR': 
            cmap = WoFSColors.temp_cmap
        elif var == 'W': 
            cmap = WoFSColors.wz_cmap_extend
            levels = [2.5, 5, 10, 15, 20, 25, 30, 35, 40]
        elif 'nmep' in var:
            cmap = WoFSColors.wz_cmap
            levels = WoFSLevels.prob_levels
        else:
            cmap = WoFSColors.wz_cmap_extend
        
        return cmap, levels
    
    def remove_ticks(self, ax):
        
        # Remove tick marks
        ax.xaxis.set_ticks([])
        ax.yaxis.set_ticks([])
            
        # Optionally, remove tick labels too
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
    
        return ax
    
    
    def update(self, t):
        for ax in self.axes.flat:
            ax.clear()

        if t == 0:
            zs = self.get_target_and_pred_pair(self.inputs, self.inputs, t=0, level=self.level)
        else:    
            zs = self.get_target_and_pred_pair(self.predictions, self.targets, t=t, level=self.level)
    
        rmse = np.sqrt(np.mean((zs[0] - zs[1])**2))
    
        for i, ax in enumerate(self.axes.flat):
            ax = self.remove_ticks(ax)

            z = zs[i].squeeze()

            
            if self.var in ['REFL_10CM', 'UP_HELI_MAX', 'COMPOSITE_REFL_10CM']:
                z = np.ma.masked_where(z < 10, z)
            
            if self.animation_type == 'wofs_wofscast_analysis_mrms' and i == 3:
                cmap = self.dz_cmap 
                levels = self.dz_levels
            else:
                cmap=self.cmap 
                levels=self.levels
            
            im = ax.contourf(z, origin='lower', aspect='equal', cmap=cmap, levels=levels)
            
            ax.set_title(self.titles[i], fontweight='bold')
            if i == 1 and self.add_rmse:
                dis_name = display_name_mapper.get(self.var, self.var)
                ax.annotate(f'RMSE of {dis_name} ({units_mapper.get(self.var, self.var)}): {rmse:.4f}', 
                            xy=(0.01, 0.95), xycoords='axes fraction', 
                            weight='bold', color='red', 
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            ax.annotate(f'Time: {self.dts[t]}', xy=(0.01, 0.01), xycoords='axes fraction', 
                        weight='bold', color='red', fontsize=10, 
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

            if self.mrms_dataset is not None and i < 3 and self.add_mrms_overlay:
                self.add_mrms_overlay(ax, z[1], t)
                
            if self.add_timing_text:
                timing_text = ['8-12 min with 60+ CPUs', '30-40 secs with 1 GPU'] 
                ax.annotate(timing_text[i], xy=(0.025, 0.95), xycoords='axes fraction',
                            weight='bold', color='gray', fontsize=12, 
                            bbox=dict(facecolor='lightblue', alpha=0.7, edgecolor='none'))
    
        if self.cbar is None:
            self.cbar = self.fig.colorbar(im, cax=self.cbar_ax, orientation='horizontal')
            self.cbar.set_label(f'{display_name_mapper.get(self.var, self.var)} ({units_mapper.get(self.var, self.var)})')

    def add_mrms_overlay(self, ax, pred, t):
        this_rmse = np.sqrt(np.mean((pred - self.mrms_dataset[t]) ** 2))

        ax.contour(self.mrms_dataset[t], 
                   origin='lower', aspect='equal', 
                   colors=['black', 'blue'], 
                   levels=[35.0, 50.0], linewidths=[1.0, 1.5])
        
        if self.add_rmse:
            ax.annotate(f'RMSE with MRMS: {this_rmse:.4f}', 
                    xy=(0.01, 0.90), xycoords='axes fraction', 
                    weight='bold', color='k', 
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
