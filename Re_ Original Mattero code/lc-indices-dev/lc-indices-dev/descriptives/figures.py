#%%
import numpy as np
import pandas as pd # Added pandas here as it's used more broadly now
import matplotlib.pyplot as plt
import warnings
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mt # Ensure this alias is used for colormaps
import matplotlib.patches as mpatches
from colour import Color # Ensure 'colour' library is installed: pip install colour
import seaborn as sns
from collections.abc import Iterable
from matplotlib.transforms import offset_copy
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, silhouette_score
from collections import defaultdict

# --- Start of Visualizer Class and Helper Functions from Visualization.py ---
LCGreen, LCYellow, LCBlue = '#4FD284', '#FFDD26', '#87ceeb'
MBRed='#AB0000'        

def get_LCColors():
    return LCGreen, LCYellow, LCBlue

def make_LCColorMap(start = LCYellow, end = LCBlue, NOfCols = 100):
    start_color = Color(start)
    colors = list(start_color.range_to(Color(end),NOfCols))
    colors = [str(C) for C in colors] 
    cmap = LinearSegmentedColormap.from_list('LCBS_cmap', colors, N=1000)        
    return(cmap)    

def make_LCDivergingMap(start = LCYellow, mid = 'white', end = LCGreen):
    start_color = Color(start)
    colors = list(start_color.range_to(Color(mid),1000))
    colors1 = [str(C) for C in colors] 
    start_color = Color(mid)
    colors = list(start_color.range_to(Color(end),1000))
    colors2 = [str(C) for C in colors] 
    colors = colors1 + colors2       
    cmap = LinearSegmentedColormap.from_list('LCBS_DIVcmap', colors, N=100) 
    return cmap

def get_LCDiscreteColors(start_color_hex = LCYellow, end_color_hex = LCBlue, NOfCols = 5):
    start_c = Color(start_color_hex)
    colors_list = list(start_c.range_to(Color(end_color_hex),NOfCols))
    colors_list = [str(C) for C in colors_list]        
    return(colors_list)

class Visualizer:
    styles={'all-star': 0, 'default':-1}

    def __init__(self, style: str='custom', custom_style: dict={}):
        if style == 'custom':
            self.style = custom_style
        else:
            shelf_style = self.styles.get(style, 'None')
            if shelf_style == 'None':
                warnings.warn('The provided style is not valid, using default',UserWarning)
                self.style = self.styles.get('default',None)
            else:
                self.style = shelf_style
    
    def create_fig(self, grid, subplots,param:dict=None,size:tuple=(3,3),title='',file: str='',dpi=350,layout='tight'):
        color_cycle_list = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'] 
        default_cycler = (cycler(color=color_cycle_list))
        
        plt.style.use('seaborn-v0_8-white') 
        plt.rcParams.update({'font.size': 10}) 
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['xtick.major.pad']='5' 
        plt.rcParams['ytick.major.pad']='5' 
        
        plt.rc('axes', prop_cycle=default_cycler)
        
        self.xtick_size=8 
        self.ytick_size=8
        self.xlabel_size=10 
        self.ylabel_size=10
        self.font='serif' 
        
        fig, axd = plt.subplot_mosaic(grid,
                              figsize=size,per_subplot_kw=param ,layout=layout)
        if title: 
            fig.suptitle(title, fontsize=14, y=0.98 if layout=='tight' else 1.0) 
        
        for subplot_def in subplots: 
            idx_keys = list(axd.keys())
            idx = subplot_def.get('idx',grid[0][0] if isinstance(grid, list) and grid and isinstance(grid[0],list) and grid[0][0] in idx_keys else (idx_keys[0] if idx_keys else None) )
            if idx is None or idx not in axd: 
                warnings.warn(f"Could not determine a valid subplot index for subplot definition: {subplot_def}. Skipping.", UserWarning)
                continue
            current_ax = axd[idx]

            is_polar = param and idx in param and param[idx].get('polar', False)
            if is_polar : 
                 current_ax.grid(True, linestyle=':', alpha=0.6) 
            elif 'seaborn-v0_8-whitegrid' not in plt.style.library : 
                 current_ax.grid(False) 

            plot_type = subplot_def.get('type','line')
            
            plot_func_map = {
                'line': self.draw_line, 'scatter': self.draw_scat, 'hist': self.draw_hist,
                'kde': self.draw_kde, 'bar': self.draw_bar, 'line_stddev': self.draw_line_stddev,
                'static_lines': self.static_lines, 'from_func': self.draw_from_func,
                'regression': self.draw_regression, 'annotation': self.draw_annotation
            }
            
            draw_function = plot_func_map.get(plot_type)
            if draw_function:
                try:
                    draw_function(current_ax, subplot_def)
                except Exception as e:
                    warnings.warn(f"Error drawing plot type '{plot_type}' for subplot '{idx}': {e}", UserWarning)
            else:
                warnings.warn(f"Unknown plot type '{plot_type}' for subplot '{idx}'. Skipping.", UserWarning)

        if file!='':
            try:
                fig.savefig(file,bbox_inches='tight' if layout=='tight' else None ,dpi=dpi)
                print(f"Figure saved to {file}")
            except Exception as e:
                print(f"Error saving figure: {e}")
        plt.close(fig) 
            
    def draw_annotation(self, ax: plt.Axes,subplot: dict):
        text=subplot.get('text','')
        x=subplot.get('x',0.5) 
        y=subplot.get('y',0.5)
        xycoords=subplot.get('xycoords','axes fraction') 
        xtext=subplot.get('x_text',x) 
        ytext=subplot.get('y_text',y) 
        textcoords=subplot.get('textcoords', xycoords) 

        align=subplot.get('align',['center','center'])
        color=subplot.get('color','black')
        arrow_props_dict = subplot.get('arrowprops', None) 
        font_size=subplot.get('font_size',plt.rcParams['font.size']) 

        if arrow_props_dict is None: 
            arrow_head=subplot.get('arrow_head',None)
            arrow_body=subplot.get('arrow_body',None)
            if arrow_head or arrow_body: 
                 arrow_opacity=subplot.get('arrow_opacity',0.5)
                 arrow_props_dict = dict(arrowstyle=arrow_head if arrow_head else '-',
                                         ls=arrow_body if arrow_body else '-',
                                         alpha=arrow_opacity, color=color,
                                         relpos=(0.5, 0.5)) 
        ax.annotate(text,
                    xy=(x,y), xycoords=xycoords,
                    xytext=(xtext,ytext), textcoords=textcoords,
                    color=color, arrowprops=arrow_props_dict,
                    ha=align[0], va=align[1], fontsize=font_size)

    def add_global_style_preplot(self,ax:plt.Axes,subplot: dict):
        color_cycler_override=subplot.get('color',None) 
        xlim=subplot.get('xlim',None)
        ylim=subplot.get('ylim',None)
        x_label_position=subplot.get('xlabel_position','bottom') 
        y_label_position=subplot.get('ylabel_position','left')
        x_tick_position=subplot.get('xtick_position','bottom')
        y_tick_position=subplot.get('ytick_position','left')
        
        ax.xaxis.set_label_position(x_label_position)
        ax.yaxis.set_label_position(y_label_position)
        ax.xaxis.set_ticks_position(x_tick_position)
        ax.yaxis.set_ticks_position(y_tick_position)
        
        if xlim is not None: ax.set_xlim(xlim)
        if ylim is not None: ax.set_ylim(ylim)
        
        if color_cycler_override is not None:
            try:
                cy = cycler('color', color_cycler_override)
                ax.set_prop_cycle(cy)
            except Exception as e:
                warnings.warn(f"Failed to set color cycler: {e}", UserWarning)

    def add_global_style(self,ax:plt.Axes,subplot: dict):
        title_str=subplot.get('title','') 
        title_pad=subplot.get('title_offset',plt.rcParams['axes.titlepad']) 
        title_fontsize = subplot.get('title_fontsize', plt.rcParams['axes.titlesize'])
        title_loc = subplot.get('title_loc', 'left') 

        legend_flag=subplot.get('show_legend',False) 
        legend_voffset=subplot.get('legend_offset',0) 
        legend_hoffset=subplot.get('legend_offset_horizontal',0) 
        legend_ncols=subplot.get('legend_ncols',0) 
        legend_loc_pref=subplot.get('legend_loc','best') 
        legend_fontsize = subplot.get('legend_fontsize', plt.rcParams['legend.fontsize'])
        legend_frame = subplot.get('legend_frame', False) 

        xlabel_str=subplot.get('xlabel','') 
        xlabel_fsize=subplot.get('xlabel_size',self.xlabel_size) 
        xlabel_font=subplot.get('xlabel_font_family',self.font) 
        xlabel_pad=subplot.get('xlabel_offset',plt.rcParams['axes.labelpad']) 

        xtick_rot_val=subplot.get('xtick_rot',0) 
        xtick_fsize=subplot.get('xtick_size',self.xtick_size)      
        xtick_labels_custom=subplot.get('xtick_lab',None) 
        xtick_coords_custom=subplot.get('xtick_coord_list',None) 
        xtick_pad_val=subplot.get('xtick_offset',plt.rcParams['xtick.major.pad']) 
        
        ylabel_str=subplot.get('ylabel','') 
        ylabel_fsize=subplot.get('ylabel_size',self.ylabel_size) 
        ylabel_font=subplot.get('ylabel_font_family',self.font) 
        ylabel_pad=subplot.get('ylabel_offset',plt.rcParams['axes.labelpad']) 

        ytick_rot_val=subplot.get('ytick_rot',0) 
        ytick_fsize=subplot.get('ytick_size',self.ytick_size)        
        ytick_labels_custom=subplot.get('ytick_lab',None) 
        ytick_coords_custom=subplot.get('ytick_coord_list',None) 
        ytick_pad_val=subplot.get('ytick_offset',plt.rcParams['ytick.major.pad']) 
        
        polar_labels_flag=subplot.get('polar_labels',False) 
        
        if xtick_coords_custom is not None: 
            ax.set_xticks(xtick_coords_custom)
        
        if xtick_labels_custom is not None:
            if len(xtick_labels_custom) == 0: 
                ax.set_xticklabels([])
                if xtick_coords_custom is None: 
                    ax.set_xticks([])
            else: 
                current_ticks = ax.get_xticks()
                if xtick_coords_custom is not None and len(xtick_coords_custom) == len(xtick_labels_custom):
                    ax.set_xticklabels(xtick_labels_custom)
                elif len(current_ticks) == len(xtick_labels_custom):
                    ax.set_xticklabels(xtick_labels_custom)
                else: 
                    # warnings.warn(f"xtick_lab length {len(xtick_labels_custom)} mismatch with current ({len(current_ticks)}) or specified ticks. Applying partially.", UserWarning)
                    try: ax.set_xticklabels(xtick_labels_custom)
                    except Exception:  ax.set_xticklabels(xtick_labels_custom[:len(current_ticks)] if len(current_ticks) < len(xtick_labels_custom) else xtick_labels_custom + [''] * (len(current_ticks) - len(xtick_labels_custom)))

        if ytick_coords_custom is not None: 
            ax.set_yticks(ytick_coords_custom)

        if ytick_labels_custom is not None:
            if len(ytick_labels_custom) == 0: 
                ax.set_yticklabels([])
                if ytick_coords_custom is None: 
                    ax.set_yticks([])
            else: 
                current_ticks = ax.get_yticks()
                if ytick_coords_custom is not None and len(ytick_coords_custom) == len(ytick_labels_custom):
                    ax.set_yticklabels(ytick_labels_custom)
                elif len(current_ticks) == len(ytick_labels_custom):
                    ax.set_yticklabels(ytick_labels_custom)
                else: 
                    # warnings.warn(f"ytick_lab length {len(ytick_labels_custom)} mismatch with current ({len(current_ticks)}) or specified ticks. Applying partially.", UserWarning)
                    try: ax.set_yticklabels(ytick_labels_custom)
                    except Exception: ax.set_yticklabels(ytick_labels_custom[:len(current_ticks)] if len(current_ticks) < len(ytick_labels_custom) else ytick_labels_custom + [''] * (len(current_ticks) - len(ytick_labels_custom)))
            
        ax.tick_params(axis='x', which='major', labelsize=xtick_fsize, pad=xtick_pad_val, rotation=xtick_rot_val)
        ax.tick_params(axis='y', which='major', labelsize=ytick_fsize, pad=ytick_pad_val, rotation=ytick_rot_val)

        if xtick_rot_val != 0: 
             plt.setp(ax.get_xticklabels(), ha="right" if xtick_rot_val > 0 else "left" if xtick_rot_val < 0 else "center", rotation_mode="anchor")

        if title_str: ax.set_title(title_str, pad=title_pad, fontsize=title_fontsize, loc=title_loc)
        if xlabel_str: ax.set_xlabel(xlabel=xlabel_str,labelpad=xlabel_pad,fontsize=xlabel_fsize,family=xlabel_font)
        if ylabel_str: ax.set_ylabel(ylabel=ylabel_str,labelpad=ylabel_pad,fontsize=ylabel_fsize,family=ylabel_font)
        
        handles, labels = ax.get_legend_handles_labels()
        if legend_flag and handles: 
            bbox_anchor_val = None
            if 'center' not in legend_loc_pref and (legend_voffset != 0 or legend_hoffset !=0): 
                bbox_x = 0.5 + legend_hoffset
                bbox_y = 1.02 + legend_voffset 
                bbox_anchor_val = (bbox_x, bbox_y)

            ax.legend(handles, labels, loc=legend_loc_pref, 
                      bbox_to_anchor=bbox_anchor_val,
                      ncols=legend_ncols if legend_ncols > 0 else None, 
                      fontsize=legend_fontsize, frameon=legend_frame)
        
        if polar_labels_flag and hasattr(ax, 'set_theta_zero_location'):
            original_labels = [label.get_text() for label in ax.get_xticklabels()]
            if not original_labels: return 

            angles_deg = ax.get_xticks()
            ax.set_thetagrids(angles_deg, labels=original_labels, fontsize=xtick_fsize)
            for label in ax.get_xticklabels(): 
                try: 
                    angle_deg_val = float(label.get_text().replace('°','')) 
                    if 90 < angle_deg_val < 270: label.set_rotation(angle_deg_val + 180)
                    else: label.set_rotation(angle_deg_val)
                    label.set_horizontalalignment('center') 
                    label.set_verticalalignment('center') 
                except ValueError: pass 

    def draw_line(self, ax: plt.Axes,subplot: dict):
        y_data=subplot.get('y',None) 
        if y_data is None: return
        y_data=np.asarray(y_data)
        if y_data.ndim == 1: y_data = y_data.reshape(-1,1) 
        if y_data.shape[0] == 0 : return 

        x_data=subplot.get('x',np.arange(y_data.shape[0])) 
        x_data = np.asarray(x_data)
        sec_y = subplot.get('secondary_y',False)
        series_names=subplot.get('series_names',None) 
        xerr=subplot.get('x_err',None)
        yerr=subplot.get('y_err',None)

        opacity_vals=subplot.get('opacity', 1.0) 
        if isinstance(opacity_vals, (int, float)): opacity_vals = np.full(y_data.shape[1] if y_data.shape[1]>0 else 1, opacity_vals)
        
        colors_list=subplot.get('color_series', []) 
        marker_styles=subplot.get('point_style',['']*y_data.shape[1] if y_data.shape[1]>0 else ['']) 
        line_styles=subplot.get('line_style',['-']*y_data.shape[1] if y_data.shape[1]>0 else ['-']) 
        line_w=subplot.get('line_width',1.5) 
        
        loc_ax=ax
        if sec_y:
            loc_ax=ax.twinx()
            loc_ax.grid(False) 
        
        self.add_global_style_preplot(loc_ax,subplot)
        
        for i in range(y_data.shape[1]): 
            label = series_names[i] if isinstance(series_names,(list, tuple)) and i < len(series_names) else (series_names if y_data.shape[1]==1 and isinstance(series_names,str) else None)
            
            current_xerr = xerr[:,i] if isinstance(xerr,np.ndarray) and xerr.ndim==2 and i<xerr.shape[1] else (xerr if isinstance(xerr,np.ndarray) and (xerr.ndim==1 and (len(xerr)==y_data.shape[0] or xerr.size==1)) else None)
            current_yerr = yerr[:,i] if isinstance(yerr,np.ndarray) and yerr.ndim==2 and i<yerr.shape[1] else (yerr if isinstance(yerr,np.ndarray) and (yerr.ndim==1 and (len(yerr)==y_data.shape[0] or yerr.size==1)) else None)

            color_to_use = None
            if colors_list and i < len(colors_list) and colors_list[i] is not None:
                color_to_use = colors_list[i]
            
            alpha = opacity_vals[i % len(opacity_vals)] if isinstance(opacity_vals, (list, np.ndarray)) and len(opacity_vals)>0 else opacity_vals
            marker = marker_styles[i % len(marker_styles)] if marker_styles and len(marker_styles)>0 else ''
            linestyle = line_styles[i % len(line_styles)] if line_styles and len(line_styles)>0 else '-'
            fmt_str = marker + linestyle 
            
            loc_ax.errorbar(x_data,y_data[:,i], xerr=current_xerr,yerr=current_yerr, label=label, capsize=3, 
                            alpha=alpha, color=color_to_use, fmt=fmt_str, linewidth=line_w, markersize=5 if marker else 0) 
                 
        self.add_global_style(loc_ax,subplot)
    
    def draw_bar(self, ax: plt.Axes,subplot: dict):
        y_data=subplot.get('y',None) 
        if y_data is None: return
        y_data = np.asarray(y_data)
        if y_data.ndim == 1: y_data = y_data.reshape(-1,1)
        if y_data.size == 0 or y_data.shape[0] == 0 or (y_data.ndim == 2 and y_data.shape[1] == 0 and y_data.shape[0] == 1 and not np.any(y_data)):
            self.add_global_style(ax,subplot) 
            return

        err_data=subplot.get('err',None) 
        if err_data is not None:
            err_data = np.asarray(err_data)
            if err_data.shape != y_data.shape:
                if err_data.size == y_data.size: err_data = err_data.reshape(y_data.shape)
                elif y_data.shape[1] == 1 and err_data.size == y_data.shape[0]: err_data = err_data.reshape(-1,1)
                elif err_data.ndim == 1 and err_data.shape[0] == y_data.shape[1] and y_data.shape[0] > 1 : 
                    err_data_reshaped = np.zeros_like(y_data, dtype=float) 
                    for i_s_err in range(y_data.shape[1]): err_data_reshaped[:,i_s_err] = err_data[i_s_err]
                    err_data = err_data_reshaped
                else:
                    warnings.warn(f"Error data shape {err_data.shape} incompatible with y_data shape {y_data.shape}. Ignoring errors.", UserWarning)
                    err_data = None
        
        x_coords=subplot.get('x',np.arange(y_data.shape[0])) 
        x_coords = np.asarray(x_coords)
        is_stacked=subplot.get('is_stacked',False) 
        num_series = y_data.shape[1]
        
        bar_width_total_alloc = 0.8 
        default_bar_width = bar_width_total_alloc
        if num_series > 0 and not is_stacked: default_bar_width = bar_width_total_alloc / num_series
        
        bar_width=subplot.get('width', default_bar_width)
        
        series_names=subplot.get('series_names',None) 
        show_bar_values=subplot.get('show_values',False) 
        bar_orientation = subplot.get('orientation','vertical') 
        category_names=subplot.get('bar_names', [str(val) for val in x_coords]) 

        hatch_patterns=subplot.get('hatch_style',None) 
        
        opacity_vals=subplot.get('opacity',1.0)
        if isinstance(opacity_vals, (int, float)): opacity_vals = np.full(num_series if num_series > 0 else 1, opacity_vals)
        
        colors_list=subplot.get('color_series',[]) 
        edge_line_width=subplot.get('edge_width',0.5 if np.any(np.asanyarray(opacity_vals)<1.0) else 0) 
        edge_line_color=subplot.get('edge_color','black')
        
        loc_ax=ax
        sec_y_flag = subplot.get('secondary_y',False)
        if sec_y_flag:
            loc_ax=ax.twinx() if bar_orientation == 'vertical' else ax.twiny()
            loc_ax.grid(False)
        
        self.add_global_style_preplot(loc_ax,subplot)
        
        num_categories = y_data.shape[0]
        if num_categories == 0: 
            self.add_global_style(loc_ax, subplot)
            return

        if is_stacked:
            current_bottom_positive = np.zeros(num_categories)
            current_bottom_negative = np.zeros(num_categories)
            for i in range(num_series):
                label = series_names[i] if isinstance(series_names,(list,tuple)) and i < len(series_names) else None
                current_y = y_data[:,i]
                current_err_i = err_data[:,i] if isinstance(err_data, np.ndarray) and err_data.ndim==2 and i<err_data.shape[1] else None
                
                color_to_use = None
                if colors_list and i < len(colors_list) and colors_list[i] is not None: color_to_use = colors_list[i]
                
                alpha = opacity_vals[i % len(opacity_vals)] if isinstance(opacity_vals, (list,np.ndarray)) and len(opacity_vals)>0 else opacity_vals
                hatch = hatch_patterns[i % len(hatch_patterns)] if isinstance(hatch_patterns, list) and len(hatch_patterns)>0 else (hatch_patterns if isinstance(hatch_patterns, str) else None)

                bar_func = loc_ax.bar if bar_orientation == 'vertical' else loc_ax.barh
                bottom_to_use = np.where(current_y >= 0, current_bottom_positive, current_bottom_negative)

                bars = bar_func(x_coords, current_y, bar_width, label=label, 
                               bottom=bottom_to_use if bar_orientation == 'vertical' else None,
                               left=bottom_to_use if bar_orientation == 'horizontal' else None,
                               yerr=current_err_i if bar_orientation == 'vertical' else None,
                               xerr=current_err_i if bar_orientation == 'horizontal' else None,
                               capsize=3 if current_err_i is not None else 0, 
                               color=color_to_use, alpha=alpha, hatch=hatch, 
                               linewidth=edge_line_width, edgecolor=edge_line_color if edge_line_width > 0 else None)
                
                if show_bar_values: loc_ax.bar_label(bars, padding=2, fmt='%.2g', label_type='center')
                current_bottom_positive += np.where(current_y >= 0, current_y, 0)
                current_bottom_negative += np.where(current_y < 0, current_y, 0)
        else: # Grouped bars
            for i in range(num_series):
                offset = (i - (num_series - 1) / 2) * bar_width 
                label = series_names[i] if isinstance(series_names,(list,tuple)) and i < len(series_names) else None
                current_y = y_data[:,i]
                current_err_i = err_data[:,i] if isinstance(err_data, np.ndarray) and err_data.ndim==2 and i<err_data.shape[1] else None
                
                color_to_use = None
                if colors_list and i < len(colors_list) and colors_list[i] is not None: color_to_use = colors_list[i]
                
                alpha = opacity_vals[i % len(opacity_vals)] if isinstance(opacity_vals, (list,np.ndarray)) and len(opacity_vals)>0 else opacity_vals
                hatch = hatch_patterns[i % len(hatch_patterns)] if isinstance(hatch_patterns, list) and len(hatch_patterns)>0 else (hatch_patterns if isinstance(hatch_patterns, str) else None)

                bar_func = loc_ax.bar if bar_orientation == 'vertical' else loc_ax.barh
                bar_positions = x_coords + offset

                bars = bar_func(bar_positions, current_y, bar_width, label=label,
                               yerr=current_err_i if bar_orientation == 'vertical' else None,
                               xerr=current_err_i if bar_orientation == 'horizontal' else None,
                               capsize=3 if current_err_i is not None else 0,
                               color=color_to_use, alpha=alpha, hatch=hatch, 
                               linewidth=edge_line_width, edgecolor=edge_line_color if edge_line_width > 0 else None)
                if show_bar_values: loc_ax.bar_label(bars, padding=3, fmt='%.2g')
        
        if bar_orientation == 'vertical':
            loc_ax.set_xticks(x_coords)
            if category_names and len(category_names) == len(x_coords): # Check length match
                 loc_ax.set_xticklabels(category_names)
        else: 
            loc_ax.set_yticks(x_coords)
            if category_names and len(category_names) == len(x_coords): # Check length match
                 loc_ax.set_yticklabels(category_names)
        
        self.add_global_style(loc_ax,subplot)
        
    def draw_regression(self, ax: plt.Axes,subplot: dict):
        res=subplot.get('reg_obj',None)
        normalize = subplot.get('normalize',False)

        if normalize:
            res.attributes['params'] = res.attributes['params'] * (res.arrays['std_dev'][:-1] / res.arrays['std_dev'][-1])
            res.attributes['std_errors'] = res.attributes['std_errors'] * (res.arrays['std_dev'][:-1] / res.arrays['std_dev'][-1])

        color_pal = subplot.get('color_pal',sns.color_palette("coolwarm_r", as_cmap=True))
        col_sigmoid=subplot.get('col_sigmoid',5)
        sec_y = subplot.get('secondary_y',False)
        star_hatches=subplot.get('star_hatches',['','.','o','O'])
        xlim=subplot.get('xlim',[-1.05*(res.attributes['params'].abs().max()+res.attributes['std_errors'].abs().max()),1.05*res.attributes['params'].abs().max()+res.attributes['std_errors'].abs().max()])
        alpha= subplot.get('opacity',1)
        edgecolors = subplot.get('edge_color','black')
        linewidths = subplot.get('edge_width',0.5)
        
        subplot['xlim'] = xlim #update xlim for global style
        
        if sec_y:
            loc_ax=ax.twinx()
            alpha=alpha/3
            loc_ax.grid(False)
        else:    
            loc_ax=ax
        
        self.add_global_style_preplot(loc_ax,subplot)
        curve = lambda x: 1/(1+np.exp(-col_sigmoid*x))  

        for idx in range(len(res.attributes['params'])):
            color = 'lightgrey' if res.attributes['pvalues'].iloc[idx] > 0.1 else color_pal(curve(res.attributes['params'].iloc[idx]))
            hatch = star_hatches[int(res.attributes['pvalues'].iloc[idx] < 0.1) + int(res.attributes['pvalues'].iloc[idx] < 0.05) + int(res.attributes['pvalues'].iloc[idx] < 0.01)]
            # Plotting the bar
            loc_ax.barh(idx, res.attributes['params'].iloc[idx], xerr=res.attributes['std_errors'].iloc[idx], color=color, capsize=5, hatch=hatch,linewidth=linewidths , edgecolor=edgecolors,error_kw={'capthick': linewidths*2, 'elinewidth': linewidths, 'ecolor': edgecolors},alpha=alpha)
        #for idx, row in res.attributes['params'].iterrows():
        #    color = 'lightgrey' if row['stars'] == 0 else color_pal(curve(row['mean']))
        #    hatch = star_hatches[int(row['stars'])]
        #    # Plotting the bar
        #    loc_ax.barh(idx, row['mean'], xerr=row['std'], color=color, capsize=5, hatch=hatch,linewidth=linewidths , edgecolor=edgecolors,error_kw={'capthick': linewidths*2, 'elinewidth': linewidths, 'ecolor': edgecolors},alpha=alpha)
        # Setting labels for y-axis
        loc_ax.set_yticks(range(len(res.attributes['params'])))
        loc_ax.set_yticklabels(res.attributes['params'].index)
        # Optional: Adding grid, labels, title, etc. for clarity
        loc_ax.axvline(0, color='black', linewidth=0.8)  # Add a line at x=0 for reference
        self.add_global_style(loc_ax,subplot)
            

    def draw_hist(self, ax: plt.Axes,subplot: dict):
        x_data=subplot.get('y',None) 
        if x_data is None: return
        bins_val=subplot.get('bins','auto') 
        is_density=subplot.get('is_density',False) 
        hist_plot_type=subplot.get('hist_type','bar') 
        bar_orientation = subplot.get('orientation','vertical') 
        opacity_val= subplot.get('opacity',0.75) 
        series_names=subplot.get('series_names',None) 
        colors_list=subplot.get('color_series', None)
        
        loc_ax=ax
        sec_y_flag = subplot.get('secondary_y',False)
        if sec_y_flag:
            loc_ax=ax.twinx() if bar_orientation == 'vertical' else ax.twiny()
            loc_ax.grid(False)
        
        self.add_global_style_preplot(loc_ax,subplot)
        is_stacked_hist = subplot.get('is_stacked', hist_plot_type == 'barstacked') 
        loc_ax.hist(x_data, bins=bins_val, density=is_density, 
                    histtype=hist_plot_type if hist_plot_type != 'barstacked' else 'bar', 
                    alpha=opacity_val, orientation=bar_orientation, label=series_names, stacked=is_stacked_hist,
                    color=colors_list) 
        self.add_global_style(loc_ax,subplot)

    def draw_kde(self, ax: plt.Axes,subplot: dict):
        y_data=subplot.get('y',None) 
        if y_data is None: return
        y_data = np.asarray(y_data)
        if y_data.ndim == 1: y_data = y_data.reshape(-1,1)
        if y_data.shape[0] == 0: return

        bw_method_val=subplot.get('bandwidth','scott') 
        use_common_norm = subplot.get('common_norm',True) 
        series_names=subplot.get('series_names',None) 
        
        opacity_vals=subplot.get('opacity', 0.7)
        if isinstance(opacity_vals, (int, float)): opacity_vals = np.full(y_data.shape[1] if y_data.shape[1]>0 else 1, opacity_vals)
        
        colors_list=subplot.get('color_series',[]) 
        fill_kde = subplot.get('fill',True) 
        
        loc_ax=ax
        sec_y_flag = subplot.get('secondary_y',False)
        if sec_y_flag:
            loc_ax=ax.twinx()
            loc_ax.grid(False)
        
        self.add_global_style_preplot(loc_ax,subplot)
        
        for i in range(y_data.shape[1]): 
            label = series_names[i] if isinstance(series_names,(list,tuple)) and i < len(series_names) else (series_names if y_data.shape[1]==1 and isinstance(series_names,str) else None)
            
            color_to_use = None
            if colors_list and i < len(colors_list) and colors_list[i] is not None: color_to_use = colors_list[i]
                
            alpha = opacity_vals[i % len(opacity_vals)] if isinstance(opacity_vals, (list, np.ndarray)) and len(opacity_vals)>0 else opacity_vals

            sns.kdeplot(y_data[:,i], bw_method=bw_method_val, fill=fill_kde, common_norm=use_common_norm,
                        alpha=alpha, color=color_to_use, label=label, ax=loc_ax, warn_singular=False) 
        self.add_global_style(loc_ax,subplot)
        
    def draw_scat(self, ax: plt.Axes,subplot: dict):
        x_data=subplot.get('x',None) 
        y_data=subplot.get('y',None) 
        if x_data is None or y_data is None: return
        x_data, y_data = np.asarray(x_data), np.asarray(y_data)
        if x_data.size == 0 or y_data.size == 0: return

        point_size=subplot.get('size',30) 
        color_data=subplot.get('color_array',None) 
        if color_data is not None: color_data = np.asarray(color_data)
        
        color_legend_labels=subplot.get('color_names',None) 
        marker_style = subplot.get('marker','o') 
        opacity_val = subplot.get('opacity', 0.7)
        
        edge_colors_val = subplot.get('edge_color','face') 
        edge_widths_val = subplot.get('edge_width',0.5 if opacity_val < 1.0 else 0) 
        cmap_name = subplot.get('cmap', 'viridis') 
        
        loc_ax=ax
        sec_y_flag = subplot.get('secondary_y',False)
        if sec_y_flag:
            loc_ax=ax.twinx()
            loc_ax.grid(False)
        
        self.add_global_style_preplot(loc_ax,subplot)
        
        # Determine if color_data is categorical for legend handling
        is_categorical_color = False
        if color_legend_labels and color_data is not None:
            # Check if color_data seems categorical (e.g., not a float array for continuous mapping)
            # This check can be tricky. A simple heuristic:
            if color_data.ndim == 1 and (color_data.dtype == object or color_data.dtype.kind in 'SU' or (np.issubdtype(color_data.dtype, np.integer) and len(np.unique(color_data)) <= 20) ): # Heuristic for categorical
                is_categorical_color = True


        if is_categorical_color:
            unique_cats = pd.unique(color_data) 
            
            prop_cycle = plt.rcParams['axes.prop_cycle']
            colors_from_cycle = [item['color'] for item in prop_cycle]
            
            color_map_func = mt.colormaps.get_cmap(cmap_name) if cmap_name else None


            for i_cat_val, cat_val_orig in enumerate(unique_cats):
                mask = (color_data == cat_val_orig)
                if not np.any(mask): continue 

                cat_label = color_legend_labels[i_cat_val % len(color_legend_labels)] if color_legend_labels and i_cat_val < len(color_legend_labels) else str(cat_val_orig)
                
                cat_plot_color = None
                if color_map_func : 
                    color_idx_norm = i_cat_val / (len(unique_cats) -1) if len(unique_cats) > 1 else 0.5
                    cat_plot_color = color_map_func(color_idx_norm)
                else: # Fallback to default cycler if no cmap
                    cat_plot_color = colors_from_cycle[i_cat_val % len(colors_from_cycle)]


                loc_ax.scatter(x_data[mask], y_data[mask], s=point_size, 
                               color=cat_plot_color, 
                               marker=marker_style, alpha=opacity_val, 
                               edgecolors=edge_colors_val, linewidths=edge_widths_val, label=cat_label)
        else: 
            scatter_plot = loc_ax.scatter(x_data,y_data,s=point_size,c=color_data,marker=marker_style,
                                          alpha=opacity_val, cmap=cmap_name, 
                                          edgecolors=edge_colors_val,linewidths=edge_widths_val,
                                          label=subplot.get('label', None)) 
            if subplot.get('show_colorbar', False) and color_data is not None and not is_categorical_color :
                try:
                    plt.colorbar(scatter_plot, ax=loc_ax, label=subplot.get('colorbar_label',''))
                except Exception as e:
                    warnings.warn(f"Could not create colorbar: {e}", UserWarning)
        
        self.add_global_style(loc_ax,subplot) 

    def draw_line_stddev(self, ax: plt.Axes,subplot: dict):
        y_data=subplot.get('y',None) 
        if y_data is None: return
        y_data = np.asarray(y_data)
        if y_data.ndim == 1: y_data = y_data.reshape(-1,1)
        if y_data.shape[0] == 0: return
        
        x_data=subplot.get('x',np.arange(y_data.shape[0])) 
        x_data = np.asarray(x_data)
        std_dev_multiplier=subplot.get('std_multiplier',1) 
        std_dev_values=subplot.get('std',None) 
        if std_dev_values is None and y_data.shape[0] > 1: 
             std_dev_values = np.std(y_data,axis=0, keepdims=True)
        elif std_dev_values is None: 
             std_dev_values = np.zeros_like(y_data) 
        std_dev_values = np.asarray(std_dev_values)

        if std_dev_values.ndim == 1 and y_data.ndim == 2 and std_dev_values.shape[0] == y_data.shape[1]: 
            std_dev_values = std_dev_values.reshape(1, -1) 
        elif std_dev_values.size == 1 and y_data.ndim ==2 : 
            std_dev_values = np.full((1,y_data.shape[1]), std_dev_values.item())
        elif std_dev_values.shape != (1, y_data.shape[1]) and std_dev_values.shape != y_data.shape: 
            warnings.warn(f"Std_dev_values shape {std_dev_values.shape} not compatible with y_data {y_data.shape}. Using zero std.", UserWarning)
            std_dev_values = np.zeros((1, y_data.shape[1]))


        series_names=subplot.get('series_names',None) 
        alpha_fill_val = subplot.get('alpha_fill', 0.2) 
        alpha_line_val = subplot.get('alpha_line', 0.6) 
        colors_list = subplot.get('color_series', [])


        loc_ax=ax
        sec_y_flag = subplot.get('secondary_y',False)
        if sec_y_flag:
            loc_ax=ax.twinx()
            loc_ax.grid(False)
        
        self.add_global_style_preplot(loc_ax,subplot)

        for i in range(y_data.shape[1]):
            label = series_names[i] if isinstance(series_names,(list,tuple)) and i < len(series_names) else (series_names if y_data.shape[1]==1 and isinstance(series_names,str) else None)
            
            current_y = y_data[:,i]
            current_std_val = std_dev_values[0,i] if std_dev_values.ndim == 2 and i < std_dev_values.shape[1] else 0
            current_std_for_fill = np.full_like(current_y, current_std_val)

            color_to_use = None
            if colors_list and i < len(colors_list) and colors_list[i] is not None: 
                color_to_use = colors_list[i]
            else: 
                # Ensure color is fetched once per series for consistency between line and fill
                prop_cycle = loc_ax.axes.prop_cycle if hasattr(loc_ax, 'axes') else plt.rcParams['axes.prop_cycle'] # More robust access
                cycled_colors = [item['color'] for item in prop_cycle]
                color_to_use = cycled_colors[i % len(cycled_colors)]


            loc_ax.plot(x_data,current_y,alpha=alpha_line_val,label=label, color=color_to_use) 
            loc_ax.fill_between(x_data, current_y + current_std_for_fill*std_dev_multiplier, 
                                current_y - current_std_for_fill*std_dev_multiplier, alpha=alpha_fill_val, color=color_to_use, edgecolor='none')
        
        self.add_global_style(loc_ax,subplot)         
    
    def static_lines(self, ax: plt.Axes,subplot: dict):
        orientation_val=subplot.get('orientation','vertical') 
        x_pos=subplot.get('x',None) 
        y_pos=subplot.get('y',None) 
        
        ymin_frac=subplot.get('ymin',0) 
        ymax_frac=subplot.get('ymax',1) 
        xmin_frac=subplot.get('xmin',0) 
        xmax_frac=subplot.get('xmax',1) 
        
        color_val=subplot.get('color','black') 
        linestyle_val=subplot.get('style','solid') 
        label_val = subplot.get('label',None) 
        linewidth_val = subplot.get('linewidth', plt.rcParams['lines.linewidth']) 

        if orientation_val=='vertical' and x_pos is not None:
            ax.axvline(x_pos, ymin=ymin_frac, ymax=ymax_frac,
                       color=color_val,linestyle=linestyle_val,label=label_val, linewidth=linewidth_val)
        elif orientation_val !='vertical' and y_pos is not None: 
            ax.axhline(y_pos, xmin=xmin_frac, xmax=xmax_frac,
                       color=color_val,linestyle=linestyle_val,label=label_val, linewidth=linewidth_val)
    
    def draw_from_func(self, ax: plt.Axes,subplot: dict):
        func_to_run=subplot.get('func',None) 
        if func_to_run is None or not callable(func_to_run): 
            warnings.warn("No valid function provided for 'draw_from_func'", UserWarning)
            self.add_global_style(ax,subplot) 
            return

        args_for_func_call=subplot.get('args',{}) 
        try:
            func_to_run(ax,**args_for_func_call) 
        except Exception as e:
            warnings.warn(f"Error executing function in 'draw_from_func': {e}", UserWarning)
        
        self.add_global_style(ax,subplot) 
    
    @staticmethod
    def available_plot_types():
        print("Implemented plots: line, bar, scatter, hist, kde, line_stddev, static_lines, from_func, regression, annotation")

#%%
dt = pd.read_csv('global_universe.csv')

vz = Visualizer()
subfigures = []

#%%
######## FIGURE 1 PANEL A,C: clustered boxplots by group (within) and year (between) of signal strength ########
dt_1a = dt[['gvkey_iid','year','signal_0','signal_1','signal_2','Industry','curcdd']].dropna(subset=['signal_0'])
dt_1a = pd.melt(dt_1a, id_vars=['gvkey_iid','year','Industry','curcdd'],
                value_vars=['signal_0','signal_1','signal_2'],
                var_name='signal_type', value_name='signal_strength')

def draw_clustered_boxplot(ax: plt.Axes, data: pd.DataFrame, stack_group_cols: str, group_col: str, year_col: str, value_col: str):
    #for stack in data[stack_group_cols].unique():
        #subset = data[data[stack_group_cols] == stack]
        sns.boxplot(
            data=data,
            x=year_col,
            y=value_col,
            hue=group_col,
            ax=ax,
            linewidth=1.5,      
            showfliers=False,   
            palette="viridis",  
            dodge=True          
        )
        #ax.set_title(f"{stack} - {group_col} by {year_col}")
        ax.set_xlabel(year_col)
        ax.set_ylabel(value_col)
        ax.legend(title=group_col)


subfigures.append({
    'idx': '1a',
    'type': 'from_func',
    'func': draw_clustered_boxplot,
    'args': {
        'data': dt_1a,
        'group_col': 'signal_type',
        'year_col': 'year',
        'value_col': 'signal_strength',
        'stack_group_cols': 'signal_type'
    }
})
# %%
grid = [['1a']]
vz.create_fig(grid,subfigures,file='fig1a.png',size=(10,10)) 
# %%
