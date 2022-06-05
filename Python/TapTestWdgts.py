# DAS-taptest widgets
# %matplotlib notebook
import ipywidgets as widgets
import numpy as np
from scipy.signal import hilbert
from scipy import signal
import datetime, pytz
import dateutil.parser
from scipy.interpolate import interp1d

# Plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
params = {
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'savefig.dpi': 300,  # to adjust notebook inline plot size
    'axes.labelsize': 12, # fontsize for x and y labels (was 10)
    'axes.titlesize': 12,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex':False
}
matplotlib.rcParams.update(params)


import utm
def azimuth_dipping_from_gps(xlon, ylat, zele):
    """
    Function by Jiaxuan Li
    get azimuth and dipping angle from gps locations: longitude, latitude, and elevation
    Args:
        xlon: longitude array in degree
        ylat: latitude array in degree
        xele: elevaton array in meter
    Returns:
        azimuth: angle zero in North direction, positive for left hand rotation w.r.t Up
        dipping: angle zero in Up direction, ranges from 0 to 180 degree
    """
    utm_reference = utm.from_latlon(ylat[0], xlon[0])
    utm_coords = utm.from_latlon(ylat, xlon, utm_reference[2], utm_reference[3])
    nx = len(xlon)
    azimuth = np.zeros(nx)
    dipping = np.zeros(nx)
    unit_z = np.array([0,0,1])
    r2d = 180/np.pi
    for i in range(nx):
        i1 = i if i == 0 else i-1
        i2 = i if i == nx-1 else i+1
        # unit vectors
        vec_en = np.array([utm_coords[0][i2]-utm_coords[0][i1], utm_coords[1][i2]-utm_coords[1][i1]])
        vec_enz = np.append(vec_en, zele[i2]-zele[i1])
        vec_en = vec_en/np.sqrt(np.sum(vec_en**2))
        vec_enz = vec_enz/np.sqrt(np.sum(vec_enz**2))
        # azimuth
        azi = np.arctan2(vec_en[0], vec_en[1])*r2d
        if azi < 0:
            azi += 360
        # dipping
        dip = np.arccos(np.dot(vec_enz, unit_z))*r2d
        #
        azimuth[i] = azi
        dipping[i] = dip
    return azimuth, dipping

def channel_spacing_from_gps(xlon, ylat, zele):
    """
    Function by Jiaxuan Li
    get channel spacing gps locations: longitude, latitude, and elevation
    Args:
        xlon: longitude array in degree
        ylat: latitude array in degree
        xele: elevaton array in meter
    Returns:
        channel_spacing: channel spacing in meters
        channel_spacing_horizontal: channel spacing within horizontal plane in meters
    """
    utm_reference = utm.from_latlon(ylat[0], xlon[0])
    utm_coords = utm.from_latlon(ylat, xlon, utm_reference[2], utm_reference[3])
    nx = len(xlon)
    channel_spacing = np.zeros(nx)
    channel_spacing_horizontal = np.zeros(nx)
    for i in range(nx-1):
        i1 = i if i == 0 else i
        i2 = i if i == nx-1 else i+1
        # unit vectors
        vec_en = np.array([utm_coords[0][i2]-utm_coords[0][i1], utm_coords[1][i2]-utm_coords[1][i1]])
        vec_enz = np.append(vec_en, zele[i2]-zele[i1])
        channel_spacing[i] = np.sqrt(np.sum(vec_enz**2))
        channel_spacing_horizontal[i] = np.sqrt(np.sum(vec_en**2))
    return channel_spacing, channel_spacing_horizontal

import requests
import urllib
import urllib3

def make_remote_request(url: str, params: dict):
    """
    Makes the remote request
    Continues making attempts until it succeeds
    """

    count = 1
    while True:
        try:
            response = requests.get((url + urllib.parse.urlencode(params)))
        except (OSError, urllib3.exceptions.ProtocolError) as error:
            print('\n')
            print('*' * 20, 'Error Occured', '*' * 20)
            print(f'Number of tries: {count}')
            print(f'URL: {url}')
            print(error)
            print('\n')
            count += 1
            continue
        break

    return response


def elevation_function(lat,lon):
    url = 'https://nationalmap.gov/epqs/pqs.php?'
    params = {'x': lon,
              'y': lat,
              'units': 'Meters',
              'output': 'json'}
    result = make_remote_request(url, params)
    return result.json()['USGS_Elevation_Point_Query_Service']['Elevation_Query']['Elevation']


class badchnnlwdgt(widgets.VBox):
    """
        Interactive plots for bad channel identification
        data: [array] DAS data array (channels,time)
        dt: [float] time sampling [s]
        min_time: [float] initial time to be displayed (default 0.0 s)
    """
    
    def __init__(self, data, dt, min_time=0.0):
        super().__init__()
        output = widgets.Output()
        self.data = data
        self.nt = data.shape[1]
        self.dt = dt
        self.nch = data.shape[0]
        self.chAx = np.arange(self.nch)
        self.tAx = np.linspace(0.0, (self.nt-1)*self.dt, self.nt)
        # Initial plot interval
        self.min_t = min_time
        self.max_t = self.min_t + 100.0
        self.it_min = int(self.min_t/self.dt+0.5)
        self.it_max = min(self.nt-1, int(self.max_t/self.dt+0.5))
        self.min_ch = 0
        self.max_ch = self.nch-1
        clipVal = np.percentile(np.absolute(self.data[self.min_ch:self.max_ch,self.it_min:self.it_max]), 98)
        self.bad_channels = np.array([], dtype=int)
        self.show_bad = True # Flag whether to show or not bad channels
        
        # Creating the initial data plot
        with output:
            self.fig, (self.ax1, self.ax2, self.ax3) = plt.subplots(3, 1, 
                                                                    figsize=(9,10.8), gridspec_kw={'height_ratios': [0.3,1.5,0.3]})
        self.im = self.ax2.imshow(self.data[self.min_ch:self.max_ch,self.it_min:self.it_max].T, 
                                 extent=[self.chAx[self.min_ch], self.chAx[self.max_ch], self.tAx[self.it_max], self.tAx[self.it_min]],
                                 aspect='auto', vmin=-clipVal, vmax=clipVal, 
                                 cmap=plt.get_cmap('seismic'))
        self.ax2.set_ylabel("Time [s]")
        self.ax2.set_xlabel("Channel number")
        self.ax2.grid()
        # Colorbar
        divider = make_axes_locatable(self.ax2)
        cax = divider.append_axes("bottom", size="2%", pad=0.7)
        cbar = plt.colorbar(self.im, orientation="horizontal", cax=cax)
        cbar.set_label('Amplitude')
        
        # Energy sum
        energy = np.sum(abs(self.data[self.min_ch:self.max_ch, self.it_min:self.it_max]), axis=1)
        self.line_en, = self.ax1.plot(range(self.min_ch,self.max_ch), energy/(energy.max()+1e-18), lw=3)
        self.ax1.autoscale(enable=True, axis='x', tight=True)
        self.ax1.set_xlabel("channel number")
        self.ax1.set_ylabel("Normalized \n Energy")
        self.ax1.grid()
        self.ax1.set_ylim([0.0, 1.0])
        
        # Channel color
        self.line_ch, = self.ax3.plot(range(self.min_ch,self.max_ch), [0.0]*len(range(self.min_ch,self.max_ch)), 'o')
        self.line_bad_ch, = self.ax3.plot([], [], 'ro')
        self.ax3.get_yaxis().set_visible(False)
        self.ax3.autoscale(enable=True, axis='x', tight=True)
        self.ax3.grid()
        self.ax3.set_xlabel("channel number")
        
        show_bad_ch = widgets.Checkbox(
            value=True,
            description='Show bad channels',
            disabled=False,
            indent=False
        )

        bad_channels_wdg = widgets.Textarea(
                    value='', 
                    description='bad channels', 
                    continuous_update=False
        )
        
        min_ch_wdg = widgets.IntSlider(
            value=0, 
            min=0, max=self.nch-1, step=1,
            description='min channel',
            continuous_update=False
        )

        max_ch_wdg = widgets.IntSlider(
            value=self.nch, 
            min=0, max=self.nch-1, step=1,
            description='max channel',
            continuous_update=False
        )
        
        min_time_wdg = widgets.FloatSlider(
            value=self.min_t, 
            min=self.tAx[0], max=self.tAx[-1], step=0.5,
            description='min time [s]',
            continuous_update=False
        )
        
        max_time_wdg = widgets.FloatSlider(
            value=self.max_t, 
            min=self.tAx[0], max=self.tAx[-1], step=0.5,
            description='max time [s]',
            continuous_update=False
        )
        
        
        bad_ch_check = widgets.HBox([bad_channels_wdg,show_bad_ch])
        ch_sliders = widgets.HBox([min_ch_wdg,max_ch_wdg])
        time_sliders = widgets.HBox([min_time_wdg,max_time_wdg])
        controls = widgets.VBox([bad_ch_check, ch_sliders, time_sliders])
        
        show_bad_ch.observe(self.update_show_bad, 'value')
        bad_channels_wdg.observe(self.update_plot_chan, 'value')
        min_ch_wdg.observe(self.update_min_ch, 'value')
        max_ch_wdg.observe(self.update_max_ch, 'value')
        min_time_wdg.observe(self.update_min_time, 'value')
        max_time_wdg.observe(self.update_max_time, 'value')
        
        # add to children
        self.children = [output, controls]

    # Methods to update the plot using widgets
    def get_channel_mask(self):
        mask = (self.chAx >= self.min_ch) * (self.chAx <= self.max_ch)
        if len(self.bad_channels) > 0: 
            mask_tmp = np.ones(self.nch, dtype=bool)
            mask_tmp[self.bad_channels] = False
            # Getting all good channels within channel range
            mask *= mask_tmp
        return mask
    def update_energy_plot(self):
        channel_mask = self.get_channel_mask()
        if np.any(channel_mask):
            if not self.show_bad:
                chAx_tmp = self.chAx[channel_mask]
                energy = np.sum(abs(self.data[chAx_tmp, self.it_min:self.it_max]), axis=1)
                min_ch = np.amin(chAx_tmp)
                max_ch = min_ch + len(energy)
                self.line_en.set_data(range(min_ch,max_ch), energy/(energy.max()+1e-18))
            else:
                energy = np.sum(abs(self.data[self.min_ch:self.max_ch, self.it_min:self.it_max]), axis=1)
                self.line_en.set_data(range(self.min_ch,self.max_ch), energy/(energy.max()+1e-18))
        else:
            print("No channel to show with these settings!")
            return
        self.ax1.relim()
        self.ax1.autoscale_view()
    
    def update_ch_plot(self):
        channel_mask = self.get_channel_mask()
        if np.any(channel_mask):
            if not self.show_bad:
                chAx_tmp = self.chAx[channel_mask]
                min_ch = np.amin(chAx_tmp)
                max_ch = min_ch + np.count_nonzero(chAx_tmp)        
                self.line_ch.set_data(range(min_ch,max_ch), [0.0]*len(range(min_ch,max_ch)))
                self.line_bad_ch.set_data([],[])
            else:
                self.line_ch.set_data(range(self.min_ch,self.max_ch), [0.0]*len(range(self.min_ch,self.max_ch)))
                mask_bad = (self.bad_channels >= self.min_ch) * (self.bad_channels <= self.max_ch)
                if len(self.bad_channels) > 0 and np.any(mask_bad):
                    self.line_bad_ch.set_data(self.bad_channels[mask_bad], [0.0]*len(self.bad_channels[mask_bad]))
                else:
                    self.line_bad_ch.set_data([],[])
        else:
            return
        self.ax3.relim()
        self.ax3.autoscale_view()
        
    def update_data_plot(self):
        channel_mask = self.get_channel_mask()
        if np.any(channel_mask):
            if not self.show_bad:
                chAx_tmp = self.chAx[channel_mask]
                min_ch = np.amin(chAx_tmp)
                max_ch = min_ch + np.count_nonzero(chAx_tmp)
                self.im.set_data(self.data[chAx_tmp,self.it_min:self.it_max].T)
                self.im.set_extent([min_ch, max_ch, self.tAx[self.it_max], self.tAx[self.it_min]])
            else:
                self.im.set_data(self.data[self.min_ch:self.max_ch,self.it_min:self.it_max].T)
                self.im.set_extent([self.chAx[self.min_ch], self.chAx[self.max_ch], self.tAx[self.it_max], self.tAx[self.it_min]])
        else:
            return
    def update_min_ch(self, value):
        self.min_ch = value["new"]
        if self.max_ch <= self.min_ch:
            print("Max channel smaller than min channel! Change sliders values")
            return
        self.update_data_plot()
        self.update_energy_plot()
        self.update_ch_plot()

    def update_max_ch(self, value):
        self.max_ch = value["new"]
        if self.max_ch <= self.min_ch:
            print("Max channel smaller than min channel! Change sliders values")
            return
        self.update_data_plot()
        self.update_energy_plot()
        self.update_ch_plot()
    
    def update_min_time(self, value):
        self.min_t = value["new"]
        if self.max_t <= self.min_t:
            print("Max time smaller than min time! Change sliders values")
            return
        self.it_min = int(self.min_t/self.dt+0.5)
        self.it_max = int(self.max_t/self.dt+0.5)
        self.update_data_plot()
        self.update_energy_plot()
        self.update_ch_plot()

    def update_max_time(self, value):
        self.max_t = value["new"]
        if self.max_t <= self.min_t:
            print("Max time smaller than min time! Change sliders values")
            return
        self.it_min = int(self.min_t/self.dt+0.5)
        self.it_max = int(self.max_t/self.dt+0.5)
        self.update_data_plot()
        self.update_energy_plot()
        self.update_ch_plot()
        
    def update_show_bad(self, value):
        self.show_bad = value["new"]
        self.update_data_plot()
        self.update_energy_plot()
        self.update_ch_plot()

    # Functions to update plot
    def update_plot_chan(self, value):
        values = value["new"].split(",")
        bad_channels = np.array([], dtype=int)
        for val in values:
            try:
                # List of bad channels
                if ":" in val:
                    strt_ch = int(val.split(":")[0])
                    lst_ch = int(val.split(":")[1])
                    bad_channels = np.append(bad_channels, np.arange(strt_ch,lst_ch))
                else:
                    bad_channels = np.append(bad_channels, int(val))
            except ValueError:
                self.bad_channels = np.array([], dtype=int)
                print("Incorrect input for bad channel!")
                self.update_data_plot()
                self.update_energy_plot()
                self.update_ch_plot()
                return
        # Remove repeated values
        self.bad_channels = np.unique(bad_channels)
        self.update_data_plot()
        self.update_energy_plot()
        self.update_ch_plot()

class taptestwdgt(widgets.VBox):
    """
        Interactive plots for bad channel identification
        data: [array] DAS data array (channels,time)
        dt: [float] time sampling [s]
        dch: [float] nominal channel sampling [m]
        gps_time: [datetime] GPS vehicle time points
        gps_lat: [float] GPS vehicle latitude positions or closest latitude of the fiber
        gps_lon: [float] GPS vehicle longitude positions or closest longitude of the fiber
        orig_time: [datetime] Time of the first time sample of the data
        min_time: [float] initial time to be displayed (default 0.0 s)
        envelope: [boolean] Plot data envelope (default True)
        pclip: [float] Percentile clipping threshold for data plotting (default 98.0)
    """
    
    def __init__(self, data, dt, dch, gps_time, gps_dist, gps_lat, gps_lon, 
                orig_time, min_time=0.0, envelope=True, pclip=98.0):
        super().__init__()
        output = widgets.Output()
        if envelope:
            env = np.abs(hilbert(data))
            env /= env.max()
            self.data = env
        else:
            self.data = data
        self.nt = data.shape[1]
        self.dt = dt
        self.nch = data.shape[0]
        self.chAx = np.arange(self.nch)
        self.tAx = np.linspace(0.0, (self.nt-1)*self.dt, self.nt)
        # Initial plot interval
        self.min_t = min_time
        self.max_t = self.min_t + 100.0
        self.it_min = int(self.min_t/self.dt+0.5)
        self.it_max = int(self.max_t/self.dt+0.5)
        self.min_ch = 0
        self.max_ch = self.nch-1
        clipVal = np.percentile(np.absolute(self.data), pclip)
        self.bad_channels = np.array([], dtype=int)
        self.show_bad = True # Flag whether to show or not bad channels
        vmax = clipVal
        vmin = 0.0 if envelope else -vmax
        # Preparing GPS data for plotting
        self.gps_time = np.array([tm.timestamp() - orig_time.timestamp() for tm in gps_time])
        self.gps_ch = gps_dist/dch
        # Interpolating to fine time sampling
        f_ch = interp1d(self.gps_time, self.gps_ch, kind='linear', bounds_error=False)
        ntFine = int((self.gps_time[-1]-self.gps_time[0])/0.1)+1
        self.gps_time_int = np.linspace(self.gps_time[0],self.gps_time[-1],ntFine)
        self.chAxTap = f_ch(self.gps_time_int)
        self.shift = 0.0
        self.show_tap = True
        self.reverse_tap = False
        self.show_map_num = False
        # Channel mapping variables
        self.min_ch_map = 0
        self.max_ch_map = self.nch
        self.mapped_channels = np.array([], dtype=int)
        self.mapped_lat = np.array([])
        self.mapped_lon = np.array([])
        self.mapped_ele = np.array([])
        # Storing GPS latitude and longitude
        if any(len(gps_lat) != len(lst) for lst in [gps_lon, self.gps_ch]):
            raise ValueError("GPS array lengths are not consistent!")
        self.gps_lat = gps_lat
        self.gps_lon = gps_lon
        # Function to interpolate positions based on GPS times
        self.f_lat = interp1d(self.gps_time, self.gps_lat, kind='linear', bounds_error=False)
        self.f_lon = interp1d(self.gps_time, self.gps_lon, kind='linear', bounds_error=False)
        # Creating the initial data plot
        with output:
            self.fig, (self.ax0, self.ax1, self.ax2, self.ax3) = plt.subplots(4, 1, 
                                                                    figsize=(9,15), gridspec_kw={'height_ratios': [2.0,0.3,1.5,0.3]})
        self.im = self.ax2.imshow(self.data[:,self.it_min:self.it_max].T, 
                                 extent=[self.chAx[0], self.chAx[-1], self.tAx[self.it_max], self.tAx[self.it_min]],
                                 aspect='auto', vmin=vmin, vmax=vmax, 
                                 cmap=plt.get_cmap('seismic'))
        # Plotting tap test line
        mask_ln_tap = (self.chAxTap+self.shift >= self.min_ch) * (self.chAxTap+self.shift <= self.max_ch)
        self.ax2.set_ylim([self.max_t,self.min_t])
        if np.any(mask_ln_tap):
            self.line_tap, = self.ax2.plot(self.chAxTap[mask_ln_tap]+self.shift,self.gps_time_int[mask_ln_tap], "r--", lw=3)
        else:
            self.line_tap, = self.ax2.plot([], [], "r--", lw=3)
        self.ax2.set_ylabel("Time [s]")
        self.ax2.set_xlabel("Channel number")
        self.ax2.grid()
        # Colorbar
        divider = make_axes_locatable(self.ax2)
        cax = divider.append_axes("bottom", size="2%", pad=0.7)
        cbar = plt.colorbar(self.im, orientation="horizontal", cax=cax)
        cbar.set_label('Amplitude')
        
        # Energy sum
        energy = np.sum(abs(self.data[self.min_ch:self.max_ch, self.it_min:self.it_max]), axis=1)
        self.line_en, = self.ax1.plot(range(self.min_ch,self.max_ch), energy/(energy.max()+1e-18), lw=3)
        self.ax1.autoscale(enable=True, axis='x', tight=True)
        self.ax1.set_xlabel("channel number")
        self.ax1.set_ylabel("Normalized \n Energy")
        self.ax1.grid()
        self.ax1.set_ylim([0.0, 1.0])
        
        # Channel color
        self.line_ch, = self.ax3.plot(range(self.min_ch,self.max_ch), [0.0]*len(range(self.min_ch,self.max_ch)), 'o')
        self.line_bad_ch, = self.ax3.plot([], [], 'ro')
        self.line_mapped, = self.ax3.plot([], [], 'go')
        self.ax3.get_yaxis().set_visible(False)
        self.ax3.autoscale(enable=True, axis='x', tight=True)
        self.ax3.grid()
        self.ax3.set_xlabel("channel number")
        
        # Mapped channels
        self.scat_ch, = self.ax0.plot([],[],'go')
        self.ann = []
        minLat = np.amin(gps_lat)
        maxLat = np.amax(gps_lat)
        minLon = np.amin(gps_lon)
        maxLon = np.amax(gps_lon)
        self.ax0.set_ylim([minLat,maxLat])
        self.ax0.set_xlim([minLon,maxLon])
        self.ax0.set_xlabel("Longitude [deg]")
        self.ax0.set_ylabel("Latitude [deg]")
        self.ax0.grid()
            
        # Bad-channel widgets
        
        show_bad_ch = widgets.Checkbox(
            value=True,
            description='Show bad channels',
            disabled=False,
            indent=False
        )

        bad_channels_wdg = widgets.Textarea(
                    value='', 
                    description='bad channels', 
                    continuous_update=False
        )
        
        # Display limit widgets
                
        min_ch_wdg = widgets.IntSlider(
            value=0, 
            min=0, max=self.nch-1, step=1,
            description='min channel',
            continuous_update=False
        )

        max_ch_wdg = widgets.IntSlider(
            value=self.nch, 
            min=0, max=self.nch-1, step=1,
            description='max channel',
            continuous_update=False
        )
        
        min_time_wdg = widgets.IntSlider(
            value=self.min_t, 
            min=self.tAx[0], max=self.tAx[-1], step=1,
            description='min time [s]',
            continuous_update=False
        )
        
        max_time_wdg = widgets.IntSlider(
            value=self.max_t, 
            min=self.tAx[0], max=self.tAx[-1], step=1,
            description='max time [s]',
            continuous_update=False
        )
        
        # Tap-test widgets
        
        ch_shift_wdg = widgets.FloatSlider(
            value=0.0,
            min=-np.amax(self.chAxTap),
            max=np.amax(self.chAxTap),
            step=0.1,
            description='Tap-test shift:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
        
        show_tap_ln = widgets.Checkbox(
            value=True,
            description='Show tap-test line',
            disabled=False,
            indent=False
        )
        
        tap_dir_wg = widgets.ToggleButton(
            value=False,
            description='Reverse direction',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Reverse tap-test direction'
        )
        
        # Tap-test mapping
        min_ch_tap_wdg = widgets.IntSlider(
            value=0, 
            min=0, max=self.nch-1, step=1,
            description='min map ch',
            continuous_update=False
        )

        max_ch_tap_wdg = widgets.IntSlider(
            value=self.nch, 
            min=0, max=self.nch-1, step=1,
            description='max map ch',
            continuous_update=False
        )
        
        map_ch_wdg = widgets.Button(description='Map channels!')
        show_mp_ch_nb = widgets.Checkbox(value=False,description='Show mapped ch #')
        
        
        bad_ch_check = widgets.HBox([bad_channels_wdg,show_bad_ch])
        ch_sliders = widgets.HBox([min_ch_wdg,max_ch_wdg])
        time_sliders = widgets.HBox([min_time_wdg,max_time_wdg])
        tap_test_controls = widgets.HBox([ch_shift_wdg,show_tap_ln, tap_dir_wg])
        tap_map_controls = widgets.HBox([min_ch_tap_wdg,max_ch_tap_wdg, map_ch_wdg, show_mp_ch_nb])
        controls = widgets.VBox([bad_ch_check, ch_sliders, time_sliders, tap_test_controls, tap_map_controls])
        
        # Functions to enable user inputs
        show_bad_ch.observe(self.update_show_bad, 'value')
        bad_channels_wdg.observe(self.update_plot_chan, 'value')
        min_ch_wdg.observe(self.update_min_ch, 'value')
        max_ch_wdg.observe(self.update_max_ch, 'value')
        min_time_wdg.observe(self.update_min_time, 'value')
        max_time_wdg.observe(self.update_max_time, 'value')
        ch_shift_wdg.observe(self.update_shift, 'value')
        show_tap_ln.observe(self.update_show_ln, "value")
        tap_dir_wg.observe(self.update_reverse_tap, "value")
        min_ch_tap_wdg.observe(self.update_min_ch_map, 'value')
        max_ch_tap_wdg.observe(self.update_max_ch_map, 'value')
        map_ch_wdg.on_click(self.map_channel)
        show_mp_ch_nb.observe(self.update_show_map_ch_num, 'value')
        
        # add to children
        self.children = [output, controls]
        
    # Methods to update the plot using widgets
    def update_mapped_ch_plot(self, map_ch, map_lon, map_lat):
        if len(self.ann) > 0:
            for ann in self.ann:
                ann.remove()
            self.ann = []
        self.scat_ch.set_data(map_lon, map_lat)
        if self.show_map_num:
            for label, x, y in zip(map_ch, map_lon, map_lat):
                self.ann.append(self.ax0.annotate(label, 
                                                  xy=(x, y), 
                                                  xytext=(-5, 5), 
                                                  textcoords='offset points', 
                                                  ha='right', va='bottom'))
        if len(map_ch) > 0:
            minLat = np.amin(map_lat)
            maxLat = np.amax(map_lat)
            minLon = np.amin(map_lon)
            maxLon = np.amax(map_lon)
            self.ax0.set_ylim([minLat,maxLat])
            self.ax0.set_xlim([minLon,maxLon])
        
    def get_channel_mask(self):
        mask = (self.chAx >= self.min_ch) * (self.chAx <= self.max_ch)
        min_ch = self.min_ch
        max_ch = self.max_ch
        if len(self.bad_channels) > 0 and not self.show_bad: 
            mask_tmp = np.ones(self.nch, dtype=bool)
            mask_tmp[self.bad_channels] = False
            # Minimum and maximum local channel indices
            channel_map = np.where(mask_tmp)[0] # indices of good channels
            chAx_tmp = np.arange(channel_map.shape[0])
            mask_new_ch = (chAx_tmp >= self.min_ch) * (chAx_tmp <= self.max_ch)
            min_ch = chAx_tmp[mask_new_ch][0]
            max_ch = chAx_tmp[mask_new_ch][-1]
            mask = channel_map[mask_new_ch]
        return mask, min_ch, max_ch
    
    def update_energy_plot(self):
        channel_mask, min_ch, max_ch = self.get_channel_mask()
        if np.any(channel_mask):
            if not self.show_bad and len(self.bad_channels) > 0:
                energy = np.sum(abs(self.data[channel_mask, self.it_min:self.it_max]), axis=1)
                self.line_en.set_data(range(min_ch,max_ch+1), energy/(energy.max()+1e-18))
            else:
                energy = np.sum(abs(self.data[self.min_ch:self.max_ch, self.it_min:self.it_max]), axis=1)
                self.line_en.set_data(range(self.min_ch,self.max_ch), energy/(energy.max()+1e-18))
        else:
            print("No channel to show with these settings!")
            return
        self.ax1.relim()
        self.ax1.autoscale_view()
    
    def update_ch_plot(self):
        channel_mask, min_ch, max_ch = self.get_channel_mask()
        if np.any(channel_mask):
            if not self.show_bad and len(self.bad_channels) > 0:
                vis_ch = range(min_ch,max_ch+1)
                self.line_ch.set_data(vis_ch, [0.0]*len(vis_ch))
                self.line_bad_ch.set_data([],[])
                if len(self.mapped_channels) > 0:
                    mask_tmp = np.ones(self.nch, dtype=bool)
                    mask_tmp[self.bad_channels] = False
                    channel_map = np.where(mask_tmp)[0] # indices of good channels
                    channel_map = channel_map[min_ch:max_ch+1]
                    # Getting info of visible mapped channels
                    mapped_ch_plt = []
                    map_lon = []
                    map_lat = []
                    map_ch = []
                    for idx, ch in enumerate(channel_map):
                        if ch in self.mapped_channels:
                            mapped_ch_plt.append(vis_ch[idx])
                            map_ch.append(ch)
                            map_lon.append(self.mapped_lon[np.where(self.mapped_channels == ch )])
                            map_lat.append(self.mapped_lat[np.where(self.mapped_channels == ch )])
                    self.line_mapped.set_data(mapped_ch_plt, [0.0]*len(mapped_ch_plt))
                    self.update_mapped_ch_plot(map_ch, map_lon, map_lat)
            else:
                self.line_ch.set_data(range(self.min_ch,self.max_ch), [0.0]*len(range(self.min_ch,self.max_ch)))
                mask_bad = (self.bad_channels >= self.min_ch) * (self.bad_channels <= self.max_ch)
                if len(self.bad_channels) > 0 and np.any(mask_bad):
                    self.line_bad_ch.set_data(self.bad_channels[mask_bad], [0.0]*len(self.bad_channels[mask_bad]))
                else:
                    self.line_bad_ch.set_data([],[])
                if len(self.mapped_channels) > 0:
                    mask_mapped = (self.mapped_channels >= self.min_ch) * (self.mapped_channels <= self.max_ch)
                    self.line_mapped.set_data(self.mapped_channels[mask_mapped], 
                                              [0.0]*len(self.mapped_channels[mask_mapped]))
                    self.update_mapped_ch_plot(self.mapped_channels[mask_mapped], 
                                               self.mapped_lon[mask_mapped], 
                                               self.mapped_lat[mask_mapped])
        else:
            return
        self.ax3.relim()
        self.ax3.autoscale_view()
        
    def update_data_plot(self):
        channel_mask, min_ch, max_ch = self.get_channel_mask()
        if np.any(channel_mask):
            if not self.show_bad and len(self.bad_channels) > 0:
                self.im.set_data(self.data[channel_mask,self.it_min:self.it_max].T)
                self.im.set_extent([min_ch, max_ch, self.tAx[self.it_max], self.tAx[self.it_min]])
            else:
                self.im.set_data(self.data[self.min_ch:self.max_ch,self.it_min:self.it_max].T)
                self.im.set_extent([self.chAx[self.min_ch], self.chAx[self.max_ch], self.tAx[self.it_max], self.tAx[self.it_min]])
        else:
            return
        
    def update_tap_line(self):
        channel_mask, min_ch, max_ch = self.get_channel_mask()
        if np.any(channel_mask):
            if self.show_tap:
                if self.reverse_tap:
                    chTap = np.flip(self.chAxTap)+self.shift
                else:
                    chTap = self.chAxTap+self.shift
                self.line_tap.set_data(chTap,self.gps_time_int)
                self.ax2.set_ylim([self.max_t,self.min_t])
                self.ax2.set_xlim([min_ch,max_ch])
            else:
                self.line_tap.set_data([], [])
        else:
            return
        return
    
    def update_plots(self):
        self.update_data_plot()
        self.update_energy_plot()
        self.update_ch_plot()
        self.update_tap_line()
    
    def update_min_ch(self, value):
        self.min_ch = value["new"]
        if self.max_ch <= self.min_ch:
            print("Max channel smaller than min channel! Change sliders values")
            return
        self.update_plots()

    def update_max_ch(self, value):
        self.max_ch = value["new"]
        if self.max_ch <= self.min_ch:
            print("Max channel smaller than min channel! Change sliders values")
            return
        self.update_plots()
        
    def update_min_time(self, value):
        self.min_t = value["new"]
        if self.max_t <= self.min_t:
            print("Max time smaller than min time! Change sliders values")
            return
        self.it_min = int(self.min_t/self.dt+0.5)
        self.it_max = int(self.max_t/self.dt+0.5)
        self.update_plots()

    def update_max_time(self, value):
        self.max_t = value["new"]
        if self.max_t <= self.min_t:
            print("Max time smaller than min time! Change sliders values")
            return
        self.it_min = int(self.min_t/self.dt+0.5)
        self.it_max = int(self.max_t/self.dt+0.5)
        self.update_plots()
        
    def update_show_bad(self, value):
        self.show_bad = value["new"]
        self.update_plots()

    # Functions to update plot
    def update_plot_chan(self, value):
        values = value["new"].split(",")
        bad_channels = np.array([], dtype=int)
        for val in values:
            try:
                # List of bad channels
                if ":" in val:
                    strt_ch = int(val.split(":")[0])
                    lst_ch = int(val.split(":")[1])
                    bad_channels = np.append(bad_channels, np.arange(strt_ch,lst_ch))
                else:
                    bad_channels = np.append(bad_channels, int(val))
            except ValueError:
                self.bad_channels = np.array([], dtype=int)
                print("Incorrect input for bad channel!")
                self.update_plots()
                return
        # Remove repeated values
        self.bad_channels = np.unique(bad_channels)
        self.update_plots()
        
    def update_shift(self, value):
        self.shift = value["new"]
        self.update_tap_line()
        
    def update_show_ln(self, value):
        self.show_tap = value["new"]
        self.update_tap_line()
    
    def update_reverse_tap(self,value):
        self.reverse_tap = value['new']
        self.update_tap_line()
        
    def update_min_ch_map(self, value):
        self.min_ch_map = value["new"]
        if self.max_ch_map <= self.min_ch_map:
            print("Max mapping channel smaller than min mapping channel! Change sliders values")
            return
        
    def update_max_ch_map(self, value):
        self.max_ch_map = value["new"]
        if self.max_ch_map <= self.min_ch_map:
            print("Max mapping channel smaller than min mapping channel! Change sliders values")
            return
        
    def update_show_map_ch_num(self, value):
        self.show_map_num = value["new"]
        self.update_ch_plot()
    
    def map_channel(self, click):
        # Mapping channels between min and max ch_map
        ch_map = np.arange(self.min_ch_map,self.max_ch_map+1)
        if self.reverse_tap:
            chTap = np.flip(self.chAxTap)+self.shift
        else:
            chTap = self.chAxTap+self.shift
        chTapMask = (chTap >= self.min_ch_map) * (chTap <= self.max_ch_map)
        chTapMaskTime = (self.gps_time_int >= self.min_t) * (self.gps_time_int <= self.max_t)
        chTapMask *= chTapMaskTime
        chTapMap = chTap[chTapMask]
        if len(chTapMap) == 0:
            print("Cannot map! Change mapping and visualization parameters!")
            return
        # Interpolate to half channel index to obtain accurate GPS time
        f_chTime = interp1d(chTap[chTapMask], self.gps_time_int[chTapMask], kind='linear', bounds_error=False)
        chTapMapIdx = chTapMap.astype(int)
        chTapMapPos = (chTapMapIdx+0.5).astype(float)
        gpsTimeMap = f_chTime(chTapMapPos)
        # Converting local index to global one
        self.mapped_lat = np.append(self.mapped_lat, self.f_lat(gpsTimeMap))
        self.mapped_lon = np.append(self.mapped_lon, self.f_lon(gpsTimeMap))
        if len(self.bad_channels) > 0: 
            mask_tmp = np.ones(self.nch, dtype=bool)
            mask_tmp[self.bad_channels] = False
            channel_map = np.where(mask_tmp)[0] # indices of good channels
            mask_idx = (chTapMapIdx >= 0) * (chTapMapIdx < channel_map.shape[0])
            chTapMapIdx = chTapMapIdx[mask_idx]
        else:
            channel_map = self.chAx
        self.mapped_channels = np.append(self.mapped_channels, channel_map[chTapMapIdx])
        # Removing repeated values
        self.mapped_channels, unq_idx = np.unique(self.mapped_channels, return_index=True)
        self.mapped_lat = self.mapped_lat[unq_idx]
        self.mapped_lon = self.mapped_lon[unq_idx]
        self.update_ch_plot()


# Useful utility functions
def find_close_ch(gps_lat, gps_lon, ch_lat, ch_lon):
    """Function to find the closed channel based on GPS position points"""
    npoints = len(gps_lat)
    close_ch = np.zeros(npoints, dtype=int)
    for idx in range(npoints):
        dist = np.sqrt((ch_lat-gps_lat[idx])**2+(ch_lon-gps_lon[idx])**2)
        idx_min = np.argmin(dist)
        close_ch[idx] = idx_min
    return close_ch