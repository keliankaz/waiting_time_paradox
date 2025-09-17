import requests
import zipfile
import geopandas as gpd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io import shapereader
import cartopy.feature as cfeature
from pathlib import Path
import os

class Qfaults:
    def __init__(self, data_dir=None):
        
        self.url = "https://earthquake.usgs.gov/static/lfs/nshm/qfaults/Qfaults_GIS.zip"
        self.data_dir = Path(data_dir) if data_dir else Path("./Qfaults_GIS")
        
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.download()
        
        self.crs = "EPSG:4326"
        self.data = gpd.read_file(self.data_dir / 'SHP/Qfaults_US_Database.shp').to_crs(self.crs)


    def download(self):
        print(f"Downloading and saving to {self.data_dir}")
        response = requests.get(self.url, stream=True)
        response.raise_for_status()  # Ensures an exception is raised for HTTP errors

        with open(self.data_dir / "Qfaults_GIS.zip", "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        with zipfile.ZipFile(self.data_dir / "Qfaults_GIS.zip", "r") as zip_ref:
            zip_ref.extractall(self.data_dir)
        
    def plot_basemap(self, ax=None, bounds=[-125, -114, 32, 42]):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

        ax.set_extent(bounds, crs=ccrs.PlateCarree())
        # Read Natural Earth state polygons

        shpfilename = shapereader.natural_earth(resolution='10m',
                                                category='cultural',
                                                name='admin_1_states_provinces')

        reader = shapereader.Reader(shpfilename)

        # Filter for California
        for record in reader.records():
            if record.attributes['name'] == 'California':
                geom = record.geometry
                ax.add_geometries([geom], ccrs.PlateCarree(),
                                edgecolor='black', facecolor='none', linewidth=0.6)

        # Add coastline for context
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        return ax
    
    def plot(self, ax=None, bounds=[-125, -114, 32, 42]):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()})

        self.plot_basemap(ax, bounds)
        self.data.plot(ax=ax, color='slategray', alpha=0.8, linewidth=0.2)
        
        return ax
    
    
#%%
if __name__ == "__main__":
    qfaults = Qfaults()
    qfaults.plot()