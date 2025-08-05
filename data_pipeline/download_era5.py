import cdsapi
from pathlib import Path 

c = cdsapi.Client()
variable = "wind" # "2m_temperature", "total_cloud_cover", "total_precipitation", "mean_sea_level_pressure", "10m_u_component_of_wind", "10m_v_component_of_wind"

#"2m_temperature",  "sea_surface_temperature", "total_precipitation"
months = ['01', '02', '03','04', '05','06','07','08', '09','10', '11', '12']
for year in range(1940, 2024):
    if not Path('../era5/dev/era5.{}.{}.zip'.format(variable, year)).is_file():
        if year >= 2015:
            x = 'operational'
        else:
            x = 'consolidated'
        c.retrieve(
            "reanalysis-era5-single-levels-monthly-means",
            #"reanalysis-era5-pressure-levels-monthly-means",
            {
                #"pressure_level": ["500"],
                'format': 'zip',
                'product_type': ["monthly_averaged_reanalysis"],
                'vertical_resolution': 'single_level',
                "variable": [ "10m_u_component_of_wind", "10m_v_component_of_wind" ],
                'year': [ year ],
                'month': months,
                "time": ["00:00"],
                "data_format": "netcdf",
                "download_format": "zip"
            },
            '../era5/dev/era5.{}.{}.zip'.format(variable, year))
        
