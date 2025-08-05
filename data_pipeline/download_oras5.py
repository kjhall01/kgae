import cdsapi
from pathlib import Path 

c = cdsapi.Client()

months = ['01', '02', '03','04', '05','06','07','08', '09','10', '11', '12']
for year in range(1958, 2015):
    if not Path('../oras5/dev/oras5.v.{}.zip'.format(year)).is_file():
        if year >= 2015:
            x = 'operational'
        else:
            x = 'consolidated'
        c.retrieve(
            'reanalysis-oras5',
            {
                'format': 'zip',
                'product_type': [x],
                #'vertical_resolution': 'single_level',
                #'variable': "depth_of_20_c_isotherm",
                "vertical_resolution": "all_levels",
                "variable": ["meridional_velocity"],

                'year': [ year ],
                'month': months,
            },
            '../oras5/dev/oras5.v.{}.zip'.format(year))
        


        dataset = "reanalysis-oras5"
