# Obtain files
# import pandas as pd
# import datapackage
#
# url = 'https://data.nav.no/api/nav-opendata/2f6ce2a2c65dd50709d389486da3947a/datapackage.json'
# package = datapackage.Package(url)
#
# save_path = "..\\data\\input\\"
# counter = 2020
# resources = package.resources
# for resource in package.resources:
#     if resource.tabular:
#       df = pd.read_csv(resource.descriptor['path'], sep=";")
#       df.to_csv(path_or_buf=save_path+str(counter), sep=",")
#       counter-=1


# Rename files
# import os
# from pathlib import Path
# path = "..\\data\\input\\"
# paths = sorted(Path(path).iterdir(), key=os.path.getmtime)
#
# counter = 2020
#
# print(paths)
#
# for f in paths:
#     os.rename(f, "..\\data\\input\\"+str(counter)+"data")
#     counter-=1



