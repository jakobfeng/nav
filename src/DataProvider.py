from builtins import print

import pandas as pd
import os
from pathlib import Path
import datapackage

# Obtain strucured files
# url = 'https://data.nav.no/api/nav-opendata/2f6ce2a2c65dd50709d389486da3947a/datapackage.json'
# package = datapackage.Package(url)
# save_path = "..\\data\\input\\struct\\"
# resources = package.resources
# for resource in package.resources:
#     if resource.tabular:
#         str_path = str(resource.descriptor['path'])
#         if "ledige_stillinger_meldt_til_nav" in str_path:
#             year = str_path[-11:-7]
#             print(year)
#             df = pd.read_csv(resource.descriptor['path'], sep=";")
#             df.to_csv(path_or_buf=save_path + year + "_data.csv", sep=";")



# Rename descriptive files
# path = "..\\data\\input\\descript\\"
# paths = sorted(Path(path).iterdir())
# for f in paths:
#     p_list = str(f).split(sep="\\")
#     year = p_list[-1][11:15]
#     print(year)
#     os.rename(f, path + year + "_descript.csv")

# Convert to csv
# path = "..\\data\\input\\struct\\"
# paths = sorted(Path(path).iterdir())
# for p in paths:
#     print(p)
#     read_file = pd.read_csv(str(p), header=0, sep=",")
#     read_file.to_csv(str(p) + ".csv", sep=",")

# Delete old files
# path = "..\\data\\input\\struct\\"
# files_in_directory = os.listdir(path)
# filtered_files = [file for file in files_in_directory if not file.endswith(".csv")]
# for file in filtered_files:
#     path_to_file = os.path.join(path, file)
#     print(path_to_file)
#     os.remove(path_to_file)
