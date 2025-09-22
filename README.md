# china-freight-AFV
# Overview
This repository contains all codes of the paper - ??

# Requirements and Installation
The whole analysis-related codes should run with a **Python** environment, regardless of operating systems theoretically. We successfully execute all the codes in both Windows (Win10, Win11) machines and a macOS (Sequoia 15.2) machine. More detailed info is as below:

# Prerequisites
It is highly recommended to install and use the following versions of python/packages to run the codes:

## 01FreightTripODGeneration.py
   - ``python``==3.9
   - ``numpy``==1.24.3
   - ``pandas``==2.0.3
   - ``psutil``==5.9.5
     
## 02DetailedFreightTripsGeneration.py
   - ``python``==3.9
   - ``qgis``==3.34.1
   - ``numpy``==1.26.4
   - ``pandas``==2.2.3
   - ``tqdm``==4.67.1
   - ``geopandas``==1.0.1
   - ``openpyxl``==3.1.5
     
## codes in the dir./SimulationOptimization/...
   - ``python``==3.9
   - ``numpy``>=1.20.0
   - ``pandas``>=1.3.0
   - ``matplotlib``>=3.4.0
   - ``seaborn``>=0.11.0
   - ``geopandas``>=0.10.0
   - ``PyYAML``>=6.0
   - ``psutil``>=5.9.0
   - ``openpyxl``>=3.0.9
   - ``pyarrow``>=6.0.0
   - ``shapely``>=1.8.0
   - ``fiona``>=1.8.20
   - ``rtree``>=1.0.0
     
# Installation
It is highly recommended to download AnaConda to create/manage Python environments. You can create a new Python environment and install required aforementioned packages via both the GUI or Command Line. Typically, the installation should be prompt (around 10-20 min from a "_clean_" machine to "_ready-to-use_" machine, but highly dependent on the Internet speed).

- via **Anaconda GUI**
  1. Open the Anaconda
  2. Find and click "_Environments_" at the left sidebar
  3. Click "_Create_" to create a new Python environment
  4. Select the created Python environment in the list, and then search and install all packages one by one.
     
- via **Command Line** (using **_Terminal_** for macOS machine and **_Anaconda Prompt_** for Windows machine, respectively)
  1. Create your new Python environment )
     ```
     conda create --name <input_your_environment_name> python=3.10.6
     ```
  2. Activate the new environment 
     ```
     conda activate <input_your_environment_name>
     ```
  3. Install all packages one by one 
     ```
     conda install <package_name>=<specific_version>
     ```

# Usage
1. Git clone/download the repository to your local disk.
2. Unzip the full datasets (which can be provided upon request, see ???)
   > The structure of the provided full datasets should look like as below:
   > 
   > ```
   > - ??
   > - ??
   > - ??
   > - ??
   > - ??
   > ```
3. Unzip each compressed dataset (``.??`` file) and drag folders/files into corresponding dir of this repo. For example, extract all files from the ``?A?.7z`` to the dir ``./data/input/?A?/``.
4. Run (all the codes that need to be run is stored in the dir ``./codes/``)
   1. **01FreightTripODGeneration.py**: run the script and some intermediate data will be produced (can be found in the dir ``./data/interim/...``) then
   2. **02DetailedFreightTripsGeneration.py**: run the script and some intermediate data will be produced (can be found in the dir ``./data/interim/...``) then
   3. **PreDistenceCorrection.py** in the dir ``./codes/SimulationOptimization/0preprocess``: run the script and some intermediate data will be produced (can be found in the dir ``./data/interim/...``) then
   4. **main.py** in the dir ``./codes/SimulationOptimization``: run the script
5. Outputs will be stored in the dir ``./data/output``, respectively.

# Contact
- Leave questions in [Issues on GitHub](https://github.com/wanganya/china-freight-AFV/issues)
- Get in touch with the Corresponding Author: [Dr. Chengxiang Zhuge](mailto:chengxiang.zhuge@polyu.edu.hk) or visit our research group website: [The TIP](https://thetipteam.editorx.io/website) for more information

# License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
