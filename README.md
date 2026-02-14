## The waiting time paradox and earthquake quienscence on the San Andreas Fault

Author: Kelian Dascher-Cousineau<sup>1</sup>, Micheal Oskin<sup>2</sup>

<sup>1</sup> Geoscience Department, Utah State University, Logan, UT, USA

<sup>2</sup> Department of Earth Sciences, University of California, Davis, CA, USA

This repository includes the code used to explore the waiting time paradox and earthquake quienscence on the San Andreas Fault.

The code includes:
1. A collection of paleoseismic trench data from the San Andreas, San Jacinto, Dead Sea, Altyn Tagh, and Alpine Faults.
2. Jupyter notebooks that explore the severity of the waiting time paradox
3. Source code to load, manipulate and visualize paleoseismic trench data.

### Installation 

The code has been tested on MacOS Sequoia 15.6 Make sure you have the latest of version of conda installed.

Install dependencies and create a new conda environment.

``` bash
make setup
conda activate paleoseismic
```

Note that this will install the latest version of OxCal. This is required to run the oxcal models.

### Usage

Refer to the following notebooks for examples of how to use the code:

 [Paleoseismic trench data](notebooks/paleoseismicity_example.ipynb)

### Available datasets:

The following datasets are available:

| Fault Name               | Dataset Names                                                                 |
|--------------------------|-------------------------------------------------------------------------------|
| Dead Sea Fault           | DSF_Qatar.txt, DSF_Beteiha.txt                                                |
| Cascadia Subduction Zone | CSZ_central.txt, CSZ_south.txt                                                |
| San Andreas Fault        | SAF_Pallett_Creek.txt, SAF_Wrightwood.txt, SAF_Frasier_Mountian.txt, SAF_Noyo_Canyon.txt |
| San Jacinto Fault        | SJF_Hog_Lake.txt, SJF_Mystic_Lake.txt                                         |
| Altin Tahg               | ATF_Lake_Paringa.txt, ATF_CopperMine10.txt, ATF_Hokuri_Creek.txt              |

### Metadata File for Trench Data

Below is the metadata for the trench data, also available in the [metadata.yaml](data/metadata.yaml) file:

| File                   | Start Time | End Time | Gaps            | Historic Events                      | Notes                                                                                                     | References |
|-------------------------|------------|----------|-----------------|--------------------------------------|-----------------------------------------------------------------------------------------------------------|------------|
| **DSF_Qatar.txt**       | -3000      | 2025     | None            | 363, 749, 1068, 1212, 1458           | Historic events in 363, 749, 1068, 1212, and 1458 A.D.                                                    | Klinger et al. (2015) |
| **CSZ_central.txt**     | None       | None     | None            | 1700                                 | Events present in central part of subduction zone. Added historic event in 1700                           | Goldfinger et al. (2012) |
| **SAF_Pallett_Creek.txt** | 700      | 2025     | None            | 1812, 1857                           | Historic events in 1812 and 1857 A.D.                                                                     | Scharer et al. (2011) |
| **DSF_Beteiha.txt**     | -6500      | 2025     | None            | -142, 130, 303, 348, 363, 500, 660, 1202, 1758 | Event chronology from Ref 49, with historic events from Ref 51: 142 B.C., 130, 303, 348, 363, 500, 660, 1202, 1758 A.D. | Wechsler et al. (2014); Lefevre et al. (2018) |
| **SAF_Wrightwood.txt**  | -3000      | 2025     | [-1500, 500]    | 1812, 1857                           | Historic events in 1812 and 1857 A.D. Old and young records combined, with gap interval removed.          | Scharer et al. (2007); Biasi et al. (2002) |
| **SAF_Frasier_Mountian.txt** | 800   | 2025     | None            | 1857                                 | Historic event in 1857 A.D. Oldest interval removed due to possible missed events.                        | Scharer et al. (2014, 2017) |
| **SJF_Hog_Lake.txt**    | None       | 2025     | None            | 1800, 1918                           | Historic events in 1800 and 1918 A.D.                                                                     | Rockwell et al. (2015) |
| **AF_Lake_Paringa.txt** | None       | None     | None            | 1717                                 | Most recent event 1717 A.D.                                                                               | Howarth et al. (2018) |
| **ATF_CopperMine10.txt**| None       | None     | None            | –                                    | Event chronology from Ref 52, with additional event from Ref 57.                                          | Yuan et al. (2018); Pinzon et al. (2024) |
| **AF_Hokuri_Creek.txt** | None       | None     | None            | 1717                                 | Most recent event 1717 A.D.                                                                               | Berryman et al. (2012) |
| **CSZ_south.txt**       | None       | None     | None            | 1700                                 | Includes additional events confined to southern part of subduction zone.                                  | Goldfinger et al. (2012) |
| **SAF_Noyo_Canyon.txt** | None       | None     | None            | 1906                                 | Historic event in 1906 A.D. 75% gaps age model.                                                           | Goldfinger et al. (2007) |
| **SJF_Mystic_Lake.txt** | None       | None     | [-900, -400]    | 1812                                 | Historic event in 1812 A.D. Interval removed at sedimentary hiatus ca. 400–900 B.C.                       | Onderdonk et al. (2018) |

