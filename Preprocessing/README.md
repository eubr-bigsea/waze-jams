# Script to create the training file


First of all, it's necessary create a file with the grids of the desired city. In order to do this, use the program `1_createGrids.py`. This is a fast step, it will not take more than a few seconds.

After that, you are able to perform the `2_preprocessing-*.py` to filter the records of a given city and create a table of traffic jams that will be used in the main application.

## 1 - CreateGrids.py


The parameters are:

 * --shapefile, -s: The Filepath of the shapefile (shp format, but the dbf file must be in the same folder)
 * --ngrid, -g:	The organization of the grid. Format: nlines ncolumns (default, 50x50)


Example:

```bash
$ python 1_createGrids.py -s $PWD/curitiba.shp -g 50 50
```

The output of this step is a csv file, where each line corresponds to a grid and contains its coordinates, the identification number, and the information whether its valid (inside the city).


West Longitude | South Latitude  | East Longitude | North Latitude |  IDgrid  | Valid
---------------|-----------------|----------------|----------------|----------|------
-49.389338636|-25.64538622|-49.3852563633|-25.6394132162|1|0
-49.3852563633|-25.64538622|-49.3811740905|-25.6394132162|2|0
-49.3811740905|-25.64538622|-49.3770918178|-25.6394132162|3|0
-49.3770918178|-25.64538622|-49.373009545|-25.6394132162|4|0
-49.373009545|-25.64538622|-49.3689272723|-25.6394132162|5|0
-49.3689272723|-25.64538622|-49.3648449996|-25.6394132162|6|1
-49.3648449996|-25.64538622|-49.3607627268,|-25.6394132162|7|1
-49.3607627268|-25.64538622|-49.3566804541|-25.6394132162|8|1
-49.3566804541|-25.64538622|-49.3525981813|-25.6394132162|9|1
...|...|...|...|...|...


The image below is an example of representation of the output that corresponds to the grids for the city of Curitiba (only valid grids).

<p align="middle">
<img  src="Curitiba_Grids_Distribuition.jpg" width="400" alt='Curitiba Grids Distribuition Map' >
</p>


## 2 - preprocessing.py

Reads input data and filters the records of a specific city  and based in the selected parameters, assign 1 to the grid (created in previous step) in which the record is contained (otherwise, 0). Note: For performance gains, are not considered grids outside the perimeter of the cities.

This step is performed in PyCOMPSs, and in order to execute it, it's necessary that the input file to be read is already divided into numFrag parts (number of cores available) if you do not use the HDFS Integration API. We encourage you to use the HDFS Integration API, because the user does not need to manually divide the file, just specify the number of divisions you want.


The parameters are:

 * input (-i): The input file path;
 * grids (-g):  The input of the grids list file;
 * city (-c): The target city;
 * window (-w): The window time in seconds to take in count (default, 3600)
 * numFrag (-f): Number of workers (cores)


Example of how submit this application in COMPSs, considering the Curitiba's city:

```bash
STARTTIME=$(date +%s)

INPUT=$1
runcompss --lang=python $PWD/2_preprocessing-fs.py \
          -i $INPUT \
          -g $PWD/Curitiba_Grids.csv \
          -c Curitiba \
          -f 4 \
          -w 3600

ENDTIME=$(date +%s)
echo "It took $(($ENDTIME - $STARTTIME)) seconds"
```

This application creates the following two files:

1. output\_counts.csv:	The number of events recorded at each time point (only for statistical analysis);
2. output\_training.csv: The output of the application itself. This file will be used as input for the model.

**Note:** This may take some time depending on the size of the input file.

### Requirements files

```bash
$ sudo pip install -r ./requirements.txt

  Shapely == 1.6.1
  numpy == 1.11.0
  pandas == 0.20.3
  pyshp == 1.2.11
```
