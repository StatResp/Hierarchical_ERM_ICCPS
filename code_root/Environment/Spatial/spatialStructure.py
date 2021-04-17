from pyproj import Proj
import numpy as np
from math import floor
import json
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib import colors
import matplotlib.pyplot as plt
import fiona
from shapely import geometry
import sys
import pickle
from datetime import datetime, timedelta

numRows = 30
numColumns = 30
gridCounter = 0
gridSize = 1609.34
xLow = 504963.24880000204
yLow = 181515.03680000082
xHigh = 553573.9182999973
yHigh = 230170.18359999958

# numClusters = 5

# declare projection scheme to change lat long to state plane coordinates
p1 = Proj('+proj=lcc +lat_1=36.41666666666666 +lat_2=35.25 +lat_0=34.33333333333334 +lon_0=-86 +x_0=600000 +y_0=0 '
          '+ellps=GRS80 +datum=NAD83 +no_defs')

# GLOBAL VARIABLES
gridPlacement = np.zeros((numRows, numColumns))
neighborPerGrid = {}


def get_valid_cells():
    global gridSize
    valid_cells = []
    # read shapefile
    # shp = fiona.open("../../../../data/geo_export_c5423d03-3502-4fe2-892e-f2b41eb9a262.shp")
    shp = fiona.open("../../../data/geo_export_c5423d03-3502-4fe2-892e-f2b41eb9a262.shp")

    numGridsX = int(floor((xHigh - xLow) / float(gridSize)))
    numGridsY = int(floor((yHigh - yLow) / float(gridSize)))
    grids = {}
    numToCoord = {}
    for counterY in range(numGridsY):
        for counterX in range(numGridsX):
            lowerLeftCoords = (xLow + counterX * gridSize, yLow + counterY * gridSize)
            if counterX == (numGridsX - 1):  # reached the end on x axis
                xCoord = xHigh
            else:
                xCoord = xLow + counterX * gridSize + gridSize
            if counterY == (numGridsY - 1):  # reached the end on y axis
                yCoord = yHigh
            else:
                yCoord = yLow + counterY * gridSize + gridSize

            upperRightCoords = (xCoord, yCoord)
            center_x = (lowerLeftCoords[0] + upperRightCoords[0]) / 2
            center_y = (lowerLeftCoords[1] + upperRightCoords[1]) / 2
            # invert mapping. map refers to x,y but numpy array refers to row,column --> y,x
            grids[(counterX, counterY)] = [[center_x, center_y]]
            numToCoord[counterY * len(grids) + counterX] = [[center_x, center_y]]

            lng, lat = p1(center_x, center_y, inverse=True)
            point = geometry.Point(lng, lat)
            inCounty = False
            for counterShape in range(len(shp)):
                shapefile_record = shp[counterShape]
                shape = geometry.asShape(shapefile_record['geometry'])
                if shape.contains(point):
                    inCounty = True
                    break

            if inCounty:
                valid_cells.append(counterY * numGridsX + counterX)

    pickle.dump(valid_cells, open('../../../data/valid_cells_out.pkl', 'wb'))

    return valid_cells


def get_mapped_cells():
    global gridSize
    valid_cells = []
    # read shapefile
    # shp = fiona.open("../../../../data/geo_export_c5423d03-3502-4fe2-892e-f2b41eb9a262.shp")
    shp = fiona.open("../../../data/geo_export_c5423d03-3502-4fe2-892e-f2b41eb9a262.shp")

    numGridsX = int(floor((xHigh - xLow) / float(gridSize)))
    numGridsY = int(floor((yHigh - yLow) / float(gridSize)))
    grids = {}
    numToCoord = {}
    for counterY in range(numGridsY):
        for counterX in range(numGridsX):
            lowerLeftCoords = (xLow + counterX * gridSize, yLow + counterY * gridSize)
            if counterX == (numGridsX - 1):  # reached the end on x axis
                xCoord = xHigh
            else:
                xCoord = xLow + counterX * gridSize + gridSize
            if counterY == (numGridsY - 1):  # reached the end on y axis
                yCoord = yHigh
            else:
                yCoord = yLow + counterY * gridSize + gridSize

            upperRightCoords = (xCoord, yCoord)
            center_x = (lowerLeftCoords[0] + upperRightCoords[0]) / 2
            center_y = (lowerLeftCoords[1] + upperRightCoords[1]) / 2
            # invert mapping. map refers to x,y but numpy array refers to row,column --> y,x
            grids[(counterX, counterY)] = [[center_x, center_y]]
            numToCoord[counterY * len(grids) + counterX] = [[center_x, center_y]]

            lng, lat = p1(center_x, center_y, inverse=True)
            point = geometry.Point(lng, lat)
            inCounty = False
            for counterShape in range(len(shp)):
                shapefile_record = shp[counterShape]
                shape = geometry.asShape(shapefile_record['geometry'])
                if shape.contains(point):
                    inCounty = True
                    break

            if inCounty:
                valid_cells.append(counterY * numGridsX + counterX)

    pickle.dump(valid_cells, open('../../../data/valid_cells_out.pkl', 'wb'))

    return valid_cells




def parseJSON():
    with open('../../../data/DavidsonMVA_ETrims_Jan2018_April2019.json') as f:
        data = json.load(f)

    df = pd.DataFrame.from_dict(data, orient='columns')

    return df


def cluster_regions(df_incidents, num_clusters):
    X = np.asarray([x for x in list(df_incidents['cell'])])
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(X)
    labels = {}
    vcounter = 0
    _valid_cells = get_valid_cells()
    for y in range(30):
        for x in range(30):
            cell = y * numRows + x
            if cell not in _valid_cells:
                # cluster labels vary from 0--numClusters-1. For invalid cells, set label to numClusters
                key = str((y * numRows) + x)   #str(x) + ',' + str(y)
                labels[key] = num_clusters
            else:
                vcounter += 1
                # assign the label from the existing point
                prediction = kmeans.predict(np.asarray([[x, y]]))[0]
                if prediction == 5:
                    print("weird prediction indeed")
                key = str((y * numRows) + x)  # str(x) + ',' + str(y)
                labels[key] = kmeans.predict(np.asarray([[x, y]]))[0]
    return labels, kmeans


def visualize_regions(data):
    # create discrete colormap
    cmap = colors.ListedColormap(['red', 'blue', 'green', 'yellow', 'black', 'purple', 'white'])
    bounds = [0, 1, 2, 3, 4, 5, 6]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots()
    ax.imshow(data, cmap=cmap, norm=norm, origin='lower')

    # draw gridlines
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    plt.show()
    print('Plotted Regions')

'''
For each cluster, calculates the mean distance of the cells from the cluster center.
Returns a dictionary with cluster_id(key) --> mean_distance(value)
'''
def get_mean_distance(labels, cluster_obj, cell_id_to_xy_coords_dict):
    # create dict for distances for each region
    dist_region = {x: 0 for x in range(cluster_obj.n_clusters)}
    dist_count = {x: 0 for x in range(cluster_obj.n_clusters)}
    # for each cell
    for cell, label in labels.items():
        # get distance to center and update dictionary
        if label == cluster_obj.n_clusters: # default cluster for invalid cells
            continue
        center = cluster_obj.cluster_centers_[label]
        # cell = [int(y) for y in cell.split(',')]
        cell_coords = cell_id_to_xy_coords_dict[cell]
        dist = np.sqrt((cell_coords[0] - center[0]) ** 2 + (cell_coords[1] - center[1]) ** 2)
        dist_region[label] += dist
        dist_count[label] += 1

    # normalize distances
    for region, dist in dist_region.items():
        dist_region[region] /= dist_count[region]

    return dist_region


'''
Visualizes incidents on a grid heat map
'''
def visualize_incidents(data):
    X = [x for x in list(df_incidents['cell'])]
    x, y = zip(*X)
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=30)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.clf()
    plt.imshow(heatmap.T, extent=extent, origin='lower')
    plt.show()
    print('Plotted Incidents')


'''
Creates a new dataframe with cell number and count of incidents
'''
def get_cell_id_and_rates(df, _valid_cells):

    # the rate is calculate in minutes - why minutes? I think seconds is better
    # total_time = (df['datetime'].max() - df['datetime'].min()).total_seconds()/60

    # - why minutes? The rest of the code assumes seconds for everything

    rate_df = df.copy(deep=True)

    # datetime - int representing millisecond epoch time
    total_time = (rate_df['datetime'].max() - rate_df['datetime'].min() )/ 1000
    # rate_df['cell_string_id'] = rate_df.apply(lambda row: row.cell[0] * numRows + row.cell[1], axis=1)

    rate_df = df.groupby(by='cell_string_id').size().reset_index(name='total_incidents')
    # rate_df['cell_num'] = rate_df.apply(lambda row: row['cell'][0]*numRows + row['cell'][1])
    rate_df['incidents_per_second'] = rate_df['total_incidents'] / total_time
    # print(rate_df.head())

    rate_dictionary = dict()
    raw_cell_ids = list(rate_df['cell_string_id'])
    valid_cell_ids = list()
    for cell_id in raw_cell_ids:
        if cell_id in _valid_cells:
            rate_dictionary[cell_id] = rate_df[rate_df['cell_string_id'] == cell_id].iloc[0]['incidents_per_second']
            valid_cell_ids.append(cell_id)

    return valid_cell_ids, rate_dictionary

    # Geoff edits:
    # print(df.columns)
    #
    # print(df['cell'].head())
    #
    # count_df = df.copy(deep=True)
    #
    # count_df['cell_string'] = [','.join(map(str, l)) for l in df['cell']]
    #
    # a = count_df.groupby(by='cell_string').sum()
    #
    # count_df = df.groupby['cell'].size().reset_index(name='count')
    # count_df['cell_num'] = count_df.apply(lambda row: row.cell[0] * numRows + row.cell[1])
    # print(count_df.head())
    # return count_df


if __name__ == "__main__":

    # generate_valid_grid_file
    get_valid_cells()
    sys.exit()

    # df_incidents = parseJSON()
    # df_incidents['cell_string_id'] = df_incidents.apply(lambda row: row.cell[1] * numRows + row.cell[0], axis=1)
    # cell_ids = list(df_incidents['cell_string_id'])
    # valid_cells = get_valid_cells()

    # cell_ids, cell_rates = get_cell_id_and_rates(df_incidents, valid_cells)
    # sys.exit()
    # visualize_incidents(df_incidents)
    # labels, cluster_obj = cluster_regions(df_incidents, 6)
    # pickle.dump((labels, cluster_obj), open('../../../data/cluster_output_6_r.pk', 'wb'))

    # result = pickle.load(open('../../../data/cluster_output.pk', 'rb'))
    # labels = result[0]
    # cluster_obj = result[1]
    # # mean_distances = get_mean_distance(labels, cluster_obj)
    # # cell_ids, cell_rates = get_cell_id_and_rates(df_incidents, valid_cells)
    # grid_labels = np.zeros([30, 30])
    # for y in range(30):
    #     for x in range(30):
    #         # key = str(x) + ',' + str(y)
    #         key = str((y * numRows) + x)
    #         grid_labels[y, x] = labels[key]
    # visualize_regions(grid_labels)
    #
    # print('done')
