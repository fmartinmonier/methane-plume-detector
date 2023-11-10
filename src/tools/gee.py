import os
import sys
import ee
import datetime
import numpy as np

def define_point(lat, lon):
    """ Defines an ee.Geometry object describing a point. """
    point = ee.Geometry.Point((lon, lat))
    return point

def define_area_of_interest(point, buffer_distance):
    """
    Defines an ee.Geometry object describing a polygon for the area of interest.
    The area is defined by first creating an ee.Geometry point object, and then
    expanding out from the point with length buffer_distance in meters.
    """
    area_bounds = point.buffer(buffer_distance).bounds().getInfo()["coordinates"][0]
    area_of_interest = ee.Geometry.Polygon(
        [[area_bounds[0], area_bounds[1], area_bounds[2], area_bounds[3]]],
        geodesic=False,
        proj=None,
    )
    return area_of_interest

def get_satellite_dictionary(satellite_name):
    """
    Get dictionary of ee information for a particular satellite.
    ****review dictionary fields TODO
    """

    full_index = {
        "S2": {
            "eepath": "COPERNICUS/S2",
            "spacecraft": "SPACECRAFT_NAME",
            "swir1": "B11",
            "swir2": "B12",
            "RGB": ["B4", "B3", "B2"],
            "nir": "B8",
            "scaling": 10000,
        },
        "L7": {
            "eepath": "LANDSAT/LE07/C02/T1_L2",
            "spacecraft": "SPACECRAFT_ID",
            "swir1": "SR_B5",
            "swir2": "SR_B7",
            "RGB": ["SR_B3", "SR_B2", "SR_B1"],
            "nir": "SR_B4",
            "scaling": 1 / (2.75e-05),
            "offset": -0.2,
        },
        "L8": {
            "eepath": "LANDSAT/LC08/C02/T1_TOA",
            "spacecraft": "SPACECRAFT_ID",
            "swir1": "B6",
            "swir2": "B7",
            "RGB": ["B4", "B3", "B2"],
            "nir": "B5",
        },
        "S3": {
            "eepath": "COPERNICUS/S3/OLCI",
            "spacecraft": "spacecraft",
            "swir1": None,
            "swir2": None,
            "RGB": ["Oa01_radiance", None, None],
        },
    }

    satellite_dictionary = full_index[satellite_name]

    return satellite_dictionary

def get_elevation(lon, lat):
    point = ee.Geometry.Point(lon, lat)

    # Load the Digital Elevation Map dataset
    dem = ee.Image('USGS/SRTMGL1_003')

    # Extract the elevation at the point of interest
    elevation = dem.sample(point, 30).first().get('elevation').getInfo()
    
    return elevation / 1000

def query_ee_band_data(
    lat,
    lon,
    lat_shift,
    lon_shift,
    facility_type,
    buffer_distance,
    start_date,
    n_days,
    satellite_name,
    cloud_threshold,
    get_no2_bands=True,
    get_ch4_bands=True,
    get_aux_bands=True,
    verbose=False,
    cache=None,
):
    """
    Function to query ee for satellite observations by point location and approximate date.

    Arguments
        lat              [float]  :  query latitude
        lon              [float]  :  query longitude
        lat_shift        [int]    :  off centering of queried facility latitude in pixels
        lon_shift        [int]    :  off centering of queried facility longitude in pixels
        buffer_distance  [float]  :  buffer distance (radius) in meters to generate ee.Polygon object for area of interest
        start_date       [str]    :  start date for ee query, in isoformat (yyyy-mm-dd)
        n_days           [int]    :  number of days to query beyond the start date until a good observation is found
        satellite_name   [str]    :  satellite name to get dictionary from get_satellite_dictionary()
        cloud_threshold  [float]  :  cloud-cover percentage above which scenes are rejected due to cloudiness
        get_no2_bands    [bool]   :  download spectral band data for NO2 retrievals?
        get_ch4_bands    [bool]   :  download spectral band data for CH4 retrievals?
        get_aux_bands    [bool]   :  download spectral band data for miscellaneous purposes, e.g. NIR band for water masking
        verbose          [bool]   :  whether or not to print messages
        cache            [str]    :  path to directory containing previous downloads, or None
    """

    # Identify dates for which data has already been downloaded
    if cache:
        cache_files = os.listdir(cache)
        known_dates = [name[:-4] for name in cache_files]
    else:
        known_dates = []

    # Get image collection based on the input point and start date + n_days
    satellite_dictionary = get_satellite_dictionary(satellite_name)
    date = ee.Date(start_date)
    point = define_point(lat, lon)
    area_of_interest = define_area_of_interest(point, buffer_distance)
    image_collection = (
        ee.ImageCollection(satellite_dictionary["eepath"])
        .filterBounds(point)
        .filterDate(date, date.advance(n_days, "day"))
        .sort("system:time_start")
    )

    # Convert the collection object to a list
    collection_list = image_collection.toList(image_collection.size().getInfo())
    collection_list_length = collection_list.size().getInfo()

    # For each image in the collection
    for i in range(collection_list_length):

        # Get the image and observation date
        image = ee.Image(collection_list.get(i))
        scene_date = datetime.datetime.utcfromtimestamp(
            image.getInfo()["properties"]["system:time_start"] / 1000
        )
        scene_date_str = scene_date.strftime("%Y-%m-%d")

        # If we've already downloaded data for this date, skip it
        if scene_date_str in known_dates:
            if verbose:
                print(
                    f"Already downloaded data for {scene_date_str}; moving to next day\n"
                )
            continue

        # Otherwise, get the data for the area of interest
        band_data = image.sampleRectangle(region=area_of_interest, defaultValue=0)
        # rgb_data1 = np.array(
        #    band_data.get(satellite_index[satellite]["RGB"][0]).getInfo()
        # )
        # try:
        #    rgb_data1=np.array(band_data.get(sat_info[sat]["RGB"][0]).getInfo())
        # except:
        #    if verbose: print (scene_date , "image clip boundaries outside. trying next image")
        #    continue

        # Determine cloud cover percentage
        if satellite_name == "S2":
            cloud_percentage = band_data.getInfo()['properties']['CLOUDY_PIXEL_PERCENTAGE']
        elif satellite_name == "L8" or satellite_name == "L7":
            cloud_percentage = band_data.getInfo()["properties"]["CLOUD_COVER"]
        if verbose:
            print(scene_date)
            print(f"Cloud percentage : {np.round(cloud_percentage,2)} ")

        successful_obs = False
        # If the image is too cloudy, skip to the next date
        if cloud_percentage > cloud_threshold:
            if verbose:
                print("Too many clouds\n")
            continue
        else:
            successful_obs = True
            # Otherwise, exit the loop
            if verbose:
                print("Sucessful observation\n")
            break
    
    if successful_obs:
        # Get image info
        image_info = image.getInfo()["properties"]

        # Get the band data for the image
        dat = {}

        # Sentinel-2
        if satellite_name == "S2":
            if get_no2_bands:
                dat["B1"] = (
                    np.array(band_data.get("B1").getInfo())
                    / satellite_dictionary["scaling"]
                )
                dat["B2"] = (
                    np.array(band_data.get("B2").getInfo())
                    / satellite_dictionary["scaling"]
                )
                dat["B3"] = (
                    np.array(band_data.get("B3").getInfo())
                    / satellite_dictionary["scaling"]
                )
                dat["B4"] = (
                    np.array(band_data.get("B4").getInfo())
                    / satellite_dictionary["scaling"]
                )
            if get_ch4_bands:
                dat["B10"] = (
                    np.array(band_data.get("B10").getInfo())
                    / satellite_dictionary["scaling"]
                )
                dat["B11"] = (
                    np.array(band_data.get("B11").getInfo())
                    / satellite_dictionary["scaling"]
                )
                dat["B12"] = (
                    np.array(band_data.get("B12").getInfo())
                    / satellite_dictionary["scaling"]
                )
            if get_aux_bands:
                dat["B8"] = (
                    np.array(band_data.get("B8").getInfo())
                    / satellite_dictionary["scaling"]
                )

        # Landsat-7
        elif satellite_name == "L7":
            if get_no2_bands:
                dat["B1"] = (
                    np.array(band_data.get("SR_B1").getInfo())
                    / satellite_dictionary["scaling"]
                    + satellite_dictionary["offset"]
                )
                dat["B2"] = (
                    np.array(band_data.get("SR_B2").getInfo())
                    / satellite_dictionary["scaling"]
                    + satellite_dictionary["offset"]
                )
                dat["B3"] = (
                    np.array(band_data.get("SR_B3").getInfo())
                    / satellite_dictionary["scaling"]
                    + satellite_dictionary["offset"]
                )
            if get_ch4_bands:
                dat["B5"] = (
                    np.array(band_data.get("SR_B5").getInfo())
                    / satellite_dictionary["scaling"]
                    + satellite_dictionary["offset"]
                )
                dat["B7"] = (
                    np.array(band_data.get("SR_B7").getInfo())
                    / satellite_dictionary["scaling"]
                    + satellite_dictionary["offset"]
                )
            if get_aux_bands:
                dat["B4"] = (
                    np.array(band_data.get("SR_B4").getInfo())
                    / satellite_dictionary["scaling"]
                    + satellite_dictionary["offset"]
                )

        # Landsat-8
        elif satellite_name == "L8":
            if get_no2_bands:
                dat["B1"] = np.array(band_data.get("B1").getInfo())
                dat["B2"] = np.array(band_data.get("B2").getInfo())
                dat["B3"] = np.array(band_data.get("B3").getInfo())
                dat["B4"] = np.array(band_data.get("B4").getInfo())
            if get_ch4_bands:
                dat["B6"] = np.array(band_data.get("B6").getInfo())
                dat["B7"] = np.array(band_data.get("B7").getInfo())
            if get_aux_bands:
                dat["B5"] = np.array(band_data.get("B5").getInfo())

        # Record scene date and various image properties
        dat["scene_date"] = scene_date
        dat["properties"] = image_info
        dat["facility_type"] = facility_type
        dat["elevation"] = get_elevation(lon, lat)
        dat["lat_shift"] = lat_shift
        dat["lon_shift"] = lon_shift

        return dat, scene_date, successful_obs
    else:
        return None, scene_date, successful_obs