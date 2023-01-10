# -*- encoding: utf-8 -*-
# @File     : main.py
# @Time     : 2023/01/10 09:32
# @Version  : 1.0
# @Author   : xyang
# @Contact  : yang_ax@foxmail.com
# @Software : Vscode



import glob
from osgeo import gdal
# 安装教程参考：https://blog.csdn.net/liubing8609/article/details/124000951
import numpy as np
from PIL import Image
import xml.etree.ElementTree as ET
import pyproj
import os
from shapely.geometry import Polygon
import math
import warnings
warnings.filterwarnings("ignore")

# 寻找外接矩形函数
def bounding_box(coordinates):
    """ 
    Calculates the bounding box of a list of (longitude, latitude) coordinates.
    Returns a tuple (min_lon, min_lat, max_lon, max_lat).
    """
    min_lon = min(coord[0] for coord in coordinates)
    max_lon = max(coord[0] for coord in coordinates)
    min_lat = min(coord[1] for coord in coordinates)
    max_lat = max(coord[1] for coord in coordinates)
    return (min_lon, max_lon, min_lat, max_lat)

# 定义经纬度转化坐标函数：
def world_to_pixel(transform, longitude, latitude):
    """
    Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
    the pixel location of a geospatial coordinate
    """
    x = (longitude - transform[0]) / transform[1]
    y = (latitude - transform[3]) / transform[5]
    return (x, y)

# 遍历得到XML文件中的所有含有经纬度的节点
def parse_coordinates(input_file, transform):
    """
    Parses an XML file and extracts all the nodes containing "Coordinates" values.
    Returns a list of (node name, node text) tuples.
    """
    # Parse the XML file
    tree = ET.parse(input_file)

    # Extract the nodes from the tree
    nodes = []
    for node in tree.iter():
        if 'Coordinates' in node.tag:
            # nodes.append((node.tag, node.text.strip()))
            once = node.text.strip()
            once = once.split() # 拆分字符串
            assert len(once) % 2 == 0 # 确保经纬度为偶数个
            pairs = []
            for i in range(0, len(once), 2):
                pairs.append((float(once[i]), float(once[i+1])))
            # 转化为像素点坐标：
            pairs = [world_to_pixel(transform, x[0], x[1]) for x in pairs]
            nodes.append(pairs)
    return nodes

def area(p):
    # Create a polygon from the coordinates
    polygon = Polygon(p)

    # Calculate the area of the polygon
    return polygon.area

def crop_satellite_image_png(dataset, transform,output_file, min_x, max_x, min_y, max_y, width, height, num_bands):
    # 条件判定，去掉超过图片边界的区域
    # Clamp the coordinates to the bounds of the image
    min_x = int(min(max(min_x, 0), width))
    max_x = int(min(max(max_x, 0), width))
    min_y = int(min(max(min_y, 0), height))
    max_y = int(min(max(max_y, 0), height))
    assert min_x<max_x
    assert min_y<max_y
    # 建立空的np数组，用于存放裁切的图片
    image = np.empty((max_y - min_y, max_x - min_x, dataset.RasterCount), dtype=np.uint8)
    # 按通道保存：
    for i in range(dataset.RasterCount):
        band = dataset.GetRasterBand(i + 1)
        image[:,:,i] = band.ReadAsArray(min_x, min_y, max_x - min_x, max_y - min_y)
    # 保存：
    # Save the image
    image = Image.fromarray(image)
    image.save(output_file)

def scale_rectangle(arito, area_ac,area_bounding, start_x, end_x , start_y,end_y): 
    # 由于维度大小相反，所以：
    # 计算缩放因子：
    scale_factor = (area_ac / area_bounding) / arito
    scale_factor = math.pow(scale_factor,0.5)
    # 计算外接矩形的中心点坐标
    center_x = (start_x + end_x) / 2
    center_y = (end_y + start_y) / 2
    assert start_x<end_x
    assert start_y<end_y
    # 缩放外接矩形
    new_start_x = center_x - scale_factor * (center_x - start_x)
    new_end_x = center_x + scale_factor * (center_x - start_x)
    new_start_y = center_y - scale_factor * (center_y - start_y)
    new_end_y = center_y + scale_factor * (center_y - start_y)

    return new_start_x, new_end_x , new_start_y, new_end_y


def crop_raster(input_path, output_path, x_min, y_min, x_max, y_max,width,height):
    """
    Crops a raster located at `input_path` and saves the result to `output_path`.
    
    The area to be cropped is specified by the `x_min`, `y_min`, `x_max`, and `y_max`
    parameters, which are the coordinates of the top-left and bottom-right corners of the
    cropped area in pixels.
    
    This function assumes that the input raster is in GeoTIFF format.
    """
    # Open the input raster
    dataset = gdal.Open(input_path)
    
    # Get the number of bands in the raster
    num_bands = dataset.RasterCount
    # dataset.RasterCount
    
    # Get the raster's spatial reference
    srs = dataset.GetProjection()
    
    # Get the raster's geotransform
    geotransform = dataset.GetGeoTransform()
    
    # 条件判定，去掉超过图片边界的区域
    # Clamp the coordinates to the bounds of the image
    x_min = min(max(x_min, 0), width)
    x_max = min(max(x_max, 0), width)
    y_min = min(max(y_min, 0), height)
    y_max = min(max(y_max, 0), height)

    # Calculate the size of the cropped area
    cols = int(x_max - x_min)
    rows = int(y_max - y_min)
    if cols == 0 or rows == 0:
        return 0
    # Calculate the top-left corner of the cropped area in geospatial coordinates
    x_min_geo = geotransform[0] + x_min * geotransform[1] + y_min * geotransform[2]
    y_max_geo = geotransform[3] + x_min * geotransform[4] + y_min * geotransform[5]
    
    # Create the output raster
    driver = gdal.GetDriverByName("GTiff")
    output = driver.Create(output_path, cols, rows, num_bands, gdal.GDT_Byte)
    
    # Set the output raster's spatial reference and geotransform
    output.SetProjection(srs)
    output.SetGeoTransform((x_min_geo, geotransform[1], geotransform[2], y_max_geo, geotransform[4], geotransform[5]))
    
    # Loop through each band
    for i in range(num_bands):
        # Get the input band
        band = dataset.GetRasterBand(i + 1)
        
        # Read the band data
        data = band.ReadAsArray(x_min, y_min, cols, rows)
        
        # Get the output band
        out_band = output.GetRasterBand(i + 1)
        
        # Write the band data to the output raster
        out_band.WriteArray(data)
    
    # Close the dataset
    dataset = None
    
    # Close the output raster
    output = None


def main(file_name, xml_name, arito):
    """

    """
    # 首先读入tif，
    dataset = gdal.Open(file_name)
    if dataset is None:
        raise Exception("Could not open the input file")
        
    transform= dataset.GetGeoTransform()  # 图片的相关经纬度信息。
    num_bands = dataset.RasterCount
    # print("Number of bands:", num_bands)
    # Get the image dimensions and transform
    width = dataset.RasterXSize
    height = dataset.RasterYSize

    # 开始解析XML文件：
    pairs = parse_coordinates(xml_name, transform) # 得到像素点坐标
    # 解析得到多对物体，对于每一个物体：

    #输出的路径为："./文件夹/图片名/类型名/（递增序号）.png"

    save_path = "./out/"+file_name.split("\\")[1]+"/"+file_name.split("\\")[-2].split(".")[0] + "/" + xml_name.split("\\")[-1].split(".")[0].split("_")[-1]+"/"
    if os.path.exists(save_path) == False:
        os.makedirs(save_path)
    for num,pair in enumerate(pairs):
        # 寻找外接矩形：
        start_x, end_x , start_y, end_y = bounding_box(pair)
        # 计算实际面积
        area_ac = area(pair)
        # 外接矩形面积：
        area_bounding = (end_x-start_x) * (end_y-start_y)
        # 面积比值：
        if area_ac<1:
            continue
        old_arito = area_ac/area_bounding
        print(area_bounding)
        if old_arito > arito and arito != False:
            # 自定义比值， 修正外接矩形坐标：
            # 更新坐标：
            # print("更新了")
            start_x, end_x , start_y,end_y = scale_rectangle(arito, area_ac, area_bounding, start_x, end_x , start_y,end_y)
            # 验证修改后，比值小于等于设定的
            area_bounding = (end_x-start_x) * (end_y-start_y)
            # assert area_ac/area_bounding <= arito
            # print(area_ac/area_bounding)

        ###################
        # 开始裁切：
        if start_x>end_x or start_y>end_y:
            # 去掉错误情况
            continue
        #crop_satellite_image_png(dataset, transform, save_path+str(num+1)+".png", start_x, end_x , start_y, end_y, width, height, num_bands)
        print("正在处理：",save_path+str(num+1)+".tif")
        crop_raster(file_name, save_path+str(num+1)+".tif", start_x, start_y, end_x, end_y,width,height)

if __name__ == "__main__":
    # 首先遍历所有 所有tif文件路径
    file_names = glob.glob("./tif/*/*/*.tif")

    # 裁切比值，设定在0.1-1 之间。 
    # 注意：如果最小外接矩形的比值已经<arito则不生效。
    # 如果最小外接矩形的比值>arito, 则会等比例扩大外接矩形。
    # 如果默认按最小外接矩形裁切，请： 
    # arito = False
    arito = 0.3
    
    # 对于每一个tif文件
    for file_name in file_names:
        # 查看他的所有XML文件 
        xml_names = glob.glob(file_name.replace(file_name.split('\\')[-1], "*.xml")) # 一张图片对应的所有XML文件
        for xml_name in xml_names:
            # 对每一个XML文件开始解析：
            main(file_name, xml_name, arito)  # 传入tif路径，xml路径
    print("Done!")
    
    
    