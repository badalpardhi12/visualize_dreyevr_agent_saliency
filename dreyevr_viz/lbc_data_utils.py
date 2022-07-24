from PIL import Image
import ast

def get_data(route_data_path, sampled_dataidx):
    route_rgb_path = route_data_path / 'rgb'
    route_rgb_pathL = route_data_path / 'rgb_left'
    route_rgb_pathR = route_data_path / 'rgb_right'
    route_measurements_path = route_data_path / 'measurements'

    rgb_img_path = route_rgb_path / '{:04d}.png'.format(sampled_dataidx)
    rgb_imgL_path = route_rgb_pathL / '{:04d}.png'.format(sampled_dataidx)
    rgb_imgR_path = route_rgb_pathR / '{:04d}.png'.format(sampled_dataidx)
    rgb_img = Image.open(str(rgb_img_path))
    rgb_imgL = Image.open(str(rgb_imgL_path))
    rgb_imgR = Image.open(str(rgb_imgR_path))

    measure_path = route_measurements_path / '{:04d}.json'.format(sampled_dataidx)
    with open(measure_path) as read_file:
        json_str = read_file.readline()
        input_data = ast.literal_eval(json_str)
    input_data['rgb'] = rgb_img
    input_data['rgb_left'] = rgb_imgL
    input_data['rgb_right'] = rgb_imgR
    return input_data