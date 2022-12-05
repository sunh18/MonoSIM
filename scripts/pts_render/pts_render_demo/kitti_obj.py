import numpy as np
import cv2

# To project a point from Velodyne coordinates into the left color image,
# you can use this formula: x = P2 * R0_rect * Tr_velo_to_cam * y


def get_calib(calib_dir):
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(calib_dir, 'r') as f:
        for i in range(7):
            line = f.readline()
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array(list(map(float, value.split(' ')))).reshape(3, 4)
                except ValueError:
                    try:
                        data[key] = np.array(list(map(float, value.split(' ')))).reshape(3, 3)
                    except ValueError:
                        pass  # casting error: data[key] already eq. value, so pass
    delta_z = data['P2'][2, 3]
    delta_x = (data['P2'][0, 3] - data['P2'][0, 2] * delta_z) / data['P2'][0, 0]
    delta_y = (data['P2'][1, 3] - data['P2'][1, 2] * delta_z) / data['P2'][1, 1]
    data['K2'] = data['P2'][0:3, 0:3]
    data['d2'] = np.array([delta_x, delta_y, delta_z]).reshape(3, 1)
    data['RT2'] = np.dot(data['R0_rect'], data['Tr_velo_to_cam'])
    data['RT2'][:, 3:] += data['d2']
    return data['K2'], data['RT2']


if __name__ == '__main__':
    K, RT = get_calib('C:/Users/Song/Desktop/000003/000003.txt')
    im = cv2.imread('C:/Users/Song/Desktop/000003/000003.png')
    h, w, c = im.shape
    pc = np.fromfile('C:/Users/Song/Desktop/000003/000003.bin',
                     dtype=np.float32, count=-1).reshape([-1, 4]).transpose()
    pc[3, :] = 1.
    cam_pc = np.dot(RT, pc)  # 刚体变换
    cam_pc = cam_pc[:, cam_pc[2, :] > 1]
    px = np.dot(K, cam_pc)
    px = np.round(px / px[2, :])
    ind = (px[0, :] < w) & (px[0, :] >= 0) & \
          (px[1, :] < h) & (px[1, :] >= 0)
    px = px[:, ind].astype(np.int)
    im[px[1, :], px[0, :]] = (255, 255, 255)
    cv2.imshow('test', im)
    cv2.waitKey(0)

