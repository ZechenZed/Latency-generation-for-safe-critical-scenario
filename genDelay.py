import argparse
import numpy as np
import logging
import h5py
from mayavi import mlab
import random
import time
import os
import cv2


def getTransform(x, y, z, pitch, yaw, roll, degrees=True):
    '''Given location x,y,z and pitch, yaw, roll, obtain the matrix that convert from local to global CS using the
    left-handed system from UE4 '''

    if degrees:
        pitch, yaw, roll = [np.radians(x) for x in [pitch, yaw, roll]]

    cy, sy = np.cos(yaw), np.sin(yaw)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)

    mat = np.array([cp * cy, cy * sp * sr - sy * cr, -cy * sp * cr - sy * sr, x, \
                    cp * sy, sy * sp * sr + cy * cr, -sy * sp * cr + cy * sr, y, \
                    sp, -cp * sr, cp * cr, z, \
                    0, 0, 0, 1], dtype=float).reshape(4, 4)

    return mat


def transformPoints(transformMatrix, pts, inverse=False):
    '''Given a transformation matrix [4,4] convert pts [N,3] or [N,4] (last coordinate is intensity)'''

    # Check if intensity is available
    if pts.shape[1] == 4:
        # split intensity from 3D coordinates, add homogeneus coordinate
        intensity = pts[:, -1, np.newaxis].copy()
        pts[:, -1] = 1
    else:
        # add homogeneus coordinate
        intensity = None
        pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)

    # perform transformation
    mat = np.array(transformMatrix)
    if inverse:
        mat = np.linalg.inv(mat)
    ptst = pts @ mat.T
    ptst = ptst[:, :3]

    # merge intensity back
    if intensity is not None:
        ptst = np.concatenate([ptst, intensity], axis=1)

    return ptst


def updateBoundingBox(x, y, z, yaw, pitch, w, l, h, vis_bb):
    # Create 8 corner points
    cpts = 0.5 * np.array(
        [[-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, 1], [1, 1, 1], [1, -1, 1], [-1, -1, 1]])
    cpts *= np.array([[w, l, h]])
    cpts = transformPoints(getTransform(x, y, z, pitch, yaw, 0), cpts)

    # list of 16 points to create whole BB
    pts = cpts[[0, 3, 7, 3, 2, 6, 2, 1, 5, 1, 0, 4, 7, 6, 5, 4], :]

    # update vis
    vis_bb.mlab_source.reset(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2])

    return pts[:, 0][0], pts[:, 0][4], pts[:, 1][0], pts[:, 1][1]


def main(args):
    # Read and load the data
    # dir_list = os.listdir('Datasets/GTdata/')
    # num_files = len(dir_list)
    img = cv2.imread('Datasets/000000.png')
    f = h5py.File('Datasets/groundtruth/' + args.filename, 'r')
    pcls = np.array(f['point_cloud'])
    lidar_pose = np.array(f['lidar_pose'])
    vbbs = np.array(f['vehicle_boundingbox'])
    pbbs = np.array(f['pedestrian_boundingbox'])
    num_cars = lidar_pose.shape[1]
    # TODO: Create delay of pcls
    cars = list(range(num_cars))
    delayed_frame = 0

    if args.count_20 != 0:  # 20 percent
        delayed_frame = 2
        deCar_num = int(args.count_20 * len(cars) / 100)
        rand_20 = random.sample(cars, deCar_num)
        for i in rand_20:
            cars.remove(i)
        for i in rand_20:
            pcls[(delayed_frame - 1):99, i, :, :] = pcls[0:(100 - delayed_frame), i, :, :]

    if args.count_40 != 0:  # 40
        delayed_frame = 4
        deCar_num = int(args.count_40 * len(cars) / 100)
        rand_40 = random.sample(cars, deCar_num)
        for i in rand_40:
            cars.remove(i)
        for i in rand_40:
            pcls[(delayed_frame - 1):99, i, :, :] = pcls[0:(100 - delayed_frame), i, :, :]

    if args.count_60 != 0:    # 60
        delayed_frame = 6
        deCar_num = int(args.count_60 * len(cars) / 100)
        rand_60 = random.sample(cars, deCar_num)
        for i in rand_60:
            cars.remove(i)
        for i in rand_60:
            pcls[(delayed_frame - 1):99, i, :, :] = pcls[0:(100 - delayed_frame), i, :, :]

    if args.count_80 != 0:    # 80
        delayed_frame = 8
        deCar_num = int(args.count_80 * len(cars) / 100)
        rand_80 = random.sample(cars, deCar_num)
        for i in rand_80:
            cars.remove(i)
        for i in rand_80:
            pcls[(delayed_frame - 1):99, i, :, :] = pcls[0:(100 - delayed_frame), i, :, :]

    if args.count_100 != 0:    # 100
        delayed_frame = 10
        deCar_num = int(args.count_100 * len(cars) / 100)
        rand_100 = random.sample(cars, deCar_num)
        for i in rand_100:
            cars.remove(i)
        for i in rand_100:
            pcls[(delayed_frame - 1):99, i, :, :] = pcls[0:(100 - delayed_frame), i, :, :]


    if delayed_frame != 0:    # Tailor the delayed frame
        pcls = pcls[delayed_frame - 1:99, :, :, :]
        lidar_pose = lidar_pose[delayed_frame - 1:99, :, :]
        vbbs = vbbs[delayed_frame - 1:99, :, :]
        pbbs = pbbs[delayed_frame - 1:99, :, :]

    # get the basic constant of the scene
    nframes = pcls.shape[0]
    nvehicles = pcls.shape[1]
    npedestrians = pbbs.shape[1]

    # Create the folder for the dataset
#    if args.datatype == 'testing':
#        os.makedirs('Datasets/kitti/testing/velodyne', exist_ok=True)
#        os.makedirs('Datasets/kitti/testing/calib', exist_ok=True)
#        os.makedirs('Datasets/kitti/testing/image_2', exist_ok=True)
#    elif args.datatype == 'training':
#        os.makedirs('Datasets/kitti/training/label_2', exist_ok=True)
##        os.makedirs('Datasets/kitti/training/velodyne', exist_ok=True)
 #       os.makedirs('Datasets/kitti/training/calib', exist_ok=True)
 #       os.makedirs('Datasets/kitti/training/image_2', exist_ok=True)

    # # Create Mayavi Visualisation
    fig = mlab.figure(size=(600, 400), bgcolor=(0.05, 0.05, 0.05))
    zeros = np.zeros(pcls.shape[1] * pcls.shape[2])
    vis = mlab.points3d(zeros, zeros, zeros, zeros, mode='point', figure=fig)
    zeros = np.zeros(16)
    vis_vbbs = [mlab.plot3d(zeros, zeros, zeros, zeros, color=(0, 1, 0), tube_radius=None, line_width=1, figure=fig) for
                i in range(nvehicles)]
    vis_pbbs = [mlab.plot3d(zeros, zeros, zeros, zeros, color=(0, 1, 1), tube_radius=None, line_width=1, figure=fig) for
                i in range(npedestrians)]

    @mlab.animate(delay=10)
    def anim():
        # mlab.options.offscreen = True
        # fig.scene.disable_render = True
        # fig.scene.anti_aliasing_frames = 0
        calib_txt = open('Datasets/calib.txt', 'r')
        x_max, x_min, y_max, y_min, z_max, z_min = np.zeros(6)

        for frame in range(nframes):
            train_new_frame = ''
            test_new_frame = ''
            frame_tt = 0

            if args.datatype == 'training':
                train = open('Datasets/kitti/ImageSets/train.txt', 'r+')
                train_list = train.read()
                frame_tt = int(len(train_list) / 7)
                train_zero_ct = len(str(frame_tt))
                train_new_frame = (6 - train_zero_ct) * str(0) + str(frame_tt)
                train.write(train_new_frame + '\n')
                train.close()
            elif args.datatype == 'testing':
                test = open('Datasets/kitti/ImageSets/test.txt', 'r+')
                test_list = test.read()
                frame_tt = int(len(test_list) / 7)
                test_zero_ct = len(str(frame_tt))
                test_new_frame = (6 - test_zero_ct) * str(0) + str(frame_tt)
                test.write(test_new_frame + '\n')
                test.close()

            frame_tt += 1
            print(f'Frame: {frame_tt}')
            for i in range(npedestrians):
                lx, rx, uy, dy = updateBoundingBox(*pbbs[frame, i], vis_pbbs[i])

            # Update the vehicle bounding boxes and pcls
            fusedPCL = []
            vbbs_txt = []

            for i in range(nvehicles):
                # Get PCL for the given vehicle in the global Coordinate System
                pcl = pcls[frame, i]
                transform = getTransform(*lidar_pose[frame, i].tolist())
                pcl_global = transformPoints(transform, pcl)
                # print(pcl_global.shape)
                fusedPCL.append(pcl_global)

                # Update the vehicle BB visualisation
                lx, rx, uy, dy = updateBoundingBox(*vbbs[frame, i].tolist(), vis_vbbs[i])
                vbbs_txt.append('Car')  # Type
                vbbs_txt.append(float(0.00))  # Truncated
                vbbs_txt.append(float(0.00))  # Occluded
                vbbs_txt.append(float(0.00))  # Alpha
                vbbs_txt.append(float(10))  # Bbox left  x
                vbbs_txt.append(float(10))  # upper y
                vbbs_txt.append(float(100))  # right x
                vbbs_txt.append(float(100))  # down  y
                vbbs_txt.append(float(vbbs[frame, i][7]))  # Dimensions Height
                vbbs_txt.append(float(vbbs[frame, i][5]))  # Width
                vbbs_txt.append(float(vbbs[frame, i][6]))  # Length
                vbbs_txt.append(float(vbbs[frame, i][0]))  # 3d Location x
                vbbs_txt.append(float(vbbs[frame, i][1]))  # y
                vbbs_txt.append(float(vbbs[frame, i][2]))  # z
                vbbs_txt.append(float(1.0))  # Rotation
                if frame == 1:
                    x_max = float(vbbs[frame, i][0])
                    x_min = float(vbbs[frame, i][0])
                    y_max = float(vbbs[frame, i][1])
                    y_min = float(vbbs[frame, i][1])
                    z_max = float(vbbs[frame, i][2])
                    z_min = float(vbbs[frame, i][2])
                else:
                    x_max = max(float(vbbs[frame, i][0]), x_max)
                    x_min = min(x_min, float(vbbs[frame, i][0]))
                    y_max = max(float(vbbs[frame, i][1]), y_max)
                    y_min = min(y_min, float(vbbs[frame, i][1]))
                    z_max = max(float(vbbs[frame, i][2]), z_max)
                    z_min = min(z_min, float(vbbs[frame, i][2]))
                # Save the vehicle bouding box


            if args.datatype == 'training':
                vbbs_txt = np.array(vbbs_txt).reshape(-1, 15)
                np.savetxt('Datasets/kitti/training/label_2/' + train_new_frame + '.txt', vbbs_txt, delimiter=' ',
                           fmt='%s')

                calib_save = open('Datasets/kitti/training/calib/' + train_new_frame + '.txt', 'w+')
                calib_save = calib_txt
                calib_save.close()

                # Update PCL visualisation with Mayavi
                fusedPCL = np.concatenate(fusedPCL, axis=0)
                fusedPCL.astype('float32').tofile('Datasets/kitti/training/velodyne/' + train_new_frame + '.bin')

                cv2.imwrite('Datasets/kitti/training/image_2/' + train_new_frame + '.png', img)

                vis.mlab_source.set(x=fusedPCL[:, 0], y=fusedPCL[:, 1], z=fusedPCL[:, 2], scalars=fusedPCL[:, 3])

            elif args.datatype == 'testing':
                vbbs_txt = np.array(vbbs_txt).reshape(-1, 15)
                np.savetxt('Datasets/kitti/testing/label_2/' + test_new_frame + '.txt', vbbs_txt, delimiter=' ',
                           fmt='%s')

                calib_save = open('Datasets/kitti/testing/calib/' + test_new_frame + '.txt', 'w+')
                calib_save = calib_txt
                calib_save.close()

                # Update PCL visualisation with Mayavi
                fusedPCL = np.concatenate(fusedPCL, axis=0)
                fusedPCL.astype('float32').tofile('Datasets/kitti/testing/velodyne/' + test_new_frame + '.bin')

                cv2.imwrite('Datasets/kitti/testing/image_2/' + test_new_frame + '.png', img)

                vis.mlab_source.set(x=fusedPCL[:, 0], y=fusedPCL[:, 1], z=fusedPCL[:, 2], scalars=fusedPCL[:, 3])

            # Set top-view if first frame
            if frame == 0:
                mlab.gcf().scene.z_plus_view()
            yield
        print("x_max:", x_max, "x_min:", x_min)
        print("y_max:", y_max, "y_min:", y_min)
        print("z_max:", z_max, "z_min:", z_min)

    anim()
    mlab.show()


if __name__ == '__main__':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '--filename',
        default='',
        type=str,
        help='File name of the target ground truth data')
    argparser.add_argument(
        '--datatype',
        default='',
        type=str,
        help='Training or Testing Data')
    argparser.add_argument(
        '--count_20',
        default=0,
        type=int,
        help='Number of car that has 20% delay time on communication')
    argparser.add_argument(
        '--count_40',
        default=0,
        type=int,
        help='Number of car that has 40% delay time on communication')
    argparser.add_argument(
        '--count_60',
        default=0,
        type=int,
        help='Number of car that has 60% delay time on communication')
    argparser.add_argument(
        '--count_80',
        default=0,
        type=int,
        help='Number of car that has 80% delay time on communication')
    argparser.add_argument(
        '--count_100',
        default=0,
        type=int,
        help='Number of car that has 100% delay time on communication')
    args = argparser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        pass
    finally:
        logging.info('Delay Completed')
