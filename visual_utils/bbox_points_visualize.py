import numpy as np
import open3d as o3d
from open3d import geometry
import torch

from PIL import Image, ImageFont, ImageDraw
from pyquaternion import Quaternion

from tqdm import tqdm


def rotate_yaw(yaw):
    return np.array([[np.cos(yaw), np.sin(yaw), 0],
                     [-np.sin(yaw), np.cos(yaw), 0],
                     [0, 0, 1]], dtype=np.float32)


def text_3d(text, pos, direction=None, degree=0.0, density=10, font='/usr/share/fonts/truetype/freefont/FreeMono.ttf',
            font_size=10):
    """
       Generate a 3D text point cloud used for visualization.
       :param text: content of the text
       :param pos: 3D xyz position of the text upper left corner
       :param direction: 3D normalized direction of where the text faces
       :param degree: in plane rotation of text
       :param font: Name of the font - change it according to your system
       :param font_size: size of the font
       :return: o3d.geoemtry.PointCloud object

    """
    if direction is None:
        direction = (0., 0., 1.)

    font_obj = ImageFont.truetype(font, font_size * density)
    font_dim = font_obj.getsize(text)

    img = Image.new('RGB', font_dim, color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((0, 0), text, font=font_obj, fill=(0, 0, 0))
    img = np.asarray(img)
    img_mask = img[:, :, 0] < 128
    indices = np.indices([*img.shape[0:2], 1])[:, img_mask, 0].reshape(3, -1).T

    pcd = o3d.geometry.PointCloud()
    pcd.colors = o3d.utility.Vector3dVector(img[img_mask, :].astype(float) / 255.0)
    pcd.points = o3d.utility.Vector3dVector(indices / 1000 / density)

    raxis = np.cross([0.0, 0.0, 1.0], direction)
    if np.linalg.norm(raxis) < 1e-6:
        raxis = (0.0, 0.0, 1.0)
    trans = (Quaternion(axis=raxis, radians=np.arccos(direction[2])) *
             Quaternion(axis=direction, degrees=degree)).transformation_matrix
    trans[0:3, 3] = np.asarray(pos)
    pcd.transform(trans)
    return pcd


def draw_bboxes(bbox3d_list,
                vis=None,
                origin=[0, 0, 0],
                rot_axis=2,
                center_mode='lidar_bottom'):

    """Draw bbox on visualizer.
    Args:
        bbox3d_list: [[bbox3d, bbox_color]]
            bbox3d (numpy.array | torch.tensor, shape=[M, 7]):
                3d bbox (x, y, z, dx, dy, dz, yaw) to visualize.
            bbox_color (tuple[float]): the color of bbox. Default: (0, 1, 0).
        vis (:obj:`open3d.visualization.Visualizer`): open3d visualizer.
        rot_axis (int): rotation axis of bbox. Default: 2.
        center_mode (bool): indicate the center of bbox is bottom center
            or gravity center. avaliable mode
            ['lidar_bottom', 'camera_bottom']. Default: 'lidar_bottom'.
    """

    if not vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=origin)

    for bbox3d, bbox_color in bbox3d_list:
        if isinstance(bbox3d, torch.Tensor):
            bbox3d = bbox3d.cpu().numpy()
        bbox3d = bbox3d.copy()

        for i in range(len(bbox3d)):
            center = bbox3d[i, 0:3]
            dim = bbox3d[i, 3:6]
            yaw = np.zeros(3)
            yaw[rot_axis] = bbox3d[i, 6]
            rot_mat = geometry.get_rotation_matrix_from_xyz(yaw)

            if center_mode == 'lidar_bottom':
                center[rot_axis] += dim[
                                        rot_axis] / 2  # bottom center to gravity center
            elif center_mode == 'camera_bottom':
                center[rot_axis] -= dim[
                                        rot_axis] / 2  # bottom center to gravity center
            box3d = geometry.OrientedBoundingBox(center, rot_mat, dim)

            line_set = geometry.LineSet.create_from_oriented_bounding_box(box3d)
            line_set.paint_uniform_color(bbox_color)
            # draw bboxes on visualizer
            vis.add_geometry(line_set)

    vis.add_geometry(mesh_frame)

    return vis


def vis_box(boxes, vis=None, labels=[], color=(1, 0, 0)):
    if not vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    # convert center to bottom center
    boxes[:, 2] -= 0.5 * boxes[:, 5]
    vis_boxes_list = [[boxes, color]]

    vis = draw_bboxes(vis_boxes_list, vis=vis)

    # add labels
    if len(labels):
        pose = boxes[:, :3] + boxes[:, 3:6] / 2
        pose[:, 2] += boxes[:, 5] / 2
        for i, label in enumerate(tqdm(labels)):
            t = "{:.2f}".format(label)
            text = text_3d(t, pos=pose[i], direction=[0.0, 0.0, 1.0], degree=-90, font_size=800, density=1)
            vis.add_geometry(text)

    return vis


def vis_pc(points, vis=None, color=None):

    if not vis:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # eg:[0, 0, 1]
    if color:
        pcd.paint_uniform_color(color)

    # mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=origin)
    # vis.add_geometry(mesh_frame)

    vis.add_geometry(pcd)

    return vis


if __name__ == "__main__":

    # visualize bouding boxes
    boxes = np.array([[1, 1, 1, 10, 10, 10, 0], [20, 20, 20, 10, 10, 10, 0]], dtype=float)
    lb = [1, 2]

    vis = vis_box(boxes, labels=lb)

    # visualize point cloud
    points = []
    for i in range(1, 5):
        points.extend(np.load("../pc_demo/{}.npy".format(i)))
    points = np.array(points)
    vis = vis_pc(points[:, :3], vis=vis)

    top_points = np.load("../pc_demo/0.npy")
    vis = vis_pc(top_points[:, :3], vis=vis, color=[1, 0, 1])

    vis.run()
