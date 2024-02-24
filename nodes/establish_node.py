import cv2
import numpy as np
import torch
from PIL import Image

# 骨骼关键点连接对 18个关键点
pose_pairs = [
    [0, 1], [0, 14], [0, 15],
    [1, 2], [1, 5],
    [2, 3],
    [3, 4],
    [4, 3],
    [5, 6],
    [6, 7],
    [7, 6],
    [8, 9], [8, 1],
    [9, 10],
    [10, 9],
    [11, 1], [11, 12],
    [12, 13],
    [13, 12],
    [14, 16],
    [15, 17],
    [16, 14],
    [17, 15]
]

# 手部关键点连接对
hand_pairs = [
    [0, 1], [1, 2], [2, 3], [3, 4],
    [0, 5], [5, 6], [6, 7], [7, 8],
    [0, 9], [9, 10], [10, 11], [11, 12],
    [0, 13], [13, 14], [14, 15], [15, 16],
    [0, 17], [17, 18], [18, 19], [19, 20]
]

# 绘制用的颜色
pose_colors = [
    (255., 0., 85.), (255., 0., 0.), (255., 85., 0.), (255., 170., 0.),
    (255., 255., 0.), (170., 255., 0.), (85., 255., 0.), (0., 255., 0.),
    (255., 0., 0.), (0., 255., 85.), (0., 255., 170.), (0., 255., 255.),
    (0., 170., 255.), (0., 85., 255.), (0., 0., 255.), (255., 0., 170.),
    (170., 0., 255.), (255., 0., 255.), (85., 0., 255.), (0., 0., 255.),
    (0., 0., 255.), (0., 0., 255.), (0., 255.,
                                     255.), (0., 255., 255.), (0., 255., 255.)
]

hand_colors = [
    (100., 100., 100.), (100, 0, 0), (150, 0, 0),
    (200, 0, 0), (255, 0, 0), (100, 100, 0), (150, 150, 0), (200, 200, 0), (255, 255, 0),
    (0, 100, 50), (0, 150, 75), (0, 200, 100), (0, 255, 125), (0, 50, 100), (0, 75, 150),
    (0, 100, 200), (0, 125, 255), (100, 0, 100), (150, 0, 150), (200, 0, 200), (255, 0, 255)
]
def pil_to_tensor_grayscale(pil_image):
    # 将PIL图像转换为NumPy数组
    numpy_image = np.array(pil_image)
    # 归一化像素值
    numpy_image = numpy_image.astype(np.float32) / 255.0
    # 添加一个通道维度 [H, W] -> [1, H, W]
    numpy_image = np.expand_dims(numpy_image, axis=0)
    # 将NumPy数组转换为PyTorch张量
    tensor_image = torch.from_numpy(numpy_image)
    return tensor_image

class Mynode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "json": ("POSE_KEYPOINT",),
            },
            "optional": {
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.0, "step": 0.1}),
                "descending_parameters": ("INT", {"default": 0, "min": 0, "step": 1}),
            }
        }

    CATEGORY = "ys/openpose"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "test"

    def test(self, image, json, scale_factor, descending_parameters):
        # 这里开始处理和画图
        data_2 = json
        required_list = [0, 14, 15, 16, 17]

        #torch转numpy
        image = image.squeeze(0).cpu().numpy()
        # 创建一张和image尺寸大小相同的纯黑背景
        image_width, image_height = image.shape[1], image.shape[0]
        # 创建纯黑背景图像
        black_image = np.zeros((image_height, image_width, 3), np.uint8)

        for d in data_2[0]['people']:
            kpt_pose = np.array(d['pose_keypoints_2d']).reshape((18, 3))
            for p in pose_pairs:
                pt1 = tuple(list(map(int, kpt_pose[p[0], 0:2])))
                c1 = kpt_pose[p[0], 2]
                pt2 = tuple(list(map(int, kpt_pose[p[1], 0:2])))
                c2 = kpt_pose[p[1], 2]

                if c1 == 0.0 or c2 == 0.0:
                    continue
                # 这里按18个关键点来说
                kpt_face = np.array(d['face_keypoints_2d']).reshape((70, 3))
                center = np.mean(kpt_face, axis=0)[0:2]

                if p[0] in required_list:
                    my_array_pt1 = np.array(pt1)
                    centered_keypoints = my_array_pt1 - center
                    # 缩放头部关键点坐标
                    scaled_keypoints = centered_keypoints * scale_factor
                    # 恢复原始坐标系
                    scaled_keypoints = scaled_keypoints + center
                    my_array_pt1 = scaled_keypoints
                    my_array_pt1[1] = my_array_pt1[1] + descending_parameters
                    pt1 = tuple(int(x) for x in my_array_pt1)
                if p[1] in required_list:
                    my_array_pt2 = np.array(pt2)
                    centered_keypoints = my_array_pt2 - center
                    # 缩放头部关键点坐标
                    scaled_keypoints = centered_keypoints * scale_factor
                    # 恢复原始坐标系
                    scaled_keypoints = scaled_keypoints + center
                    my_array_pt2 = scaled_keypoints
                    my_array_pt2[1] = my_array_pt2[1] + descending_parameters
                    pt2 = tuple(int(x) for x in my_array_pt2)

                color = tuple(list(map(int, pose_colors[p[0]])))
                black_image = cv2.line(black_image, pt1, pt2, color, thickness=4)
                black_image = cv2.circle(black_image, pt1, 4, color, thickness=4, lineType=8, shift=0)
                black_image = cv2.circle(black_image, pt2, 4, color, thickness=4, lineType=8, shift=0)

            kpt_left_hand = np.array(d['hand_left_keypoints_2d']).reshape((21, 3))
            for q in hand_pairs:
                pt1 = tuple(list(map(int, kpt_left_hand[q[0], 0:2])))
                c1 = kpt_left_hand[p[0], 2]
                pt2 = tuple(list(map(int, kpt_left_hand[q[1], 0:2])))
                c2 = kpt_left_hand[q[1], 2]

                color = (0, 0, 0)  # hei色
                black_image = cv2.circle(black_image, pt1, 4, color, thickness=4, lineType=8, shift=0)
                black_image = cv2.circle(black_image, pt2, 4, color, thickness=4, lineType=8, shift=0)

                if c1 == 0.0 or c2 == 0.0:
                    continue

                color = tuple(list(map(int, hand_colors[q[0]])))
                black_image = cv2.line(black_image, pt1, pt2, color, thickness=4)

            kpt_right_hand = np.array(d['hand_right_keypoints_2d']).reshape((21, 3))
            for k in hand_pairs:
                pt1 = tuple(list(map(int, kpt_right_hand[k[0], 0:2])))
                c1 = kpt_right_hand[k[0], 2]
                pt2 = tuple(list(map(int, kpt_right_hand[k[1], 0:2])))
                c2 = kpt_right_hand[k[1], 2]

                if c1 == 0.0 or c2 == 0.0:
                    continue

                color = (0, 0, 0)  # 蓝色
                black_image = cv2.circle(black_image, pt1, 4, color, thickness=4, lineType=8, shift=0)
                black_image = cv2.circle(black_image, pt2, 4, color, thickness=4, lineType=8, shift=0)

                color = tuple(list(map(int, hand_colors[q[0]])))
                black_image = cv2.line(black_image, pt1, pt2, color, thickness=4)

            # 绘脸
            # 1、脸部轮廓收缩
            kpt_face = np.array(d['face_keypoints_2d']).reshape((70, 3))
            center = np.mean(kpt_face, axis=0)
            for i in range(len(kpt_face)):
                # 前17个关键点是脸部轮廓
                # 将中心点作为原点
                centered_keypoints = kpt_face[i] - center
                # 缩放头部关键点坐标
                scaled_keypoints = centered_keypoints * scale_factor
                # 恢复原始坐标系
                scaled_keypoints = scaled_keypoints + center
                kpt_face[i] = scaled_keypoints
                # 2、pose的颈部网上的坐标和所有脸部的坐标往下移（y轴加上一个数，传入参数）,pose在上面实现了
                kpt_face[i][1] = kpt_face[i][1] + descending_parameters

            for item in kpt_face:
                x = int(item[0])
                y = int(item[1])
                c1 = item[2]
                if c1 == 0.0:
                    continue

                radius = 2  # 点的半径
                thickness = -1  # 填充点
                color = (255, 255, 255)  # 白色
                black_image = cv2.circle(black_image, (x, y), radius, color, thickness)

        result = cv2.cvtColor(black_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(result)
        torch_img = pil_to_tensor_grayscale(pil_image)
        # 转换为PyTorch张量
        return (torch_img,)

NODE_CLASS_MAPPINGS = {
    "18_keypoints_json": Mynode
}