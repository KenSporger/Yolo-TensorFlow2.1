import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# IOU计算
# input box shape: (N,S,S,B,4)
# output shape: (N,S,S,B)
def calcu_iou(box1, box2):
    # x,y,w,h->左上角右下角坐标，便于计算交集
    # shape:(N,S,S,B,(x1,y1,x2,y2))
    box1_corners = tf.stack([
        box1[..., 0] - box1[..., 2] / 2.0,
        box1[..., 1] - box1[..., 3] / 2.0,
        box1[..., 0] + box1[..., 2] / 2.0,
        box1[..., 1] + box1[..., 3] / 2.0
    ],axis=-1)
    box2_corners = tf.stack([
        box2[..., 0] - box2[..., 2] / 2.0,
        box2[..., 1] - box2[..., 3] / 2.0,
        box2[..., 0] + box2[..., 2] / 2.0,
        box2[..., 1] + box2[..., 3] / 2.0
    ],axis=-1)
    # 获取交集角点
    # shape:(N,S,S,B,(x1,y1,x2,y2))
    intersection_corners = tf.concat([
        tf.maximum(box1_corners[...,:2], box2_corners[...,:2]),
        tf.minimum(box1_corners[...,2:], box2_corners[...,2:])
    ], axis=-1)

    intersection_area = tf.maximum(0, intersection_corners[...,2] - intersection_corners[...,0]) *\
                        tf.maximum(0, intersection_corners[...,3] - intersection_corners[...,1])
    union_area = box1[...,2] * box1[...,3] + box2[...,2] * box2[...,3] - intersection_area

    return tf.clip_by_value(intersection_area / tf.maximum(union_area , 1e-10), 0.0, 1.0)

Sx, Sy = (7,7)
W, H = (448,448)
nboxes = 2
classes = 10
grid_x = tf.linspace(0.,Sx-1,Sx)
grid_y = tf.linspace(0.,Sy-1,Sy)
offset_x, offset_y = tf.meshgrid(grid_x, grid_y)
offset_x =tf.expand_dims(offset_x, axis=-1)
offset_y =tf.expand_dims(offset_y, axis=-1)
lambda_coord = 5
lambda_noobj = 0.5


# input shape: (N,S,S,nB*1+nB*4+C)
# input shape:(N,S,S,5+C)
def calcu_loss(y_pred, y_true):
    # box置信度:(N,S,S,B)
    pred_conf = y_pred[...,:nboxes]
    # 类别:(N,S,S,C)
    pred_classes = y_pred[...,nboxes*5:]
    # box位置大小
    pred_grid_coord = y_pred[...,nboxes:nboxes*5]
    # (N,S,S,B,4)
    pred_grid_coord = tf.reshape(pred_grid_coord, [-1, Sx, Sy, nboxes, 4])
    # 归一化：x,y,w,h为全局相对偏移
    pred_global_coord = tf.stack([
        (pred_grid_coord[...,0] + offset_x) / Sx,
        (pred_grid_coord[...,1] + offset_y) / Sy,
        pred_grid_coord[...,2],
        pred_grid_coord[...,3]
    ],axis=-1)

    # (N,S,S,1)
    target_conf = tf.expand_dims(y_true[...,0], axis=-1)
    # (N, S, S, C)
    target_class = y_true[...,5:]
    target_coord = y_true[...,1:5]
    # (N,S,S,1,4)
    target_global_coord = tf.reshape(target_coord, [-1, Sx, Sy, 1, 4])
    # target_global_coord = tf.stack([
    #     target_coord[...,0] / W,
    #     target_coord[..., 1] / H,
    #     target_coord[...,2] / W,
    #     target_coord[..., 3] / H,
    # ],axis=-1)
    target_grid_coord = tf.stack([
        target_global_coord[...,0] * Sx - offset_x,
        target_global_coord[..., 1] * Sy - offset_y,
        target_global_coord[...,2],
        target_global_coord[..., 3],
    ],axis=-1)

    # (N,S,S,B)
    pred_ious = calcu_iou(pred_global_coord, target_global_coord)
    # (N,S,S,B)
    box_maxious = tf.reduce_max(pred_ious, axis=-1, keepdims=True)
    # 负责预测物体的box:iou在grid中最大 and grid有真实物体
    obj_boxmask = tf.cast(pred_ious >= box_maxious, tf.float32) * target_conf
    # 不负责预测物体的box
    noobj_boxmask = tf.ones_like(obj_boxmask) - obj_boxmask

    # 概率误差,target_conf=0 or 1
    class_loss = tf.reduce_mean(tf.reduce_sum(tf.square(
        target_conf * (pred_classes - target_class)
    ), axis=[1,2,3]))

    # 含物体的置信度误差
    objconf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(
        obj_boxmask * (pred_conf - pred_ious)
    ), axis=[1,2,3]))

    # 不含物体的置信度误差
    noobjconf_loss = tf.reduce_mean(tf.reduce_sum(tf.square(
        noobj_boxmask * (pred_conf - pred_ious)
    ), axis=[1,2,3]))

    # 定位误差
    obj_boxmask = tf.expand_dims(obj_boxmask, axis=-1)
    location_loss = tf.reduce_mean(tf.reduce_sum(tf.square(
        obj_boxmask * (
            tf.stack([
                pred_grid_coord[..., 0],
                pred_grid_coord[..., 1],
                tf.sqrt(pred_grid_coord[..., 2]),
                tf.sqrt(pred_grid_coord[..., 3])
            ], axis=-1) -
            tf.stack([
                target_grid_coord[..., 0],
                target_grid_coord[..., 1],
                tf.sqrt(target_grid_coord[..., 2]),
                tf.sqrt(target_grid_coord[..., 3])
            ],axis=-1)
        )
    ), axis=[1,2,3,4]))

    total_loss = lambda_coord * location_loss + objconf_loss + lambda_noobj * noobjconf_loss + class_loss

    return total_loss



if __name__ == '__main__':
    pred = tf.random.uniform([1, Sx, Sy, nboxes*5+classes])
    target = tf.random.uniform([1, Sx, Sy, 5+classes])
    print(calcu_loss(pred, target))



