from PIL import Image
import numpy as np
import cv2
import copy
import time
import os

# Silence Numba compiler debug output
os.environ["NUMBA_DEBUG"] = "0"
os.environ["NUMBA_DEBUG_FRONTEND"] = "0"
os.environ["NUMBA_DUMP_BYTECODE"] = "0"
os.environ["NUMBA_DUMP_CFG"] = "0"
os.environ["NUMBA_DUMP_IR"] = "0"
os.environ["NUMBA_DUMP_LLVM"] = "0"
os.environ["NUMBA_DUMP_ASSEMBLY"] = "0"

from numba import njit


# @njit(cache=True, fastmath=True)
# def blend_numba(crop16, original16, mask16, blended):
#     h, w, _ = crop16.shape

#     for y in range(h):
#         for x in range(w):
#             m = mask16[y, x]
#             inv_m = 255 - m

#             for c in range(3):
#                 blended[y, x, c] = (
#                     crop16[y, x, c] * m
#                     + original16[y, x, c] * inv_m
#                 ) // 255

@njit(cache=True, fastmath=True)
def blend_numba_inplace(crop, original, mask):
    h, w, _ = crop.shape

    for y in range(h):
        for x in range(w):
            m = mask[y, x]
            inv_m = 255 - m

            for c in range(3):
                crop[y, x, c] = (
                    crop[y, x, c] * m
                    + original[y, x, c] * inv_m
                ) // 255


def get_crop_box(box, expand):
    x, y, x1, y1 = box
    x_c, y_c = (x+x1)//2, (y+y1)//2
    w, h = x1-x, y1-y
    s = int(max(w, h)//2*expand)
    crop_box = [x_c-s, y_c-s, x_c+s, y_c+s]
    return crop_box, s


def face_seg(image, mode="raw", fp=None):
    """
    对图像进行面部解析，生成面部区域的掩码。

    Args:
        image (PIL.Image): 输入图像。

    Returns:
        PIL.Image: 面部区域的掩码图像。
    """
    seg_image = fp(image, mode=mode)  # 使用 FaceParsing 模型解析面部
    if seg_image is None:
        print("error, no person_segment")  # 如果没有检测到面部，返回错误
        return None

    seg_image = seg_image.resize(image.size)  # 将掩码图像调整为输入图像的大小
    return seg_image


def get_image(image, face, face_box, upper_boundary_ratio=0.5, expand=1.5, mode="raw", fp=None):
    """
    将裁剪的面部图像粘贴回原始图像，并进行一些处理。

    Args:
        image (numpy.ndarray): 原始图像（身体部分）。
        face (numpy.ndarray): 裁剪的面部图像。
        face_box (tuple): 面部边界框的坐标 (x, y, x1, y1)。
        upper_boundary_ratio (float): 用于控制面部区域的保留比例。
        expand (float): 扩展因子，用于放大裁剪框。
        mode: 融合mask构建方式 

    Returns:
        numpy.ndarray: 处理后的图像。
    """
    # 将 numpy 数组转换为 PIL 图像
    body = Image.fromarray(image[:, :, ::-1])  # 身体部分图像(整张图)
    face = Image.fromarray(face[:, :, ::-1])  # 面部图像

    x, y, x1, y1 = face_box  # 获取面部边界框的坐标
    crop_box, s = get_crop_box(face_box, expand)  # 计算扩展后的裁剪框
    x_s, y_s, x_e, y_e = crop_box  # 裁剪框的坐标
    face_position = (x, y)  # 面部在原始图像中的位置

    # 从身体图像中裁剪出扩展后的面部区域（下巴到边界有距离）
    face_large = body.crop(crop_box)
        
    ori_shape = face_large.size  # 裁剪后图像的原始尺寸

    # 对裁剪后的面部区域进行面部解析，生成掩码
    mask_image = face_seg(face_large, mode=mode, fp=fp)
    
    mask_small = mask_image.crop((x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 裁剪出面部区域的掩码
    
    mask_image = Image.new('L', ori_shape, 0)  # 创建一个全黑的掩码图像
    mask_image.paste(mask_small, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))  # 将面部掩码粘贴到全黑图像上
    
    
    # 保留面部区域的上半部分（用于控制说话区域）
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)  # 计算上半部分的边界
    modified_mask_image = Image.new('L', ori_shape, 0)  # 创建一个新的全黑掩码图像
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))  # 粘贴上半部分掩码
    
    
    # 对掩码进行高斯模糊，使边缘更平滑
    blur_kernel_size = int(0.05 * ori_shape[0] // 2 * 2) + 1  # 计算模糊核大小
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)  # 高斯模糊
    #mask_array = np.array(modified_mask_image)
    mask_image = Image.fromarray(mask_array)  # 将模糊后的掩码转换回 PIL 图像
    
    # 将裁剪的面部图像粘贴回扩展后的面部区域
    face_large.paste(face, (x - x_s, y - y_s, x1 - x_s, y1 - y_s))
    
    body.paste(face_large, crop_box[:2], mask_image)
    
    body = np.array(body)  # 将 PIL 图像转换回 numpy 数组

    return body[:, :, ::-1]  # 返回处理后的图像（BGR 转 RGB）



# def get_image_blending(image, face, face_box, mask_array, crop_box, mask_bbox):

#     x, y, x1, y1 = face_box
#     x_s, y_s, x_e, y_e = crop_box

#     # ---------------------------------------------------------
#     # 1. Crop de la zone visage
#     # ---------------------------------------------------------

#     # crop_x1 = max(0, x_s)
#     # crop_y1 = max(0, y_s)
#     # crop_x2 = min(image.shape[1], x_e)
#     # crop_y2 = min(image.shape[0], y_e)

#     # offset_x = x - x_s
#     # offset_y = y - y_s

#     # face_w = x1 - x
#     # face_h = y1 - y

#     # crop = image[
#     #     crop_y1:crop_y2,
#     #     crop_x1:crop_x2
#     # ].copy()

#     # 1. Crop
#     crop_x1 = max(0, x_s)
#     crop_y1 = max(0, y_s)
#     crop_x2 = min(image.shape[1], x_e)
#     crop_y2 = min(image.shape[0], y_e)

#     crop = image[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ].copy()

   

#     # Mask coordinates relative to the actual clipped crop
#     mask_offset_x = crop_x1 - x_s
#     mask_offset_y = crop_y1 - y_s

#     mask_array = mask_array[
#         mask_offset_y:mask_offset_y + crop.shape[0],
#         mask_offset_x:mask_offset_x + crop.shape[1]
#     ]


#     # 2. Paste face
#     local_x1 = x - crop_x1
#     local_y1 = y - crop_y1

#     face_h, face_w = face.shape[:2]

#     src_x1 = max(0, -local_x1)
#     src_y1 = max(0, -local_y1)

#     dst_x1 = max(0, local_x1)
#     dst_y1 = max(0, local_y1)

#     paste_w = min(
#         face_w - src_x1,
#         crop.shape[1] - dst_x1
#     )

#     paste_h = min(
#         face_h - src_y1,
#         crop.shape[0] - dst_y1
#     )

#     if paste_w > 0 and paste_h > 0:
#         crop[
#             dst_y1:dst_y1 + paste_h,
#             dst_x1:dst_x1 + paste_w
#         ] = face[
#             src_y1:src_y1 + paste_h,
#             src_x1:src_x1 + paste_w
#         ]

#     # ---------------------------------------------------------
#     # 3. Resize mask si nécessaire
#     # ---------------------------------------------------------

#     if mask_array.shape[:2] != crop.shape[:2]:

#         print("\n========== MASK/CROP DEBUG ==========")
#         print(f"image shape       : {image.shape}")
#         print(f"face_box          : {face_box}")
#         print(f"crop_box          : {crop_box}")
#         print(f"crop actual       : {crop.shape}")
#         print(f"mask actual       : {mask_array.shape}")
#         print(f"crop coords       : ({crop_x1}, {crop_y1}) -> ({crop_x2}, {crop_y2})")
#         print(f"theoretical size  : ({y_e-y_s}, {x_e-x_s})")
#         print(f"mask_bbox         : {mask_bbox}")
#         print("====================================\n")
#         raise RuntimeError(
#         f"Mask/crop mismatch: "
#         f"mask={mask_array.shape[:2]}, "
#         f"crop={crop.shape[:2]}"
#         )


#     if mask_bbox is None:
#         return image

#     mask_x1, mask_y1, mask_x2, mask_y2 = mask_bbox

#     # mask_bbox was defined in the original theoretical crop coordinates.
#     # Shift it to the actual clipped crop coordinates.
#     mask_x1 -= mask_offset_x
#     mask_x2 -= mask_offset_x
#     mask_y1 -= mask_offset_y
#     mask_y2 -= mask_offset_y

#     # Clip bbox to the actual crop
#     mask_x1 = max(0, mask_x1)
#     mask_y1 = max(0, mask_y1)
#     mask_x2 = min(crop.shape[1], mask_x2)
#     mask_y2 = min(crop.shape[0], mask_y2)

#     if mask_x2 <= mask_x1 or mask_y2 <= mask_y1:
#         return image

#     # ---------------------------------------------------------
#     # 5. Blending uniquement dans la bbox du masque
#     # ---------------------------------------------------------

#     mask_roi = mask_array[
#         mask_y1:mask_y2,
#         mask_x1:mask_x2
#     ]

#     crop_roi = crop[
#         mask_y1:mask_y2,
#         mask_x1:mask_x2
#     ]

#     original_roi = image[
#         crop_y1 + mask_y1:crop_y1 + mask_y2,
#         crop_x1 + mask_x1:crop_x1 + mask_x2
#     ]

#     # uint16 pour éviter le overflow uint8
#     mask16 = mask_roi.astype(np.uint16)[..., None]
#     inv_mask16 = (255 - mask_roi).astype(np.uint16)[..., None]

#     crop16 = crop_roi.astype(np.uint16)
#     original16 = original_roi.astype(np.uint16)

#     blended = (
#         crop16 * mask16
#         + original16 * inv_mask16
#     ) // 255

#     crop[
#         mask_y1:mask_y2,
#         mask_x1:mask_x2
#     ] = blended.astype(np.uint8)

#     # ---------------------------------------------------------
#     # 6. Remettre le crop dans l'image
#     # ---------------------------------------------------------

#     output = image.copy()

#     output[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ] = crop

#     return output

def get_image_blending(
    image,
    face,
    face_box,
    mask_array,
    crop_box,
    mask_bbox,
    profile=None
):

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box

    # ---------------------------------------------------------
    # 1. Crop
    # ---------------------------------------------------------

    t0 = time.perf_counter()

    crop_x1 = max(0, x_s)
    crop_y1 = max(0, y_s)
    crop_x2 = min(image.shape[1], x_e)
    crop_y2 = min(image.shape[0], y_e)

    # crop = image[
    #     crop_y1:crop_y2,
    #     crop_x1:crop_x2
    # ].copy()
    crop = image[
    crop_y1:crop_y2,
    crop_x1:crop_x2
    ].copy()

    # Keep original crop so the cached frame can be restored
    original_crop = crop.copy()

    mask_offset_x = crop_x1 - x_s
    mask_offset_y = crop_y1 - y_s

    mask_array = mask_array[
        mask_offset_y:mask_offset_y + crop.shape[0],
        mask_offset_x:mask_offset_x + crop.shape[1]
    ]

    t1 = time.perf_counter()

    # ---------------------------------------------------------
    # 2. Paste face
    # ---------------------------------------------------------

    local_x1 = x - crop_x1
    local_y1 = y - crop_y1

    face_h, face_w = face.shape[:2]

    src_x1 = max(0, -local_x1)
    src_y1 = max(0, -local_y1)

    dst_x1 = max(0, local_x1)
    dst_y1 = max(0, local_y1)

    paste_w = min(
        face_w - src_x1,
        crop.shape[1] - dst_x1
    )

    paste_h = min(
        face_h - src_y1,
        crop.shape[0] - dst_y1
    )

    if paste_w > 0 and paste_h > 0:
        crop[
            dst_y1:dst_y1 + paste_h,
            dst_x1:dst_x1 + paste_w
        ] = face[
            src_y1:src_y1 + paste_h,
            src_x1:src_x1 + paste_w
        ]

    t2 = time.perf_counter()

    # ---------------------------------------------------------
    # 3. Check mask
    # ---------------------------------------------------------

    if mask_array.shape[:2] != crop.shape[:2]:

        print("\n========== MASK/CROP DEBUG ==========")
        print(f"image shape       : {image.shape}")
        print(f"face_box          : {face_box}")
        print(f"crop_box          : {crop_box}")
        print(f"crop actual       : {crop.shape}")
        print(f"mask actual       : {mask_array.shape}")
        print(
            f"crop coords       : "
            f"({crop_x1}, {crop_y1}) -> ({crop_x2}, {crop_y2})"
        )
        print(
            f"theoretical size  : "
            f"({y_e-y_s}, {x_e-x_s})"
        )
        print(f"mask_bbox         : {mask_bbox}")
        print("====================================\n")

        raise RuntimeError(
            f"Mask/crop mismatch: "
            f"mask={mask_array.shape[:2]}, "
            f"crop={crop.shape[:2]}"
        )

    t3 = time.perf_counter()

    # ---------------------------------------------------------
    # 4. BBox
    # ---------------------------------------------------------

    if mask_bbox is None:
        return image, None, None

    mask_x1, mask_y1, mask_x2, mask_y2 = mask_bbox

    mask_x1 -= mask_offset_x
    mask_x2 -= mask_offset_x
    mask_y1 -= mask_offset_y
    mask_y2 -= mask_offset_y

    mask_x1 = max(0, mask_x1)
    mask_y1 = max(0, mask_y1)
    mask_x2 = min(crop.shape[1], mask_x2)
    mask_y2 = min(crop.shape[0], mask_y2)

    if mask_x2 <= mask_x1 or mask_y2 <= mask_y1:
        return image, None, None
    t4 = time.perf_counter()

    # ---------------------------------------------------------
    # 5. Extract ROIs
    # ---------------------------------------------------------

    mask_roi = mask_array[
        mask_y1:mask_y2,
        mask_x1:mask_x2
    ]

    crop_roi = crop[
        mask_y1:mask_y2,
        mask_x1:mask_x2
    ]

    original_roi = image[
        crop_y1 + mask_y1:crop_y1 + mask_y2,
        crop_x1 + mask_x1:crop_x1 + mask_x2
    ]

    t5 = time.perf_counter()

    # ---------------------------------------------------------
    # 6. Blending
    # ---------------------------------------------------------

    # mask16 = mask_roi.astype(np.uint16)[..., None]
    # inv_mask16 = (255 - mask_roi).astype(np.uint16)[..., None]

    # crop16 = crop_roi.astype(np.uint16)
    # original16 = original_roi.astype(np.uint16)

    # blended = (
    #     crop16 * mask16
    #     + original16 * inv_mask16
    # ) // 255




    # mask16 = mask_roi.astype(np.uint16)

    # crop16 = crop_roi.astype(np.uint16)
    # original16 = original_roi.astype(np.uint16)

    # # Blend channel-by-channel using OpenCV
    # blended = np.empty_like(crop16, dtype=np.uint16)

    # for c in range(3):
    #     blended[:, :, c] = (
    #         crop16[:, :, c] * mask16
    #         + original16[:, :, c] * (255 - mask16)
    #     ) // 255


    t_blend = time.perf_counter()



    # t = time.perf_counter()

    # mask16 = mask_roi.astype(np.uint16)
    # if profile is not None:
    #     profile["astype_mask"] += time.perf_counter() - t

    # t = time.perf_counter()
    # crop16 = crop_roi.astype(np.uint16)
    # if profile is not None:
    #     profile["astype_crop"] += time.perf_counter() - t

    # t = time.perf_counter()
    # original16 = original_roi.astype(np.uint16)
    # if profile is not None:
    #     profile["astype_original"] += time.perf_counter() - t

    # blended = np.empty_like(crop16, dtype=np.uint16)

    # t = time.perf_counter()
    # # for c in range(3):
    # #     blended[:, :, c] = (
    # #         crop16[:, :, c] * mask16
    # #         + original16[:, :, c] * (255 - mask16)
    # #     ) // 255

    # blend_numba(crop16, original16, mask16, blended)

    # if profile is not None:
    #     profile["math"] += time.perf_counter() - t

    # if profile is not None:
    #     profile["blend"] += time.perf_counter() - t_blend



    blend_numba_inplace(
    crop_roi,
    original_roi,
    mask_roi
    )

    if profile is not None:
        profile["math"] += time.perf_counter() - t_blend

    if profile is not None:
        profile["blend"] += time.perf_counter() - t_blend




    t6 = time.perf_counter()


    # crop[
    #     mask_y1:mask_y2,
    #     mask_x1:mask_x2
    # ] = blended.astype(np.uint8)

    # t7 = time.perf_counter()

    # # ---------------------------------------------------------
    # # 7. Output
    # # ---------------------------------------------------------

    # output = image.copy()

    # output[
    #     crop_y1:crop_y2,
    #     crop_x1:crop_x2
    # ] = crop

    # t8 = time.perf_counter()


    # ---------------------------------------------------------
    # 6b. Write blended ROI
    # ---------------------------------------------------------

    t_write_roi_start = time.perf_counter()

    # crop[
    #     mask_y1:mask_y2,
    #     mask_x1:mask_x2
    # ] = blended.astype(np.uint8)

    t_write_roi_end = time.perf_counter()

    # ---------------------------------------------------------
    # 7. Output
    # ---------------------------------------------------------

    t_output_start = time.perf_counter()

    # output = image.copy()

    # output[
    #     crop_y1:crop_y2,
    #     crop_x1:crop_x2
    # ] = crop
 
    image[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ] = crop
    

    t_output_end = time.perf_counter()

    # ---------------------------------------------------------
    # PROFILING
    # ---------------------------------------------------------

    if profile is not None:

        profile["crop"] += t1 - t0
        profile["paste"] += t2 - t1
        profile["mask_check"] += t3 - t2
        profile["bbox"] += t4 - t3
        profile["roi"] += t5 - t4
       
        profile["write_roi"] += t_write_roi_end - t_write_roi_start
        profile["output_copy"] += t_output_end - t_output_start

    

  
    return image, original_crop, (
        crop_y1,
        crop_y2,
        crop_x1,
        crop_x2
        )


def get_image_prepare_material(image, face_box, upper_boundary_ratio=0.5, expand=1.5, fp=None, mode="raw"):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    blur_kernel_size = int(0.1 * ori_shape[0] // 2 * 2) + 1
    # mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    # return mask_array, crop_box
    mask_array = cv2.GaussianBlur(
        np.array(modified_mask_image),
        (blur_kernel_size, blur_kernel_size),
        0
    )

    # Calcul de la bbox du masque UNE SEULE FOIS
    ys, xs = np.where(mask_array > 0)

    if len(xs) > 0:
        mask_bbox = (
            xs.min(),
            ys.min(),
            xs.max() + 1,
            ys.max() + 1
        )
    else:
        mask_bbox = None

    return mask_array, crop_box, mask_bbox
