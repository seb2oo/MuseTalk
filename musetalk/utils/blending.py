from PIL import Image
import numpy as np
import cv2
import copy
import time


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


# def get_image_blending(image, face, face_box, mask_array, crop_box):
#     body = Image.fromarray(image[:,:,::-1])
#     face = Image.fromarray(face[:,:,::-1])

#     x, y, x1, y1 = face_box
#     x_s, y_s, x_e, y_e = crop_box
#     face_large = body.crop(crop_box)

#     mask_image = Image.fromarray(mask_array)
#     mask_image = mask_image.convert("L")
#     face_large.paste(face, (x-x_s, y-y_s, x1-x_s, y1-y_s))
#     body.paste(face_large, crop_box[:2], mask_image)
#     body = np.array(body)
#     return body[:,:,::-1]

# def get_image_blending(image, face, face_box, mask_array, crop_box):

#     x, y, x1, y1 = face_box
#     x_s, y_s, x_e, y_e = crop_box

#     # --------------------------------------------------
#     # Crop region
#     # --------------------------------------------------

#     crop_x1 = max(0, x_s)
#     crop_y1 = max(0, y_s)
#     crop_x2 = min(image.shape[1], x_e)
#     crop_y2 = min(image.shape[0], y_e)

#     # Offset of face inside crop
#     offset_x = x - x_s
#     offset_y = y - y_s

#     face_w = x1 - x
#     face_h = y1 - y

#     # --------------------------------------------------
#     # Copy crop
#     # --------------------------------------------------

#     crop = image[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ].copy()

#     # --------------------------------------------------
#     # Insert generated face
#     # --------------------------------------------------

#     local_x1 = offset_x
#     local_y1 = offset_y

#     local_x2 = local_x1 + face_w
#     local_y2 = local_y1 + face_h

#     src_x1 = max(0, -local_x1)
#     src_y1 = max(0, -local_y1)

#     src_x2 = min(
#         face_w,
#         crop.shape[1] - local_x1
#     )

#     src_y2 = min(
#         face_h,
#         crop.shape[0] - local_y1
#     )

#     if src_x2 > src_x1 and src_y2 > src_y1:

#         dst_x1 = max(0, local_x1)
#         dst_y1 = max(0, local_y1)

#         crop[
#             dst_y1:dst_y1 + (src_y2 - src_y1),
#             dst_x1:dst_x1 + (src_x2 - src_x1)
#         ] = face[
#             src_y1:src_y2,
#             src_x1:src_x2
#         ]

#     # --------------------------------------------------
#     # Blend
#     # --------------------------------------------------

#     mask = mask_array

#     if mask.shape[:2] != crop.shape[:2]:
#         mask = cv2.resize(
#             mask,
#             (crop.shape[1], crop.shape[0]),
#             interpolation=cv2.INTER_LINEAR
#         )

#     mask = mask.astype(np.float32) / 255.0
#     mask = mask[..., None]

#     original_crop = image[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ].astype(np.float32)

#     crop_float = crop.astype(np.float32)

#     blended = (
#         crop_float * mask
#         + original_crop * (1.0 - mask)
#     )

#     blended = np.clip(
#         blended,
#         0,
#         255
#     ).astype(np.uint8)

#     # --------------------------------------------------
#     # Put crop back
#     # --------------------------------------------------

#     output = image.copy()

#     output[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ] = blended

#     return output




# def get_image_blending(image, face, face_box, mask_array, crop_box):

#     t0 = time.perf_counter()

#     x, y, x1, y1 = face_box
#     x_s, y_s, x_e, y_e = crop_box

#     # --------------------------------------------------
#     # 1. Crop coordinates
#     # --------------------------------------------------

#     crop_x1 = max(0, x_s)
#     crop_y1 = max(0, y_s)
#     crop_x2 = min(image.shape[1], x_e)
#     crop_y2 = min(image.shape[0], y_e)

#     offset_x = x - x_s
#     offset_y = y - y_s

#     face_w = x1 - x
#     face_h = y1 - y

#     t1 = time.perf_counter()

#     # --------------------------------------------------
#     # 2. Crop copy
#     # --------------------------------------------------

#     crop = image[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ].copy()

#     t2 = time.perf_counter()

#     # --------------------------------------------------
#     # 3. Insert generated face
#     # --------------------------------------------------

#     local_x1 = offset_x
#     local_y1 = offset_y

#     src_x1 = max(0, -local_x1)
#     src_y1 = max(0, -local_y1)

#     src_x2 = min(
#         face_w,
#         crop.shape[1] - local_x1
#     )

#     src_y2 = min(
#         face_h,
#         crop.shape[0] - local_y1
#     )

#     if src_x2 > src_x1 and src_y2 > src_y1:

#         dst_x1 = max(0, local_x1)
#         dst_y1 = max(0, local_y1)

#         crop[
#             dst_y1:dst_y1 + (src_y2 - src_y1),
#             dst_x1:dst_x1 + (src_x2 - src_x1)
#         ] = face[
#             src_y1:src_y2,
#             src_x1:src_x2
#         ]

#     t3 = time.perf_counter()

#     # --------------------------------------------------
#     # 4. Resize mask if necessary
#     # --------------------------------------------------

#     mask = mask_array

#     if mask.shape[:2] != crop.shape[:2]:
#         mask = cv2.resize(
#             mask,
#             (crop.shape[1], crop.shape[0]),
#             interpolation=cv2.INTER_LINEAR
#         )

#     t4 = time.perf_counter()

#     # --------------------------------------------------
#     # 5. Mask conversion
#     # --------------------------------------------------

#     mask = mask.astype(np.float32) / 255.0
#     mask = mask[..., None]

#     t5 = time.perf_counter()

#     # --------------------------------------------------
#     # 6. Original crop conversion
#     # --------------------------------------------------

#     original_crop = image[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ].astype(np.float32)

#     t6 = time.perf_counter()

#     # --------------------------------------------------
#     # 7. Generated crop conversion
#     # --------------------------------------------------

#     crop_float = crop.astype(np.float32)

#     t7 = time.perf_counter()

#     # --------------------------------------------------
#     # 8. Actual blend
#     # --------------------------------------------------

#     blended = (
#         crop_float * mask
#         + original_crop * (1.0 - mask)
#     )

#     t8 = time.perf_counter()

#     # --------------------------------------------------
#     # 9. Convert back uint8
#     # --------------------------------------------------

#     blended = np.clip(
#         blended,
#         0,
#         255
#     ).astype(np.uint8)

#     t9 = time.perf_counter()

#     # --------------------------------------------------
#     # 10. Put crop back
#     # --------------------------------------------------

#     output = image.copy()

#     t10 = time.perf_counter()

#     output[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ] = blended

#     t11 = time.perf_counter()

#     # --------------------------------------------------
#     # Profiling
#     # --------------------------------------------------

#     if not hasattr(get_image_blending, "_count"):
#         get_image_blending._count = 0

#     get_image_blending._count += 1

#     if get_image_blending._count <= 10:

#         print("\n========== BLENDING PROFILE ==========")

#         print(f"Coordinates       : {(t1-t0)*1000:.3f} ms")
#         print(f"Crop copy         : {(t2-t1)*1000:.3f} ms")
#         print(f"Insert face       : {(t3-t2)*1000:.3f} ms")
#         print(f"Mask resize       : {(t4-t3)*1000:.3f} ms")
#         print(f"Mask float        : {(t5-t4)*1000:.3f} ms")
#         print(f"Original float    : {(t6-t5)*1000:.3f} ms")
#         print(f"Crop float        : {(t7-t6)*1000:.3f} ms")
#         print(f"Blend calculation : {(t8-t7)*1000:.3f} ms")
#         print(f"Uint8 conversion  : {(t9-t8)*1000:.3f} ms")
#         print(f"Output copy       : {(t10-t9)*1000:.3f} ms")
#         print(f"Final insertion   : {(t11-t10)*1000:.3f} ms")

#         print("--------------------------------------")
#         print(f"TOTAL             : {(t11-t0)*1000:.3f} ms")
#         print("======================================")

#     return output


# def get_image_blending(image, face, face_box, mask_array, crop_box):

#     x, y, x1, y1 = face_box
#     x_s, y_s, x_e, y_e = crop_box

#     # --------------------------------------------------
#     # Crop coordinates
#     # --------------------------------------------------

#     crop_x1 = max(0, x_s)
#     crop_y1 = max(0, y_s)
#     crop_x2 = min(image.shape[1], x_e)
#     crop_y2 = min(image.shape[0], y_e)

#     offset_x = x - x_s
#     offset_y = y - y_s

#     face_w = x1 - x
#     face_h = y1 - y

#     # --------------------------------------------------
#     # Crop
#     # --------------------------------------------------

#     crop = image[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ].copy()

#     # --------------------------------------------------
#     # Insert generated face
#     # --------------------------------------------------

#     local_x1 = offset_x
#     local_y1 = offset_y

#     src_x1 = max(0, -local_x1)
#     src_y1 = max(0, -local_y1)

#     src_x2 = min(
#         face_w,
#         crop.shape[1] - local_x1
#     )

#     src_y2 = min(
#         face_h,
#         crop.shape[0] - local_y1
#     )

#     if src_x2 > src_x1 and src_y2 > src_y1:

#         dst_x1 = max(0, local_x1)
#         dst_y1 = max(0, local_y1)

#         crop[
#             dst_y1:dst_y1 + (src_y2 - src_y1),
#             dst_x1:dst_x1 + (src_x2 - src_x1)
#         ] = face[
#             src_y1:src_y2,
#             src_x1:src_x2
#         ]

#     # --------------------------------------------------
#     # Prepare mask
#     # --------------------------------------------------

#     mask = mask_array

#     if mask.shape[:2] != crop.shape[:2]:
#         mask = cv2.resize(
#             mask,
#             (crop.shape[1], crop.shape[0]),
#             interpolation=cv2.INTER_LINEAR
#         )

#     # --------------------------------------------------
#     # Integer blending
#     # --------------------------------------------------

#     # uint8 -> uint16 to avoid overflow
#     mask16 = mask.astype(np.uint16)

#     inv_mask16 = 255 - mask16

#     crop16 = crop.astype(np.uint16)

#     original16 = image[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ].astype(np.uint16)

#     # Add channel dimension
#     mask16 = mask16[..., None]
#     inv_mask16 = inv_mask16[..., None]

#     blended = (
#         crop16 * mask16
#         + original16 * inv_mask16
#     ) // 255

#     blended = blended.astype(np.uint8)

#     # --------------------------------------------------
#     # Put crop back
#     # --------------------------------------------------

#     output = image.copy()

#     output[
#         crop_y1:crop_y2,
#         crop_x1:crop_x2
#     ] = blended

#     return output

def get_image_blending(image, face, face_box, mask_array, crop_box):

    x, y, x1, y1 = face_box
    x_s, y_s, x_e, y_e = crop_box

    crop_x1 = max(0, x_s)
    crop_y1 = max(0, y_s)
    crop_x2 = min(image.shape[1], x_e)
    crop_y2 = min(image.shape[0], y_e)

    offset_x = x - x_s
    offset_y = y - y_s

    face_w = x1 - x
    face_h = y1 - y

    crop = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ].copy()

    local_x1 = offset_x
    local_y1 = offset_y

    src_x1 = max(0, -local_x1)
    src_y1 = max(0, -local_y1)

    src_x2 = min(
        face_w,
        crop.shape[1] - local_x1
    )

    src_y2 = min(
        face_h,
        crop.shape[0] - local_y1
    )

    if src_x2 > src_x1 and src_y2 > src_y1:

        dst_x1 = max(0, local_x1)
        dst_y1 = max(0, local_y1)

        crop[
            dst_y1:dst_y1 + (src_y2 - src_y1),
            dst_x1:dst_x1 + (src_x2 - src_x1)
        ] = face[
            src_y1:src_y2,
            src_x1:src_x2
        ]

    # =========================================================
    # MASK CACHE
    # =========================================================

    # Cache directement sur l'objet numpy original.
    # On utilise un attribut externe via dictionnaire global.
    global _MASK_CACHE

    try:
        _MASK_CACHE
    except NameError:
        _MASK_CACHE = {}

    mask_key = id(mask_array)

    cached = _MASK_CACHE.get(mask_key)

    if cached is None:

        mask16 = mask_array.astype(np.uint16)
        inv_mask16 = 255 - mask16

        # Ajouter la dimension des canaux une seule fois
        mask16 = mask16[..., None]
        inv_mask16 = inv_mask16[..., None]

        cached = (mask16, inv_mask16)

        _MASK_CACHE[mask_key] = cached

    else:
        mask16, inv_mask16 = cached


    if mask_array.size > 0:
        nonzero_ratio = np.count_nonzero(mask_array) / mask_array.size

        if not hasattr(get_image_blending, "_debug_count"):
            get_image_blending._debug_count = 0

        if get_image_blending._debug_count < 10:
            print(
                f"[MASK] shape={mask_array.shape} "
                f"nonzero={nonzero_ratio * 100:.2f}% "
                f"min={mask_array.min()} "
                f"max={mask_array.max()}"
            )
            get_image_blending._debug_count += 1

    # =========================================================
    # RESIZE DU MASQUE
    # =========================================================

    # Normalement le masque est déjà de la bonne taille.
    # On conserve cette sécurité.
    if mask16.shape[:2] != crop.shape[:2]:

        mask_resized = cv2.resize(
            mask_array,
            (crop.shape[1], crop.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        mask16 = mask_resized.astype(np.uint16)
        inv_mask16 = 255 - mask16

        mask16 = mask16[..., None]
        inv_mask16 = inv_mask16[..., None]

    # =========================================================
    # BLENDING INTEGER
    # =========================================================

    crop16 = crop.astype(np.uint16)

    original16 = image[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ].astype(np.uint16)

    blended = (
        crop16 * mask16
        + original16 * inv_mask16
    ) // 255

    blended = blended.astype(np.uint8)

    # =========================================================
    # OUTPUT
    # =========================================================

    output = image.copy()

    output[
        crop_y1:crop_y2,
        crop_x1:crop_x2
    ] = blended

    return output


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
    mask_array = cv2.GaussianBlur(np.array(modified_mask_image), (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
