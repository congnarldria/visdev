import os
import sys
from pathlib import Path

os.environ.setdefault("OS", "Windows_NT")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine.parser import ParseError, parse_program
from app.engine.samples_gen import ensure_samples
from app.engine.session import Session


def run_source(source: str) -> Session:
    session = Session(source, ROOT)
    session.run()
    return session


def test_hdevelop_batch1():
    import math

    import numpy as np

    from app.engine.chapters.tuple_ops import EXISTING, HALCON_TUPLE, IMPLEMENTED, SKIPPED
    from app.engine.objects import Image
    from app.engine.operators import REGISTRY

    registered = {name for name in REGISTRY if name.startswith("tuple_")}
    assert HALCON_TUPLE - SKIPPED == registered
    assert IMPLEMENTED.isdisjoint(EXISTING)
    assert IMPLEMENTED.isdisjoint(SKIPPED)

    source = """
T := [10, 20, 30]
A := T[1]
Vizion.tuple_select (T, 0, B)
Vizion.tuple_select (T, 1, C)
Vizion.tuple_gen_sequence (1, 5, 2, Seq)
Vizion.tuple_add (Seq, 10, Plus)
Vizion.tuple_sub (Plus, 1, Minus)
Vizion.tuple_less_elem (Seq, 4, Less)
Vizion.tuple_equal (Seq, [1, 3, 5], Same)
Vizion.tuple_find ([4, 8, 4], 4, Found)
Vizion.tuple_find ([1, 2], 9, Missing)
Vizion.tuple_sort ([3, 1, 2], Sorted)
Vizion.tuple_string (12.5, '.1f', Text)
Vizion.tuple_regexp_match ('ab12', '[0-9]+', Digits)
Vizion.tuple_substr ('HALCON', 0, 2, Head)
Vizion.tuple_gen_const (3, 7, Const)
"""
    session = run_source(source)
    assert session.error is None, session.error
    assert session.vars["A"] == 10
    assert session.vars["B"] == 10
    assert session.vars["C"] == 20
    assert session.vars["Seq"] == [1, 3, 5]
    assert session.vars["Plus"] == [11, 13, 15]
    assert session.vars["Minus"] == [10, 12, 14]
    assert session.vars["Less"] == [1, 1, 0]
    assert session.vars["Same"] == 1
    assert session.vars["Found"] == [0, 2]
    assert session.vars["Missing"] == -1
    assert session.vars["Sorted"] == [1, 2, 3]
    assert session.vars["Text"] == "12.5"
    assert session.vars["Digits"] == "12"
    assert session.vars["Head"] == "HAL"
    assert session.vars["Const"] == [7, 7, 7]

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_rgb1_to_gray_tuple_add_shift_local_threshold.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["First"] == 11
    gray = sample.vars["Gray"].gray()
    summed = sample.vars["Sum"].gray()
    assert np.array_equal(gray, summed)
    moved = sample.vars["Moved"].gray()
    assert moved[30, 50] == summed[18, 50]
    assert sample.vars["Bright"].count() >= 1
    assert abs(sample.vars["Mat"][2] - 12) < 1e-6

    blank = np.zeros((32, 32), np.uint8)
    blank[8:24, 8:24] = 180
    picture = Image(blank)
    basins, lines = REGISTRY["watersheds"].fn(picture, ctx=None).iconic
    assert basins.count() >= 1
    grown = REGISTRY["regiongrowing"].fn(picture, 16, 16, 5, 10, ctx=None).iconic[0]
    assert grown.count() == 1
    assert abs(REGISTRY["region_features"].fn(grown, "row1", ctx=None).control[0][0] - 8) <= 1

    identity = REGISTRY["hom_mat2d_identity"].fn(ctx=None).control[0]
    rotated = REGISTRY["hom_mat2d_rotate"].fn(identity, math.pi / 2, 0, 0, ctx=None).control[0]
    row, col = REGISTRY["affine_trans_pixel"].fn(rotated, 0, 10, ctx=None).control
    assert abs(row - (-10)) < 1e-6
    assert abs(col - 0) < 1e-6

    from app.engine.objects import Region, XLD

    mask = np.zeros((20, 20), np.uint8)
    mask[2:5, 3:6] = 255
    moved_region = REGISTRY["move_region"].fn(Region(20, 20, [mask]), 4, 1, ctx=None).iconic[0]
    assert moved_region.objects[0][6, 4] == 255
    assert moved_region.objects[0][2, 3] == 0
    bone = REGISTRY["skeleton"].fn(Region(20, 20, [mask]), ctx=None).iconic[0]
    assert bone.count() == 1
    contour = XLD(20, 20, [[[0, 0], [10, 0], [10, 30], [0, 30]]])
    clipped = REGISTRY["clip_contours_xld"].fn(contour, 0, 0, 20, 20, ctx=None).iconic[0]
    assert clipped.contours and all(point[1] <= 20 for line in clipped.contours for point in line)
    rounded = run_source("Vizion.tuple_round ([-1.5, 1.5], Round)\nVizion.tuple_split ('a,b', ',', Parts)\n")
    assert rounded.error is None, rounded.error
    assert rounded.vars["Round"] == [-2, 2]
    assert rounded.vars["Parts"] == ["a", "b"]


def _first(value):
    if isinstance(value, list):
        assert value, value
        return value[0]
    return value


def test_hdevelop_batch2():
    import math

    from app.engine.objects import Image
    from app.engine.operators import REGISTRY

    assert "'area'" in REGISTRY["select_shape"].control_in[0].choices
    assert "'contlength'" in REGISTRY["select_contours_xld"].control_in[0].choices
    assert "'green'" in REGISTRY["dev_set_color"].control_in[0].choices
    assert "'douglas'" in REGISTRY["gen_polygons_xld"].control_in[0].choices
    assert "'all'" in REGISTRY["measure_pos"].control_in[3].choices
    assert "'bilinear'" in REGISTRY["gen_measure_rectangle2"].control_in[7].choices
    assert "'none'" in REGISTRY["create_scaled_shape_model"].control_in[7].choices

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_rgb1_to_gray_measure_metrology_scaled_shape.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    columns = [float(item) for item in sample.vars["ColEdge"]]
    assert any(abs(item - 40) < 4 for item in columns), columns
    assert any(abs(item - 100) < 4 for item in columns), columns
    assert abs(float(_first(sample.vars["Intra"])) - 60) < 8
    assert abs(float(_first(sample.vars["LineCol"])) - 40) < 5
    circle = [float(item) for item in sample.vars["CircleParam"]]
    assert abs(circle[0] - 70) < 6 and abs(circle[1] - 180) < 6 and abs(circle[2] - 28) < 6, circle
    radii = sorted(float(item) for item in sample.vars["EllipseParam"][3:5])
    assert abs(radii[0] - 22) < 8 and abs(radii[1] - 36) < 8, sample.vars["EllipseParam"]
    assert len(sample.vars["Contour"].contours) == 3
    assert abs(float(_first(sample.vars["FoundRow"])) - 70) < 12
    assert abs(float(_first(sample.vars["FoundCol"])) - 180) < 12
    assert 0.85 <= float(_first(sample.vars["FoundScale"])) <= 1.15
    assert float(_first(sample.vars["FoundScore"])) > 0.4
    assert sample.vars["Metric"] == "use_polarity"
    assert float(sample.vars["ScaleMin"]) < 1 < float(sample.vars["ScaleMax"])

    scene_data = np.full((120, 160), 40, np.uint8)
    cv2.circle(scene_data, (80, 50), 18, 220, -1)
    scene = Image(scene_data)
    template = Image(scene_data[30:71, 60:101].copy())
    model = REGISTRY["create_ncc_model"].fn(template, 1, 0.0, 0.3, 0.15, "use_polarity", ctx=None).control[0]
    model.native = None
    levels, start, extent, step, metric = REGISTRY["get_ncc_model_params"].fn(model, ctx=None).control
    assert levels == 1 and metric == "use_polarity"
    assert abs(start) < 1e-9 and abs(extent - 0.3) < 1e-9 and abs(step - 0.15) < 1e-6
    origin_row, origin_col = REGISTRY["get_ncc_model_origin"].fn(model, ctx=None).control
    rows, cols, _angles, _scores = REGISTRY["find_ncc_model"].fn(
        scene, model, 0.0, 0.3, 0.5, 1, 0.5, "interpolation", 0, ctx=None
    ).control
    assert abs(_first(rows) - (30 + origin_row)) < 2, rows
    assert abs(_first(cols) - (60 + origin_col)) < 2, cols
    REGISTRY["set_ncc_model_origin"].fn(model, 0, 0, ctx=None)
    shifted, _cols, _angles, _scores = REGISTRY["find_ncc_model"].fn(
        scene, model, 0.0, 0.3, 0.5, 1, 0.5, "interpolation", 0, ctx=None
    ).control
    assert abs(_first(shifted) - (_first(rows) - origin_row)) < 1.5
    REGISTRY["clear_ncc_model"].fn(model, ctx=None)
    try:
        REGISTRY["find_ncc_model"].fn(scene, model, 0.0, 0.3, 0.5, 1, 0.5, "none", 0, ctx=None)
    except RuntimeError as exc:
        assert "clear" in str(exc)
    else:
        raise AssertionError("clear 之後仍可搜尋")

    handle = REGISTRY["gen_measure_rectangle2"].fn(50, 80, 0, 30, 2, 160, 120, "bilinear", ctx=None).control[0]
    REGISTRY["close_measure"].fn(handle, ctx=None)
    try:
        REGISTRY["measure_pos"].fn(scene, handle, 1, 10, "all", "all", ctx=None)
    except RuntimeError as exc:
        assert "close_measure" in str(exc)
    else:
        raise AssertionError("關閉後仍能量測")
    from app.engine.operators import _with_origin

    class _Origin:
        origin_row = 10.0
        origin_col = 4.0
        ref_row = 0.0
        ref_col = 0.0
        cleared = False

    shifted_row, shifted_col = _with_origin([30.0], [40.0], _Origin(), [2.0])
    assert shifted_row == [10.0] and shifted_col == [32.0]
    assert math.isfinite(origin_col)


def test_hdevelop_batch3():
    import math

    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_create_matrix_rigid_barcode_text.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(_first(sample.vars["Eye00"])) - 1) < 1e-6
    assert abs(float(_first(sample.vars["Row"])) - 0) < 1e-6
    assert abs(float(_first(sample.vars["Col"])) - 10) < 1e-6
    assert abs(float(_first(sample.vars["Qx"])) - 0) < 1e-6
    assert abs(float(_first(sample.vars["Qy"])) - 1) < 1e-6
    assert abs(float(_first(sample.vars["Qz"])) - 0) < 1e-6
    bar = sample.vars["BarText"]
    bar = bar[0] if isinstance(bar, list) else bar
    assert bar == "4006381333931", bar
    code = sample.vars["CodeText"]
    code = code[0] if isinstance(code, list) else code
    assert code == "VISDEV-3", code
    words = " ".join(str(item) for item in sample.vars["Words"])
    assert "VISDEV" in words, words

    pose = REGISTRY["create_pose"].fn(1, 2, 3, 0.2, -0.3, 0.4, "Rp+T", "gba", "point", ctx=None).control[0]
    mat = REGISTRY["pose_to_hom_mat3d"].fn(pose, ctx=None).control[0]
    back = REGISTRY["hom_mat3d_to_pose"].fn(mat, ctx=None).control[0]
    again = REGISTRY["pose_to_hom_mat3d"].fn(back, ctx=None).control[0]
    assert np.allclose(mat, again, atol=1e-6), (mat, again)
    shifted = REGISTRY["convert_pose_type"].fn(pose, "R(p-T)", "gba", "point", ctx=None).control[0]
    shifted_mat = REGISTRY["pose_to_hom_mat3d"].fn(shifted, ctx=None).control[0]
    assert np.allclose(mat, shifted_mat, atol=1e-5)
    values, vectors = REGISTRY["eigenvalues_symmetric_matrix"].fn(
        REGISTRY["create_matrix"].fn(2, 2, [2, 1, 1, 2], ctx=None).control[0], ctx=None
    ).control
    assert abs(float(values.data.reshape(-1)[0]) - 3) < 1e-6
    assert math.isfinite(float(vectors.data[0, 0]))


def test_hdevelop_batch4():
    import math

    from app.engine.chapters.code_ops import render_code128
    from app.engine.objects import Image
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_open_framegrabber_file_camera_Code39_quaternion_cloud.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    text = sample.vars["Text39"]
    text = text[0] if isinstance(text, list) else text
    assert text == "VISDEV", text
    assert abs(float(_first(sample.vars["Qx"]))) < 1e-5
    assert abs(float(_first(sample.vars["Qy"])) - 1) < 1e-5
    assert abs(float(_first(sample.vars["Qz"]))) < 1e-5
    center = sample.vars["Center"]
    assert abs(float(center[2]) - 5) < 1e-6, center
    assert float(sample.vars["ClassId"]) == 1
    assert float(sample.vars["Score"]) > 0.5

    exhausted = run_source("Vizion.open_framegrabber ('File', 'samples/code39.png', Acq)\nVizion.grab_image (Image, Acq)\nVizion.grab_image (Image2, Acq)\n")
    assert exhausted.error and "沒有下一張" in exhausted.error

    folder = ROOT / "workspace" / "batch4_frames"
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "a.png"), np.full((20, 20), 10, np.uint8))
    cv2.imwrite(str(folder / "b.png"), np.full((20, 20), 200, np.uint8))
    frames = run_source(
        "Vizion.open_framegrabber ('File', 'workspace/batch4_frames', 'gray', 0, 0, Acq)\n"
        "Vizion.grab_image (A, Acq)\nVizion.grab_image (B, Acq)\nVizion.close_framegrabber (Acq)\n"
    )
    assert frames.error is None, frames.error
    assert int(frames.vars["A"].gray().mean()) == 10
    assert int(frames.vars["B"].gray().mean()) == 200

    model = REGISTRY["create_bar_code_model"].fn("", "", ctx=None).control[0]
    found = REGISTRY["find_bar_code"].fn(Image(render_code128("Hi")), model, "Code 128", ctx=None).control[0]
    assert found == ["Hi"], found
    code_param = next(param for param in REGISTRY["find_bar_code"].control_in if param.name == "CodeType")
    assert "'Code 128'" in code_param.choices and "'Code 39'" in code_param.choices
    name_param = next(param for param in REGISTRY["open_framegrabber"].control_in if param.name == "Name")
    assert name_param.choices == ["'File'", "'OpenCV'"]

    left = REGISTRY["axis_angle_to_quat"].fn(0, 0, 1, math.pi / 2, ctx=None).control[0]
    composed = REGISTRY["quat_compose"].fn(left, left, ctx=None).control[0]
    qx, qy, qz = REGISTRY["quat_rotate_point"].fn(composed, 1, 0, 0, ctx=None).control
    assert abs(float(qx) + 1) < 1e-5 and abs(float(qy)) < 1e-5 and abs(float(qz)) < 1e-5
    pose = REGISTRY["quat_to_pose"].fn(left, ctx=None).control[0]
    back = np.asarray(REGISTRY["pose_to_quat"].fn(pose, ctx=None).control[0], np.float64)
    original = np.asarray(left, np.float64)
    if float(np.dot(original, back)) < 0:
        back = -back
    assert np.allclose(original, back, atol=1e-5)

    cloud = REGISTRY["gen_object_model_3d_from_points"].fn([0, 3], [0, 0], [0, 0], ctx=None).control[0]
    other = REGISTRY["gen_object_model_3d_from_points"].fn(0, 0, 4, ctx=None).control[0]
    distance = REGISTRY["distance_object_model_3d"].fn(cloud, other, ctx=None).control[0]
    assert abs(float(distance) - 4) < 1e-6

    cam = [800, 800, 320, 240, 0, 0, 0, 0, 0]
    world_x, world_y = REGISTRY["image_points_to_world_plane"].fn(
        cam, [0, 0, 400, 0, 0, 0, 0], 240 + 800 * 20 / 400, 320 + 800 * 10 / 400, 1, ctx=None
    ).control
    assert abs(float(world_x) - 10) < 1e-3 and abs(float(world_y) - 20) < 1e-3

    rows, cols, square = 4, 5, 10.0
    obj = np.zeros((rows * cols, 3), np.float64)
    obj[:, 0] = np.tile(np.arange(cols), rows) * square
    obj[:, 1] = np.repeat(np.arange(rows), cols) * square
    matrix = np.array([[800.0, 0, 320], [0, 800.0, 240], [0, 0, 1]])
    calib = REGISTRY["create_calib_data"].fn("calibration_object", 1, 1, ctx=None).control[0]
    REGISTRY["set_calib_data_calib_object"].fn(calib, 0, "4,5,10", ctx=None)
    views = [
        (np.array([[0.1], [0.05], [0.02]]), np.array([[0.0], [0.0], [500.0]])),
        (np.array([[-0.12], [0.08], [0.04]]), np.array([[8.0], [-6.0], [520.0]])),
        (np.array([[0.04], [-0.15], [-0.05]]), np.array([[-10.0], [5.0], [480.0]])),
    ]
    for rvec, tvec in views:
        projected, _ = cv2.projectPoints(obj, rvec, tvec, matrix, np.zeros(5))
        points = projected.reshape(-1, 2)
        assert points[:, 0].min() > 0 and points[:, 0].max() < 640
        assert points[:, 1].min() > 0 and points[:, 1].max() < 480
        REGISTRY["set_calib_data_observ_points"].fn(calib, 0, 0, points[:, 1].tolist(), points[:, 0].tolist(), ctx=None)
    error = REGISTRY["calibrate_cameras"].fn(calib, ctx=None).control[0]
    params = REGISTRY["get_calib_data"].fn(calib, "camera", 0, "params", ctx=None).control[0]
    assert float(error) < 0.5, error
    assert abs(float(params[0]) - 800) < 20, params

    border = 80
    square_px = 40
    board = np.full((6 * square_px + border * 2, 8 * square_px + border * 2), 255, np.uint8)
    for row in range(6):
        for col in range(8):
            if (row + col) % 2 == 0:
                y0 = border + row * square_px
                x0 = border + col * square_px
                board[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    plate = REGISTRY["create_calib_data"].fn("calibration_object", 1, 1, ctx=None).control[0]
    REGISTRY["set_calib_data_calib_object"].fn(plate, 0, "5,7,40", ctx=None)
    REGISTRY["find_calib_object"].fn(Image(board), plate, 0, 0, "", "", ctx=None)
    assert len(plate.observations) == 1
    assert len(plate.observations[0][1]) == 35


def test_hdevelop_batch5():
    from app.engine.objects import Image
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gray_erosion_rect_ellipse_contrast.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    eroded = sample.vars["Eroded"].gray()
    assert int(eroded[60, 80]) == 200
    assert int(eroded[30, 40]) == 0
    assert int(sample.vars["Stretched"].gray()[60, 80]) == 255
    assert abs(float(sample.vars["Dist"]) - 10) < 1e-6
    assert abs(float(sample.vars["RowProj"])) < 1e-6
    assert abs(float(sample.vars["ColProj"]) - 4) < 1e-6
    assert int(sample.vars["Oval"].union_mask()[60, 80]) == 255

    red = Image(np.full((4, 4), 255, np.uint8))
    green = Image(np.zeros((4, 4), np.uint8))
    blue = Image(np.zeros((4, 4), np.uint8))
    hue, saturation, value = REGISTRY["trans_from_rgb"].fn(red, green, blue, "hsv", ctx=None).iconic
    assert int(hue.data[0, 0]) == 0
    assert int(saturation.data[0, 0]) == 255
    assert int(value.data[0, 0]) == 255

    plain = np.zeros((40, 40), np.uint8)
    plain[10:30, 10:30] = 180
    opened = REGISTRY["gray_opening_rect"].fn(Image(plain), 5, ctx=None).iconic[0]
    assert int(opened.data[20, 20]) == 180
    assert int(opened.data[10, 10]) == 180
    assert int(opened.data[9, 9]) == 0

    scene = np.arange(60 * 80, dtype=np.uint16).reshape(60, 80) % 251
    scene = scene.astype(np.uint8)
    camera = [90, 90, 40, 30, 0.2, 0.01, 0.0, 0.0, 0.0]
    cleared = REGISTRY["change_radial_distortion_cam_par"].fn("none", camera, 0, ctx=None).control[0]
    assert cleared[4:] == [0, 0, 0, 0, 0]
    rectified = REGISTRY["change_radial_distortion_image"].fn(Image(scene), camera, cleared, ctx=None).iconic[0]
    matrix = np.array([[90.0, 0, 40], [0, 90.0, 30], [0, 0, 1]])
    expected = cv2.undistort(scene, matrix, np.array([0.2, 0.01, 0.0, 0.0, 0.0]), None, matrix)
    assert np.array_equal(rectified.data, expected)

    ordered = run_source(
        "Vizion.gen_image_const (Image, 'byte', 100, 80)\n"
        "Vizion.gen_rectangle1 (Top, 5, 8, 15, 18)\n"
        "Vizion.gen_rectangle1 (Bottom, 40, 60, 55, 75)\n"
        "Vizion.union2 (Top, Bottom, Both)\n"
        "Vizion.connection (Both, Parts)\n"
        "Vizion.sort_region (Parts, Sorted, 'character', 'true', 'row')\n"
        "Vizion.area_center (Sorted, Area, Row, Column)\n"
        "Vizion.shape_trans (Top, Rect, 'rectangle1')\n"
        "Vizion.area_center (Rect, Area2, Row2, Column2)\n"
        "Vizion.clip_region (Bottom, Clipped, 40, 60, 44, 64)\n"
        "Vizion.area_center (Clipped, Area3, Row3, Column3)\n"
        "Vizion.gen_rectangle2 (Tilt, 40, 50, 0, 10, 4)\n"
    )
    assert ordered.error is None, ordered.error
    assert float(ordered.vars["Row"][0]) < float(ordered.vars["Row"][1])
    assert int(ordered.vars["Area2"][0]) == 11 * 11
    assert int(ordered.vars["Area3"][0]) == 25
    assert int(ordered.vars["Tilt"].union_mask()[40, 50]) == 255
    assert int(ordered.vars["Tilt"].union_mask()[40, 70]) == 0
    choices = next(param for param in REGISTRY["shape_trans"].control_in if param.name == "Type").choices
    assert "'inner_circle'" in choices and "'rectangle2'" in choices


def test_hdevelop_batch6():
    from app.engine.objects import Image, Region
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_get_image_size_gray_tophat.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Width"] == 80 and sample.vars["Height"] == 60
    assert float(sample.vars["Gray"]) == 180
    assert float(sample.vars["MinGray"]) == 180 and float(sample.vars["MaxGray"]) == 180
    assert float(sample.vars["Range"]) == 0
    assert sample.vars["Count"] == 31 * 41
    assert float(sample.vars["Peak"]) == 0

    gradient = np.tile(np.linspace(0, 200, 80), (40, 1)).astype(np.uint8)
    flat = REGISTRY["illuminate"].fn(Image(gradient), 21, 21, 0, ctx=None).iconic[0].data
    original = float(gradient[:, -1].mean() - gradient[:, 0].mean())
    spread = float(flat[:, -1].mean() - flat[:, 0].mean())
    assert spread < original * 0.35

    ring = np.zeros((80, 80), np.uint8)
    cv2.circle(ring, (40, 40), 20, 255, 2)
    polar = REGISTRY["polar_trans_image_ext"].fn(
        Image(ring), 40, 40, 0, 6.28318530718, 18, 22, 90, 8, "bilinear", ctx=None
    ).iconic[0]
    assert polar.width == 90 and polar.height == 8
    assert float(polar.data.mean()) > 40

    constant = Image(np.full((20, 20), 40, np.uint8))
    lap = REGISTRY["laplace"].fn(constant, "absolute", 3, "n_4", ctx=None).iconic[0]
    assert int(lap.data[10, 10]) == 0

    values = np.arange(101, dtype=np.uint8).reshape(1, 101)
    area = Region(101, 1, [np.full((1, 101), 255, np.uint8)])
    low, high, span = REGISTRY["min_max_gray"].fn(area, Image(values), 0, ctx=None).control
    assert low == 0 and high == 100 and span == 100
    low, high, _span = REGISTRY["min_max_gray"].fn(area, Image(values), 50, ctx=None).control
    assert 20 < float(low) < 30 and 70 < float(high) < 80

    picked = run_source(
        "Vizion.gen_image_const (Image, 'byte', 80, 40)\n"
        "Vizion.gen_rectangle1 (Bright, 0, 0, 11, 11)\n"
        "Vizion.gen_rectangle1 (Dark, 2, 30, 12, 40)\n"
        "Vizion.paint_region (Bright, Image, Mid, 200, 'fill')\n"
        "Vizion.paint_region (Dark, Mid, Marked, 20, 'fill')\n"
        "Vizion.union2 (Bright, Dark, Both)\n"
        "Vizion.connection (Both, Parts)\n"
        "Vizion.select_gray (Parts, Marked, Kept, 'mean', 'and', 100, 255)\n"
        "Vizion.count_obj (Kept, KeptCount)\n"
        "Vizion.partition_rectangle (Bright, Tiles, 6, 6)\n"
        "Vizion.count_obj (Tiles, TileCount)\n"
        "Vizion.gen_rectangle1 (Wall, 0, 18, 20, 22)\n"
        "Vizion.expand_region (Dark, Wall, Grown, 3, 'image')\n"
        "Vizion.area_center (Dark, Area0, Row0, Column0)\n"
        "Vizion.area_center (Grown, Area1, Row1, Column1)\n"
        "Vizion.reduce_domain (Marked, Bright, Reduced)\n"
        "Vizion.crop_domain (Reduced, Part)\n"
        "Vizion.get_image_size (Part, PartWidth, PartHeight)\n"
        "Vizion.full_domain (Reduced, Whole)\n"
        "Vizion.get_domain (Whole, Domain)\n"
        "Vizion.area_center (Domain, DomainArea, DomainRow, DomainColumn)\n"
        "Vizion.gen_region_polygon (Poly, [5, 5, 20], [5, 25, 15])\n"
        "Vizion.area_center (Poly, PolyArea, PolyRow, PolyColumn)\n"
        "Vizion.change_domain (Marked, Dark, Changed)\n"
        "Vizion.reduce_domain (Changed, Bright, Overlap)\n"
        "Vizion.get_domain (Overlap, OverlapDomain)\n"
        "Vizion.count_obj (OverlapDomain, OverlapCount)\n"
    )
    assert picked.error is None, picked.error
    assert picked.vars["KeptCount"] == 1
    assert picked.vars["TileCount"] == 4
    assert int(picked.vars["Area1"][0]) > int(picked.vars["Area0"][0])
    assert picked.vars["PartWidth"] == 12 and picked.vars["PartHeight"] == 12
    assert int(picked.vars["DomainArea"][0]) == 80 * 40
    assert int(picked.vars["PolyArea"][0]) > 0
    assert picked.vars["OverlapCount"] == 0
    laplace_type = next(param for param in REGISTRY["laplace"].control_in if param.name == "ResultType")
    assert laplace_type.choices == ["'absolute'", "'signed'"]


def test_hdevelop_batch7():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_get_grayval_ramp_diameter_shape.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Gray"] == 20 and sample.vars["CopiedGray"] == 20
    assert sample.vars["BothCount"] == 2 and sample.vars["FirstCount"] == 1
    assert int(_first(sample.vars["BigArea"])) == 11 * 17
    assert int(sample.vars["HistoSum"]) == 11 * 17
    assert abs(float(sample.vars["RowSum"]) - 510) < 1e-6
    assert 16 < float(_first(sample.vars["Diameter"])) < 21
    assert float(_first(sample.vars["Ra"])) > float(_first(sample.vars["Rb"]))
    assert abs(float(_first(sample.vars["Phi"])) - float(_first(sample.vars["Angle"]))) < 1e-6
    assert sample.vars["Inside"] == 1 and sample.vars["Outside"] == 0
    assert int(_first(sample.vars["DotArea"])) == 2
    assert sample.vars["PartWidth"] == 10 and sample.vars["PartHeight"] == 8
    assert int(sample.vars["DevGray"]) == 3
    assert int(sample.vars["SpanGray"]) == 8
    assert 8 <= int(sample.vars["AmpGray"]) <= 32
    assert 120 <= int(sample.vars["DirGray"]) <= 136
    assert 48 < float(_first(sample.vars["CircLen"])) < 55
    assert abs(float(_first(sample.vars["RectLen"])) - 56) < 0.05
    assert abs(float(sample.vars["SegDist"]) - 5) < 1e-6
    assert float(sample.vars["CrossDist"]) == 0
    shapes = next(param for param in REGISTRY["select_shape_std"].control_in if param.name == "Shape").choices
    assert shapes == ["'max_area'", "'rectangle'", "'convex'"]
    order = next(param for param in REGISTRY["gen_circle_contour_xld"].control_in if param.name == "PointOrder").choices
    assert "'positive'" in order and "'negative'" in order


def test_hdevelop_batch8():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_scale_image_range_project_arrow.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert int(sample.vars["Top"]) == 255
    assert 128 < int(sample.vars["Deriv"]) < 160
    assert 120 <= int(sample.vars["Deriv2"]) <= 136
    assert int(sample.vars["EdgeEnergy"]) == 0
    assert int(sample.vars["LevelGray"]) == 40
    assert int(_first(sample.vars["HoleArea"])) == 710
    assert int(_first(sample.vars["FullArea"])) == 875
    assert float(sample.vars["InsideMin"]) == 0
    assert float(sample.vars["InsideMax"]) > 30
    assert 13 < float(sample.vars["OutsideMin"]) < 16
    assert abs(float(sample.vars["Angle"])) < 1e-6
    assert abs(float(sample.vars["Length"]) - 10) < 1e-6
    assert abs(float(sample.vars["Phi"])) < 1e-6
    assert abs(float(sample.vars["ColCenter"]) - 5) < 1e-6
    assert abs(float(sample.vars["ParRow"]) - 14) < 0.2
    assert abs(float(_first(sample.vars["Row1"])) - 10) < 1e-6
    assert abs(float(_first(sample.vars["Column2"])) - 20) < 1e-6
    assert abs(float(_first(sample.vars["CircleRadius"])) - 10) < 0.2
    assert abs(float(sample.vars["ContourMin"]) - 4) < 0.3
    assert float(sample.vars["ContourMax"]) > 18
    assert sample.vars["ArrowCount"] == 3
    assert int(sample.vars["ArrowGray"]) == 200
    assert sample.vars["RedGray"] == 255 and sample.vars["GreenGray"] == 0 and sample.vars["BlueGray"] == 0
    assert abs(float(sample.vars["RowT"]) - 17) < 1e-3
    assert abs(float(sample.vars["ColT"]) - 17) < 1e-3
    assert int(sample.vars["MovedGray"]) == 255
    assert int(sample.vars["OldGray"]) == 0
    counts = [0] * 256
    counts[30] = 100
    counts[220] = 100
    low, high = REGISTRY["histo_to_thresh"].fn(counts, 4, ctx=None).control
    assert low[0] == 0 and high[-1] == 255 and len(low) >= 2
    kinds = next(param for param in REGISTRY["derivate_gauss"].control_in if param.name == "Component").choices
    assert "'gradient'" in kinds and "'laplace'" in kinds


def test_hdevelop_batch9():
    from app.engine.objects import Image
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_distance_rr_align_corner_region_distance.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert 10 <= float(sample.vars["DistMin"]) <= 14
    assert float(sample.vars["DistMax"]) > 30
    assert abs(float(sample.vars["RowA"]) - 7) < 1e-3
    assert abs(float(sample.vars["ColA"]) - 7) < 1e-3
    assert abs(float(sample.vars["RowS"])) < 1e-3
    assert abs(float(sample.vars["ColS"]) - 20) < 1e-3
    assert abs(float(sample.vars["RowR"]) + 10) < 1e-3
    assert abs(float(sample.vars["ColR"])) < 1e-3
    assert abs(float(sample.vars["RowM"])) < 1e-3
    assert abs(float(sample.vars["ColM"]) + 10) < 1e-3
    assert sample.vars["CornerCount"] >= 2
    assert 8 <= float(sample.vars["CornerTop"]) <= 20
    assert sample.vars["PeakInside"] == 1
    assert int(sample.vars["ErodedGray"]) == 0
    assert int(sample.vars["FlatGray"]) == 80
    assert sample.vars["ContourCount"] > 4
    assert sample.vars["KeptCount"] == 1
    assert float(_first(sample.vars["M02"])) > float(_first(sample.vars["M20"]))
    assert sample.vars["MovedIn"] == 1 and sample.vars["MovedOut"] == 0
    assert abs(float(sample.vars["MeanRow"]) - 13) < 0.2
    assert abs(float(sample.vars["MeanCol"]) - 16) < 0.2
    assert float(_first(sample.vars["SkelLen"])) > 20
    assert sample.vars["Zero"] == 0
    assert sample.vars["Channels"] == 3
    bayer = np.zeros((8, 8), np.uint8)
    bayer[0::2, 0::2] = 255
    color = REGISTRY["cfa_to_rgb"].fn(Image(bayer), "bayer_rg", ctx=None).iconic[0]
    assert int(color.data[4, 4, 2]) > 200
    assert int(color.data[4, 4, 0]) < 40
    bayer_choices = next(param for param in REGISTRY["cfa_to_rgb"].control_in if param.name == "CFAType").choices
    assert "'bayer_rg'" in bayer_choices and "'bayer_gb'" in bayer_choices


def test_hdevelop_batch10():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_region_points_power_ridge_intersection.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert float(sample.vars["Longest"]) > 20
    assert sample.vars["DarkInside"] == 1
    assert int(sample.vars["FilledGray"]) > 40
    assert sample.vars["FlippedInside"] == 1
    assert abs(float(_first(sample.vars["RectRow"])) - 30) < 1.5
    assert abs(float(_first(sample.vars["RectCol"])) - 40) < 1.5
    assert 10 < float(_first(sample.vars["Len1"])) < 14
    assert 3 < float(_first(sample.vars["Len2"])) < 7
    assert sample.vars["SubCount"] >= 2
    assert int(sample.vars["PowGray"]) == 64
    assert int(sample.vars["RampGray"]) == 10
    assert int(sample.vars["RankLow"]) == 9
    assert abs(float(sample.vars["CrossRow"]) - 5) < 1e-3
    assert abs(float(sample.vars["CrossCol"]) - 5) < 1e-3
    assert sample.vars["Overlap"] == 1
    assert sample.vars["Hits"] == 2
    columns = sorted(abs(float(item)) for item in sample.vars["CircCol"])
    assert abs(columns[0] - 5) < 1e-3 and abs(columns[1] - 5) < 1e-3
    assert abs(float(sample.vars["LineDist"]) - 5) < 1e-6
    assert float(sample.vars["CrossDist"]) == 0
    assert sample.vars["WideW"] == 12 and sample.vars["WideH"] == 6
    modes = next(param for param in REGISTRY["lines_gauss"].control_in if param.name == "LightDark").choices
    assert modes == ["'light'", "'dark'", "'both'"]
    assert "'true'" in next(param for param in REGISTRY["points_harris"].control_in if param.name == "SubPixel").choices


def test_hdevelop_batch11():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_get_region_runs_filter_skeleton_circle.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["RunCount"] == 3
    assert sample.vars["RunTop"] == 10 and sample.vars["RunBot"] == 12
    assert sample.vars["RunLeft"] == 10 and sample.vars["RunRight"] == 20
    assert int(_first(sample.vars["RebuiltArea"])) == 33
    assert sample.vars["ClipIn"] == 1 and sample.vars["ClipOut"] == 0
    assert abs(float(sample.vars["InRow"]) - 30) <= 1
    assert abs(float(sample.vars["InCol"]) - 40) <= 1
    assert 11 <= float(sample.vars["InRad"]) <= 13
    assert int(_first(sample.vars["EndArea"])) == 4
    assert int(_first(sample.vars["JuncArea"])) == 1
    assert sample.vars["JuncInside"] == 1
    assert sample.vars["GridOn"] == 1 and sample.vars["GridOff"] == 0
    assert sample.vars["CellCount"] == 12
    assert 9.5 < float(sample.vars["FitRow"]) < 11.5
    assert float(sample.vars["FitColMin"]) < 8
    assert float(sample.vars["FitColMax"]) > 40
    assert abs(float(_first(sample.vars["Phi"]))) < 0.25
    assert sample.vars["CircHits"] == 2 and sample.vars["SegHits"] == 2
    circ_rows = sorted(abs(float(item)) for item in sample.vars["CircRow"])
    circ_cols = sorted(float(item) for item in sample.vars["CircCol"])
    assert abs(circ_rows[0] - 4) < 1e-3 and abs(circ_rows[1] - 4) < 1e-3
    assert abs(circ_cols[0] - 3) < 1e-3 and abs(circ_cols[1] - 3) < 1e-3
    seg_cols = sorted(abs(float(item)) for item in sample.vars["SegCol"])
    assert abs(seg_cols[0] - 5) < 1e-3 and abs(seg_cols[1] - 5) < 1e-3
    assert int(sample.vars["SoftGray"]) == 80
    assert int(sample.vars["SmoothGray"]) == 128
    assert int(sample.vars["EdgeGray"]) == 128
    assert int(sample.vars["SameGray"]) == 10
    assert next(param for param in REGISTRY["gen_grid_region"].control_in if param.name == "Type").choices == ["'lines'", "'rects'"]
    assert "'mirror'" in next(param for param in REGISTRY["convol_image"].control_in if param.name == "Margin").choices
    assert "'drop'" in next(param for param in REGISTRY["regress_contours_xld"].control_in if param.name == "Mode").choices


def test_hdevelop_batch12():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_inner_rectangle1_holes_rounded.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert int(_first(sample.vars["InR1"])) == 10 and int(_first(sample.vars["InC1"])) == 10
    assert int(_first(sample.vars["InR2"])) == 40 and int(_first(sample.vars["InC2"])) == 50
    assert int(_first(sample.vars["HoleArea"])) == 121
    assert sample.vars["OuterCount"] == 1 and sample.vars["HoleCount"] == 2
    assert sample.vars["HamDist"] == 165
    assert abs(float(sample.vars["HamSim"]) - 132 / 297) < 1e-6
    assert abs(float(_first(sample.vars["S03"]))) < 1
    assert abs(float(_first(sample.vars["S30"]))) < 1
    assert float(_first(sample.vars["H03"])) > 0
    assert sample.vars["PartCount"] == 2
    areas = sorted(int(item) for item in sample.vars["PartArea"])
    assert areas == [231, 336]
    assert float(_first(sample.vars["Ra"])) > float(_first(sample.vars["Rb"]))
    assert abs(float(_first(sample.vars["AxisPhi"]))) < 0.25
    rows = [float(item) for item in sample.vars["RoundRows"]]
    cols = [float(item) for item in sample.vars["RoundCols"]]
    assert abs(rows[0]) < 1e-3 and abs(cols[0]) < 1e-3
    assert abs(rows[-1] - 20) < 1e-3 and abs(cols[-1] - 20) < 1e-3
    corner = min((row * row + (col - 20) ** 2) ** 0.5 for row, col in zip(rows, cols))
    assert corner > 1.5
    assert sample.vars["ClipCount"] == 3
    assert sample.vars["ClipFirst"] == 1 and sample.vars["ClipLast"] == 3
    assert abs(float(sample.vars["ShortMin"]) - 10) < 1e-3
    assert abs(float(sample.vars["ShortMax"]) - 30) < 1e-3
    assert abs(float(_first(sample.vars["CrossRow"])) - 5) < 1e-3
    assert abs(float(_first(sample.vars["CrossCol"])) - 5) < 1e-3
    modes = next(param for param in REGISTRY["gen_contour_region_xld"].control_in if param.name == "Mode").choices
    assert modes == ["'border'", "'border_holes'", "'center'"]
    assert "'length'" in next(param for param in REGISTRY["clip_end_points_contours_xld"].control_in if param.name == "Mode").choices


def test_hdevelop_batch13():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_check_difference_chars_corners.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["DiffIn"] == 1 and sample.vars["DiffOut"] == 0 and sample.vars["DarkIn"] == 0
    assert sample.vars["DotIn"] == 1 and sample.vars["DotOut"] == 0
    assert sample.vars["CharIn"] == 1 and sample.vars["CharOut"] == 0
    assert 21 < float(sample.vars["Used"]) < 170
    assert float(_first(sample.vars["Energy"])) > 0.99
    assert abs(float(_first(sample.vars["Contrast"]))) < 1e-6
    assert float(_first(sample.vars["Homogeneity"])) > 0.99
    assert abs(float(_first(sample.vars["Correlation"]))) < 1e-6
    assert float(_first(sample.vars["EdgeContrast"])) > 0.3
    assert sample.vars["OnlyLeft"] == 1 and sample.vars["Overlap"] == 0 and sample.vars["OnlyRight"] == 1
    assert sample.vars["IndexIn"] == 1 and sample.vars["IndexOut"] == 0 and sample.vars["IndexSecond"] == 2
    assert sample.vars["FarCount"] == 0 and sample.vars["CloseCount"] == 1
    assert abs(float(sample.vars["StraightMax"])) < 0.05
    assert abs(float(sample.vars["TurnMax"]) - 1.5707963) < 0.05
    assert int(_first(sample.vars["OpenFlag"])) == 0
    assert int(_first(sample.vars["ClosedFlag"])) == 1
    assert next(param for param in REGISTRY["check_difference"].control_in if param.name == "Mode").choices == ["'abs'", "'light'", "'dark'"]
    assert "'all'" in next(param for param in REGISTRY["dots_image"].control_in if param.name == "FilterType").choices
    assert "'90'" in next(param for param in REGISTRY["cooc_feature_image"].control_in if param.name == "Direction").choices
    assert "'rel'" in next(param for param in REGISTRY["get_contour_angle_xld"].control_in if param.name == "Mode").choices


def test_hdevelop_batch14():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_empty_region_edge_plateau_flow.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["EmptyCount"] == 0
    assert sample.vars["PickedCount"] == 1
    assert sample.vars["PickedOld"] == 0 and sample.vars["PickedNew"] == 1
    assert sample.vars["FlatIn"] == 1 and sample.vars["FlatEdge"] == 0 and sample.vars["FlatBack"] == 1
    assert int(sample.vars["EdgePeak"]) > 20
    assert int(sample.vars["FlatAmp"]) == 0
    dirs = [float(item) for item in sample.vars["EdgeDirs"]]
    assert any(abs(item - 128) <= 8 for item in dirs)
    assert int(sample.vars["ClosedOn"]) == 1 and sample.vars["ClosedOff"] == 0
    assert int(sample.vars["LowCount"]) > int(sample.vars["HighCount"])
    assert int(sample.vars["HighCount"]) == 1
    assert int(sample.vars["CornerCount"]) >= 2
    assert 6 <= int(sample.vars["CornerTop"]) <= 20
    assert 130 <= int(sample.vars["FlowCol"]) <= 140
    assert 120 <= int(sample.vars["FlowRow"]) <= 136
    assert next(param for param in REGISTRY["edges_image"].control_in if param.name == "Filter").choices == ["'canny'", "'sobel'"]
    assert "'none'" in next(param for param in REGISTRY["edges_image"].control_in if param.name == "NMS").choices


def test_hdevelop_batch15():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_distance_transform_highpass_corner.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["NearIn"] == 0
    assert sample.vars["NearOut"] == 10
    assert sample.vars["CityOut"] == 10
    assert sample.vars["ChessOut"] == 10
    assert sample.vars["InsideDist"] == 11
    assert sample.vars["InsideOut"] == 0
    assert sample.vars["BinIn"] == 200 and sample.vars["BinOut"] == 40
    assert sample.vars["MeanLeft"] == 80 and sample.vars["MeanRight"] == 180 and sample.vars["MeanOut"] == 0
    assert sample.vars["HighZero"] == 128
    assert sample.vars["HighPeak"] == 255 and sample.vars["HighBack"] == 128
    assert sample.vars["MedGray"] == 10
    assert int(sample.vars["SpikeLeft"]) < 75
    assert int(sample.vars["EdgeKeep"]) > 240
    assert 36 <= int(sample.vars["Quiet"]) <= 50
    assert sample.vars["PeakCount"] == 1
    assert abs(float(_first(sample.vars["PeakRows"])) - 30) < 0.2
    assert 40.05 < float(_first(sample.vars["PeakCols"])) < 41.0
    assert int(sample.vars["FastCount"]) >= 8
    assert 18 <= int(sample.vars["FastTop"]) <= 24
    assert next(param for param in REGISTRY["distance_transform"].control_in if param.name == "Metric").choices == ["'euclidean'", "'city-block'", "'chessboard'"]
    assert "'false'" in next(param for param in REGISTRY["distance_transform"].control_in if param.name == "Foreground").choices
    assert "'parabolic'" in next(param for param in REGISTRY["anisotropic_diffusion"].control_in if param.name == "Mode").choices
    assert "'gauss'" in next(param for param in REGISTRY["local_max_sub_pix"].control_in if param.name == "Filter").choices
    assert "'cyclic'" in next(param for param in REGISTRY["median_rect"].control_in if param.name == "Margin").choices


def test_hdevelop_batch16():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_eliminate_runs_plane_entropy.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["GoneCount"] == 0
    assert int(_first(sample.vars["WideArea"])) == 231
    assert abs(float(_first(sample.vars["MRow"])) - 14.5) < 1e-6
    assert abs(float(_first(sample.vars["MCol"])) - 14.5) < 1e-6
    assert abs(float(_first(sample.vars["Alpha"]))) < 1e-6
    assert abs(float(_first(sample.vars["Beta"])) - 1) < 1e-6
    assert abs(float(_first(sample.vars["Mean"])) - 14.5) < 1e-6
    assert abs(float(_first(sample.vars["Deviation"]))) < 1e-4
    assert float(_first(sample.vars["BumpDev"])) > 5
    assert abs(float(_first(sample.vars["GrayArea"])) - 1450) < 1e-3
    assert abs(float(_first(sample.vars["GrayRow"])) - 14.5) < 1e-4
    assert abs(float(_first(sample.vars["GrayCol"])) - 15.06897) < 1e-3
    assert abs(float(_first(sample.vars["Entropy"])) - 1) < 1e-6
    assert sample.vars["BandZero"] == 128
    assert sample.vars["DotPeak"] == 255 and sample.vars["DotBack"] == 128
    assert sample.vars["WideCenter"] == 128
    assert sample.vars["ValleyCount"] == 1
    assert abs(float(_first(sample.vars["ValleyRows"])) - 30) < 0.2
    assert 40.05 < float(_first(sample.vars["ValleyCols"])) < 41.0
    assert abs(float(sample.vars["Sx"]) - 2) < 1e-6
    assert abs(float(sample.vars["Sy"]) - 3) < 1e-6
    assert abs(float(sample.vars["Phi"]) - 0.5) < 1e-6
    assert abs(float(sample.vars["Theta"])) < 1e-6
    assert sample.vars["Tx"] == 4 and sample.vars["Ty"] == -2
    assert "'e'" in next(param for param in REGISTRY["entropy_gray"].control_in if param.name == "Base").choices
    assert "'gauss'" in next(param for param in REGISTRY["local_min_sub_pix"].control_in if param.name == "Filter").choices


def test_hdevelop_batch17():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_sqrt_image_log_saddle_trim.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["RootGray"] == 9 and sample.vars["RootZero"] == 0
    assert sample.vars["LogMid"] == 128 and sample.vars["LogHigh"] == 255 and sample.vars["LogZero"] == 0
    assert sample.vars["LogRaw"] == 6
    assert sample.vars["StillGray"] == 100
    assert int(sample.vars["SharpRight"]) - int(sample.vars["SharpLeft"]) > int(sample.vars["SoftRight"]) - int(sample.vars["SoftLeft"])
    assert int(sample.vars["SaddleCount"]) >= 1
    assert abs(float(_first(sample.vars["SaddleRows"])) - 30) < 0.4
    assert abs(float(_first(sample.vars["SaddleCols"])) - 40) < 0.4
    noise = [int(item) for item in sample.vars["NoiseVals"]]
    again = [int(item) for item in sample.vars["NoiseAgain"]]
    assert noise == again
    assert sample.vars["QuietGray"] == 100
    assert any(item != 100 for item in noise)
    assert all(80 <= item <= 120 for item in noise)
    assert 4 <= int(sample.vars["PolyCount"]) <= 8
    assert int(sample.vars["PolyTop"]) <= 12 and int(sample.vars["PolyBottom"]) >= 38
    assert int(_first(sample.vars["PruneArea"])) == 15
    assert sample.vars["PartCount"] == 1 and sample.vars["HoleCount"] == 1
    assert int(_first(sample.vars["HoleArea"])) == 121
    assert sample.vars["PairCount"] == 2 and sample.vars["PairHoleCount"] == 0
    assert "'natural'" in next(param for param in REGISTRY["log_image"].control_in if param.name == "Mode").choices
    assert "'canny'" in next(param for param in REGISTRY["shock_filter"].control_in if param.name == "Mode").choices
    assert "'gaussian'" in next(param for param in REGISTRY["add_noise_white"].control_in if param.name == "Type").choices


def test_hdevelop_batch18():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gray_features_distance_auto_threshold.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    gray = [float(item) for item in sample.vars["GrayVals"]]
    assert gray == [10, 19, 14.5]
    assert abs(float(_first(sample.vars["SideLength"])) - 80) < 1e-6
    assert abs(float(_first(sample.vars["SquareArea"])) - 400) < 1e-6
    assert sample.vars["PickCount"] == 1 and sample.vars["MissCount"] == 0
    assert abs(float(_first(sample.vars["LineDist"])) - 10) < 1e-6
    assert abs(float(_first(sample.vars["SegDist"])) - 10) < 1e-6
    assert abs(float(_first(sample.vars["EndDist"])) - (10 ** 2 + 8 ** 2) ** 0.5) < 1e-6
    assert abs(float(_first(sample.vars["ContDist"])) - 10) < 1e-6
    assert sample.vars["ModeCount"] == 2
    assert sample.vars["DarkIndex"] == 1 and sample.vars["BrightIndex"] == 2
    assert sample.vars["DotIn"] == 1 and sample.vars["BlockIn"] == 0
    assert "'gray_median'" in next(param for param in REGISTRY["gray_features"].control_in if param.name == "Features").choices
    assert "'circularity'" in next(param for param in REGISTRY["xld_features"].control_in if param.name == "Features").choices


def test_hdevelop_batch19():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gamma_image_zero_cross_sector.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["GammaGray"] == 64
    assert sample.vars["BinFlatGray"] == 100
    assert sample.vars["BinCenter"] == 64
    assert sample.vars["BinFar"] == 0
    assert sample.vars["IsoGray"] == 100
    assert 10 < int(sample.vars["IsoCenter"]) < 150
    assert sample.vars["IsoFar"] == 0
    assert sample.vars["PaintIn"] == 200 and sample.vars["PaintOut"] == 10
    assert sample.vars["OnLine"] == 1 and sample.vars["OffLine"] == 0
    assert int(_first(sample.vars["CrossArea"])) == 60
    assert sample.vars["LobeCount"] == 2
    assert sample.vars["HotIndex"] == 1 and sample.vars["ColdIndex"] == 2
    assert sample.vars["WarmIndex"] == 0 and sample.vars["SpeckIndex"] == 0
    assert sample.vars["RightIn"] == 1 and sample.vars["DownIn"] == 1
    assert sample.vars["UpOut"] == 0 and sample.vars["LeftOut"] == 0
    assert sample.vars["IsSubset"] == 1 and sample.vars["NotSubset"] == 0
    assert sample.vars["IsEqual"] == 1 and sample.vars["NotEqual"] == 0
    assert "'false'" in next(param for param in REGISTRY["gamma_image"].control_in if param.name == "Encode").choices
    assert "5" in next(param for param in REGISTRY["binomial_filter"].control_in if param.name == "MaskWidth").choices


def test_hdevelop_batch20():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_exp_image_polar_tophat.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["ExpAtZero"] == 0
    assert sample.vars["ExpAtMid"] == 15
    assert sample.vars["ExpAtHigh"] == 255
    assert sample.vars["ExpNatural"] == 1
    assert sample.vars["ExpAtFive"] == 148
    assert sample.vars["PolarIn"] == 1 and sample.vars["PolarOut"] == 0
    rows = [float(item) for item in sample.vars["PolarRows"]]
    cols = [float(item) for item in sample.vars["PolarCols"]]
    assert abs(rows[0] - 10) < 1e-4 and abs(cols[0]) < 1e-4
    assert abs(rows[1] - 10) < 1e-4 and abs(cols[1] - 10) < 1e-4
    assert abs(rows[2] - 10) < 1e-4 and abs(cols[2] - 30) < 1e-4
    assert sample.vars["WideIn"] == 1 and sample.vars["ThinIn"] == 1
    assert sample.vars["UpOut"] == 0 and sample.vars["LeftOut"] == 0 and sample.vars["FarDown"] == 0
    assert sample.vars["SpeckKept"] == 1 and sample.vars["BlockGone"] == 0
    assert sample.vars["GapFilled"] == 1 and sample.vars["BodyKept"] == 0
    assert sample.vars["LabelA"] == 1 and sample.vars["LabelB"] == 2 and sample.vars["LabelBack"] == 0
    assert sample.vars["LabelCount"] == 2
    assert sample.vars["IndexA"] == 1 and sample.vars["IndexB"] == 2
    assert abs(float(_first(sample.vars["LineDist"])) - 10) < 1e-6
    assert abs(float(_first(sample.vars["EndDist"])) - (10 ** 2 + 8 ** 2) ** 0.5) < 1e-6
    assert abs(float(_first(sample.vars["CrossDist"]))) < 1e-6
    assert "'natural'" in next(param for param in REGISTRY["exp_image"].control_in if param.name == "Mode").choices


def test_hdevelop_batch21():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_reduce_domain_paint_corner_intersection.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["InDomain"] == 200 and sample.vars["OutDomain"] == 10
    assert sample.vars["FlatResp"] == 128 and sample.vars["InsideResp"] == 128
    assert int(sample.vars["CornerResp"]) > 128
    assert int(sample.vars["EdgeResp"]) < 128
    assert int(sample.vars["CornerResp"]) > int(sample.vars["EdgeResp"])
    assert sample.vars["RankFlat"] == 100
    assert sample.vars["RankHole"] == 0 and sample.vars["RankNeighbor"] == 255 and sample.vars["RankFar"] == 0
    assert abs(float(_first(sample.vars["HitRow"])) - 10) < 1e-6
    assert abs(float(_first(sample.vars["HitCol"])) - 20) < 1e-6
    assert abs(float(_first(sample.vars["SegRow"])) - 10) < 1e-6
    assert abs(float(_first(sample.vars["SegCol"])) - 20) < 1e-6
    assert sample.vars["MissCount"] == 0
    assert abs(float(_first(sample.vars["BoxArea"])) - 400) < 1e-6
    assert abs(float(_first(sample.vars["BoxLength"])) - 80) < 1e-6
    assert sample.vars["PairCount"] == 1 and sample.vars["BackCount"] == 255 and sample.vars["EmptyCount"] == 0
    assert "'margin'" in next(param for param in REGISTRY["overpaint_region"].control_in if param.name == "Type").choices
    assert "5" in next(param for param in REGISTRY["corner_response"].control_in if param.name == "Size").choices
    assert "'outer_circle'" in next(param for param in REGISTRY["shape_trans_xld"].control_in if param.name == "Type").choices
    assert "3" in next(param for param in REGISTRY["rank_image"].control_in if param.name == "Radius").choices


def test_hdevelop_batch22():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_abs_image_direction_crossing.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["AbsAtZero"] == 128 and sample.vars["AbsAtMid"] == 0 and sample.vars["AbsAtLow"] == 28
    assert sample.vars["RightDir"] == 128 and sample.vars["DownDir"] == 191
    assert abs(float(sample.vars["MidGray"]) - 10.5) < 1e-6
    assert sample.vars["NearGray"] == 11
    assert sample.vars["InPaint"] == 200 and sample.vars["OutPaint"] == 10
    assert abs(float(sample.vars["CrossRow"])) < 1e-6 and abs(float(sample.vars["CrossCol"]) - 20) < 1e-6
    assert sample.vars["CrossHit"] == 1 and sample.vars["ParHit"] == 0
    assert abs(float(_first(sample.vars["MaxOn"]))) < 1e-4
    assert abs(float(_first(sample.vars["MaxOff"])) - 10) < 1e-6
    flags = [int(item) for item in sample.vars["SelfFlags"]]
    assert flags == [0, 1]
    assert sample.vars["CritCount"] == 4 and sample.vars["LineCount"] == 0
    rows = [float(item) for item in sample.vars["CritRow"]]
    cols = [float(item) for item in sample.vars["CritCol"]]
    assert min(rows) == 10 and max(rows) == 30 and min(cols) == 10 and max(cols) == 30
    assert "'nearest'" in next(param for param in REGISTRY["get_grayval_interpolated"].control_in if param.name == "Interpolation").choices


def test_hdevelop_batch23():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_prewitt_amp_polar_split.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["PrewittEdge"] == 40 and sample.vars["PrewittFlat"] == 0
    assert sample.vars["KirschEdge"] == 40 and sample.vars["KirschFlat"] == 0
    assert sample.vars["RobertsSqrtEdge"] == 57 and sample.vars["RobertsSumEdge"] == 80
    assert abs(float(sample.vars["HitRow"]) - 10) < 1e-6 and abs(float(sample.vars["HitCol"]) - 20) < 1e-6
    assert sample.vars["HitFlag"] == 1 and sample.vars["MissFlag"] == 0
    assert abs(float(_first(sample.vars["DistOn"]))) < 1e-6
    assert abs(float(_first(sample.vars["DistCenter"])) - 5) < 1e-6
    assert abs(float(_first(sample.vars["DistOut"])) - 6) < 1e-6
    assert sample.vars["PieceCount"] == 4 and sample.vars["LinePieces"] == 1
    lengths = [float(item) for item in sample.vars["PieceLen"]]
    assert len(lengths) == 4 and all(abs(item - 20) < 1e-4 for item in lengths)
    assert sample.vars["BackInside"] == 1 and sample.vars["BackCenter"] == 0
    assert sample.vars["BackGray"] == 255 and sample.vars["CenterGray"] == 0
    assert "'sum_abs'" in next(param for param in REGISTRY["roberts"].control_in if param.name == "FilterType").choices
    assert "'nearest_neighbor'" in next(param for param in REGISTRY["polar_trans_image_inv"].control_in if param.name == "Interpolation").choices


def test_hdevelop_batch24():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_frei_amp_robinson_azimuth_circle.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["FreiEdge"] == 49 and sample.vars["FreiFlat"] == 0
    assert sample.vars["RobinsonEdge"] == 160 and sample.vars["RobinsonFlat"] == 0
    assert sample.vars["RobinsonRight"] == 128
    assert sample.vars["LeftCount"] == 1 and sample.vars["AboveCount"] == 1
    assert sample.vars["BelowCount"] == 1 and sample.vars["RightCount"] == 0
    assert sample.vars["OnLeft"] == 1 and sample.vars["OnMark"] == 0
    rows = [float(item) for item in sample.vars["BackRow"]]
    cols = [float(item) for item in sample.vars["BackCol"]]
    assert all(abs(got - want) < 1e-4 for got, want in zip(rows, [30, 40, 20]))
    assert all(abs(got - want) < 1e-4 for got, want in zip(cols, [50, 40, 40]))
    hit_cols = sorted(float(item) for item in sample.vars["CircleCol"])
    hit_rows = [float(item) for item in sample.vars["CircleRow"]]
    assert len(hit_cols) == 2 and abs(hit_cols[0] - 30) < 1e-4 and abs(hit_cols[1] - 50) < 1e-4
    assert all(abs(item - 30) < 1e-4 for item in hit_rows)
    assert sample.vars["MissCount"] == 0
    assert sample.vars["BoxCount"] == 5 and sample.vars["SpokeCount"] == 3
    assert sample.vars["QuietGray"] == 100
    assert float(sample.vars["DiffMin"]) == 0 and float(sample.vars["DiffMax"]) == 0
    assert int(_first(sample.vars["SaltArea"])) == 2400 and int(_first(sample.vars["PepperArea"])) == 2400
    assert "'above'" in next(param for param in REGISTRY["select_region_spatial"].control_in if param.name == "Direction").choices


def test_hdevelop_batch25():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_bit_not_lut_bits_pyramid.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["NotGray"] == 245
    assert sample.vars["AndGray"] == 0 and sample.vars["OrGray"] == 255 and sample.vars["XorGray"] == 255
    assert sample.vars["ShiftGray"] == 8 and sample.vars["UnshiftGray"] == 4
    assert sample.vars["LutGray"] == 245
    assert sample.vars["PlainGray"] == 6 and sample.vars["MinGray"] == 6 and sample.vars["MaxGray"] == 7
    assert sample.vars["ConstGray"] == 10
    assert sample.vars["PyrWidth"] == 40 and sample.vars["PyrHeight"] == 30
    modes = next(param for param in REGISTRY["gen_gauss_pyramid"].control_in if param.name == "Mode").choices
    assert "'none'" in modes and "'min'" in modes and "'max'" in modes
    assert "3" in next(param for param in REGISTRY["bit_lshift"].control_in if param.name == "Shift").choices


def test_hdevelop_batch26():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_measure_rectangle2_profile_funct1d.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["ProfileCount"] == 41
    assert abs(float(sample.vars["ProfileFirst"]) - 30) < 1e-6
    assert abs(float(sample.vars["ProfileMid"]) - 40) < 1e-6
    assert abs(float(sample.vars["ProfileLast"]) - 50) < 1e-6
    assert abs(float(_first(sample.vars["ThreshRow"])) - 30) < 1e-4
    assert abs(float(_first(sample.vars["ThreshCol"])) - 40) < 1e-4
    assert abs(float(_first(sample.vars["ThreshDist"])) - 10) < 1e-4
    assert abs(float(sample.vars["MovedFirst"]) - 40) < 1e-6
    assert abs(float(sample.vars["MovedLast"]) - 60) < 1e-6
    assert sample.vars["ArcCount"] == 63
    assert abs(float(sample.vars["ArcFirst"]) - 50) < 1e-4
    assert abs(float(sample.vars["ArcMid"]) - 40) < 1e-4
    assert abs(float(sample.vars["ArcLast"]) - 30) < 1e-4
    assert abs(float(sample.vars["MidY"]) - 20) < 1e-6
    assert abs(float(sample.vars["LowY"]) - 10) < 1e-6
    assert abs(float(sample.vars["HighY"])) < 1e-6
    assert abs(float(sample.vars["MirrorY"]) - 20) < 1e-6
    assert abs(float(sample.vars["S1"]) - 10) < 1e-6
    assert abs(float(sample.vars["S2"]) - 20) < 1e-6
    assert abs(float(sample.vars["S3"]) - 30) < 1e-6
    assert abs(float(sample.vars["ZeroX"]) - 1) < 1e-6
    assert "'first'" in next(param for param in REGISTRY["measure_thresh"].control_in if param.name == "Select").choices
    assert "'nearest_neighbor'" in next(param for param in REGISTRY["gen_measure_arc"].control_in if param.name == "Interpolation").choices
    assert "'zero'" in next(param for param in REGISTRY["get_y_value_funct_1d"].control_in if param.name == "Border").choices


def test_hdevelop_batch27():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_create_funct_1d_pairs_integral_derivative_inverse.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Area"]) - 40) < 1e-6
    assert abs(float(sample.vars["Flat2"])) < 1e-6
    assert abs(float(sample.vars["InvY"]) - 1) < 1e-6
    assert abs(float(sample.vars["MixedY"]) - 20) < 1e-6
    assert abs(float(sample.vars["ArrayX0"])) < 1e-6
    assert abs(float(sample.vars["ArrayY1"]) - 8) < 1e-6
    assert abs(float(sample.vars["ArrayMid"]) - 8) < 1e-6
    assert abs(float(sample.vars["Slope0"]) - 2) < 1e-6
    assert abs(float(sample.vars["Slope1"]) - 1) < 1e-6
    assert abs(float(sample.vars["Slope2"])) < 1e-6
    assert abs(float(sample.vars["SecondMid"]) + 1) < 1e-6
    assert abs(float(sample.vars["MinAt"]) - 2) < 1e-6
    assert abs(float(sample.vars["MaxAt"]) - 1) < 1e-6
    assert abs(float(sample.vars["Abs0"]) - 1) < 1e-6
    assert abs(float(sample.vars["AbsMid"])) < 1e-6
    assert abs(float(sample.vars["Abs2"]) - 1) < 1e-6
    assert "'second'" in next(param for param in REGISTRY["derivate_funct_1d"].control_in if param.name == "Mode").choices
    assert "'error'" in next(param for param in REGISTRY["compose_funct_1d"].control_in if param.name == "Border").choices


def test_hdevelop_batch28():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_fft_image_shift_phase_correlation.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["DcReal"]) - 48000) < 1e-4
    assert abs(float(sample.vars["DcImag"])) < 1e-4
    assert abs(float(sample.vars["AcReal"])) < 1e-4
    assert abs(float(sample.vars["AcImag"])) < 1e-4
    assert abs(float(sample.vars["ShiftReal"]) - 48000) < 1e-4
    assert abs(float(sample.vars["ShiftImag"])) < 1e-4
    assert sample.vars["BackGray"] == 10
    assert sample.vars["HalfGray"] == 5
    assert abs(float(sample.vars["ProdReal"]) - 48000) < 1e-4
    assert abs(float(sample.vars["ProdImag"])) < 1e-4
    assert abs(float(sample.vars["MidReal"]) - 250) < 1e-4
    assert abs(float(sample.vars["MidZero"])) < 1e-4
    assert abs(float(sample.vars["CornerReal"]) - 250) < 1e-4
    assert abs(float(sample.vars["CornerOff"])) < 1e-4
    assert sample.vars["PeakRow"] == 5 and sample.vars["PeakCol"] == 2
    assert abs(float(sample.vars["PeakScore"]) - 1) < 1e-4
    assert "'corner'" in next(param for param in REGISTRY["fftshift"].control_in if param.name == "Mode").choices


def test_hdevelop_batch29():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_fft_image_spectrum_mask_convolve_energy.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["ConjReal"]) - 48000) < 1e-4
    assert abs(float(sample.vars["ConjImag"])) < 1e-4
    assert sample.vars["SumGray"] == 20
    assert sample.vars["ZeroGray"] == 0
    assert sample.vars["DcGray"] == 10
    assert sample.vars["HighGray"] == 0
    assert sample.vars["RealGray"] == 10
    assert sample.vars["ImagGray"] == 0
    assert abs(float(sample.vars["Energy"]) - 480000) < 1e-3
    assert sample.vars["QuotGray"] == 10
    assert sample.vars["RealPeak"] == 255 and sample.vars["RealFlat"] == 0
    assert sample.vars["LogPeak"] == 255 and sample.vars["LogFlat"] == 0
    assert sample.vars["MovedGray"] == 255 and sample.vars["OldGray"] == 0
    modes = next(param for param in REGISTRY["mask_fft"].control_in if param.name == "Mode").choices
    assert "'no_dc'" in modes and "'imag'" in modes
    views = next(param for param in REGISTRY["complex_to_image"].control_in if param.name == "Mode").choices
    assert "'log'" in views and "'abs'" in views


def test_hdevelop_batch30():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_fft_image_correlation_shift_pad.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["LowGray"] == 40 and sample.vars["LowEdge"] == 40
    assert sample.vars["HighDark"] == 0 and sample.vars["HighBright"] == 40
    assert abs(float(sample.vars["DcEnergy"]) - 7489200) < 1e-2
    assert sample.vars["RollGray"] == 255 and sample.vars["RollOld"] == 0
    assert sample.vars["PhaseGray"] == 255 and sample.vars["PhaseOld"] == 0
    assert abs(float(sample.vars["UnitReal"]) - 1) < 1e-4
    assert abs(float(sample.vars["UnitImag"])) < 1e-4
    assert abs(float(sample.vars["UnitMag"]) - 1) < 1e-4
    assert sample.vars["MatchRow"] == 8 and sample.vars["MatchCol"] == 9
    assert abs(float(sample.vars["MatchScore"]) - 65025) < 1e-3
    assert sample.vars["ScorePeak"] == 255 and sample.vars["ScoreOff"] == 0
    assert sample.vars["ConstCorner"] == 0 and sample.vars["ConstMoved"] == 200
    assert sample.vars["ConstWidth"] == 4 and sample.vars["ConstHeight"] == 5
    assert sample.vars["MirrorCorner"] == 200 and sample.vars["MirrorMoved"] == 200
    assert sample.vars["CycOld"] == 200 and sample.vars["CycWrap"] == 200
    assert sample.vars["CycWidth"] == 5 and sample.vars["CycHeight"] == 4
    assert "'high'" in next(param for param in REGISTRY["freq_filter"].control_in if param.name == "Mode").choices
    pads = next(param for param in REGISTRY["pad_image"].control_in if param.name == "Mode").choices
    assert "'mirror'" in pads and "'cyclic'" in pads


def test_hdevelop_batch31():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_object_model_3d_from_points_plane_box.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Nx"])) < 1e-6
    assert abs(float(sample.vars["Ny"])) < 1e-6
    assert abs(float(sample.vars["Nz"]) - 1) < 1e-6
    assert abs(float(sample.vars["PlaneDist"]) - 3) < 1e-6
    assert abs(float(sample.vars["OnDist"])) < 1e-6
    assert sample.vars["PlaneCount"] == 4
    assert abs(float(sample.vars["OffDist"]) + 2) < 1e-6
    assert abs(float(sample.vars["NearDist"]) - 4) < 1e-6
    assert abs(float(sample.vars["MovedX0"]) - 10) < 1e-6
    assert abs(float(sample.vars["MovedX1"]) - 13) < 1e-6
    assert abs(float(sample.vars["MovedY0"]) + 1) < 1e-6
    assert abs(float(sample.vars["MovedZ0"]) - 2) < 1e-6
    assert abs(float(sample.vars["FlipX0"])) < 1e-6
    assert abs(float(sample.vars["FlipX1"]) + 3) < 1e-6
    assert abs(float(sample.vars["KeptX0"])) < 1e-6
    assert abs(float(sample.vars["KeptX1"]) - 2) < 1e-6
    assert sample.vars["KeptCount"] == 2
    assert sample.vars["BoxCount"] == 8
    assert abs(float(sample.vars["MinX"])) < 1e-6 and abs(float(sample.vars["MinY"]) - 1) < 1e-6
    assert abs(float(sample.vars["MinZ"]) - 2) < 1e-6
    assert abs(float(sample.vars["MaxX"]) - 4) < 1e-6
    assert abs(float(sample.vars["MaxY"]) - 5) < 1e-6 and abs(float(sample.vars["MaxZ"]) - 6) < 1e-6
    axes = next(param for param in REGISTRY["mirror_object_model_3d"].control_in if param.name == "Axis").choices
    assert "'y'" in axes and "'z'" in axes


def test_hdevelop_batch32():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_object_model_3d_from_points_scale_project_axis.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["ScaleX"]) - 6) < 1e-6
    assert abs(float(sample.vars["HeldX"]) - 2) < 1e-6
    assert abs(float(sample.vars["TurnX"])) < 1e-6
    assert abs(float(sample.vars["TurnY"]) - 1) < 1e-6
    assert abs(float(sample.vars["FlatX"]) - 1) < 1e-6
    assert abs(float(sample.vars["FlatY"]) - 2) < 1e-6
    assert abs(float(sample.vars["FlatZ"])) < 1e-6
    assert sample.vars["KeptCount"] == 1
    assert abs(float(sample.vars["KeptZ"]) - 3) < 1e-6
    assert abs(float(sample.vars["FarDist"]) - 5) < 1e-6
    assert abs(float(sample.vars["RestX0"])) < 1e-6
    assert abs(float(sample.vars["RestX1"]) - 2) < 1e-6
    assert abs(float(sample.vars["OrderX0"]) - 2) < 1e-6
    assert abs(float(sample.vars["OrderX1"]) - 1) < 1e-6
    assert abs(float(sample.vars["Ax"]) - 1) < 1e-6
    assert abs(float(sample.vars["Ay"])) < 1e-6
    assert abs(float(sample.vars["Az"])) < 1e-6
    assert "'xy'" in next(param for param in REGISTRY["project_object_model_3d"].control_in if param.name == "Plane").choices
    orders = next(param for param in REGISTRY["sort_object_model_3d"].control_in if param.name == "Order").choices
    assert "'descending'" in orders


def test_hdevelop_batch33():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_object_model_3d_from_points_intersect_grid_axis.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["CommonX"]) - 1) < 1e-6
    assert abs(float(sample.vars["CommonY"])) < 1e-6
    assert abs(float(sample.vars["CommonZ"])) < 1e-6
    assert abs(float(sample.vars["MidX0"]) + 1) < 1e-6
    assert abs(float(sample.vars["MidX1"]) - 1) < 1e-6
    assert abs(float(sample.vars["CutX"]) - 2) < 1e-6
    assert abs(float(sample.vars["SparseX0"])) < 1e-6
    assert abs(float(sample.vars["SparseX1"]) - 3) < 1e-6
    assert abs(float(sample.vars["CellX0"]) - 1) < 1e-6
    assert abs(float(sample.vars["CellX1"]) - 3) < 1e-6
    assert abs(float(sample.vars["Gap"]) - 9) < 1e-6
    assert sample.vars["SheetCount"] == 4
    assert abs(float(sample.vars["SheetX0"])) < 1e-6
    assert abs(float(sample.vars["SheetX1"]) - 1) < 1e-6
    assert abs(float(sample.vars["SheetY2"]) - 1) < 1e-6
    assert abs(float(sample.vars["SheetZ0"]) - 3) < 1e-6
    assert abs(float(sample.vars["Spread1"]) - 8) < 1e-6
    assert abs(float(sample.vars["Spread2"])) < 1e-6
    assert abs(float(sample.vars["Spread3"])) < 1e-6
    modes = next(param for param in REGISTRY["voxel_object_model_3d"].control_in if param.name == "Mode").choices
    assert "'point'" in modes and "'center'" in modes


def test_hdevelop_batch34():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_object_model_3d_from_region_cloud_aligned_to_image.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["RegionCount"] == 9
    assert abs(float(sample.vars["Rx0"]) - 20) < 1e-6
    assert abs(float(sample.vars["Ry0"]) - 10) < 1e-6
    assert abs(float(sample.vars["Rz0"])) < 1e-6
    assert abs(float(sample.vars["BackArea"]) - 9) < 1e-6
    assert abs(float(sample.vars["Xx0"]) - 20) < 1e-6
    assert abs(float(sample.vars["Xy0"]) - 10) < 1e-6
    assert abs(float(sample.vars["Xx1"]) - 40) < 1e-6
    assert abs(float(sample.vars["Xy1"]) - 30) < 1e-6
    assert sample.vars["ImageCount"] == 4
    assert abs(float(sample.vars["PixZ1"]) - 40) < 1e-6
    assert abs(float(sample.vars["PixY2"]) - 1) < 1e-6
    assert abs(float(sample.vars["PixZ2"]) - 10) < 1e-6
    assert sample.vars["PaintZ"] == 200 and sample.vars["PaintGray"] == 80
    assert abs(float(sample.vars["ReadZ"]) - 200) < 1e-6
    assert sample.vars["InsideCount"] == 9
    assert sample.vars["OutsideCount"] == 1 and sample.vars["MissCount"] == 0
    assert abs(float(sample.vars["LiftX"]) - 1) < 1e-6
    assert abs(float(sample.vars["LiftY"])) < 1e-6
    assert abs(float(sample.vars["LiftZ"]) - 40) < 1e-6
    assert "'z'" in next(param for param in REGISTRY["paint_object_model_3d"].control_in if param.name == "Mode").choices
    sides = next(param for param in REGISTRY["select_object_model_3d_region"].control_in if param.name == "Mode").choices
    assert "'outside'" in sides


def test_hdevelop_batch35():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_fit_image_plane_height_normal_shade.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Alpha"])) < 1e-6
    assert abs(float(sample.vars["Beta"]) - 1) < 1e-6
    assert abs(float(sample.vars["Gamma"])) < 1e-6
    assert sample.vars["FlatCorner"] == 128 and sample.vars["FlatMid"] == 128
    assert abs(float(sample.vars["Deviation"])) < 1e-6
    assert abs(float(sample.vars["PlaneNx"]) + (2 ** -0.5)) < 1e-6
    assert abs(float(sample.vars["PlaneNy"])) < 1e-6
    assert abs(float(sample.vars["PlaneNz"]) - (2 ** -0.5)) < 1e-6
    assert sample.vars["SlopeMid"] == 45 and sample.vars["SlopeGentle"] == 27
    assert sample.vars["SlopeZero"] == 0
    assert sample.vars["NxMid"] == 38 and sample.vars["NyMid"] == 128 and sample.vars["NzMid"] == 218
    assert sample.vars["UpNx"] == 128 and sample.vars["UpNy"] == 128 and sample.vars["UpNz"] == 255
    assert sample.vars["SkyGray"] == 255
    assert sample.vars["ShadeMid"] == 180 and sample.vars["ShadeSide"] == 0
    assert sample.vars["ShadeFull"] == 218 and sample.vars["ShadeBack"] == 38
    assert sample.vars["AboveCount"] == 0 and sample.vars["ReliefMid"] == 87
    assert abs(float(sample.vars["BumpA"])) < 1e-6 and abs(float(sample.vars["BumpB"])) < 1e-6
    assert abs(float(sample.vars["BumpG"]) - 1) < 1e-6
    assert abs(float(sample.vars["BumpDev"]) - (8 ** 0.5)) < 1e-6
    assert sample.vars["BumpMid"] == 136 and sample.vars["BumpCorner"] == 127
    assert sample.vars["AboveN"] == 1 and sample.vars["AboveR"] == 1 and sample.vars["AboveC"] == 1
    assert sample.vars["BelowN"] == 8
    shades = next(param for param in REGISTRY["shade_height_field"].control_in if param.name == "Mode").choices
    assert "'clamp'" in shades and "'full'" in shades
    sides = next(param for param in REGISTRY["above_image_plane"].control_in if param.name == "Mode").choices
    assert "'below'" in sides


def test_hdevelop_batch36():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_sub_plane_image_reference_tilt_volume.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["LevelCorner"] == 128 and sample.vars["LevelMid"] == 128
    assert sample.vars["Shift0"] == 118 and sample.vars["Shift40"] == 158
    assert sample.vars["RaisedGray"] == 16 and sample.vars["ClipGray"] == 255
    assert sample.vars["Made0"] == 5 and sample.vars["Made3"] == 8
    assert abs(float(sample.vars["OneGray"]) - 8) < 1e-6
    assert abs(float(sample.vars["Many0"]) - 8) < 1e-6
    assert abs(float(sample.vars["Many1"]) - 9) < 1e-6
    assert abs(float(sample.vars["TiltX"]) - 45) < 1e-6 and abs(float(sample.vars["AzX"])) < 1e-6
    assert abs(float(sample.vars["TiltY"]) - 45) < 1e-6 and abs(float(sample.vars["AzY"]) - 90) < 1e-6
    assert abs(float(sample.vars["Tilt0"])) < 1e-6 and abs(float(sample.vars["Az0"])) < 1e-6
    assert abs(float(sample.vars["TiltBack"]) - 45) < 1e-6 and abs(float(sample.vars["AzBack"]) - 180) < 1e-6
    assert abs(float(sample.vars["TiltUp"]) - 45) < 1e-6 and abs(float(sample.vars["AzUp"]) - 270) < 1e-6
    assert abs(float(sample.vars["AzDiag"]) - 45) < 1e-6
    assert abs(float(sample.vars["TiltDiag"]) - float(np_degrees_diag())) < 1e-6
    assert sample.vars["BandN"] == 4800 and sample.vars["SwapN"] == 4800 and sample.vars["OffCount"] == 0
    assert abs(float(sample.vars["VolAbove"])) < 1e-6 and abs(float(sample.vars["VolSigned"])) < 1e-6
    assert sample.vars["HighRow"] == 0 and sample.vars["HighCol"] == 79 and abs(float(sample.vars["HighZ"]) - 79) < 1e-6
    assert sample.vars["LowRow"] == 0 and sample.vars["LowCol"] == 0 and abs(float(sample.vars["LowZ"])) < 1e-6
    assert sample.vars["KeptMid"] == 137 and sample.vars["KeptCorner"] == 128
    assert abs(float(sample.vars["BumpAbove"]) - 9) < 1e-6 and abs(float(sample.vars["BumpBelow"])) < 1e-6
    assert abs(float(sample.vars["FitAbove"]) - 8) < 1e-6 and abs(float(sample.vars["FitSigned"])) < 1e-6
    assert sample.vars["PeakRow"] == 1 and sample.vars["PeakCol"] == 1 and abs(float(sample.vars["PeakZ"]) - 9) < 1e-6
    modes = next(param for param in REGISTRY["volume_image_plane"].control_in if param.name == "Mode").choices
    assert "'signed'" in modes and "'below'" in modes
    poles = next(param for param in REGISTRY["extreme_image_plane"].control_in if param.name == "Mode").choices
    assert "'low'" in poles


def test_hdevelop_batch37():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_fit_plane_region_plane_inside_region.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Alpha"])) < 1e-6
    assert abs(float(sample.vars["Beta"]) - 1) < 1e-6
    assert abs(float(sample.vars["Gamma"])) < 1e-6
    assert abs(float(sample.vars["PartDev"])) < 1e-6
    assert abs(float(sample.vars["PartMean"])) < 1e-6
    assert abs(float(sample.vars["PartVol"])) < 1e-6
    assert sample.vars["HighRow"] == 10 and sample.vars["HighCol"] == 50
    assert abs(float(sample.vars["HighZ"]) - 50) < 1e-6
    assert sample.vars["LowRow"] == 10 and sample.vars["LowCol"] == 20
    assert abs(float(sample.vars["LowZ"]) - 20) < 1e-6
    assert sample.vars["BandN"] == 651 and sample.vars["SwapN"] == 651 and sample.vars["OffCount"] == 0
    assert sample.vars["InGray"] == 128 and sample.vars["OutGray"] == 40
    assert abs(float(sample.vars["KeepA"])) < 1e-6
    assert abs(float(sample.vars["KeepB"]) - 1) < 1e-6
    assert abs(float(sample.vars["KeepG"])) < 1e-6
    assert abs(float(sample.vars["BorderA"])) < 1e-6 and abs(float(sample.vars["BorderB"])) < 1e-6
    assert abs(float(sample.vars["BorderG"])) < 1e-6
    assert abs(float(sample.vars["FlatA"])) < 1e-6 and abs(float(sample.vars["FlatB"])) < 1e-6
    assert abs(float(sample.vars["FlatG"]) - 1) < 1e-6
    assert abs(float(sample.vars["InlierA"])) < 1e-6 and abs(float(sample.vars["InlierB"])) < 1e-6
    assert abs(float(sample.vars["InlierG"])) < 1e-6
    assert abs(float(sample.vars["DevAll"]) - 3) < 1e-6
    assert abs(float(sample.vars["MeanAll"]) - 1) < 1e-6
    assert abs(float(sample.vars["VolAll"]) - 9) < 1e-6
    modes = next(param for param in REGISTRY["volume_plane_region"].control_in if param.name == "Mode").choices
    assert "'signed'" in modes and "'below'" in modes
    poles = next(param for param in REGISTRY["extreme_plane_region"].control_in if param.name == "Mode").choices
    assert "'low'" in poles
    empty = run_source(
        "Vizion.gen_image_const (Image, 'byte', 10, 10)\n"
        "Vizion.gen_rectangle1 (Box, 1, 1, 4, 4)\n"
        "Vizion.difference (Box, Box, Empty)\n"
        "Vizion.fit_plane_region (Empty, Image, Alpha, Beta, Gamma)\n"
    )
    assert empty.error and "區域是空的" in empty.error
    line = run_source(
        "Vizion.gen_image_const (Image, 'byte', 10, 10)\n"
        "Vizion.gen_region_line (Line, 4, 0, 4, 9)\n"
        "Vizion.fit_plane_region (Line, Image, Alpha, Beta, Gamma)\n"
    )
    assert line.error and "排不成一個平面" in line.error


def test_hdevelop_batch38():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_fit_plane_objects_plane_per_object.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Beta1"]) - 1) < 1e-6
    assert abs(float(sample.vars["Beta3"])) < 1e-6
    assert abs(float(sample.vars["Gamma3"]) - 1) < 1e-6
    assert abs(float(sample.vars["Flat1"])) < 1e-6
    assert abs(float(sample.vars["Flat3"]) - (8 ** 0.5)) < 1e-6
    assert abs(float(sample.vars["Mean1"]) - 15) < 1e-6
    assert abs(float(sample.vars["Mean2"]) - 45) < 1e-6
    assert abs(float(sample.vars["Mean3"]) - 1) < 1e-6
    assert abs(float(sample.vars["Vol1"]) - 1815) < 1e-6
    assert abs(float(sample.vars["Vol2"]) - 5445) < 1e-6
    assert abs(float(sample.vars["Vol3"]) - 9) < 1e-6
    assert sample.vars["HighR1"] == 10 and sample.vars["HighC1"] == 20
    assert abs(float(sample.vars["HighZ1"]) - 20) < 1e-6
    assert sample.vars["HighC2"] == 50
    assert sample.vars["HighR3"] == 41 and sample.vars["HighC3"] == 41
    assert abs(float(sample.vars["HighZ3"]) - 9) < 1e-6
    assert sample.vars["LowC1"] == 10 and abs(float(sample.vars["LowZ1"]) - 10) < 1e-6
    assert sample.vars["WarpCount"] == 1 and sample.vars["WarpN"] == 9
    assert sample.vars["SwapCount"] == 1 and sample.vars["PlanarCount"] == 2
    assert abs(float(sample.vars["TallC"]) - 45) < 1e-6
    assert sample.vars["FlatGray"] == 128 and sample.vars["BumpGray"] == 136
    assert sample.vars["EdgeGray"] == 127 and sample.vars["OutGray"] == 5
    modes = next(param for param in REGISTRY["volume_plane_objects"].control_in if param.name == "Mode").choices
    assert "'signed'" in modes and "'below'" in modes
    poles = next(param for param in REGISTRY["extreme_plane_objects"].control_in if param.name == "Mode").choices
    assert "'low'" in poles
    line = run_source(
        "Vizion.gen_image_const (Image, 'byte', 10, 10)\n"
        "Vizion.gen_region_line (Line, 4, 0, 4, 9)\n"
        "Vizion.fit_plane_objects (Line, Image, Alpha, Beta, Gamma)\n"
    )
    assert line.error and "第 1 個物件" in line.error and "排不成一個平面" in line.error


def test_hdevelop_batch39():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_fit_quad_region_parabola_quadric.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["A"])) < 1e-6
    assert abs(float(sample.vars["B"]) - 1) < 1e-6
    assert abs(float(sample.vars["C"])) < 1e-6
    assert abs(float(sample.vars["D"])) < 1e-6
    assert abs(float(sample.vars["E"]) + 20) < 1e-6
    assert abs(float(sample.vars["F"]) - 100) < 1e-6
    assert abs(float(sample.vars["OB1"]) - 1) < 1e-6
    assert abs(float(sample.vars["OE1"]) + 20) < 1e-6
    assert abs(float(sample.vars["OF1"]) - 100) < 1e-6
    assert abs(float(sample.vars["QF1"])) < 1e-6
    assert abs(float(sample.vars["PF1"]) - (14 / 5) ** 0.5) < 1e-6
    assert abs(float(sample.vars["Dev0"])) < 1e-6
    assert abs(float(sample.vars["Dev1"]) - 1) < 1e-6
    assert abs(float(sample.vars["OneG"]) - 4) < 1e-6
    assert abs(float(sample.vars["Many0"])) < 1e-6
    assert abs(float(sample.vars["Many1"]) - 16) < 1e-6
    assert sample.vars["SubGray"] == 128 and sample.vars["LevelGray"] == 128
    assert sample.vars["OutGray"] == 30
    assert sample.vars["KeptCount"] == 1 and sample.vars["SwapCount"] == 1 and sample.vars["DropCount"] == 0
    assert len(REGISTRY["fit_quad_region"].control_out) == 6
    line = run_source(
        "Vizion.gen_image_const (Image, 'byte', 10, 10)\n"
        "Vizion.gen_region_line (Line, 4, 0, 4, 9)\n"
        "Vizion.fit_quad_region (Line, Image, A, B, C, D, E, F)\n"
    )
    assert line.error and "二次曲面" in line.error
    short = run_source(
        "Vizion.gen_image_const (Image, 'byte', 10, 10)\n"
        "Vizion.gen_rectangle1 (Tiny, 0, 0, 0, 1)\n"
        "Vizion.fit_quad_region (Tiny, Image, A, B, C, D, E, F)\n"
    )
    assert short.error and "至少要六個像素" in short.error


def test_hdevelop_batch40():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_add_plane_image_height_map_difference.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["UnionMean"]) - (90 / 27)) < 1e-6
    assert abs(float(sample.vars["UnionDev"]) - (1300 / 27) ** 0.5) < 1e-6
    assert abs(float(sample.vars["VolAbove"]) - 100) < 1e-6
    assert abs(float(sample.vars["VolBelow"]) - 10) < 1e-6
    assert abs(float(sample.vars["VolSigned"]) - 90) < 1e-6
    assert sample.vars["HighRow"] == 11 and sample.vars["HighCol"] == 11
    assert abs(float(sample.vars["HighDiff"]) - 20) < 1e-6
    assert sample.vars["LowRow"] == 51 and sample.vars["LowCol"] == 51
    assert abs(float(sample.vars["LowDiff"]) + 10) < 1e-6
    assert abs(float(sample.vars["Mean1"]) - (100 / 9)) < 1e-6
    assert abs(float(sample.vars["Mean2"])) < 1e-6
    assert abs(float(sample.vars["Mean3"]) + (10 / 9)) < 1e-6
    assert sample.vars["HighR1"] == 11 and sample.vars["HighC1"] == 11
    assert abs(float(sample.vars["HighD1"]) - 20) < 1e-6
    assert sample.vars["HighR2"] == 30 and sample.vars["HighC2"] == 40
    assert sample.vars["TallCount"] == 1 and sample.vars["SwapCount"] == 1
    assert abs(float(sample.vars["FlatC"]) - 41) < 1e-6
    assert sample.vars["PeakN"] == 1 and sample.vars["PeakR"] == 11 and sample.vars["PeakC"] == 11
    assert sample.vars["ViewHigh"] == 148 and sample.vars["ViewLow"] == 118
    modes = next(param for param in REGISTRY["volume_height_diff"].control_in if param.name == "Mode").choices
    assert "'signed'" in modes and "'below'" in modes
    poles = next(param for param in REGISTRY["extreme_height_diff"].control_in if param.name == "Mode").choices
    assert "'low'" in poles
    mismatch = run_source(
        "Vizion.gen_image_const (Wide, 'byte', 10, 10)\n"
        "Vizion.gen_image_const (Narrow, 'byte', 8, 8)\n"
        "Vizion.gen_rectangle1 (Box, 0, 0, 1, 1)\n"
        "Vizion.mean_height_diff (Box, Wide, Narrow, Mean)\n"
    )
    assert mismatch.error and "兩張影像的尺寸要相同" in mismatch.error


def test_hdevelop_batch41():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_sub_local_mean_local_mean_roughness.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["CenterGray"] == 155
    assert sample.vars["SideGray"] == 125
    assert sample.vars["FlatGray"] == 128
    assert abs(float(sample.vars["SpotRa"]) - (160 / 27)) < 1e-6
    assert abs(float(sample.vars["FlatRa"])) < 1e-6
    assert abs(float(sample.vars["SpotRq"]) - (800 / 9) ** 0.5) < 1e-6
    assert abs(float(sample.vars["Rp"]) - (80 / 3)) < 1e-6
    assert abs(float(sample.vars["Rv"]) + (10 / 3)) < 1e-6
    assert abs(float(sample.vars["Rz"]) - 30) < 1e-6
    assert abs(float(sample.vars["Ra1"])) < 1e-6
    assert abs(float(sample.vars["Ra2"]) - (160 / 27)) < 1e-6
    assert sample.vars["RoughCount"] == 1 and sample.vars["RoughN"] == 9
    assert sample.vars["SwapCount"] == 1 and sample.vars["SmoothCount"] == 1
    assert sample.vars["HighRow"] == 30 and sample.vars["HighCol"] == 40
    assert abs(float(sample.vars["HighRes"]) - (80 / 3)) < 1e-6
    assert sample.vars["PeakN"] == 1 and sample.vars["PeakR"] == 30 and sample.vars["PeakC"] == 40
    widths = next(param for param in REGISTRY["sub_local_mean"].control_in if param.name == "MaskWidth").choices
    assert "3" in widths and "9" in widths
    poles = next(param for param in REGISTRY["extreme_roughness"].control_in if param.name == "Mode").choices
    assert "'low'" in poles
    even = run_source(
        "Vizion.gen_image_const (Image, 'byte', 10, 10)\n"
        "Vizion.sub_local_mean (Image, Level, 4, 3)\n"
    )
    assert even.error and "奇數" in even.error


def test_hdevelop_batch42():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_sub_local_median_median_local_bump.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["CenterGray"] == 158
    assert sample.vars["SideGray"] == 128
    assert sample.vars["FlatGray"] == 128
    assert abs(float(sample.vars["SpotRa"]) - (10 / 3)) < 1e-6
    assert abs(float(sample.vars["FlatRa"])) < 1e-6
    assert abs(float(sample.vars["SpotRq"]) - 10) < 1e-6
    assert abs(float(sample.vars["Rp"]) - 30) < 1e-6
    assert abs(float(sample.vars["Rv"])) < 1e-6
    assert abs(float(sample.vars["Rz"]) - 30) < 1e-6
    assert abs(float(sample.vars["Ra1"])) < 1e-6
    assert abs(float(sample.vars["Ra2"]) - (10 / 3)) < 1e-6
    assert sample.vars["RoughCount"] == 1 and sample.vars["RoughN"] == 9
    assert sample.vars["SwapCount"] == 1 and sample.vars["SmoothCount"] == 1
    assert sample.vars["HighRow"] == 30 and sample.vars["HighCol"] == 40
    assert abs(float(sample.vars["HighRes"]) - 30) < 1e-6
    assert sample.vars["LowRow"] == 29 and sample.vars["LowCol"] == 39
    assert abs(float(sample.vars["LowRes"])) < 1e-6
    assert sample.vars["PeakN"] == 1 and sample.vars["PeakR"] == 30 and sample.vars["PeakC"] == 40
    sizes = next(param for param in REGISTRY["sub_local_median"].control_in if param.name == "MaskSize").choices
    assert "3" in sizes and "9" in sizes
    poles = next(param for param in REGISTRY["extreme_median_roughness"].control_in if param.name == "Mode").choices
    assert "'low'" in poles
    even = run_source(
        "Vizion.gen_image_const (Image, 'byte', 10, 10)\n"
        "Vizion.sub_local_median (Image, Level, 4)\n"
    )
    assert even.error and "奇數" in even.error


def test_hdevelop_batch43():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gray_spot_white_spots_smaller_than_kernel.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["WhitePeak"] == 30
    assert sample.vars["WhiteSide"] == 0
    assert sample.vars["WhitePit"] == 0
    assert sample.vars["BlackPit"] == 30
    assert sample.vars["BlackSide"] == 0
    assert sample.vars["BlackPeak"] == 0
    assert abs(float(sample.vars["VolWhite"]) - 30) < 1e-6
    assert abs(float(sample.vars["VolBlackOnBright"])) < 1e-6
    assert abs(float(sample.vars["MeanWhite"]) - (10 / 3)) < 1e-6
    assert abs(float(sample.vars["VolBlack"]) - 30) < 1e-6
    assert abs(float(sample.vars["VolWhiteOnDark"])) < 1e-6
    assert abs(float(sample.vars["White1"]) - 30) < 1e-6
    assert abs(float(sample.vars["White2"])) < 1e-6
    assert abs(float(sample.vars["Black1"])) < 1e-6
    assert abs(float(sample.vars["Black2"]) - 30) < 1e-6
    assert sample.vars["HighRow"] == 20 and sample.vars["HighCol"] == 25
    assert abs(float(sample.vars["HighH"]) - 30) < 1e-6
    assert sample.vars["PitRow"] == 40 and sample.vars["PitCol"] == 50
    assert abs(float(sample.vars["PitH"]) - 30) < 1e-6
    assert sample.vars["KeepCount"] == 1 and sample.vars["SwapCount"] == 1
    assert sample.vars["DarkCount"] == 1
    assert sample.vars["BrightN"] == 1 and sample.vars["BrightR"] == 20 and sample.vars["BrightC"] == 25
    assert sample.vars["DarkN"] == 1 and sample.vars["DarkR"] == 40 and sample.vars["DarkC"] == 50
    sizes = next(param for param in REGISTRY["gray_spot_white"].control_in if param.name == "MaskSize").choices
    assert "3" in sizes and "9" in sizes
    modes = next(param for param in REGISTRY["spot_volume"].control_in if param.name == "Mode").choices
    assert "'black'" in modes
    even = run_source(
        "Vizion.gen_image_const (Image, 'byte', 10, 10)\n"
        "Vizion.gray_spot_white (Image, Spots, 4)\n"
    )
    assert even.error and "奇數" in even.error


def test_hdevelop_batch44():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gray_opening_rect_opening_reconstruct.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["OpenSq"] == 200
    assert sample.vars["OpenArm"] == 0
    assert sample.vars["OpenSpeck"] == 0
    assert sample.vars["RecSq"] == 200
    assert sample.vars["RecArm"] == 200
    assert sample.vars["RecSpeck"] == 0
    assert sample.vars["GainN"] == 10
    assert abs(float(sample.vars["GainR"]) - 23) < 1e-6
    assert abs(float(sample.vars["GainC"]) - 31.5) < 1e-6
    assert abs(float(sample.vars["GainVol"]) - 2000) < 1e-6
    assert sample.vars["CloseSq"] == 0
    assert sample.vars["CloseArm"] == 200
    assert sample.vars["CloseHole"] == 200
    assert sample.vars["RecDarkArm"] == 0
    assert sample.vars["RecHole"] == 200
    assert sample.vars["PitN"] == 10
    assert abs(float(sample.vars["PitVol"]) - 2000) < 1e-6
    sizes = next(param for param in REGISTRY["gray_opening_rect"].control_in if param.name == "MaskSize").choices
    assert "3" in sizes and "9" in sizes
    even = run_source(
        "Vizion.gen_image_const (Image, 'byte', 10, 10)\n"
        "Vizion.gray_opening_rect (Image, Opened, 4)\n"
    )
    assert even.error and "奇數" in even.error


def test_hdevelop_batch45():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_thickness_max_how_thick.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["MaxR1"] == 12 and sample.vars["MaxC1"] == 12 and sample.vars["MaxD1"] == 3
    assert sample.vars["MaxR2"] == 41 and sample.vars["MaxC2"] == 11 and sample.vars["MaxD2"] == 2
    assert sample.vars["MaxR3"] == 14 and sample.vars["MaxC3"] == 54 and sample.vars["MaxD3"] == 5
    assert sample.vars["NeckR1"] == 12 and sample.vars["NeckC1"] == 12 and sample.vars["NeckD1"] == 3
    assert sample.vars["NeckR2"] == 41 and sample.vars["NeckC2"] == 11 and sample.vars["NeckD2"] == 2
    assert sample.vars["NeckR3"] == 14 and sample.vars["NeckC3"] == 58 and sample.vars["NeckD3"] == 2
    assert abs(float(sample.vars["Mean1"]) - 3) < 1e-6
    assert abs(float(sample.vars["Mean2"]) - 2) < 1e-6
    assert abs(float(sample.vars["Ratio1"]) - 1) < 1e-6
    assert abs(float(sample.vars["Ratio2"]) - 1) < 1e-6
    assert abs(float(sample.vars["Ratio3"]) - 0.4) < 1e-6
    assert abs(float(sample.vars["Sum1"]) - 35) < 1e-6
    assert abs(float(sample.vars["Sum2"]) - 82) < 1e-6
    assert sample.vars["ThickCount"] == 1 and sample.vars["SwapCount"] == 1 and sample.vars["ThinCount"] == 2
    assert sample.vars["CoreN"] == 1 and sample.vars["CoreR"] == 12 and sample.vars["CoreC"] == 12
    assert sample.vars["RawCore"] == 5 and sample.vars["RawNeck"] == 2 and sample.vars["RawOut"] == 0
    assert sample.vars["ViewCore"] == 200 and sample.vars["ViewNeck"] == 80
    modes = next(param for param in REGISTRY["thickness_image"].control_in if param.name == "Metric").choices
    assert "'chessboard'" in modes and "'euclidean'" in modes
    empty = run_source("Vizion.gen_rectangle1 (Box, 0, 0, 1, 1)\nVizion.difference (Box, Box, Empty)\nVizion.thickness_max (Empty, Row, Column, Distance)\n")
    assert empty.error and "區域是空的" in empty.error


def test_hdevelop_batch46():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_area_intersection_overlap_area.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Inter1"] == 50 and sample.vars["Inter2"] == 25
    assert sample.vars["Union1"] == 150 and sample.vars["Union2"] == 121
    assert sample.vars["Only1"] == 50 and sample.vars["Only2"] == 0
    assert abs(float(sample.vars["Iou1"]) - (1 / 3)) < 1e-6
    assert abs(float(sample.vars["Iou2"]) - (25 / 121)) < 1e-6
    assert abs(float(sample.vars["Cover1"]) - 0.5) < 1e-6
    assert abs(float(sample.vars["Cover2"]) - 1) < 1e-6
    assert sample.vars["Inside1"] == 0 and sample.vars["Inside2"] == 1
    assert sample.vars["KeepCount"] == 1 and sample.vars["SwapCount"] == 1 and sample.vars["HalfCount"] == 1
    assert sample.vars["KeepN"] == 25 and sample.vars["KeepR"] == 42 and sample.vars["KeepC"] == 42
    assert sample.vars["Sym1"] == 100 and sample.vars["Sym2"] == 96
    assert sample.vars["HitGray"] == 255 and sample.vars["MissGray"] == 0
    broadcast = run_source(
        "Vizion.gen_image_const (Image, 'byte', 80, 60)\n"
        "Vizion.gen_rectangle1 (Big, 0, 0, 20, 20)\n"
        "Vizion.gen_rectangle1 (InBox, 2, 2, 4, 4)\n"
        "Vizion.gen_rectangle1 (OutBox, 30, 30, 32, 32)\n"
        "Vizion.concat_obj (InBox, OutBox, Parts)\n"
        "Vizion.region_cover (Parts, Big, Cover)\n"
        "CoverIn := Cover[1]\n"
        "CoverOut := Cover[2]\n"
    )
    assert broadcast.error is None, broadcast.error
    assert broadcast.vars["CoverIn"] == 1 and broadcast.vars["CoverOut"] == 0
    bad = run_source(
        "Vizion.gen_image_const (Image, 'byte', 40, 40)\n"
        "Vizion.gen_rectangle1 (A, 1, 1, 2, 2)\n"
        "Vizion.gen_rectangle1 (B, 3, 3, 4, 4)\n"
        "Vizion.gen_rectangle1 (C, 5, 5, 6, 6)\n"
        "Vizion.gen_rectangle1 (D, 7, 7, 8, 8)\n"
        "Vizion.concat_obj (A, B, Pair)\n"
        "Vizion.concat_obj (C, D, More)\n"
        "Vizion.concat_obj (More, A, Three)\n"
        "Vizion.area_intersection (Pair, Three, Area)\n"
    )
    assert bad.error and "物件數" in bad.error


def test_hdevelop_batch47():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_distance_objects_how_far_apart.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Min1"] == 11 and sample.vars["Min2"] == 1 and sample.vars["Min3"] == 0
    assert sample.vars["Far1"] == 20 and sample.vars["Far2"] == 5 and sample.vars["Far3"] == 5
    assert sample.vars["NearR1"] == 10 and sample.vars["NearC1"] == 19
    assert sample.vars["NearR2"] == 10 and sample.vars["NearC2"] == 30 and sample.vars["NearD1"] == 11
    assert sample.vars["TouchR"] == 40 and sample.vars["TouchC"] == 14 and sample.vars["TouchD"] == 1
    assert sample.vars["OverR"] == 50 and sample.vars["OverC"] == 45 and sample.vars["OverD"] == 0
    assert sample.vars["FarR1"] == 10 and sample.vars["FarC1"] == 10 and sample.vars["FarP1"] == 20
    assert abs(float(sample.vars["OffR1"])) < 1e-6
    assert abs(float(sample.vars["OffC1"]) - 20) < 1e-6
    assert abs(float(sample.vars["OffC2"]) - 5) < 1e-6
    assert abs(float(sample.vars["Center1"]) - 20) < 1e-6
    assert abs(float(sample.vars["Center2"]) - 5) < 1e-6
    assert sample.vars["GapCount"] == 1 and sample.vars["SwapCount"] == 1
    assert sample.vars["TouchCount"] == 1 and sample.vars["OverCount"] == 1
    assert sample.vars["EdgeN"] == 10
    assert abs(float(sample.vars["EdgeR"]) - 14.5) < 1e-6
    assert abs(float(sample.vars["EdgeC"]) - 19) < 1e-6
    assert sample.vars["EdgeGray"] == 255 and sample.vars["InsideGray"] == 0
    modes = next(param for param in REGISTRY["distance_objects"].control_in if param.name == "Metric").choices
    assert "'chessboard'" in modes and "'euclidean'" in modes
    same = run_source(
        "Vizion.gen_image_const (Image, 'byte', 80, 60)\n"
        "Vizion.gen_rectangle1 (A, 10, 10, 19, 19)\n"
        "Vizion.gen_rectangle1 (B, 10, 30, 19, 39)\n"
        "Vizion.distance_objects (A, B, 'euclidean', Distance)\n"
    )
    assert same.error is None and abs(float(same.vars["Distance"][0]) - 11) < 1e-6
    bad = run_source(
        "Vizion.gen_image_const (Image, 'byte', 40, 40)\n"
        "Vizion.gen_rectangle1 (A, 1, 1, 2, 2)\n"
        "Vizion.gen_rectangle1 (B, 4, 4, 5, 5)\n"
        "Vizion.gen_rectangle1 (C, 8, 8, 9, 9)\n"
        "Vizion.concat_obj (A, B, Pair)\n"
        "Vizion.concat_obj (Pair, C, Three)\n"
        "Vizion.distance_objects (Pair, Three, Distance)\n"
    )
    assert bad.error and "物件數" in bad.error


def test_hdevelop_batch48():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_region_angle_which_side.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Ang1"] == 0 and sample.vars["Ang2"] == 180
    assert sample.vars["Ang3"] == 90 and sample.vars["Ang4"] == 270
    assert sample.vars["RightCount"] == 1 and sample.vars["LeftCount"] == 1
    assert sample.vars["BelowCount"] == 1 and sample.vars["AboveCount"] == 1
    assert abs(float(sample.vars["NearR"]) - 34.5) < 1e-6
    assert abs(float(sample.vars["NearC"]) - 24) < 1e-6
    assert abs(float(sample.vars["FarR"]) - 16) < 1e-6
    assert abs(float(sample.vars["FarC"]) - 34.5) < 1e-6
    assert sample.vars["CloseN"] == 50
    assert abs(float(sample.vars["CloseR"]) - 34.5) < 1e-6
    assert abs(float(sample.vars["CloseC"]) - 24) < 1e-6
    assert sample.vars["FarN"] == 50
    assert abs(float(sample.vars["FarMostR"]) - 16) < 1e-6
    assert abs(float(sample.vars["FarMostC"]) - 34.5) < 1e-6
    assert sample.vars["Gap1"] == 100 and sample.vars["Gap2"] == 30
    assert sample.vars["Gap3"] == 80 and sample.vars["Gap4"] == 110
    assert sample.vars["Wide1"] == 10 and sample.vars["Wide2"] == 3
    assert sample.vars["Wide3"] == 8 and sample.vars["Wide4"] == 11
    assert sample.vars["GapN"] == 100
    assert sample.vars["GapGray"] == 255 and sample.vars["RefGray"] == 0
    sides = next(param for param in REGISTRY["select_side"].control_in if param.name == "Side").choices
    assert "'left'" in sides and "'above'" in sides
    orders = next(param for param in REGISTRY["sort_distance"].control_in if param.name == "Order").choices
    assert "'near'" in orders and "'far'" in orders


def test_hdevelop_batch49():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_partition_nearest_gap_to_nearest.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["GrownN"] == 2 and sample.vars["BorderN"] == 2
    assert sample.vars["InsideN"] == 2 and sample.vars["TieCount"] == 1
    assert sample.vars["G1"] == 1500 and sample.vars["G2"] == 3300
    assert sample.vars["T1"] == 1500 and sample.vars["T2"] == 3300
    assert abs(float(sample.vars["GR"]) - 29.5) < 1e-6
    assert abs(float(sample.vars["GC1"]) - 12) < 1e-6
    assert abs(float(sample.vars["GC2"]) - 52) < 1e-6
    assert sample.vars["OwnL"] == 1 and sample.vars["OwnR"] == 2
    assert sample.vars["CornerL"] == 1 and sample.vars["CornerR"] == 2
    assert sample.vars["B1"] == 60 and sample.vars["B2"] == 60
    assert abs(float(sample.vars["BC1"]) - 24) < 1e-6
    assert abs(float(sample.vars["BC2"]) - 25) < 1e-6
    assert sample.vars["BigN"] == 3300 and abs(float(sample.vars["BigC"]) - 52) < 1e-6
    assert sample.vars["SmallN"] == 1500 and abs(float(sample.vars["SmallC"]) - 12) < 1e-6
    assert sample.vars["In1"] == 300 and sample.vars["In2"] == 300
    assert abs(float(sample.vars["InC1"]) - 22) < 1e-6
    assert abs(float(sample.vars["InC2"]) - 27) < 1e-6
    assert sample.vars["OffStrip"] == 0 and sample.vars["OnStrip"] == 1
    assert sample.vars["LabelL"] == 1 and sample.vars["LabelR"] == 2
    assert sample.vars["TieN"] == 60
    assert abs(float(sample.vars["TieR"]) - 29.5) < 1e-6
    assert abs(float(sample.vars["TieC"]) - 11) < 1e-6
    assert sample.vars["CutGray"] == 255 and sample.vars["CutGray2"] == 255
    assert sample.vars["SideGray"] == 0
    metrics = next(param for param in REGISTRY["partition_nearest"].control_in if param.name == "Metric").choices
    assert "'city-block'" in metrics and "'euclidean'" in metrics
    which = next(param for param in REGISTRY["select_territory"].control_in if param.name == "Which").choices
    assert "'large'" in which and "'small'" in which
    overlap = run_source(
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.gen_rectangle1 (A, 0, 0, 9, 9)\n"
        "Vizion.gen_rectangle1 (B, 0, 0, 9, 9)\n"
        "Vizion.concat_obj (A, B, Both)\n"
        "Vizion.partition_area (Both, 'city-block', Area)\n"
        "A1 := Area[1]\n"
        "A2 := Area[2]\n"
        "Vizion.partition_nearest (Both, Grown, 'city-block')\n"
        "Vizion.count_obj (Grown, N)\n"
    )
    assert overlap.error is None, overlap.error
    assert overlap.vars["A1"] == 200 and overlap.vars["A2"] == 0 and overlap.vars["N"] == 1
    empty = run_source(
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.gen_rectangle1 (Box, 0, 0, 2, 2)\n"
        "Vizion.difference (Box, Box, Empty)\n"
        "Vizion.partition_nearest (Empty, Grown, 'city-block')\n"
    )
    assert empty.error and "空" in empty.error


def test_hdevelop_batch50():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_sort_feature_pick_by_feature.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["D1"] == 100 and sample.vars["D2"] == 48 and sample.vars["D3"] == 16
    assert abs(float(sample.vars["DR"]) - 14.5) < 1e-6
    assert abs(float(sample.vars["DC"]) - 14.5) < 1e-6
    assert abs(float(sample.vars["DC3"]) - 31.5) < 1e-6
    assert sample.vars["U1"] == 16
    assert abs(float(sample.vars["UR"]) - 11.5) < 1e-6
    assert abs(float(sample.vars["UC"]) - 31.5) < 1e-6
    assert abs(float(sample.vars["UC2"]) - 41.5) < 1e-6
    assert sample.vars["TopN"] == 2 and sample.vars["Top1"] == 100 and sample.vars["Top2"] == 48
    assert sample.vars["LargeN"] == 100
    assert abs(float(sample.vars["LargeR"]) - 14.5) < 1e-6
    assert abs(float(sample.vars["LargeC"]) - 14.5) < 1e-6
    assert sample.vars["SmallN"] == 16
    assert abs(float(sample.vars["SmallR"]) - 11.5) < 1e-6
    assert abs(float(sample.vars["SmallC"]) - 31.5) < 1e-6
    assert sample.vars["R1"] == 1 and sample.vars["R2"] == 3
    assert sample.vars["R3"] == 2 and sample.vars["R4"] == 4
    assert sample.vars["SecN"] == 48 and abs(float(sample.vars["SecR"]) - 32.5) < 1e-6
    assert sample.vars["Pick1"] == 100 and sample.vars["Pick2"] == 16
    assert abs(float(sample.vars["PickC"]) - 31.5) < 1e-6
    assert sample.vars["I1"] == 1 and sample.vars["I2"] == 3
    assert sample.vars["I3"] == 2 and sample.vars["I4"] == 4
    assert sample.vars["AboveN"] == 2 and sample.vars["BelowN"] == 2
    assert sample.vars["Ab1"] == 100 and sample.vars["Ab2"] == 48
    assert sample.vars["Low"] == 16 and sample.vars["High"] == 100
    assert sample.vars["HitGray"] == 255 and sample.vars["MissGray"] == 0
    features = next(param for param in REGISTRY["sort_feature"].control_in if param.name == "Feature").choices
    assert "'area'" in features and "'circularity'" in features
    orders = next(param for param in REGISTRY["sort_feature"].control_in if param.name == "Order").choices
    assert "'down'" in orders and "'up'" in orders
    which = next(param for param in REGISTRY["select_extreme"].control_in if param.name == "Which").choices
    assert "'max'" in which and "'min'" in which
    sides = next(param for param in REGISTRY["select_relative"].control_in if param.name == "Side").choices
    assert "'above'" in sides and "'below'" in sides
    too_many = run_source(
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.gen_rectangle1 (A, 0, 0, 2, 2)\n"
        "Vizion.select_top (A, Top, 3, 'area', 'down')\n"
    )
    assert too_many.error and "超出" in too_many.error
    empty = run_source(
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.gen_rectangle1 (Box, 0, 0, 2, 2)\n"
        "Vizion.difference (Box, Box, Empty)\n"
        "Vizion.sort_feature (Empty, Sorted, 'area', 'down')\n"
    )
    assert empty.error and "空" in empty.error


def test_hdevelop_batch51():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_region_defects_fill_notches.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["BiteN"] == 2 and sample.vars["ChipN"] == 2 and sample.vars["DeepN"] == 1
    assert sample.vars["BA1"] == 100 and sample.vars["BA2"] == 10
    assert abs(float(sample.vars["BR1"]) - 24.5) < 1e-6
    assert abs(float(sample.vars["BC1"]) - 34.5) < 1e-6
    assert abs(float(sample.vars["BR2"]) - 13) < 1e-6
    assert abs(float(sample.vars["BC2"]) - 66) < 1e-6
    assert sample.vars["C1"] == 1 and sample.vars["C2"] == 0 and sample.vars["C3"] == 1
    assert sample.vars["DA1"] == 100 and sample.vars["DA2"] == 10
    assert abs(float(sample.vars["DP1"]) - 10) < 1e-6
    assert abs(float(sample.vars["DP2"]) - 4) < 1e-6
    assert sample.vars["PR1"] == 20 and sample.vars["PC1"] == 30
    assert abs(float(sample.vars["PD1"]) - 10) < 1e-6
    assert sample.vars["PR2"] == 14 and sample.vars["PC2"] == 65
    assert abs(float(sample.vars["PD2"]) - 4) < 1e-6
    assert sample.vars["Chip1"] == 800 and sample.vars["Chip2"] == 375
    assert sample.vars["DeepA"] == 100 and sample.vars["MostN"] == 100
    assert sample.vars["F1"] == 900 and sample.vars["F2"] == 100 and sample.vars["F3"] == 385
    assert sample.vars["GapGray"] == 255 and sample.vars["BodyGray"] == 0
    metrics = next(param for param in REGISTRY["region_defects"].control_in if param.name == "Metric").choices
    assert "'city-block'" in metrics and "'euclidean'" in metrics
    empty = run_source(
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.gen_rectangle1 (Box, 0, 0, 2, 2)\n"
        "Vizion.difference (Box, Box, Empty)\n"
        "Vizion.region_defects (Empty, Defects, 1, 'city-block')\n"
    )
    assert empty.error and "空" in empty.error


def test_hdevelop_batch52():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_select_inner_who_contains_whom.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["InnerN"] == 2 and sample.vars["In1"] == 196 and sample.vars["In2"] == 16
    assert sample.vars["OuterN"] == 2 and sample.vars["Out1"] == 1200 and sample.vars["Out2"] == 25
    assert sample.vars["HolderN"] == 2 and sample.vars["Hold1"] == 1200 and sample.vars["Hold2"] == 196
    assert sample.vars["LeafN"] == 16
    assert abs(float(sample.vars["LeafR"]) - 23.5) < 1e-6
    assert abs(float(sample.vars["LeafC"]) - 23.5) < 1e-6
    assert sample.vars["MidN"] == 196 and abs(float(sample.vars["MidR"]) - 24.5) < 1e-6
    assert sample.vars["DeepN"] == 1 and sample.vars["TopN"] == 2
    assert sample.vars["H1"] == 0 and sample.vars["H2"] == 1
    assert sample.vars["H3"] == 2 and sample.vars["H4"] == 0
    assert sample.vars["K1"] == 2 and sample.vars["K2"] == 1
    assert sample.vars["K3"] == 0 and sample.vars["K4"] == 0
    assert sample.vars["D1"] == 0 and sample.vars["D2"] == 1
    assert sample.vars["D3"] == 2 and sample.vars["D4"] == 0
    assert sample.vars["LeafGray"] == 255 and sample.vars["FrameGray"] == 0
    assert "select_inner" in REGISTRY and "nesting_depth" in REGISTRY
    same = run_source(
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.gen_rectangle1 (A, 0, 0, 4, 4)\n"
        "Vizion.gen_rectangle1 (B, 0, 0, 4, 4)\n"
        "Vizion.concat_obj (A, B, Parts)\n"
        "Vizion.select_inner (Parts, Inner)\n"
        "Vizion.count_obj (Inner, N)\n"
        "Vizion.container_index (Parts, Holder)\n"
        "H1 := Holder[1]\n"
        "H2 := Holder[2]\n"
    )
    assert same.error is None, same.error
    assert same.vars["N"] == 0 and same.vars["H1"] == 0 and same.vars["H2"] == 0
    empty = run_source(
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.gen_rectangle1 (Box, 0, 0, 2, 2)\n"
        "Vizion.difference (Box, Box, Empty)\n"
        "Vizion.select_inner (Empty, Inner)\n"
    )
    assert empty.error and "空" in empty.error


def test_hdevelop_batch53():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_select_border_touches_border.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["TouchN"] == 2 and sample.vars["Touch1"] == 100 and sample.vars["Touch2"] == 50
    assert sample.vars["InnerN"] == 2 and sample.vars["Inner1"] == 100 and sample.vars["Inner2"] == 80
    assert sample.vars["TopN"] == 1 and sample.vars["TopA"] == 100 and sample.vars["RightN"] == 0
    assert sample.vars["KeepN"] == 1 and sample.vars["KeepA"] == 100
    assert sample.vars["BandN"] == 1 and sample.vars["BandA"] == 80
    assert sample.vars["FarN"] == 100
    assert abs(float(sample.vars["FarR"]) - 29.5) < 1e-6
    assert abs(float(sample.vars["FarC"]) - 39.5) < 1e-6
    assert sample.vars["D1"] == 0 and sample.vars["D2"] == 0
    assert sample.vars["D3"] == 25 and sample.vars["D4"] == 2
    assert sample.vars["GT1"] == 0 and sample.vars["GT2"] == 20
    assert sample.vars["GT3"] == 25 and sample.vars["GT4"] == 40
    assert sample.vars["GB1"] == 50 and sample.vars["GB3"] == 25 and sample.vars["GB4"] == 10
    assert sample.vars["GL2"] == 0 and sample.vars["GL3"] == 35
    assert sample.vars["GR2"] == 75 and sample.vars["GR4"] == 2
    assert sample.vars["PR1"] == 0 and sample.vars["PC1"] == 20 and sample.vars["PS1"] == "top"
    assert sample.vars["PR2"] == 20 and sample.vars["PC2"] == 0 and sample.vars["PS2"] == "left"
    assert sample.vars["PR3"] == 25 and sample.vars["PC3"] == 35 and sample.vars["PS3"] == "top"
    assert sample.vars["PR4"] == 40 and sample.vars["PC4"] == 77 and sample.vars["PS4"] == "right"
    assert sample.vars["EdgeGray"] == 255 and sample.vars["MidGray"] == 0
    modes = next(param for param in REGISTRY["select_border"].control_in if param.name == "Mode").choices
    assert "'touch'" in modes and "'inner'" in modes
    sides = next(param for param in REGISTRY["select_border_side"].control_in if param.name == "Side").choices
    assert "'top'" in sides and "'right'" in sides
    corner = run_source(
        "Vizion.gen_image_const (Image, 'byte', 30, 20)\n"
        "Vizion.gen_rectangle1 (Corner, 0, 0, 3, 3)\n"
        "Vizion.border_point (Corner, Row, Col, Side)\n"
        "R := Row[1]\n"
        "C := Col[1]\n"
        "S := Side[1]\n"
        "Vizion.border_gap (Corner, Top, Bottom, Left, Right)\n"
        "T := Top[1]\n"
        "L := Left[1]\n"
    )
    assert corner.error is None, corner.error
    assert corner.vars["R"] == 0 and corner.vars["C"] == 0 and corner.vars["S"] == "top"
    assert corner.vars["T"] == 0 and corner.vars["L"] == 0
    bad = run_source(
        "Vizion.gen_image_const (Image, 'byte', 30, 20)\n"
        "Vizion.gen_rectangle1 (Corner, 0, 0, 3, 3)\n"
        "Vizion.select_margin (Corner, Kept, -1)\n"
    )
    assert bad.error and "負" in bad.error


def test_hdevelop_batch54():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_neighbor_distance_objects_touch.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["D1"] == 1 and sample.vars["D2"] == 1
    assert sample.vars["D3"] == 1 and sample.vars["D4"] == 16
    assert sample.vars["I1"] == 2 and sample.vars["I2"] == 1
    assert sample.vars["I3"] == 1 and sample.vars["I4"] == 3
    assert sample.vars["NR1"] == 10 and sample.vars["NC1"] == 19 and sample.vars["NG1"] == 1
    assert sample.vars["NR2"] == 10 and sample.vars["NC2"] == 20
    assert sample.vars["NR4"] == 40 and sample.vars["NC4"] == 40 and sample.vars["NG4"] == 16
    assert sample.vars["K1"] == 2 and sample.vars["K2"] == 2
    assert sample.vars["K3"] == 2 and sample.vars["K4"] == 0
    assert sample.vars["TouchN"] == 3
    assert sample.vars["Touch1"] == 100 and sample.vars["Touch2"] == 50 and sample.vars["Touch3"] == 25
    assert sample.vars["AloneN"] == 1 and sample.vars["AloneA"] == 25
    assert abs(float(sample.vars["AloneR"]) - 42) < 1e-6
    assert abs(float(sample.vars["AloneC"]) - 42) < 1e-6
    assert sample.vars["ContactN"] == 3
    assert sample.vars["CA1"] == 10 and sample.vars["CA2"] == 14 and sample.vars["CA3"] == 5
    assert abs(float(sample.vars["CR1"]) - 14.5) < 1e-6
    assert abs(float(sample.vars["CC1"]) - 19) < 1e-6
    assert sample.vars["MergeN"] == 2 and sample.vars["M1"] == 175 and sample.vars["M2"] == 25
    assert abs(float(sample.vars["MR"]) - (2725 / 175)) < 1e-4
    assert abs(float(sample.vars["MC"]) - (3100 / 175)) < 1e-4
    assert sample.vars["TouchGray"] == 255 and sample.vars["BodyGray"] == 0 and sample.vars["FarGray"] == 0
    metrics = next(param for param in REGISTRY["neighbor_distance"].control_in if param.name == "Metric").choices
    assert "'chessboard'" in metrics and "'city-block'" in metrics
    apart = run_source(
        "Vizion.gen_image_const (Image, 'byte', 40, 30)\n"
        "Vizion.gen_rectangle1 (A, 0, 0, 4, 4)\n"
        "Vizion.gen_rectangle1 (B, 0, 6, 4, 10)\n"
        "Vizion.concat_obj (A, B, Parts)\n"
        "Vizion.select_touching (Parts, Touched, 0, 'chessboard')\n"
        "Vizion.count_obj (Touched, N)\n"
    )
    assert apart.error is None, apart.error
    assert apart.vars["N"] == 0
    single = run_source(
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.gen_rectangle1 (A, 0, 0, 2, 2)\n"
        "Vizion.neighbor_distance (A, 'chessboard', Distance)\n"
    )
    assert single.error and "兩個" in single.error


def test_hdevelop_batch55():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_shape_score_most_like_template.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["S1"] == 0
    assert abs(float(sample.vars["S2"])) < 1e-12
    assert abs(float(sample.vars["S3"]) - 0.001793327482459972) < 1e-9
    assert abs(float(sample.vars["S4"]) - 3.461924208773209) < 1e-9
    assert sample.vars["BestN"] == 100
    assert abs(float(sample.vars["BestR"]) - 14.5) < 1e-6
    assert abs(float(sample.vars["BestC"]) - 44.5) < 1e-6
    assert sample.vars["WorstN"] == 150
    assert abs(float(sample.vars["WorstR"]) - 32) < 1e-6
    assert abs(float(sample.vars["WorstC"]) - 24.5) < 1e-6
    assert sample.vars["Near1"] == 100 and sample.vars["Near2"] == 225
    assert abs(float(sample.vars["NearC"]) - 44.5) < 1e-6
    assert sample.vars["Far1"] == 150 and abs(float(sample.vars["FarR"]) - 32) < 1e-6
    assert sample.vars["R1"] == 1 and sample.vars["R2"] == 2
    assert sample.vars["R3"] == 3 and sample.vars["R4"] == 4
    assert sample.vars["I1"] == 1 and sample.vars["I2"] == 2
    assert sample.vars["I3"] == 3 and sample.vars["I4"] == 4
    assert sample.vars["KeepN"] == 3
    assert sample.vars["Keep1"] == 100 and sample.vars["Keep2"] == 225 and sample.vars["Keep3"] == 185
    assert abs(float(sample.vars["H1"]) - (1 / 6)) < 1e-9
    assert abs(float(sample.vars["SameH"]) - float(sample.vars["H1"])) < 1e-12
    assert sample.vars["LikeGray"] == 255 and sample.vars["BarGray"] == 0
    methods = next(param for param in REGISTRY["shape_score"].control_in if param.name == "Method").choices
    assert "'i1'" in methods and "'i3'" in methods
    orders = next(param for param in REGISTRY["sort_shape"].control_in if param.name == "Order").choices
    assert "'near'" in orders and "'far'" in orders
    bad = run_source(
        "Vizion.gen_image_const (Image, 'byte', 40, 20)\n"
        "Vizion.gen_rectangle1 (A, 0, 0, 4, 4)\n"
        "Vizion.gen_rectangle1 (B, 0, 8, 4, 12)\n"
        "Vizion.gen_rectangle1 (C, 8, 0, 12, 4)\n"
        "Vizion.concat_obj (A, B, Two)\n"
        "Vizion.concat_obj (Two, C, Three)\n"
        "Vizion.shape_score (Three, Two, 'i1', Score)\n"
    )
    assert bad.error and "範本" in bad.error


def test_hdevelop_batch56():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_align_delta_move_onto_template.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["DR1"] == 0 and sample.vars["DC1"] == -30
    assert sample.vars["DR2"] == -28 and sample.vars["DC2"] == -28
    assert abs(float(sample.vars["DR3"]) + 17.5) < 1e-6
    assert sample.vars["DC3"] == -45
    assert sample.vars["O1"] == 100 and sample.vars["O2"] == 36 and sample.vars["O3"] == 50
    assert sample.vars["C1"] == 1 and abs(float(sample.vars["C2"]) - 0.36) < 1e-6
    assert abs(float(sample.vars["C3"]) - 0.5) < 1e-6
    assert sample.vars["MA1"] == 0 and sample.vars["MA2"] == 64 and sample.vars["MA3"] == 100
    assert sample.vars["MissN"] == 2 and sample.vars["MissA1"] == 64 and sample.vars["MissA2"] == 100
    assert sample.vars["KeepN"] == 1 and sample.vars["KeepA"] == 100
    assert abs(float(sample.vars["KeepC"]) - 44.5) < 1e-6
    assert sample.vars["Move1"] == 100
    assert abs(float(sample.vars["MoveR"]) - 14.5) < 1e-6
    assert abs(float(sample.vars["MoveC"]) - 14.5) < 1e-6
    assert abs(float(sample.vars["Phi"]) + 1.5707963267948966) < 1e-6
    assert sample.vars["CrossN"] == 25 and sample.vars["TurnN"] == 100
    assert sample.vars["TurnA"] == 100
    assert abs(float(sample.vars["TurnR"]) - 52) < 1e-6
    assert abs(float(sample.vars["TurnC"]) - 19.5) < 1e-6
    assert sample.vars["OnGray"] == 255 and sample.vars["OffGray"] == 0
    modes = next(param for param in REGISTRY["align_region"].control_in if param.name == "Mode").choices
    assert "'center'" in modes and "'rigid'" in modes
    bad = run_source(
        "Vizion.gen_image_const (Image, 'byte', 40, 20)\n"
        "Vizion.gen_rectangle1 (A, 0, 0, 3, 3)\n"
        "Vizion.gen_rectangle1 (B, 0, 8, 3, 11)\n"
        "Vizion.gen_rectangle1 (C, 8, 0, 11, 3)\n"
        "Vizion.concat_obj (A, B, Two)\n"
        "Vizion.concat_obj (Two, C, Three)\n"
        "Vizion.align_delta (Three, Two, DRow, DColumn)\n"
    )
    assert bad.error and "範本" in bad.error


def test_hdevelop_batch57():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_grid_size_rows_and_columns.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["GridRows"] == 2 and sample.vars["GridCols"] == 3
    assert sample.vars["RowPitch"] == 20 and sample.vars["ColPitch"] == 20
    assert [sample.vars[name] for name in ("R1", "R4", "R5")] == [1, 2, 2]
    assert [sample.vars[name] for name in ("K1", "K2", "K3", "K4", "K5")] == [1, 2, 3, 1, 3]
    assert abs(float(sample.vars["S1C"]) - 11.5) < 1e-6
    assert abs(float(sample.vars["S3C"]) - 51.5) < 1e-6
    assert abs(float(sample.vars["S4C"]) - 11.5) < 1e-6
    assert abs(float(sample.vars["S5R"]) - 31.5) < 1e-6
    assert sample.vars["MissN"] == 1 and sample.vars["MissA"] == 16
    assert abs(float(sample.vars["MissR"]) - 31.5) < 1e-6
    assert abs(float(sample.vars["MissC"]) - 31.5) < 1e-6
    assert sample.vars["CellA"] == 16
    assert abs(float(sample.vars["CellR"]) - 31.5) < 1e-6
    assert abs(float(sample.vars["CellC"]) - 51.5) < 1e-6
    assert abs(float(sample.vars["HoleR"]) - 31.5) < 1e-6
    assert abs(float(sample.vars["HoleC"]) - 31.5) < 1e-6
    assert [sample.vars[name] for name in ("G1", "G2", "G3", "G0")] == [1, 2, 3, 0]
    gap = next(param for param in REGISTRY["grid_index"].control_in if param.name == "Gap")
    assert gap.default == "0"
    empty = run_source(
        "Vizion.gen_image_const (Image, 'byte', 80, 60)\n"
        "Vizion.gen_rectangle1 (A, 10, 10, 13, 13)\n"
        "Vizion.gen_rectangle1 (B, 10, 30, 13, 33)\n"
        "Vizion.gen_rectangle1 (C, 10, 50, 13, 53)\n"
        "Vizion.gen_rectangle1 (D, 30, 10, 33, 13)\n"
        "Vizion.gen_rectangle1 (E, 30, 50, 33, 53)\n"
        "Vizion.concat_obj (A, B, Pair)\n"
        "Vizion.concat_obj (Pair, C, Three)\n"
        "Vizion.concat_obj (Three, D, Four)\n"
        "Vizion.concat_obj (Four, E, Parts)\n"
        "Vizion.select_grid (Parts, Hole, 2, 2)\n"
    )
    assert empty.error and "沒有物件" in empty.error


def test_hdevelop_batch58():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gray_offset_brightness_side.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["DR1"] == 0 and sample.vars["DC1"] == 3
    assert sample.vars["DR2"] == 0 and sample.vars["DC2"] == 0
    assert sample.vars["DR3"] == -3 and sample.vars["DC3"] == 0
    assert sample.vars["S1"] == 3 and sample.vars["S2"] == 0 and sample.vars["S3"] == 3
    assert sample.vars["P1"] == 0 and sample.vars["P2"] == 0 and sample.vars["P3"] == 270
    assert sample.vars["Side1"] == "right" and sample.vars["Side2"] == "center" and sample.vars["Side3"] == "above"
    balance = 1.0 - 3.0 / 9.5
    assert abs(float(sample.vars["B1"]) - balance) < 1e-6
    assert sample.vars["B2"] == 1
    assert abs(float(sample.vars["B3"]) - balance) < 1e-6
    assert sample.vars["BA"] == 200 and sample.vars["DA"] == 200
    assert abs(float(sample.vars["BR"]) - 19.5) < 1e-6 and abs(float(sample.vars["BC"]) - 24.5) < 1e-6
    assert abs(float(sample.vars["DR"]) - 19.5) < 1e-6 and abs(float(sample.vars["DC"]) - 14.5) < 1e-6
    assert sample.vars["KeepN"] == 1 and sample.vars["KeepA"] == 400
    assert abs(float(sample.vars["KeepC"]) - 19.5) < 1e-6
    assert sample.vars["MidN"] == 1 and sample.vars["MidA"] == 200
    assert abs(float(sample.vars["MidC"]) - 49.5) < 1e-6
    assert sample.vars["LeftGray"] == 50 and sample.vars["RightGray"] == 200
    sides = next(param for param in REGISTRY["select_bright"].control_in if param.name == "Side").choices
    assert "'right'" in sides and "'center'" in sides
    bad = run_source(
        "Vizion.gen_rectangle1 (Region, 0, 0, 3, 3)\n"
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.gray_offset (Region, Image, DRow, DColumn)\n"
    )
    assert bad.error and "尺寸" in bad.error


def test_hdevelop_batch59():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_edge_contrast_edge_clarity.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["C1"] == 200 and sample.vars["C2"] == 0 and sample.vars["C3"] == -160
    assert sample.vars["A1"] == 200 and sample.vars["A2"] == 0 and sample.vars["A3"] == 160
    assert sample.vars["I1"] == 200 and sample.vars["I2"] == 100 and sample.vars["I3"] == 20
    assert sample.vars["O1"] == 0 and sample.vars["O2"] == 100 and sample.vars["O3"] == 180
    assert sample.vars["KeepN"] == 2
    assert abs(float(sample.vars["Keep1"]) - 19.5) < 1e-6
    assert abs(float(sample.vars["Keep2"]) - 49.5) < 1e-6
    assert sample.vars["SoftN"] == 1 and sample.vars["SoftA"] == 400
    assert abs(float(sample.vars["SoftC"]) - 49.5) < 1e-6
    assert sample.vars["BestA"] == 400 and abs(float(sample.vars["BestR"]) - 19.5) < 1e-6
    assert sample.vars["InnerA"] == 76
    assert abs(float(sample.vars["InnerR"]) - 19.5) < 1e-6
    assert abs(float(sample.vars["InnerC"]) - 19.5) < 1e-6
    assert sample.vars["OuterA"] == 84
    assert [sample.vars[name] for name in ("SharpGray", "OutsideGray", "DarkGray", "HaloGray")] == [200, 0, 20, 180]
    rings = next(param for param in REGISTRY["edge_ring"].control_in if param.name == "Which").choices
    assert "'inner'" in rings and "'outer'" in rings
    bad = run_source(
        "Vizion.gen_rectangle1 (Region, 0, 0, 3, 3)\n"
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.edge_contrast (Region, Image, Contrast)\n"
    )
    assert bad.error and "尺寸" in bad.error


def test_hdevelop_batch60():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_edge_min_weakest_edge.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["M1"] == -10 and sample.vars["M2"] == 200 and sample.vars["M3"] == 0
    assert sample.vars["F1"] == 10 and sample.vars["F2"] == 200 and sample.vars["F3"] == 0
    assert sample.vars["MR1"] == 20 and sample.vars["MC1"] == 34
    assert sample.vars["MR2"] == 15 and sample.vars["MC2"] == 50
    assert sample.vars["MR3"] == 38 and sample.vars["MC3"] == 8
    assert sample.vars["Side1"] == "right" and sample.vars["Side2"] == "above" and sample.vars["Side3"] == "above"
    assert sample.vars["N1"] == 5 and sample.vars["N2"] == 0 and sample.vars["N3"] == 76
    assert sample.vars["WeakObj"] == 2 and sample.vars["WA1"] == 5 and sample.vars["WA2"] == 76
    assert sample.vars["WR1"] == 22 and sample.vars["WC1"] == 34
    assert sample.vars["ClosedN"] == 1 and sample.vars["ClosedA"] == 400
    assert abs(float(sample.vars["ClosedR"]) - 24.5) < 1e-6
    assert abs(float(sample.vars["ClosedC"]) - 59.5) < 1e-6
    assert sample.vars["BrokenN"] == 2
    assert abs(float(sample.vars["BR1"]) - 24.5) < 1e-6
    assert abs(float(sample.vars["BR2"]) - 47.5) < 1e-6
    assert sample.vars["EdgeGray"] == 200 and sample.vars["LeakGray"] == 210
    bad = run_source(
        "Vizion.gen_rectangle1 (Region, 0, 0, 3, 3)\n"
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.edge_min (Region, Image, Contrast)\n"
    )
    assert bad.error and "尺寸" in bad.error
    assert REGISTRY["weak_count"].control_in[0].default == "30"


def test_hdevelop_batch61():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_band_count_bright_enough.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert [sample.vars[name] for name in ("N1", "N2", "N3", "S1")] == [100, 100, 4, 100]
    assert abs(float(sample.vars["R1"]) - 0.5) < 1e-9
    assert sample.vars["R2"] == 1 and abs(float(sample.vars["R3"]) - 0.02) < 1e-9
    assert sample.vars["BA1"] == 100 and abs(float(sample.vars["BandC"]) - 24.5) < 1e-6
    assert sample.vars["BA3"] == 4
    assert sample.vars["KeepN"] == 1 and sample.vars["KeepA"] == 100
    assert abs(float(sample.vars["KeepR"]) - 34.5) < 1e-6
    assert abs(float(sample.vars["KeepC"]) - 14.5) < 1e-6
    assert sample.vars["BR1"] == 10 and sample.vars["BC1"] == 20 and sample.vars["BG1"] == 180
    assert sample.vars["BR3"] == 30 and sample.vars["BC3"] == 40 and sample.vars["BG3"] == 255
    assert sample.vars["DR1"] == 10 and sample.vars["DC1"] == 10 and sample.vars["DG1"] == 40
    assert sample.vars["DR2"] == 30 and sample.vars["DG2"] == 200
    assert sample.vars["DR3"] == 30 and sample.vars["DC3"] == 42 and sample.vars["DG3"] == 20
    assert sample.vars["PA"] == 4
    assert abs(float(sample.vars["PR"]) - 30.5) < 1e-6
    assert abs(float(sample.vars["PC"]) - 40.5) < 1e-6
    assert sample.vars["FlatN"] == 1 and sample.vars["FlatA"] == 100
    assert [sample.vars[name] for name in ("DimGray", "LitGray", "HotGray")] == [40, 180, 255]
    assert REGISTRY["select_band"].control_in[-1].default == "0.5"
    bad = run_source(
        "Vizion.gen_rectangle1 (Region, 0, 0, 3, 3)\n"
        "Vizion.gen_image_const (Image, 'byte', 20, 10)\n"
        "Vizion.band_count (Region, Image, 0, 10, Count)\n"
    )
    assert bad.error and "尺寸" in bad.error


def test_hdevelop_batch62():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gray_mode_main_gray.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert [sample.vars[name] for name in ("G1", "G2", "G3")] == [40, 200, 20]
    assert [sample.vars[name] for name in ("C1", "C2", "C3")] == [100, 100, 196]
    assert abs(float(sample.vars["R1"]) - 0.5) < 1e-9
    assert sample.vars["R2"] == 1 and abs(float(sample.vars["R3"]) - 0.98) < 1e-9
    assert [sample.vars[name] for name in ("H1", "H2", "H3")] == [180, 200, 255]
    assert [sample.vars[name] for name in ("S1", "S2", "S3")] == [100, 0, 4]
    assert [sample.vars[name] for name in ("D1", "D2", "D3")] == [140, 0, 235]
    assert sample.vars["MA1"] == 100
    assert abs(float(sample.vars["MR1"]) - 14.5) < 1e-6
    assert abs(float(sample.vars["MC1"]) - 14.5) < 1e-6
    assert sample.vars["MA3"] == 196
    assert abs(float(sample.vars["MR3"]) - (6778 / 196)) < 1e-6
    assert abs(float(sample.vars["MC3"]) - (9738 / 196)) < 1e-6
    assert sample.vars["KeepN"] == 1 and sample.vars["KeepA"] == 100 and sample.vars["SwapN"] == 1
    assert abs(float(sample.vars["KeepR"]) - 34.5) < 1e-6
    assert abs(float(sample.vars["KeepC"]) - 14.5) < 1e-6
    assert sample.vars["MixN"] == 1 and sample.vars["MixA"] == 200
    assert abs(float(sample.vars["MixR"]) - 14.5) < 1e-6
    assert abs(float(sample.vars["MixC"]) - 19.5) < 1e-6
    assert REGISTRY["select_mixed"].control_in[0].default == "1"


def test_hdevelop_batch63():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_row_runs_horizontal_and_vertical.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert [sample.vars[name] for name in ("RN1", "RN2", "RN3")] == [3, 1, 1]
    assert [sample.vars[name] for name in ("CN1", "CN2", "CN3")] == [1, 3, 1]
    assert [sample.vars[name] for name in ("RL1", "RL2", "RL3")] == [6, 15, 10]
    assert [sample.vars[name] for name in ("CL1", "CL2", "CL3")] == [20, 6, 10]
    assert sample.vars["RG1"] == 2 and sample.vars["RG2"] == 0
    assert sample.vars["CG2"] == 2 and sample.vars["CG3"] == 0
    assert sample.vars["RowKeep"] == 1 and sample.vars["RowA"] == 240
    assert abs(float(sample.vars["RowR"]) - 20.5) < 1e-6
    assert abs(float(sample.vars["RowC"]) - 19.5) < 1e-6
    assert sample.vars["ColKeep"] == 1 and sample.vars["ColA"] == 210
    assert sample.vars["ColR"] == 47
    assert abs(float(sample.vars["ColC"]) - (4185 / 210)) < 1e-6
    assert sample.vars["TA"] == 120
    assert abs(float(sample.vars["TR"]) - 26.5) < 1e-6
    assert abs(float(sample.vars["TC"]) - 19.5) < 1e-6
    assert sample.vars["WA"] == 90 and sample.vars["WR"] == 47
    assert abs(float(sample.vars["WC"]) - 26.5) < 1e-6
    directions = next(param for param in REGISTRY["select_stripes"].control_in if param.name == "Direction").choices
    assert "'row'" in directions and "'column'" in directions


def test_hdevelop_batch64():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_symmetry_score_left_right_symmetry.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["HC1"]) - (368 / 384)) < 1e-9
    assert sample.vars["HC2"] == 1 and sample.vars["HR1"] == 1 and sample.vars["HR2"] == 1
    assert abs(float(sample.vars["HB1"]) - (368 / 384)) < 1e-9
    assert sample.vars["MC1"] == 16 and sample.vars["MC2"] == 0 and sample.vars["MR1"] == 0
    assert abs(float(sample.vars["CR1"]) - 19.5) < 1e-6 and abs(float(sample.vars["CC1"]) - 19.5) < 1e-6
    assert abs(float(sample.vars["CR2"]) - 14.5) < 1e-6 and abs(float(sample.vars["CC2"]) - 54.5) < 1e-6
    assert sample.vars["AsymN"] == 1 and sample.vars["AA"] == 16
    assert abs(float(sample.vars["AR"]) - 19.5) < 1e-6 and abs(float(sample.vars["AC"]) - 11.5) < 1e-6
    assert sample.vars["CA"] == 368
    assert sample.vars["KeepN"] == 1 and sample.vars["KeepA"] == 100
    assert abs(float(sample.vars["KeepR"]) - 14.5) < 1e-6 and abs(float(sample.vars["KeepC"]) - 54.5) < 1e-6
    assert sample.vars["OddN"] == 1 and sample.vars["OddA"] == 384
    assert abs(float(sample.vars["OddR"]) - 19.5) < 1e-6
    assert abs(float(sample.vars["OddC"]) - (7360 / 384)) < 1e-6
    assert sample.vars["FlipGray"] == 255 and sample.vars["GoneGray"] == 0
    axes = next(param for param in REGISTRY["symmetry_score"].control_in if param.name == "Axis").choices
    assert "'column'" in axes and "'row'" in axes and "'both'" in axes


def test_hdevelop_batch65():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_top_point_four_endpoints.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["TR1"] == 10 and sample.vars["TC1"] == 10
    assert sample.vars["TR2"] == 30 and sample.vars["TC2"] == 41
    assert sample.vars["BR1"] == 19 and sample.vars["BC1"] == 10
    assert sample.vars["LR1"] == 10 and sample.vars["LC1"] == 10
    assert sample.vars["LR2"] == 31 and sample.vars["LC2"] == 40
    assert sample.vars["RR1"] == 10 and sample.vars["RC1"] == 29
    assert sample.vars["RR2"] == 30 and sample.vars["RC2"] == 49
    assert sample.vars["TA1"] == 20 and sample.vars["TER1"] == 10
    assert abs(float(sample.vars["TEC1"]) - 19.5) < 1e-6
    assert sample.vars["TA2"] == 9 and abs(float(sample.vars["TEC2"]) - 45) < 1e-6
    assert sample.vars["LA"] == 9 and sample.vars["LER"] == 35 and sample.vars["LEC"] == 40
    assert sample.vars["RA"] == 10
    assert abs(float(sample.vars["RER"]) - 14.5) < 1e-6 and sample.vars["REC"] == 29
    assert sample.vars["EdgeGray"] == 255 and sample.vars["BelowGray"] == 0


def test_hdevelop_batch66():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hole_count_hole_count_and_size.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["C1"] == 1 and sample.vars["C2"] == 0 and sample.vars["C3"] == 2
    assert sample.vars["OC"] == 0
    assert abs(float(sample.vars["R1"]) - (100 / 900)) < 1e-9
    assert sample.vars["R2"] == 0
    assert abs(float(sample.vars["R3"]) - (125 / 1200)) < 1e-9
    assert sample.vars["HoleN"] == 3
    assert sample.vars["FA"] == 100
    assert abs(float(sample.vars["FR"]) - 24.5) < 1e-6 and abs(float(sample.vars["FC"]) - 24.5) < 1e-6
    assert sample.vars["SA"] == 25 and sample.vars["SR"] == 32 and sample.vars["SC"] == 62
    assert sample.vars["BigN"] == 2 and sample.vars["BA"] == 100
    assert abs(float(sample.vars["BR"]) - 19.5) < 1e-6 and abs(float(sample.vars["BC"]) - 59.5) < 1e-6
    assert sample.vars["HoledN"] == 1 and sample.vars["HA"] == 1075
    assert abs(float(sample.vars["HRow"]) - (32650 / 1075)) < 1e-6
    assert abs(float(sample.vars["HCol"]) - (69900 / 1075)) < 1e-6
    assert sample.vars["SolidN"] == 1 and sample.vars["SolidA"] == 100
    assert abs(float(sample.vars["SolidR"]) - 46.5) < 1e-6 and abs(float(sample.vars["SolidC"]) - 14.5) < 1e-6
    assert sample.vars["HollowN"] == 1 and sample.vars["HollowA"] == 800
    assert abs(float(sample.vars["HollowR"]) - 24.5) < 1e-6
    assert sample.vars["FillA"] == 900 and abs(float(sample.vars["FillR"]) - 24.5) < 1e-6
    assert sample.vars["OA"] == 230 and sample.vars["OFA"] == 230
    assert sample.vars["HoleGray"] == 255 and sample.vars["WallGray"] == 0


def test_hdevelop_batch67():
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_window_inside_stays_inside_window.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["IA1"] == 100 and sample.vars["IA2"] == 100 and sample.vars["IA3"] == 0
    assert sample.vars["OA1"] == 0 and sample.vars["OA2"] == 100 and sample.vars["OA3"] == 100
    assert sample.vars["R1"] == 1 and abs(float(sample.vars["R2"]) - 0.5) < 1e-9 and sample.vars["R3"] == 0
    assert sample.vars["D1"] == 0 and sample.vars["D2"] == 10 and sample.vars["D3"] == 15
    assert sample.vars["CD2"] == 10 and sample.vars["CD3"] == 30
    assert sample.vars["OverN"] == 2 and sample.vars["OVA1"] == 100 and sample.vars["OVA2"] == 100
    assert abs(float(sample.vars["OVR1"]) - 14.5) < 1e-6 and abs(float(sample.vars["OVC1"]) - 34.5) < 1e-6
    assert abs(float(sample.vars["OVR2"]) - 49.5) < 1e-6
    assert sample.vars["InsideN"] == 1 and sample.vars["InA"] == 100
    assert abs(float(sample.vars["InR"]) - 26.5) < 1e-6 and abs(float(sample.vars["InC"]) - 26.5) < 1e-6
    assert sample.vars["OutsideN"] == 1 and sample.vars["OutA"] == 100
    assert abs(float(sample.vars["OutR"]) - 49.5) < 1e-6
    assert sample.vars["CrossN"] == 1 and sample.vars["CA"] == 200
    assert abs(float(sample.vars["CR"]) - 19.5) < 1e-6 and abs(float(sample.vars["CC"]) - 34.5) < 1e-6
    assert sample.vars["OutGray"] == 255 and sample.vars["InGray"] == 0
    metric = next(param for param in REGISTRY["window_depth"].control_in if param.name == "Metric").choices
    assert "'chessboard'" in metric and "'city-block'" in metric


def test_hdevelop_batch68():
    import math

    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_major_phi_major_axis_direction.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["P1"] == 0 and abs(float(sample.vars["P2"]) - math.pi / 2) < 1e-9
    assert sample.vars["P3"] == 0
    assert abs(float(sample.vars["P4"]) - 0.517887199575716) < 1e-9
    assert sample.vars["L1"] == 20 and sample.vars["L2"] == 20 and sample.vars["L3"] == 10
    assert abs(float(sample.vars["L4"]) - 26.298069033812887) < 1e-6
    assert sample.vars["M1"] == 5 and abs(float(sample.vars["M4"]) - 8.456131597523353) < 1e-6
    assert sample.vars["R1"] == 4 and sample.vars["R3"] == 1
    assert abs(float(sample.vars["R4"]) - 3.109940843578536) < 1e-6
    assert sample.vars["ER1"] == 12 and sample.vars["EC1"] == 10
    assert sample.vars["ER2"] == 12 and sample.vars["EC2"] == 29
    assert sample.vars["VR1"] == 10 and sample.vars["VC1"] == 52
    assert sample.vars["VR2"] == 29 and sample.vars["VC2"] == 52
    assert sample.vars["TR1"] == 22 and sample.vars["TC1"] == 50
    assert sample.vars["TR2"] == 38 and sample.vars["TC2"] == 70
    assert sample.vars["LevelN"] == 2
    assert sample.vars["UprightN"] == 1 and sample.vars["UA"] == 100
    assert abs(float(sample.vars["UR"]) - 19.5) < 1e-6 and sample.vars["UC"] == 52
    assert sample.vars["SlantN"] == 1 and sample.vars["SA"] == 193
    assert abs(float(sample.vars["SR"]) - 30) < 1e-6 and abs(float(sample.vars["SC"]) - 60) < 1e-6
    assert sample.vars["LooseN"] == 3
    assert sample.vars["LongN"] == 3 and sample.vars["LA"] == 193
    assert abs(float(sample.vars["LR"]) - 30) < 1e-6
    assert sample.vars["SpineA"] == 20 and sample.vars["SpineR"] == 12
    assert abs(float(sample.vars["SpineC"]) - 19.5) < 1e-6
    assert sample.vars["AxisGray"] == 255 and sample.vars["OffGray"] == 0
    directions = next(param for param in REGISTRY["select_axis"].control_in if param.name == "Direction").choices
    assert "'horizontal'" in directions and "'vertical'" in directions and "'diagonal'" in directions


def test_hdevelop_batch69():
    import numpy as np

    from app.engine.objects import Image
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_edge_enhance_sharpen_edges_keep_flat.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Flat"] == 100 and sample.vars["DarkSharp"] == 73 and sample.vars["BrightSharp"] == 207
    assert sample.vars["SpikeGate"] == 106 and sample.vars["SpikeOpen"] == 111 and sample.vars["SpikeBox"] == 112
    assert sample.vars["DarkCap"] == 88 and sample.vars["BrightCap"] == 192
    assert sample.vars["BandFlat"] == 100 and sample.vars["DarkBand"] == 87 and sample.vars["BrightBand"] == 193
    assert sample.vars["MaskFlat"] == 100 and sample.vars["DarkMask"] == 73
    assert sample.vars["SpikeKept"] == 101 and sample.vars["EdgeKept"] == 180
    assert sample.vars["ResFlat"] == 128 and sample.vars["ResDark"] == 101 and sample.vars["MapBright"] == 155
    assert abs(float(sample.vars["MeanS"]) - 0.675) < 1e-6 and sample.vars["MaxS"] == 27
    assert abs(float(sample.vars["MeanB"]) - 1.3) < 1e-6 and sample.vars["MaxB"] == 19
    assert sample.vars["HotN"] == 1 and sample.vars["HotA"] == 120
    assert abs(float(sample.vars["HotR"]) - 29.5) < 1e-6 and abs(float(sample.vars["HotC"]) - 39.5) < 1e-6
    defaults = [param.default for param in REGISTRY["edge_enhance"].control_in]
    assert defaults == ["1.2", "1", "2", "0"]
    flat = np.zeros((16, 16, 3), np.uint8)
    flat[:, :] = (20, 40, 90)
    same = REGISTRY["edge_enhance"].fn(Image(flat), 1.2, 1, 2, 0, ctx=None).iconic[0]
    assert np.array_equal(same.data, flat)
    color = flat.copy()
    color[:, 8:] = (20, 40, 200)
    edged = REGISTRY["edge_enhance"].fn(Image(color), 1.2, 1, 8, 0, ctx=None).iconic[0]
    assert np.array_equal(edged.data[:, 0], color[:, 0])
    assert np.array_equal(edged.data[:, -1], color[:, -1])
    assert not np.array_equal(edged.data[:, 7], color[:, 7])


def test_csharp_region_tuple():
    from app.engine.objects import Region

    source = """
Vizion.gen_image_const (Image, 'byte', 80, 60)
Vizion.gen_rectangle1 (ho_A, 10, 10, 19, 19)
Vizion.gen_rectangle1 (ho_B, 30, 30, 39, 49)
ho_Regions := [ho_A, ho_B]
Vizion.count_obj (ho_Regions, hv_Count)
Vizion.area_center (ho_Regions, Area, Row, Column)
hv_Area := Area
A1 := Area[1]
A2 := Area[2]
Z := hv_Area[0]
N := hv_Area[1]
ho_One := ho_Regions[1]
ho_Two := ho_Regions[2]
Vizion.area_center (ho_One, OneArea, OneRow, OneCol)
Vizion.area_center (ho_Two, TwoArea, TwoRow, TwoCol)
Vizion.dev_display (ho_Regions)
"""
    sample = run_source(source)
    assert sample.error is None, sample.error
    assert sample.vars["hv_Count"] == 2
    assert isinstance(sample.vars["ho_Regions"], Region)
    assert sample.vars["ho_Regions"].count() == 2
    assert sample.vars["A1"] == 100 and sample.vars["A2"] == 200
    assert sample.vars["Z"] == 100 and sample.vars["N"] == 200
    assert sample.vars["OneArea"] == [100]
    assert abs(float(sample.vars["OneRow"][0]) - 14.5) < 1e-6
    assert sample.vars["TwoArea"] == [200]
    assert abs(float(sample.vars["TwoRow"][0]) - 34.5) < 1e-6
    assert abs(float(sample.vars["TwoCol"][0]) - 39.5) < 1e-6
    state = sample.public_state()
    iconic = next(item for item in state["iconic"] if item["name"] == "ho_Regions")
    assert iconic["kind"] == "region" and "2 個物件" in iconic["info"]
    objects = {shape.get("object") for shape in state["display"]["shapes"] if shape.get("kind") == "region"}
    assert objects >= {0, 1}
    colors = {shape.get("color") for shape in state["display"]["shapes"] if shape.get("kind") == "region"}
    assert len(colors) >= 2


def test_csharp_xld_tuple():
    from app.engine.objects import XLD

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_contour_polygon_xld_contour_tuple_stays_hobject.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert isinstance(sample.vars["ho_XLDs"], XLD)
    assert len(sample.vars["ho_XLDs"].contours) == 2
    assert sample.vars["ho_XLDs"].multi_color
    assert sample.vars["hv_Count"] == 2
    assert sample.vars["First"] == 60
    assert sample.vars["Length"][1] == 50
    assert sample.vars["OneN"] == 1
    assert isinstance(sample.vars["ho_Kept"], XLD)
    assert len(sample.vars["ho_Kept"].contours) == 1
    state = sample.public_state()
    iconic = next(item for item in state["iconic"] if item["name"] == "ho_XLDs")
    assert iconic["role"] == "HObject" and iconic["kind"] == "xld" and "2 條輪廓" in iconic["info"]
    shapes = [shape for shape in state["display"]["shapes"] if shape.get("kind") == "xld"]
    assert {shape.get("object") for shape in shapes} >= {0, 1}
    assert len({shape.get("color") for shape in shapes}) >= 2
    preview = sample.view_object("ho_XLDs")
    preview_shapes = [shape for shape in preview["display"]["shapes"] if shape.get("kind") == "xld"]
    assert len({shape.get("color") for shape in preview_shapes}) >= 2


def test_open_hdevelop_skips_password():
    from app.engine.hdevelop_file import convert_hdevelop

    raw = """<?xml version="1.0" encoding="UTF-8"?>
<hdevelop file_version="1.2" halcon_version="21.05.0.0">
<procedure name="main">
<interface/>
<body>
<c>* 開啟 HALCON 程式</c>
<l>gen_image_const (Image, 'byte', 8, 8)</l>
<l>cut (Image, Region)//二值化</l>
<l>area_center (Region, Area, Row, Column)</l>
</body>
</procedure>
<procedure name="cut">
<interface>
<io><par name="Image" base_type="iconic" dimension="0"/></io>
<oo><par name="Region" base_type="iconic" dimension="0"/></oo>
</interface>
<body>
<l>threshold (Image, Region, 0, 10)</l>
</body>
</procedure>
<procedure name="secret">
<protection version="2">QUJDREVG</protection>
<body>
<l>threshold (Image, Hidden, 0, 1)</l>
</body>
</procedure>
</hdevelop>
""".encode("utf-8")
    source, skipped = convert_hdevelop(raw)
    assert skipped == ["secret"]
    assert "程序 secret 有密碼，沒有打開" in source
    assert "Hidden" not in source and "QUJDREVG" not in source
    assert "Vizion.gen_image_const" in source and "Vizion.threshold" in source
    assert "procedure cut (Image : Region :  : )" in source
    assert "* 二值化" in source
    sample = run_source(source)
    assert sample.error is None, sample.error
    assert sample.vars["Area"] == [64]
    locked, names = convert_hdevelop(b"\x00\x01secret-bytes")
    assert names == ["整份程式"]
    assert "沒有打開" in locked
    assert "secret-bytes" not in locked
    library = """<?xml version="1.0" encoding="UTF-8"?>
<hdevelop file_version="1.0" halcon_version="11.0.3.4">
<library/>
<procedure name="locked_lib">
<interface>
<io><par name="Image"/></io>
</interface>
<xb>CIPHERTEXT</xb>
</procedure>
<procedure name="plain_lib">
<interface>
<oo><par name="Region"/></oo>
</interface>
<body>
<l>gen_image_const (Image, 'byte', 4, 4)</l>
<l>threshold (Image, Region, 0, 10)</l>
</body>
</procedure>
</hdevelop>
""".encode("utf-8")
    lib_source, lib_skipped = convert_hdevelop(library)
    assert lib_skipped == ["locked_lib"]
    assert "程序 locked_lib 有密碼，沒有打開" in lib_source
    assert "CIPHERTEXT" not in lib_source
    assert "procedure plain_lib ( : Region :  : )" in lib_source
    assert "Vizion.threshold" in lib_source


def test_hdevelop_batch70():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_area_center_catch_error_after_procedure.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Area"] == [400]
    assert abs(float(sample.vars["Row"][0]) - 19.5) < 1e-6
    assert abs(float(sample.vars["Column"][0]) - 19.5) < 1e-6
    assert sample.vars["Answer"] == 42
    assert sample.vars["Temp"] == 9
    assert sample.vars["Safe"] == 1
    assert sample.vars["Kept"] == 43
    assert "Missing" in sample.vars["Exception"]
    early = run_source("early (4, Out)\nprocedure early (: : Value : Result)\n    Result := Value + 1\nendprocedure\n")
    assert early.error is None, early.error
    assert early.vars["Out"] == 5
    skipped = run_source("try\n    Seen := 3\ncatch (Exception)\n    Seen := 9\nendtry\n")
    assert skipped.error is None, skipped.error
    assert skipped.vars["Seen"] == 3 and "Exception" not in skipped.vars
    bare = run_source("Bad := Missing\n")
    assert bare.error and bare.vars.get("Bad") is None


def test_hdevelop_batch71():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_diff_region_difference_from_golden.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Area"] == [100]
    assert abs(float(sample.vars["Row"][0]) - 14.5) < 1e-6
    assert abs(float(sample.vars["Column"][0]) - 14.5) < 1e-6
    assert sample.vars["Maximum"] == 80
    assert abs(float(sample.vars["Mean"]) - (8010 / 4800)) < 1e-9
    negative = run_source(
        "Vizion.gen_image_const (A, 'byte', 8, 8)\nVizion.gen_image_proto (A, B, 1)\nVizion.diff_region (A, B, D, -1)\n"
    )
    assert "不能是負的" in (negative.error or "")


def test_hdevelop_batch72():
    import cv2

    from app.engine.chapters.code_ops import _decode_qr

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_qrcode_print_qr_read_back.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Text"] == "VISDEV"
    assert len(sample.vars["Symbol"].contours) == 1
    raw = cv2.QRCodeEncoder.create().encode("VISDEV")
    hits = _decode_qr(raw)
    assert hits and hits[0][0] == "VISDEV"


def test_var_roles():
    sample = run_source(
        "Vizion.gen_image_const (Base, 'byte', 20, 20)\n"
        "Vizion.gen_rectangle1 (Box, 0, 0, 4, 4)\n"
        "ho_Box := Box\n"
        "hv_N := 3\n"
        "Vizion.dev_display (ho_Box)\n"
    )
    assert sample.error is None, sample.error
    state = sample.public_state()
    ho = next(item for item in state["iconic"] if item["name"] == "ho_Box")
    box = next(item for item in state["iconic"] if item["name"] == "Box")
    hv = next(item for item in state["control"] if item["name"] == "hv_N")
    assert ho["role"] == "HObject" and ho["kind"] == "region"
    assert box["role"] == "region"
    assert hv["type"] == "HTuple"


def test_hdevelop_batch73():
    import math

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_ridge_region_dark_scratch_direction.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Area"] == [40]
    assert abs(float(sample.vars["Row"][0]) - 29.5) < 1e-6
    assert abs(float(sample.vars["Column"][0]) - 40) < 1e-6
    assert int(sample.vars["Peak"]) == 56 and int(sample.vars["Side"]) == 0
    assert sample.vars["Count"] == 1
    assert abs(float(sample.vars["Length"]) - 39) < 1e-6
    assert abs(float(sample.vars["Phi"]) - math.pi / 2) < 1e-9
    assert abs(float(sample.vars["CenterRow"]) - 29.5) < 1e-6
    assert abs(float(sample.vars["CenterCol"]) - 40) < 1e-4
    assert sample.vars["WrongCount"] == 0 and sample.vars["FlatCount"] == 0
    assert sample.vars["LightArea"] == [69]
    assert abs(float(sample.vars["LightRow"][0]) - 30) < 1e-6
    assert abs(float(sample.vars["LightCol"][0]) - 40) < 1e-6
    assert 60 < float(sample.vars["Longest"]) < 70
    assert sample.vars["BlockCount"] == 0
    negative = run_source("Vizion.gen_image_const (A, 'byte', 16, 16)\nVizion.ridge_region (A, R, 1, -1, 'dark')\n")
    assert "不能是負的" in (negative.error or "")


def test_hdevelop_batch74():
    import math

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_edge_region_edge_between_pixels.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Area"] == [60]
    assert abs(float(sample.vars["Row"][0]) - 29.5) < 1e-6
    assert abs(float(sample.vars["Column"][0]) - 40) < 1e-6
    assert int(sample.vars["Peak"]) == 58 and int(sample.vars["Side"]) == 0
    assert sample.vars["Count"] == 1
    assert abs(float(sample.vars["Length"]) - 59) < 1e-6
    assert abs(float(sample.vars["Phi"]) - math.pi / 2) < 1e-9
    assert abs(float(sample.vars["CenterRow"]) - 29.5) < 1e-6
    assert abs(float(sample.vars["CenterCol"]) - 39.5) < 1e-6
    assert abs(float(sample.vars["Longest"]) - 59) < 1e-6
    assert sample.vars["AcrossCount"] == 1
    assert abs(float(sample.vars["AcrossLength"]) - 79) < 1e-6
    assert abs(float(sample.vars["AcrossPhi"])) < 1e-9
    assert abs(float(sample.vars["AcrossRow"]) - 29.5) < 1e-6
    assert abs(float(sample.vars["AcrossCol"]) - 39.5) < 1e-4
    assert sample.vars["SoftCount"] == 0 and sample.vars["FlatCount"] == 0
    negative = run_source("Vizion.gen_image_const (A, 'byte', 16, 16)\nVizion.edge_region (A, R, 1, -1)\n")
    assert "不能是負的" in (negative.error or "")


def test_hdevelop_batch75():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_phase_correlation_shift_under_one_pixel.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Row"]) - 1.5) < 0.02
    assert sample.vars["Column"] == 0
    assert abs(float(sample.vars["SideCol"]) - 1.5) < 0.02
    assert sample.vars["SideRow"] == 0
    assert sample.vars["FarRow"] == 3 and sample.vars["FarCol"] == 4
    assert sample.vars["SameRow"] == 0 and sample.vars["SameCol"] == 0
    assert float(sample.vars["SameScore"]) > 0.9
    assert abs(float(sample.vars["BackRow"]) - float(sample.vars["Row"])) < 1e-9
    assert int(sample.vars["Orig"]) == 200 and int(sample.vars["Restored"]) == 200
    mismatch = run_source(
        "Vizion.gen_image_const (A, 'byte', 16, 16)\nVizion.gen_image_const (B, 'byte', 20, 16)\n"
        "Vizion.phase_correlation (A, B, Row, Column, Score)\n"
    )
    assert "尺寸要相同" in (mismatch.error or "")


def test_hdevelop_batch76():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_contour_polygon_xld_line_outlier_stays_straight.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Row1"][0]) - 10) < 1e-6
    assert abs(float(sample.vars["Row2"][0]) - 10) < 1e-6
    columns = sorted(float(item) for item in (sample.vars["Col1"][0], sample.vars["Col2"][0]))
    assert abs(columns[0]) < 1e-6 and abs(columns[1] - 30) < 1e-6
    ends = sorted(
        (float(sample.vars[row][0]), float(sample.vars[col][0]))
        for row, col in (("D1", "C1"), ("D2", "C2"))
    )
    assert abs(ends[0][0]) < 1e-6 and abs(ends[0][1]) < 1e-6
    assert abs(ends[1][0] - 30) < 1e-6 and abs(ends[1][1] - 30) < 1e-6


def test_hdevelop_batch77():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_contour_polygon_xld_arc_outlier.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Row"][0]) - 80) < 1e-4
    assert abs(float(sample.vars["Column"][0]) - 90) < 1e-4
    assert abs(float(sample.vars["Radius"][0]) - 40) < 1e-4
    flat = run_source(
        "Vizion.gen_contour_polygon_xld (Flat, [0, 1, 2], [0, 1, 2])\n"
        "Vizion.fit_circle_contour_xld (Flat, Miss, MRow, MCol, MRad)\n"
    )
    assert "沒有可以擬合圓" in (flat.error or "")


def test_hdevelop_batch78():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_contour_region_xld_boundary_at_half_pixel.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["CenterCount"] == 1
    assert sample.vars["RowMin"] == 9.5 and sample.vars["ColMin"] == 9.5
    assert sample.vars["RowMax"] == 29.5 and sample.vars["ColMax"] == 29.5
    assert sample.vars["BorderMin"] == 10 and sample.vars["BorderMax"] == 29
    assert sample.vars["FrameCount"] == 1
    assert sample.vars["FrameMin"] == 9.5 and sample.vars["FrameMax"] == 40.5
    center_radius = float(sample.vars["CRad"][0])
    border_radius = float(sample.vars["BRad"][0])
    assert abs(center_radius - 15) < 0.15
    assert abs(border_radius - 14.56) < 0.02
    assert abs(center_radius - 15) < abs(border_radius - 15)


def test_hdevelop_batch79():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_contour_polygon_xld_gauss_weight_ignores_outlier.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["RowMin"] == 10 and sample.vars["RowMax"] == 10
    assert sample.vars["ColMin"] == 0 and sample.vars["ColMax"] == 30
    assert abs(float(sample.vars["PlainMean"]) - 12.4) < 1e-6
    assert sample.vars["CleanMin"] == 10 and sample.vars["CleanMax"] == 10


def test_hdevelop_batch80():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_derivate_gauss_slope1_derivative_is_129.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert int(sample.vars["Deriv"]) == 129 and int(sample.vars["ZeroY"]) == 128
    assert int(sample.vars["Deriv2"]) == 128 and int(sample.vars["Amp"]) == 1
    assert int(sample.vars["DerivY"]) == 129 and int(sample.vars["ZeroX"]) == 128
    bad = run_source("Vizion.gen_image_const (A, 'byte', 16, 16)\nVizion.derivate_gauss (A, D, 0, 'x')\n")
    assert "必須大於 0" in (bad.error or "")


def test_hdevelop_batch81():
    import numpy as np

    from app.engine.chapters.measure_ops import _gather
    from app.engine.gauge import sample_bicubic

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_get_grayval_interpolated_half_pixel_is_not_the_average.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["OnPixel"] == 200
    assert sample.vars["Cubic"] == 112.5
    assert sample.vars["Linear"] == 100
    assert sample.vars["Near"] == 0
    gray = np.zeros((32, 32), np.uint8)
    gray[10, 10] = 200
    cubic = _gather(gray, None, np.array([10.5]), np.array([10.0]), "bicubic")
    linear = _gather(gray, None, np.array([10.5]), np.array([10.0]), "bilinear")
    assert abs(float(cubic[0]) - 112.5) < 1e-9
    assert abs(float(linear[0]) - 100) < 1e-9
    color = np.zeros((8, 8, 3), np.uint8)
    color[3, 3] = (10, 20, 30)
    got = sample_bicubic(color, np.array([3.0]), np.array([3.0]))
    assert list(got[0]) == [10, 20, 30]
    bad = run_source("Vizion.gen_image_const (A, 'byte', 16, 16)\nVizion.get_grayval_interpolated (A, 1, 1, 'cubic', G)\n")
    assert "只能是" in (bad.error or "")


def test_hdevelop_batch82():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_teach_shape_model_least_squares_aligns_shape.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    truth = 59.9
    interpolated = float(sample.vars["ICol"][0])
    fitted = float(sample.vars["LCol"][0])
    assert abs(fitted - truth) < 0.1
    assert abs(fitted - truth) < abs(interpolated - truth)
    assert abs(float(sample.vars["LRow"][0]) - 44.5) < 0.05
    assert abs(float(sample.vars["LAng"][0])) < 1e-3


def test_hdevelop_batch83():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gauss_image_least_squares_row_near_39.25.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    truth_row = 39.25
    truth_col = 47.85
    fitted_row = float(sample.vars["LRow"][0])
    fitted_col = float(sample.vars["LCol"][0])
    assert abs(fitted_row - truth_row) < 0.1
    assert abs(float(sample.vars["IRow"][0]) - 39.23) < 1e-9
    assert abs(fitted_row - truth_row) < abs(float(sample.vars["IRow"][0]) - truth_row)
    assert abs(fitted_col - truth_col) < 0.15
    assert float(sample.vars["IScore"][0]) > float(sample.vars["LScore"][0])
    assert abs(float(sample.vars["LAng"][0])) < 1e-6


def test_hdevelop_batch84():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_create_metrology_model_caliper_averages_spike.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert float(sample.vars["ThinBegin"][0]) == 45.5
    assert float(sample.vars["ThinEnd"][0]) == 45.5
    assert float(sample.vars["WideBegin"][0]) == 39.5
    assert float(sample.vars["WideEnd"][0]) == 39.5


def test_hdevelop_batch85():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_points_foerstner_corner_inside_pixel.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error

    def one(value):
        if isinstance(value, (list, tuple)):
            return float(value[0])
        return float(value)

    still_row = one(sample.vars["StillRow"])
    still_col = one(sample.vars["StillCol"])
    corner_row = one(sample.vars["CornerRow"])
    corner_col = one(sample.vars["CornerCol"])
    truth_row = still_row + 0.4
    truth_col = still_col + 0.35
    assert abs(still_row - 24.13) < 0.05
    assert abs(still_col - 36.13) < 0.05
    assert abs(corner_row - truth_row) < 0.2
    assert abs(corner_col - truth_col) < 0.2
    assert abs(corner_row - truth_row) < abs(25 - truth_row)
    assert abs(corner_col - truth_col) < abs(37 - truth_col)


def test_hdevelop_batch86():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_vector_to_hom_mat2d_affine_ignores_outlier.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Affine"] == [1, 0, 2, 0, 1, 5]
    assert sample.vars["Row0"] == 2 and sample.vars["Col0"] == 5
    assert sample.vars["Row1"] == 12 and sample.vars["Col1"] == 20


def test_hdevelop_batch87():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hom_vector_to_proj_hom_mat2d_projective_ignores_outlier.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Row0"] == 3 and sample.vars["Col0"] == 4
    assert sample.vars["Row1"] == 13 and sample.vars["Col1"] == 14


def test_hdevelop_batch88():
    import math

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_ellipse_contour_xld_ellipse_arc_outlier.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["BRow"][0]) - 80) < 0.02
    assert abs(float(sample.vars["BCol"][0]) - 90) < 0.02
    assert abs(float(sample.vars["BR1"][0]) - 40) < 0.02
    assert abs(float(sample.vars["BR2"][0]) - 18) < 0.02
    delta = abs(float(sample.vars["BPhi"][0]) - 0.4) % math.pi
    assert min(delta, math.pi - delta) < 0.02


def test_hdevelop_batch89():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_tuple_concat_rectangle_edge_outlier.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["FRow"][0] == 50 and sample.vars["FCol"][0] == 80
    assert sample.vars["FPhi"][0] == 0
    assert sample.vars["FL1"][0] == 36 and sample.vars["FL2"][0] == 18
    assert sample.vars["SRow"][0] == 43 and sample.vars["SL2"][0] == 25


def test_hdevelop_batch90():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_fit_plane_region_spike_does_not_tilt_plane.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Alpha"])) < 1e-9
    assert abs(float(sample.vars["Beta"])) < 1e-9
    assert abs(float(sample.vars["Gamma"])) < 1e-9
    assert abs(float(sample.vars["RawA"]) + 20.0 / 3.0) < 1e-6
    assert abs(float(sample.vars["RawB"]) + 20.0 / 3.0) < 1e-6
    assert abs(float(sample.vars["RawG"]) - 460.0 / 9.0) < 1e-6


def test_hdevelop_batch91():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_object_model_3d_from_points_cloud_plane_ignores_outlier.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Nx"])) < 1e-9
    assert abs(float(sample.vars["Ny"])) < 1e-9
    assert abs(float(sample.vars["Nz"]) - 1) < 1e-9
    assert abs(float(sample.vars["Dist"])) < 1e-9


def test_hdevelop_batch92():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_object_model_3d_from_points_axis_ignores_side_point.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Ax"]) - 1) < 1e-9
    assert abs(float(sample.vars["Ay"])) < 1e-9
    assert abs(float(sample.vars["Az"])) < 1e-9


def test_hdevelop_batch93():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_contour_polygon_xld_line_normal_ignores_corner.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["LCol"][1]) - 28.85060845773462) < 1e-6
    assert abs(float(sample.vars["LCol"][3]) - 71.14939154226538) < 1e-6
    assert sample.vars["RCol"][1] == 30 and sample.vars["RCol"][3] == 70
    assert sample.vars["RRow"][1] == 44 and sample.vars["RRow"][2] == 56


def test_hdevelop_batch94():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_points_harris_harris_corner_inside_pixel.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["StillIRow"] == 25 and sample.vars["StillICol"] == 37
    assert sample.vars["MoveIRow"] == 25 and sample.vars["MoveICol"] == 37
    still_row = float(sample.vars["StillRow"])
    still_col = float(sample.vars["StillCol"])
    corner_row = float(sample.vars["CornerRow"])
    corner_col = float(sample.vars["CornerCol"])
    truth_row = still_row + 0.4
    truth_col = still_col + 0.35
    assert abs(still_row - 24.13157) < 0.01
    assert abs(still_col - 36.13157) < 0.01
    assert abs(corner_row - truth_row) < abs(25 - truth_row)
    assert abs(corner_col - truth_col) < abs(37 - truth_col)
    assert abs(corner_row - truth_row) < 0.2
    assert abs(corner_col - truth_col) < 0.2


def test_hdevelop_batch95():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_contour_polygon_xld_smooth_keeps_endpoints.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert all(abs(float(item) - 10) < 1e-6 for item in sample.vars["Row"])
    assert [abs(float(item) - expected) < 1e-6 for item, expected in zip(sample.vars["Col"], [0, 10, 20, 30, 40])] == [True] * 5


def test_hdevelop_batch96():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_zoom_image_size_zoomed_half_pixel_is_112.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Mid"] == 112
    assert sample.vars["Average"] == 100
    assert sample.vars["Kept"] == 200


def test_hdevelop_batch97():
    import math

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_elliptic_axis_wide_bar_0_tall_bar_90.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["WideP"] == 0 and sample.vars["WideA"] == 0
    assert abs(float(sample.vars["TallP"]) - math.pi / 2) < 1e-12
    assert abs(float(sample.vars["TallA"]) - math.pi / 2) < 1e-12
    assert abs(float(sample.vars["WRa"]) - math.sqrt(133)) < 1e-9
    assert abs(float(sample.vars["WRb"]) - math.sqrt(8)) < 1e-9


def test_hdevelop_batch98():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hom_mat2d_identity_pulled_back_half_pixel_is_112.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Mid"] == 112
    assert sample.vars["Average"] == 100
    assert sample.vars["Pulled"] == 112
    assert sample.vars["Row"] == 0
    assert abs(float(sample.vars["Column"]) - 0.5) < 1e-9


def test_hdevelop_batch99():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_local_max_sub_pix_gauss_peak_at_30.33298.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    center = 30 + 1 / 3
    assert sample.vars["FRow"] == 20 and sample.vars["GRow"] == 20
    assert abs(float(sample.vars["FCol"]) - (30 + 1 / 6)) < 1e-9
    assert abs(float(sample.vars["GCol"]) - 30.33298435807228) < 1e-9
    assert abs(float(sample.vars["GCol"]) - center) < abs(float(sample.vars["FCol"]) - center)


def test_hdevelop_batch100():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_points_lepetit_square_corner_inside_pixel.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["BRow"]) - 24.094262617339446) < 1e-9
    assert abs(float(sample.vars["BCol"]) - 36.094262648985456) < 1e-9


def test_hdevelop_batch101():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_create_scaled_shape_model_scale_refined_to_1.08895.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    refined = float(sample.vars["S"])
    still = float(sample.vars["Still"])
    assert still == 1.0
    assert abs(refined - 1.088948198788332) < 1e-9
    assert abs(refined - 1.1) < abs(still - 1.1)


def test_expand_region_splits_overlap():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_expand_region_region_splits_overlap.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Count"] == 2
    assert int(sample.vars["LeftArea"]) == 12
    assert int(sample.vars["RightArea"]) == 8


def test_gen_image_const_real_is_zero():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_image_const_real_zero.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Type"] == "real"
    assert sample.vars["Gray"] == 0
    assert sample.vars["Width"] == 4 and sample.vars["Height"] == 3
    assert sample.vars["ByteType"] == "byte"
    assert sample.vars["ByteGray"] == 0


def test_convert_image_type_real_and_byte():
    import numpy as np

    from app.engine.objects import Image
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_convert_image_type_real_byte_roundtrip.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["RealType"] == "real"
    assert sample.vars["ByteType"] == "byte"
    assert int(sample.vars["Original"]) == 200
    assert int(sample.vars["RealGray"]) == 200
    assert int(sample.vars["ByteGray"]) == 200
    assert int(sample.vars["Outside"]) == 0

    convert = REGISTRY["convert_image_type"].fn
    source = Image(np.array([[0, 200]], np.uint8))
    copied = convert(source, "byte", ctx=None).iconic[0]
    assert copied.data.dtype == np.uint8
    assert int(copied.data[0, 1]) == 200
    assert copied.data is not source.data
    source.data[0, 1] = 7
    assert int(copied.data[0, 1]) == 200

    real = convert(Image(np.array([[10, 200]], np.uint8)), "real", ctx=None).iconic[0]
    assert real.data.dtype == np.float32
    assert float(real.data[0, 0]) == 10 and float(real.data[0, 1]) == 200

    rounded = convert(
        Image(np.array([[-1.2, 1.4, 10.0, 254.6, 300.0]], np.float32)),
        "byte",
        ctx=None,
    ).iconic[0]
    assert rounded.data.dtype == np.uint8
    assert list(int(item) for item in rounded.data[0]) == [0, 1, 10, 255, 255]

    domain = np.zeros((1, 2), np.uint8)
    domain[0, 1] = 255
    kept = convert(Image(np.array([[3, 4]], np.uint8), domain), "real", ctx=None).iconic[0]
    assert kept.domain is not None and int(kept.domain[0, 1]) == 255
    color = np.zeros((1, 1, 3), np.uint8)
    color[0, 0] = (30, 20, 10)
    channels = convert(Image(color), "real", ctx=None).iconic[0]
    assert channels.data.shape == (1, 1, 3)
    assert [float(item) for item in channels.data[0, 0]] == [30.0, 20.0, 10.0]


def test_gen_image_surface_real_keeps_values_past_255():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_image_surface_real_keeps_310.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert int(sample.vars["ByteGray"]) == 255
    assert sample.vars["RealType"] == "real"
    assert int(sample.vars["RealGray"]) == 310
    assert int(sample.vars["ByteBowlGray"]) == 255
    assert int(sample.vars["RealBowlGray"]) == 404


def test_three_checkerboard_views_share_one_calib():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_find_calib_object_three_views_one_board.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert int(sample.vars["AfterOne"]) == 1
    assert int(sample.vars["Count"]) == 3
    assert float(sample.vars["Error"]) < 5
    assert len(sample.vars["Pose0"]) == 7
    assert len(sample.vars["Pose2"]) == 7
    assert len(sample.vars["CamParam"]) == 9
    assert float(sample.vars["CamParam"][0]) > 200
    assert int(sample.vars["ImageWidth"]) == 428 and int(sample.vars["ImageHeight"]) == 356
    assert int(sample.vars["LastWidth"]) == 428 and int(sample.vars["LastHeight"]) == 356
    again = run_source(
        "Vizion.read_image (Front, 'samples/calib_board_front.png')\n"
        "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)\n"
        "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,36')\n"
        "Vizion.find_calib_object (Front, CalibDataID, 0, 0, '', '')\n"
        "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,36')\n"
        "Vizion.get_calib_data (CalibDataID, 'model', 0, 'num_poses', Count)\n"
    )
    assert again.error is None, again.error
    assert int(again.vars["Count"]) == 0
    missing = run_source(
        "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)\n"
        "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,36')\n"
        "Vizion.get_calib_data (CalibDataID, 'model', 0, 'image_size', Size)\n"
    )
    assert missing.error is not None and "還沒有影像" in missing.error
    extent = run_source((ROOT / "samples" / "ViZion_set_calib_data_observ_points_image_size_is_extent.viz").read_text(encoding="utf-8"))
    assert extent.error is None, extent.error
    assert int(extent.vars["Width"]) == 21 and int(extent.vars["Height"]) == 11
    assert int(extent.vars["Count"]) == 1
    kept = run_source(
        "Vizion.read_image (Front, 'samples/calib_board_front.png')\n"
        "Vizion.gen_image_const (Blank, 'byte', 80, 40)\n"
        "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)\n"
        "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,36')\n"
        "Vizion.find_calib_object (Front, CalibDataID, 0, 0, '', '')\n"
        "Vizion.find_calib_object (Blank, CalibDataID, 0, 0, '', '')\n"
    )
    assert kept.error is not None and "找不到棋盤格" in kept.error
    assert len(kept.vars["CalibDataID"].observations) == 1
    assert kept.vars["CalibDataID"].observations[0][2:] == (428, 356)
    descr = run_source(
        "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)\n"
        "Vizion.set_calib_data_calib_object (CalibDataID, 0, 'board.cpd')\n"
    )
    assert descr.error is not None and "讀不到描述檔" in descr.error
    still = run_source(
        "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)\n"
        "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,10')\n"
        "Vizion.get_calib_data (CalibDataID, 'model', 0, 'num_poses', Count)\n"
    )
    assert still.error is None, still.error
    assert int(still.vars["Count"]) == 0


def test_outlier_marks_are_the_points_that_stand_away():
    import numpy as np

    from app.engine.chapters.calib_ops import _outlier_mask

    detected = np.array([[0, 0], [10, 0], [20, 0], [30, 0], [40, 0]], np.float64)
    close = detected.copy()
    assert _outlier_mask(detected, close) is None
    far = detected.copy()
    far[4, 1] = 8
    mask = _outlier_mask(detected, far)
    assert mask is not None and mask.tolist() == [False, False, False, False, True]


def test_clicking_a_calib_panel_shows_that_image():
    session = run_source(
        "Vizion.read_image (Front, 'samples/calib_board_front.png')\n"
        "Vizion.read_image (Left, 'samples/calib_board_left.png')\n"
        "Vizion.read_image (Right, 'samples/calib_board_right.png')\n"
        "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)\n"
        "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,36')\n"
        "Vizion.find_calib_object (Front, CalibDataID, 0, 0, '', '')\n"
        "Vizion.find_calib_object (Left, CalibDataID, 0, 0, '', '')\n"
        "Vizion.find_calib_object (Right, CalibDataID, 0, 0, '', '')\n"
        "Vizion.calibrate_cameras (CalibDataID, Error)\n"
    )
    assert session.error is None, session.error
    stacked = session.public_state()["display"]
    panels = stacked["calibPanels"]
    assert [panel["index"] for panel in panels] == [0, 1, 2]
    assert stacked["calibNote"]
    mean_error = float(session.vars["Error"])
    session.show_calib_view(1)
    one = session.public_state()["display"]
    assert one["width"] == 428 and one["height"] == 356
    assert one["calibBack"] is True
    labels = [shape["text"] for shape in one["shapes"] if shape["type"] == "text"]
    assert any("第 2 組" in text for text in labels)
    assert any("回到全部" in text for text in labels)
    assert float(session.vars["Error"]) == mean_error
    session.show_calib_view(None)
    again = session.public_state()["display"]
    assert again["height"] > 356
    assert [panel["index"] for panel in again["calibPanels"]] == [0, 1, 2]

    calib = session.vars["CalibDataID"]
    shifted = calib.observation_projected[0].copy()
    shifted[0] = shifted[0] + [40.0, 0.0]
    calib.observation_projected[0] = shifted
    session.show_calib_view(None)
    marked = session.public_state()["display"]
    colors = {shape["color"] for shape in marked["shapes"] if shape["type"] == "polyline"}
    assert "#ff3b30" in colors
    assert any("紅色" in shape["text"] for shape in marked["shapes"] if shape["type"] == "text")
    assert float(session.vars["Error"]) == mean_error


def test_long_trace_and_thumb_cache_stay_bounded():
    from app.engine.session import _remember_thumb

    session = Session("for Index := 1 to 2500 by 1\nValue := Index\nendfor\n", ROOT)
    session.trace = True
    session.run()
    assert session.error is None, session.error
    assert session.vars["Value"] == 2500
    assert 1500 <= len(session.log) <= 2000
    assert any("Value := 2500" in line for line in session.log[-5:])
    assert session.log[-1].startswith("執行完成")

    thumbs = {}
    for index in range(80):
        _remember_thumb(thumbs, ("image", index), lambda index=index: f"t{index}")
    assert len(thumbs) <= 48
    assert thumbs[("image", 79)] == "t79"
    assert ("image", 0) not in thumbs
    oldest = next(iter(thumbs))
    kept = thumbs[oldest]
    _remember_thumb(thumbs, oldest, lambda: "unused")
    for index in range(80, 90):
        _remember_thumb(thumbs, ("image", index), lambda index=index: f"t{index}")
    assert thumbs[oldest] == kept
    assert len(thumbs) <= 48


def test_inspect_gauge_template_measures_the_edge():
    session = run_source((ROOT / "samples" / "inspect_gauge.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert session.vars["PCol"] == [107.5]
    assert session.vars["LRow1"] == 199.5
    assert session.vars["LCol1"] == 380.0
    assert session.vars["LCol2"] == 480.0
    assert abs(float(session.vars["Gap"]) - 286.8213729832559) < 1e-6
    assert session.display.base is not None


def test_inspect_caliper_template_measures_diameter_and_size():
    session = run_source((ROOT / "samples" / "inspect_caliper.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert abs(float(session.vars["Diameter"]) - 83.92768304367884) < 1e-6
    assert float(session.vars["Width"]) == 140.0
    assert float(session.vars["Height"]) == 90.0
    assert session.vars["Judge"] == "合格"
    assert int(session.vars["CircleUsed"]) == 66
    assert int(session.vars["RectUsed"]) > 0
    assert session.display.base is not None


def test_gauge_preview_reports_the_rectangle_width():
    from app.main import SESSIONS, GaugePreviewIn, gauge_preview

    session = run_source("Vizion.read_image (Image, 'samples/gauge.png')\nVizion.dev_display (Image)\n")
    assert session.error is None, session.error
    SESSIONS[session.id] = session
    try:
        found = gauge_preview(GaugePreviewIn(
            id=session.id, row1=244.5, col1=340, row2=244.5, col2=520,
            sigma=1, threshold=8, transition="all", select="all",
        ))
    finally:
        SESSIONS.pop(session.id, None)
    assert len(found["hits"]) >= 2
    assert abs(float(found["span"]) - 140) < 1.5


def test_inspect_defect_template_finds_dark_and_bright():
    session = run_source((ROOT / "samples" / "inspect_defect.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert int(session.vars["Count"]) == 1
    assert session.vars["Area"][0] == 3550
    assert int(session.vars["BrightCount"]) == 1
    assert session.vars["BrightArea"][0] == 144
    assert session.vars["Judge"] == "合格"
    assert session.display.base is not None


def test_inspect_color_template_measures_green_area_and_hue():
    session = run_source((ROOT / "samples" / "inspect_color.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert int(session.vars["Count"]) == 1
    assert int(session.vars["AreaValue"]) == 1200
    assert int(session.vars["Hue"]) == 60
    assert session.vars["Judge"] == "合格"
    colors = {shape.get("color") for shape in session.public_state()["display"]["shapes"]}
    assert "#00e05a" in colors
    assert session.display.base is not None
    assert session.display.base.channels == 3
    tight = (ROOT / "samples" / "inspect_color.viz").read_text(encoding="utf-8").replace("HueMax := 70", "HueMax := 55")
    failed = run_source(tight)
    assert failed.error is None, failed.error
    assert int(failed.vars["Hue"]) == 60
    assert int(failed.vars["AreaValue"]) == 1200
    assert failed.vars["Judge"] == "不合格"
    failed_colors = {shape.get("color") for shape in failed.public_state()["display"]["shapes"]}
    assert "#ff2a2a" in failed_colors


def test_inspect_chars_template_reads_visdev_168():
    session = run_source((ROOT / "samples" / "inspect_chars.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert session.vars["Chars"] == "VISDEV 168"
    assert int(session.vars["Count"]) == 1
    assert float(session.vars["Score"]) >= 0.5
    assert session.vars["Judge"] == "合格"
    assert session.display.base is not None
    assert len(session.vars["Outline"].contours) >= 1
    texts = [shape.get("text") for shape in session.public_state()["display"]["shapes"]]
    assert "VISDEV 168" in texts
    wrong = (ROOT / "samples" / "inspect_chars.viz").read_text(encoding="utf-8").replace("Expect := 'VISDEV 168'", "Expect := 'NOPE'")
    failed = run_source(wrong)
    assert failed.error is None, failed.error
    assert failed.vars["Chars"] == "VISDEV 168"
    assert failed.vars["Expect"] == "NOPE"
    assert failed.vars["Judge"] == "不合格"


def test_gauge_line_failure_keeps_the_image_and_says_what_to_try():
    session = run_source(
        "Vizion.read_image (Image, 'samples/gauge.png')\n"
        "Vizion.dev_display (Image)\n"
        "Vizion.gauge_line (Image, Line, 10, 10, 10, 80, 4, 1, 80, 'all', 'best', R1, C1, R2, C2, Angle, Used)\n"
    )
    assert session.error
    assert session.error_line == 3
    assert "直線量規" in session.error
    assert "降低 Threshold" in session.error
    assert session.vars["Image"].width == 640
    assert session.display.base is not None


def test_inspect_stripe_template_measures_the_wide_band():
    session = run_source((ROOT / "samples" / "inspect_stripe.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert int(session.vars["EdgeCount"]) == 8
    assert float(session.vars["Left"]) == 127.5
    assert float(session.vars["Right"]) == 149.5
    assert float(session.vars["Width"]) == 22.0
    assert session.vars["Judge"] == "合格"
    colors = {shape.get("color") for shape in session.public_state()["display"]["shapes"]}
    assert "#ffe14a" in colors
    assert session.display.base is not None


def test_stripe_outside_tolerance_turns_the_marks_red():
    source = (ROOT / "samples" / "inspect_stripe.viz").read_text(encoding="utf-8")
    source = source.replace("WidthMin := 20", "WidthMin := 1").replace("WidthMax := 24", "WidthMax := 2")
    session = run_source(source)
    assert session.error is None, session.error
    assert session.vars["Judge"] == "不合格"
    colors = {shape.get("color") for shape in session.public_state()["display"]["shapes"]}
    assert "#ff2a2a" in colors


def test_inspect_line_template_counts_saves_ng_and_writes_csv():
    import pytest
    import shutil

    folder = ROOT / "workspace" / "line_ng"
    report = ROOT / "workspace" / "line_result.csv"
    shutil.rmtree(folder, ignore_errors=True)
    report.unlink(missing_ok=True)
    try:
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "old.png"), np.zeros((4, 4), np.uint8))
        session = run_source((ROOT / "samples" / "inspect_line.viz").read_text(encoding="utf-8"))
        assert session.error is None, session.error
        assert int(session.vars["Total"]) == 5
        assert int(session.vars["OkCount"]) == 3
        assert int(session.vars["NgCount"]) == 2
        assert session.vars["Judge"] == "不合格"
        assert session.vars["LastNg"] == "ng_02.png"
        lines = report.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "index,file,judge,count"
        assert "1,ng_01.png,NG,0" in lines
        assert "3,ok_01.png,OK,1" in lines
        assert sorted(path.name for path in folder.glob("*.png")) == ["ng_01.png", "ng_02.png"]
        assert not (folder / "old.png").exists()
        assert session.display.base is not None
        before = session.display.base.data.copy()
        assert session.show_last_ng() == "ng_02.png"
        assert session.public_state()["ngShowing"] is True
        assert session.public_state()["ngFile"] == "ng_02.png"
        from app.engine.render import read_cv_image

        saved = read_cv_image(folder / "ng_02.png")
        assert saved is not None and np.array_equal(session.display.base.data, saved)
        assert saved.ndim == 3 and saved.shape[2] == 3
        assert saved.shape[:2] == before.shape[:2]
        red = (saved[:, :, 2].astype(np.int16) > saved[:, :, 1].astype(np.int16) + 30) & (
            saved[:, :, 2].astype(np.int16) > saved[:, :, 0].astype(np.int16) + 30
        )
        assert int(red.sum()) > 8
        raw = read_cv_image(ROOT / "samples" / "line_batch" / "ng_02.png")
        assert raw is not None and raw.ndim == 2
        assert not np.array_equal(saved[:, :, 0], raw)
        missing = run_source("Vizion.dump_window ('workspace/line_ng/blank.png')\n")
        assert missing.error and "還沒有畫面" in missing.error
        assert not (folder / "blank.png").exists()
        assert int(session.vars["OkCount"]) == 3
        assert session.vars["Judge"] == "不合格"
        assert session.show_last_ng() == ""
        assert session.public_state()["ngShowing"] is False
        assert np.array_equal(session.display.base.data, before)
        saved_csv = report.read_bytes()
        preview = session.line_csv_preview()
        assert preview["header"] == "index,file,judge,count"
        assert preview["total"] == 5
        assert preview["ngTotal"] == 2
        assert preview["truncated"] is False
        assert preview["onlyNg"] is False
        assert "1,ng_01.png,NG,0" in preview["rows"]
        assert "3,ok_01.png,OK,1" in preview["rows"]
        ng_only = session.line_csv_preview(only_ng=True)
        assert ng_only["rows"] == ["1,ng_01.png,NG,0", "2,ng_02.png,NG,0"]
        assert ng_only["opens"] == ["ng_01.png", "ng_02.png"]
        assert ng_only["ngTotal"] == 2
        assert ng_only["onlyNg"] is True
        assert report.read_bytes() == saved_csv
        stats = session.line_stats()
        assert stats["total"] == 5 and stats["ok"] == 3 and stats["ng"] == 2
        assert stats["yield"] == 60 and stats["lastNg"] == "ng_02.png"
        assert stats["noJudge"] is False
        assert report.read_bytes() == saved_csv
        assert int(session.vars["OkCount"]) == 3
        program = session.display.base.data.copy()
        assert session.show_ng_file("ng_01.png") == "ng_01.png"
        assert session.show_ng_file("ng_02.png") == "ng_02.png"
        assert session.public_state()["ngFile"] == "ng_02.png"
        assert int(session.vars["OkCount"]) == 3
        assert session.show_last_ng() == ""
        assert np.array_equal(session.display.base.data, program)
        with pytest.raises(RuntimeError, match="檔名不正確"):
            session.show_ng_file("..\\secret.png")
        with pytest.raises(RuntimeError, match="找不到不合格圖"):
            session.show_ng_file("ok_01.png")
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        report.unlink(missing_ok=True)


def test_line_result_goes_out_on_tcp_and_serial():
    import shutil

    from app.engine.chapters.line_out import format_line_result
    from app.engine.chapters.serial_ops import _open_windows

    assert "CreateFileW" in _open_windows.__code__.co_names

    serial = run_source(
        "Vizion.open_serial ('loopback', Port)\n"
        "Vizion.send_line_result (Port, '不合格', 1, 2, 'ng_01.png')\n"
        "Vizion.read_serial (Port, 24, Line)\n"
    )
    assert serial.error is None, serial.error
    assert serial.vars["Line"] == "VISDEV,NG,1,2,ng_01.png\n"
    told = run_source(
        "Vizion.open_serial ('loopback', Port)\n"
        "Vizion.send_text_result (Port, 'VISDEV 168')\n"
        "Vizion.send_line_result (Port, 'NG', 0, 1, 'ng_01.png')\n"
        "Vizion.read_serial (Port, 47, Line)\n"
    )
    assert told.error is None, told.error
    assert told.vars["Line"] == "VISDEV,TEXT,VISDEV 168\nVISDEV,NG,0,1,ng_01.png\n"
    assert format_line_result("合格", 3, 0, "ok_03.png") == "VISDEV,OK,3,0,ok_03.png\n"
    refused = run_source(
        "Vizion.open_serial ('loopback', Port)\n"
        "Vizion.send_line_result (Port, 'maybe', 0, 0, 'a.png')\n"
    )
    assert refused.error and "判定只能是" in refused.error
    comma = run_source(
        "Vizion.open_serial ('loopback', Port)\n"
        "Vizion.send_line_result (Port, 'OK', 0, 0, 'a,b.png')\n"
    )
    assert comma.error and "檔名" in comma.error

    folder = ROOT / "workspace" / "line_ng"
    report = ROOT / "workspace" / "line_out.csv"
    shutil.rmtree(folder, ignore_errors=True)
    report.unlink(missing_ok=True)
    try:
        session = run_source((ROOT / "samples" / "inspect_line_out.viz").read_text(encoding="utf-8"))
        assert session.error is None, session.error
        assert int(session.vars["OkCount"]) == 3
        assert int(session.vars["NgCount"]) == 2
        assert session.vars["Judge"] == "不合格"
        assert session.vars["Wire"] == (
            "VISDEV,NG,0,1,ng_01.png\n"
            "VISDEV,NG,0,2,ng_02.png\n"
            "VISDEV,OK,1,2,ok_01.png\n"
            "VISDEV,OK,2,2,ok_02.png\n"
            "VISDEV,OK,3,2,ok_03.png\n"
        )
        assert session.vars["Link"].closed is True
        serial = run_source(
            (ROOT / "samples" / "inspect_line_out.viz").read_text(encoding="utf-8").replace(
                "OutKind := 'tcp'", "OutKind := 'serial'", 1
            )
        )
        assert serial.error is None, serial.error
        assert serial.vars["Wire"] == session.vars["Wire"]
        assert int(serial.vars["OkCount"]) == 3
        assert serial.vars["Link"].closed is True
        import socket
        import struct
        import threading

        ready = threading.Event()
        holder = {}
        regs = {}

        def serve():
            server = socket.socket()
            server.bind(("127.0.0.1", 0))
            server.listen(1)
            server.settimeout(3)
            holder["port"] = server.getsockname()[1]
            ready.set()
            for _ in range(12):
                try:
                    conn, _addr = server.accept()
                except socket.timeout:
                    break
                data = b""
                while len(data) < 12:
                    chunk = conn.recv(12 - len(data))
                    if not chunk:
                        break
                    data += chunk
                function = data[7]
                address, number = struct.unpack(">HH", data[8:12])
                if function == 6:
                    regs[address] = number
                    conn.sendall(data)
                elif function == 3:
                    stored = regs.get(address, 0)
                    trans = struct.unpack(">H", data[:2])[0]
                    conn.sendall(struct.pack(">HHHBBBH", trans, 0, 5, 1, 3, 2, stored))
                conn.close()
            server.close()

        threading.Thread(target=serve, daemon=True).start()
        assert ready.wait(2)
        modbus = run_source(
            (ROOT / "samples" / "inspect_line_out.viz").read_text(encoding="utf-8")
            .replace("OutKind := 'tcp'", "OutKind := 'modbus'", 1)
            .replace("OutHost := 'loopback'", "OutHost := '127.0.0.1'", 1)
            .replace("OutPort := 0", f"OutPort := {holder['port']}", 1)
        )
        assert modbus.error is None, modbus.error
        assert int(modbus.vars["OkCount"]) == 3
        assert int(modbus.vars["NgCount"]) == 2
        assert int(modbus.vars["Register"]) == 1
        assert regs[16] == 1
        lines = report.read_text(encoding="utf-8").splitlines()
        assert "1,ng_01.png,NG,0" in lines
        assert "3,ok_01.png,OK,1" in lines
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        report.unlink(missing_ok=True)


def test_modbus_writes_one_register_and_leaves_com_alone():
    import socket
    import struct
    import threading

    import pytest

    from app.engine.chapters.serial_ops import _open_windows

    assert "CreateFileW" in _open_windows.__code__.co_names
    got = {}
    ready = threading.Event()

    def serve():
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        got["port"] = server.getsockname()[1]
        ready.set()
        conn, _addr = server.accept()
        data = b""
        while len(data) < 12:
            chunk = conn.recv(12 - len(data))
            if not chunk:
                break
            data += chunk
        got["request"] = data
        conn.sendall(data)
        conn.close()
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    assert ready.wait(2)
    session = run_source(
        f"Vizion.write_modbus_register ('127.0.0.1', {got['port']}, 16, 2)\n"
    )
    assert session.error is None, session.error
    request = got["request"]
    assert request[7] == 6
    assert struct.unpack(">HH", request[8:12]) == (16, 2)
    refused = run_source("Vizion.write_modbus_register ('127.0.0.1', 502, 70000, 1)\n")
    assert refused.error and "65535" in refused.error
    with pytest.raises(RuntimeError, match="連不上 Modbus"):
        from app.engine.chapters.modbus_ops import write_holding_register

        write_holding_register("127.0.0.1", 1, 0, 1)


def test_modbus_reads_and_writes_several_registers():
    import socket
    import struct
    import threading

    from app.engine.chapters.serial_ops import _open_windows

    assert "CreateFileW" in _open_windows.__code__.co_names
    ready = threading.Event()
    regs = {}
    holder = {}

    def serve():
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(3)
        holder["port"] = server.getsockname()[1]
        ready.set()
        for _ in range(4):
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                break
            conn.settimeout(2)
            head = b""
            while len(head) < 7:
                chunk = conn.recv(7 - len(head))
                if not chunk:
                    break
                head += chunk
            length = struct.unpack(">H", head[4:6])[0]
            rest = b""
            while len(rest) < length - 1:
                chunk = conn.recv(length - 1 - len(rest))
                if not chunk:
                    break
                rest += chunk
            data = head + rest
            function = data[7]
            address = struct.unpack(">H", data[8:10])[0]
            if function == 16:
                count = struct.unpack(">H", data[10:12])[0]
                raw = data[13:13 + count * 2]
                regs[address] = list(struct.unpack(">" + "H" * count, raw))
                trans = struct.unpack(">H", data[:2])[0]
                conn.sendall(struct.pack(">HHHBBHH", trans, 0, 6, 1, 16, address, count))
            elif function == 3:
                count = struct.unpack(">H", data[10:12])[0]
                values = []
                for offset in range(count):
                    stored = regs.get(address)
                    values.append(stored[offset] if stored and offset < len(stored) else 0)
                trans = struct.unpack(">H", data[:2])[0]
                body = struct.pack(">" + "H" * count, *values)
                conn.sendall(struct.pack(">HHHBBB", trans, 0, 3 + len(body), 1, 3, len(body)) + body)
            conn.close()
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    assert ready.wait(2)
    port = holder["port"]
    session = run_source(
        f"Vizion.write_modbus_registers ('127.0.0.1', {port}, 16, [2, 3, 1])\n"
        f"Vizion.read_modbus_registers ('127.0.0.1', {port}, 16, 3, Values)\n"
    )
    assert session.error is None, session.error
    assert [int(item) for item in session.vars["Values"]] == [2, 3, 1]
    too_many = run_source(f"Vizion.read_modbus_registers ('127.0.0.1', {port}, 16, 20, Values)\n")
    assert too_many.error and "16" in too_many.error


def test_modbus_reads_back_the_register_it_wrote():
    import socket
    import struct
    import threading

    from app.engine.chapters.serial_ops import _open_windows

    assert "CreateFileW" in _open_windows.__code__.co_names
    ready = threading.Event()
    regs = {}
    holder = {}

    def serve():
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(3)
        holder["port"] = server.getsockname()[1]
        ready.set()
        for _ in range(12):
            try:
                conn, _addr = server.accept()
            except socket.timeout:
                break
            conn.settimeout(2)
            data = b""
            while len(data) < 12:
                chunk = conn.recv(12 - len(data))
                if not chunk:
                    break
                data += chunk
            function = data[7]
            address, number = struct.unpack(">HH", data[8:12])
            if function == 6:
                regs[address] = number
                conn.sendall(data)
            elif function == 3:
                stored = regs.get(address, 0)
                if holder.get("lie"):
                    stored = (stored + 1) & 0xFFFF
                trans = struct.unpack(">H", data[:2])[0]
                conn.sendall(struct.pack(">HHHBBBH", trans, 0, 5, 1, 3, 2, stored))
            conn.close()
        server.close()

    threading.Thread(target=serve, daemon=True).start()
    assert ready.wait(2)
    port = holder["port"]
    session = run_source(
        f"Vizion.read_modbus_register ('127.0.0.1', {port}, 16, Register)\n"
        f"Vizion.write_modbus_register ('127.0.0.1', {port}, 16, 2)\n"
        f"Vizion.read_modbus_register ('127.0.0.1', {port}, 16, Register)\n"
        f"Vizion.write_modbus_judge ('127.0.0.1', {port}, 16, '不合格', Checked)\n"
        f"Vizion.write_modbus_judge ('127.0.0.1', {port}, 16, '合格', CheckedOk)\n"
    )
    assert session.error is None, session.error
    assert int(session.vars["Register"]) == 2
    assert int(session.vars["Checked"]) == 2
    assert int(session.vars["CheckedOk"]) == 1
    assert regs[16] == 1
    holder["lie"] = True
    mismatched = run_source(f"Vizion.write_modbus_judge ('127.0.0.1', {port}, 16, 'NG', Again)\n")
    assert mismatched.error and "不一樣" in mismatched.error


def test_last_ng_and_csv_stay_inside_the_workspace():
    import pytest
    from fastapi import HTTPException

    from app.main import SESSIONS, show_last_ng_view, show_line_csv

    session = run_source("OkCount := 3\n")
    session.vars["LastNg"] = "..\\secret.png"
    session.vars["CsvFile"] = "workspace/../../etc/passwd"
    try:
        with pytest.raises(RuntimeError, match="檔名不正確"):
            session.show_last_ng()
        assert session.vars["OkCount"] == 3
        with pytest.raises(RuntimeError, match="明細路徑不正確"):
            session.line_csv_preview()
        session.vars["CsvFile"] = "/etc/passwd"
        with pytest.raises(RuntimeError, match="明細路徑不正確"):
            session.line_csv_preview()
        session.vars["LastNg"] = ""
        with pytest.raises(RuntimeError, match="沒有不合格圖"):
            session.show_last_ng()
        report = ROOT / "workspace" / "_line_preview.csv"
        report.write_text(
            "index,file,judge,count\n" + "".join(f"{i},a.png,NG,0\n" for i in range(1, 26)),
            encoding="utf-8",
        )
        try:
            session.vars["CsvFile"] = "workspace\\_line_preview.csv"
            preview = session.line_csv_preview()
            assert preview["total"] == 25
            assert preview["truncated"] is True
            assert len(preview["rows"]) == 20
            assert preview["rows"][0] == "6,a.png,NG,0"
            assert preview["rows"][-1] == "25,a.png,NG,0"
            ng_only = session.line_csv_preview(only_ng=True)
            assert ng_only["ngTotal"] == 25
            assert ng_only["rows"][0] == "6,a.png,NG,0"
            assert ng_only["opens"][0] == "a.png"
            SESSIONS[session.id] = session
            routed = show_line_csv(session.id, onlyNg=True)
            assert routed["onlyNg"] is True
            assert routed["rows"][-1] == "25,a.png,NG,0"
            assert report.read_bytes().count(b"\n") == 26
            mixed = ROOT / "workspace" / "_line_mixed.csv"
            mixed_lines = ["index,file,judge,count"]
            mixed_lines.extend(f"{i},ng_{i}.png,NG,0" for i in range(1, 6))
            mixed_lines.extend(f"{i},ok_{i}.png,OK,1" for i in range(6, 31))
            mixed.write_text("\n".join(mixed_lines) + "\n", encoding="utf-8")
            try:
                session.vars["CsvFile"] = "workspace/_line_mixed.csv"
                recent = session.line_csv_preview()
                assert recent["ngTotal"] == 5
                assert all(",OK," in row for row in recent["rows"])
                assert recent["opens"] == [""] * len(recent["rows"])
                failed = session.line_csv_preview(only_ng=True)
                assert failed["rows"] == [f"{i},ng_{i}.png,NG,0" for i in range(1, 6)]
                assert failed["opens"] == [f"ng_{i}.png" for i in range(1, 6)]
                assert failed["truncated"] is False
                assert mixed.read_bytes().decode("utf-8").count(",OK,") == 25
                stats = session.line_stats()
                assert stats["total"] == 30 and stats["ok"] == 25 and stats["ng"] == 5
                assert stats["yield"] == 83.3 and stats["lastNg"] == "ng_5.png"
                assert mixed.read_bytes().decode("utf-8").count(",OK,") == 25
            finally:
                mixed.unlink(missing_ok=True)
        finally:
            report.unlink(missing_ok=True)
        SESSIONS[session.id] = session
        session.running = True
        with pytest.raises(HTTPException) as caught:
            show_last_ng_view(session.id)
        assert caught.value.status_code == 409
    finally:
        session.running = False
        SESSIONS.pop(session.id, None)


def test_camera_line_stops_and_keeps_the_counts():
    import asyncio
    import inspect
    import shutil
    import time

    import numpy as np

    from app.engine.chapters import camera_ops
    from app.main import SESSIONS, ExecIn, exec_program, stop

    opened = inspect.getsource(camera_ops._open_camera)
    assert opened.index("grab_bgr") < opened.index("CAP_DSHOW") < opened.rindex("VideoCapture(index)")

    class FakeCap:
        def __init__(self):
            self.n = 0
            self.opened = True
            self.released = False

        def isOpened(self):
            return self.opened

        def read(self):
            time.sleep(0.05)
            self.n += 1
            image = np.zeros((40, 40), np.uint8)
            if self.n % 2 == 0:
                image[10:30, 10:30] = 255
            else:
                image[16:24, 16:24] = 255
            return True, image

        def release(self):
            self.released = True
            self.opened = False

        def set(self, *_args):
            return False

    fake = FakeCap()
    folder = ROOT / "workspace" / "line_ng"
    report = ROOT / "workspace" / "line_camera.csv"
    shutil.rmtree(folder, ignore_errors=True)
    report.unlink(missing_ok=True)
    original = camera_ops._open_camera
    camera_ops._open_camera = lambda *_args, **_kwargs: fake
    started = None
    session = None
    try:
        source = (ROOT / "samples" / "inspect_line_camera.viz").read_text(encoding="utf-8")
        started = asyncio.run(exec_program(ExecIn(source=source, action="run")))
        session = SESSIONS[started["id"]]
        deadline = time.time() + 2
        while time.time() < deadline:
            done = int(session.vars.get("OkCount", 0) or 0) + int(session.vars.get("NgCount", 0) or 0)
            if done >= 1:
                break
            time.sleep(0.05)
        stop(started["id"])
        session.runner.join(timeout=3)
        assert session.error is None, session.error
        assert session.paused == "stopped"
        grabbed = int(session.vars["OkCount"]) + int(session.vars["NgCount"])
        assert grabbed >= 1
        assert int(session.vars["Total"]) == grabbed
        assert session.vars["Report"].closed is True
        assert fake.released is True
        lines = report.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "index,file,judge,count"
        assert len(lines) == grabbed + 1
        assert lines[1] == "1,camera_1.png,NG,0"
        assert all(row.split(",")[2] in ("OK", "NG") for row in lines[1:])
    finally:
        camera_ops._open_camera = original
        if session is not None:
            stop(session.id)
            runner = getattr(session, "runner", None)
            if runner is not None:
                runner.join(timeout=2)
            SESSIONS.pop(session.id, None)
        shutil.rmtree(folder, ignore_errors=True)
        report.unlink(missing_ok=True)


def test_camera_grab_failure_keeps_counts_and_closes_the_file():
    import shutil

    from app.engine.chapters import camera_ops

    class FakeCap:
        def __init__(self):
            self.n = 0
            self.released = False

        def isOpened(self):
            return True

        def read(self):
            self.n += 1
            if self.n > 2:
                return False, None
            image = np.zeros((40, 40), np.uint8)
            if self.n % 2 == 0:
                image[10:30, 10:30] = 255
            else:
                image[16:24, 16:24] = 255
            return True, image

        def release(self):
            self.released = True

        def set(self, *_args):
            return False

    fake = FakeCap()
    folder = ROOT / "workspace" / "line_ng"
    report = ROOT / "workspace" / "line_camera.csv"
    shutil.rmtree(folder, ignore_errors=True)
    report.unlink(missing_ok=True)
    original = camera_ops._open_camera
    camera_ops._open_camera = lambda *_args, **_kwargs: fake
    try:
        folder.mkdir(parents=True)
        cv2.imwrite(str(folder / "old.png"), np.zeros((4, 4), np.uint8))
        session = run_source((ROOT / "samples" / "inspect_line_camera.viz").read_text(encoding="utf-8"))
        assert session.error
        assert "讀不到畫面" in session.error
        assert "samples/line_batch" in session.error
        assert int(session.vars["OkCount"]) == 1
        assert int(session.vars["NgCount"]) == 1
        assert int(session.vars["Total"]) == 2
        assert session.vars["Judge"] == "不合格"
        assert session.vars["LastNg"] == "camera_1.png"
        assert session.vars["Report"].closed is True
        assert session.vars["Acq"].closed is True
        assert fake.released is True
        assert session.display.base is not None
        lines = report.read_text(encoding="utf-8").splitlines()
        assert lines == [
            "index,file,judge,count",
            "1,camera_1.png,NG,0",
            "2,camera_2.png,OK,1",
        ]
        assert sorted(path.name for path in folder.glob("*.png")) == ["camera_1.png"]
    finally:
        camera_ops._open_camera = original
        shutil.rmtree(folder, ignore_errors=True)
        report.unlink(missing_ok=True)


def test_inspect_locate_template_finds_four():
    session = run_source((ROOT / "samples" / "inspect_locate.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert int(session.vars["Count"]) == 4
    scores = session.vars["Score"]
    assert len(scores) == 4
    assert min(float(item) for item in scores) > 0.9
    assert session.display.base is not None


def test_project_rename_keeps_programs_and_copies_workspace_images():
    import shutil

    import cv2
    import numpy as np

    from app.main import (
        create_project_record,
        project_programs,
        read_project_program,
        rename_project_record,
        write_project_program,
    )

    name = "beta_case_rename"
    renamed = "beta_case_renamed"
    folder = ROOT / "workspace" / "projects" / name
    target = ROOT / "workspace" / "projects" / renamed
    image = ROOT / "workspace" / "beta_mark.png"
    shutil.rmtree(folder, ignore_errors=True)
    shutil.rmtree(target, ignore_errors=True)
    image.unlink(missing_ok=True)
    try:
        cv2.imwrite(str(image), np.full((5, 9), 40, np.uint8))
        create_project_record(name)
        saved = write_project_program(
            name,
            "read_mark.viz",
            "Vizion.read_image (Image, 'workspace/beta_mark.png')\n"
            "Vizion.read_image (Sample, 'samples/parts.png')\n",
        )
        assert saved["images"] == ["beta_mark.png"]
        assert "read_image (Image, 'beta_mark.png')" in saved["source"]
        assert "samples/parts.png" in saved["source"]
        assert (folder / "beta_mark.png").is_file()
        assert not (folder / "parts.png").exists()
        moved = rename_project_record(name, renamed)
        assert moved["name"] == renamed
        assert not folder.exists()
        assert [item["name"] for item in project_programs(renamed)["programs"]] == ["read_mark.viz"]
        text = read_project_program(renamed, "read_mark.viz")["source"]
        session = Session(text, ROOT)
        session.project_name = renamed
        session.run()
        assert session.error is None, session.error
        assert session.vars["Image"].width == 9
        assert session.vars["Sample"].width > 9
        try:
            rename_project_record(renamed, renamed)
        except ValueError:
            raise AssertionError("同名應該留下")
        assert project_programs(renamed)["name"] == renamed
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        shutil.rmtree(target, ignore_errors=True)
        image.unlink(missing_ok=True)


def test_backslash_path_reads_the_workspace_image():
    import cv2
    import numpy as np

    from app.main import create_project_record, reveal_project_folder

    image = ROOT / "workspace" / "beta_slash.png"
    folder_name = "beta_slash_case"
    folder = ROOT / "workspace" / "projects" / folder_name
    image.unlink(missing_ok=True)
    import shutil
    shutil.rmtree(folder, ignore_errors=True)
    try:
        cv2.imwrite(str(image), np.full((4, 11), 70, np.uint8))
        program = "Vizion.read_image (Image, 'workspace" + "\\" + "beta_slash.png')\n"
        session = Session(program, ROOT)
        session.run()
        assert session.error is None, session.error
        assert session.vars["Image"].width == 11
        create_project_record(folder_name)
        nested = folder / "shot.png"
        cv2.imwrite(str(nested), np.full((3, 5), 20, np.uint8))
        local = Session("Vizion.read_image (Image, 'shot.png')\n", ROOT)
        local.project_name = folder_name
        local.run()
        assert local.error is None, local.error
        assert local.vars["Image"].width == 5
        info = reveal_project_folder(folder_name)
        assert info["name"] == folder_name
        assert Path(info["path"]).is_dir()
        assert isinstance(info["opened"], bool)
    finally:
        image.unlink(missing_ok=True)
        shutil.rmtree(folder, ignore_errors=True)


def test_project_holds_two_programs_and_its_image():
    import shutil

    import cv2
    import numpy as np

    from app.main import (
        create_project_record,
        list_project_records,
        project_programs,
        read_project_program,
        write_project_program,
    )

    name = "beta_case_hold"
    folder = ROOT / "workspace" / "projects" / name
    shutil.rmtree(folder, ignore_errors=True)
    try:
        created = create_project_record(name)
        assert created["name"] == name
        assert any(item["name"] == name for item in list_project_records())
        write_project_program(name, "first", "Count := 1\n")
        write_project_program(name, "second.hdv", "Count := 2\n")
        listed = [item["name"] for item in project_programs(name)["programs"]]
        assert listed == ["first.viz", "second.hdv"]
        assert "Count := 2" in read_project_program(name, "second.hdv")["source"]
        try:
            create_project_record("../outside")
        except ValueError as exc:
            assert "不正確" in str(exc)
        else:
            raise AssertionError("專案名稱應該拒絕路徑")
        cv2.imwrite(str(folder / "mark.png"), np.full((8, 6), 180, np.uint8))
        local = Session("Vizion.read_image (Image, 'mark.png')\n", ROOT)
        local.project_name = name
        local.run()
        assert local.error is None, local.error
        assert local.vars["Image"].height == 8
        assert local.vars["Image"].width == 6
        shared = Session("Vizion.read_image (Image, 'samples/parts.png')\n", ROOT)
        shared.project_name = name
        shared.run()
        assert shared.error is None, shared.error
        assert shared.vars["Image"].width > 8
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def test_subpixel_step_lands_on_the_half_pixel():
    import numpy as np

    from app.engine.chapters.xld_ops import edges_sub_pix
    from app.engine.objects import Image

    session = run_source((ROOT / "samples" / "inspect_edge.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert int(session.vars["Count"]) == 1
    ends = {
        (round(float(session.vars["RowBegin"][0]), 4), round(float(session.vars["ColBegin"][0]), 4)),
        (round(float(session.vars["RowEnd"][0]), 4), round(float(session.vars["ColEnd"][0]), 4)),
    }
    assert ends == {(0.0, 15.5), (23.0, 15.5)}
    noise = np.full((24, 32), 128, np.uint8)
    noise[3, 4] = 131
    noise[10, 20] = 125
    empty = edges_sub_pix(Image(noise), "canny", 1.0, 20, 40, ctx=None)
    assert empty.iconic[0].contours == []


def test_inspect_fit_template_rejects_the_outliers():
    session = run_source((ROOT / "samples" / "inspect_fit.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    ends = {
        (round(session.vars["RowBegin"][0], 6), round(session.vars["ColBegin"][0], 6)),
        (round(session.vars["RowEnd"][0], 6), round(session.vars["ColEnd"][0], 6)),
    }
    assert ends == {(0.0, 0.0), (0.0, 30.0)}
    assert abs(session.vars["CenterRow"][0] - 50) < 1e-6
    assert abs(session.vars["CenterCol"][0] - 80) < 1e-6
    assert abs(session.vars["Radius"][0] - 20) < 1e-6


def test_inspect_missing_template_accepts_eight_parts():
    session = run_source((ROOT / "samples" / "inspect_missing.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert int(session.vars["Count"]) == 8
    assert session.vars["Judge"] == "合格"
    assert session.display.base is not None
    short = (ROOT / "samples" / "inspect_missing.viz").read_text(encoding="utf-8").replace("CountMin := 8", "CountMin := 9")
    failed = run_source(short)
    assert failed.error is None, failed.error
    assert int(failed.vars["Count"]) == 8
    assert failed.vars["Judge"] == "不合格"


def test_hole_line_uses_the_same_center_and_diameter_rules():
    import shutil

    from app.engine.render import read_cv_image

    folder = ROOT / "workspace" / "line_ng"
    report = ROOT / "workspace" / "line_holes.csv"
    shutil.rmtree(folder, ignore_errors=True)
    report.unlink(missing_ok=True)
    try:
        session = run_source((ROOT / "samples" / "inspect_line_holes.viz").read_text(encoding="utf-8"))
        assert session.error is None, session.error
        assert int(session.vars["Total"]) == 5
        assert int(session.vars["OkCount"]) == 2
        assert int(session.vars["NgCount"]) == 3
        assert session.vars["Judge"] == "不合格"
        assert session.vars["LastNg"] == "ng_wide.png"
        rows = {}
        lines = report.read_text(encoding="utf-8").splitlines()
        assert lines[0] == "index,file,judge,count,shift,dia"
        for line in lines[1:]:
            index, name, judge, count, shift, dia = line.split(",")
            rows[name] = (judge, int(count), float(shift), float(dia), int(index))
        assert rows["ng_miss.png"][:3] == ("NG", 3, 70.0)
        assert rows["ng_shift.png"][0] == "NG" and rows["ng_shift.png"][1] == 4
        assert abs(rows["ng_shift.png"][2] - 10) < 1e-6
        assert rows["ng_wide.png"][0] == "NG" and abs(rows["ng_wide.png"][3] - 40) < 0.01
        assert rows["ok_01.png"][0] == "OK" and rows["ok_01.png"][1] == 4 and rows["ok_01.png"][2] == 0
        assert abs(rows["ok_01.png"][3] - 28) < 0.01
        assert rows["ok_02.png"][0] == "OK"
        assert sorted(path.name for path in folder.glob("*.png")) == ["ng_miss.png", "ng_shift.png", "ng_wide.png"]
        saved = read_cv_image(folder / "ng_wide.png")
        assert saved is not None and saved.ndim == 3
        red = (saved[:, :, 2].astype(np.int16) > saved[:, :, 1].astype(np.int16) + 30) & (
            saved[:, :, 2].astype(np.int16) > saved[:, :, 0].astype(np.int16) + 30
        )
        assert int(red.sum()) > 8
        assert session.show_last_ng() == "ng_wide.png"
        assert int(session.vars["OkCount"]) == 2
        stats = session.line_stats()
        assert stats["total"] == 5 and stats["ok"] == 2 and stats["ng"] == 3
        assert stats["yield"] == 40 and stats["lastNg"] == "ng_wide.png"
    finally:
        shutil.rmtree(folder, ignore_errors=True)
        report.unlink(missing_ok=True)


def test_inspect_holes_template_counts_four():
    session = run_source((ROOT / "samples" / "inspect_holes.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert int(session.vars["Count"]) == 4
    assert [int(item) for item in session.vars["Area"]] == [613, 613, 613, 613]
    assert float(session.vars["Shift"]) == 0
    assert abs(float(session.vars["Dia"]) - 28) < 0.01
    assert session.vars["Judge"] == "合格"
    assert len(session.vars["Cross"].contours) == 8
    short = (ROOT / "samples" / "inspect_holes.viz").read_text(encoding="utf-8").replace("CountMax := 4", "CountMax := 3")
    failed = run_source(short)
    assert failed.error is None, failed.error
    assert int(failed.vars["Count"]) == 4
    assert failed.vars["Judge"] == "不合格"
    moved = (ROOT / "samples" / "inspect_holes.viz").read_text(encoding="utf-8").replace(
        "WantRow := [50, 50, 120, 120]", "WantRow := [55, 50, 120, 120]", 1
    )
    off = run_source(moved)
    assert off.error is None, off.error
    assert int(off.vars["Count"]) == 4
    assert abs(float(off.vars["Shift"]) - 5) < 1e-6
    assert off.vars["Judge"] == "不合格"
    narrow = (ROOT / "samples" / "inspect_holes.viz").read_text(encoding="utf-8").replace(
        "DiaMax := 30", "DiaMax := 20", 1
    )
    small = run_source(narrow)
    assert small.error is None, small.error
    assert abs(float(small.vars["Dia"]) - 28) < 0.01
    assert small.vars["Judge"] == "不合格"
    assert float(session.vars["CenterOk"]) > 0


def test_inspect_parts_template_counts_eight_parts():
    session = run_source((ROOT / "samples" / "inspect_parts.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert int(session.vars["Count"]) == 8
    areas = session.vars["Area"]
    assert areas[0] == 2988
    assert len(areas) == 8
    assert session.display.base is not None


def test_running_large_image_is_a_preview_until_the_run_finishes():
    session = run_source("Vizion.gen_image_const (Image, 'byte', 1200, 800)\n")
    assert session.error is None, session.error
    full = session.public_state()["display"]
    assert full["width"] == 1200 and full["height"] == 800
    assert full["image"].startswith("data:image/png")
    assert not full.get("preview")
    session.running = True
    preview = session.public_state()["display"]
    assert preview["width"] <= 640 and preview["height"] <= 640
    assert preview["width"] == 640
    assert preview["image"].startswith("data:image/jpeg")
    assert preview.get("preview") is True
    assert session.vars["Image"].width == 1200
    session.running = False
    again = session.public_state()["display"]
    assert again["width"] == 1200 and again["height"] == 800
    assert again["image"].startswith("data:image/png")
    assert not again.get("preview")
    small = run_source("Vizion.gen_image_const (Image, 'byte', 320, 200)\n")
    small.running = True
    shown = small.public_state()["display"]
    assert shown["width"] == 320 and shown["image"].startswith("data:image/png")


def test_stop_during_wait_replaces_the_preview_with_the_full_image():
    import asyncio
    import time

    from app.main import SESSIONS, ExecIn, exec_program, session_state, stop

    source = "Vizion.gen_image_const (Image, 'byte', 1200, 800)\nVizion.wait_seconds (3)\n"
    started = asyncio.run(exec_program(ExecIn(source=source, action="run")))
    sid = started["id"]
    session = SESSIONS[sid]
    seen_preview = False
    final = None
    try:
        for _ in range(80):
            state = session_state(sid)
            display = state.get("display") or {}
            if state.get("running") and display.get("preview"):
                seen_preview = True
                stop(sid)
            if not state.get("running"):
                final = state
                break
            time.sleep(0.05)
        assert seen_preview
        assert final is not None and final.get("running") is False
        assert final["display"]["width"] == 1200 and final["display"]["height"] == 800
        assert final["display"]["image"].startswith("data:image/png")
        assert not final["display"].get("preview")
        assert session.vars["Image"].width == 1200
    finally:
        stop(sid)
        runner = getattr(session, "runner", None)
        if runner is not None:
            runner.join(timeout=2)
        SESSIONS.pop(sid, None)


def test_polling_encodes_the_preview_outside_the_session_lock():
    import base64

    import app.engine.render as render
    from app.main import _snapshot

    session = run_source("Vizion.gen_image_const (Image, 'byte', 1200, 800)\n")
    assert session.error is None, session.error
    session.running = True
    held = []
    calls = []
    real_jpeg = render._jpeg_data_url

    def spy(array):
        calls.append(array.shape[:2])
        session.vars["Image"].data[:] = 255
        acquired = session.lock.acquire(blocking=False)
        held.append(acquired)
        if acquired:
            session.lock.release()
        return real_jpeg(array)

    render._jpeg_data_url = spy
    try:
        state = _snapshot(session)
        again = _snapshot(session)
    finally:
        render._jpeg_data_url = real_jpeg
    assert held == [True]
    assert calls == [(427, 640)]
    assert state["running"] is True
    assert state["display"]["preview"] is True
    assert state["display"]["width"] == 640
    assert again["displayRev"] == state["displayRev"]
    assert again["display"]["image"] == state["display"]["image"]
    raw = base64.b64decode(state["display"]["image"].split(",", 1)[1])
    view = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    assert int(view.max()) < 8
    assert session.vars["Image"].width == 1200


def test_variable_thumbs_are_encoded_outside_the_session_lock():
    import app.engine.session as engine_session
    from app.main import _snapshot

    session = run_source("Vizion.gen_image_const (Image, 'byte', 900, 700)\n")
    assert session.error is None, session.error
    session._thumb_cache.clear()
    held = []
    arrays = []
    real = engine_session.jpeg_thumb

    def spy(array, max_edge=112):
        arrays.append(int(array.max()))
        session.vars["Image"].data[:] = 255
        acquired = session.lock.acquire(blocking=False)
        held.append(acquired)
        if acquired:
            session.lock.release()
        return real(array, max_edge)

    engine_session.jpeg_thumb = spy
    try:
        state = _snapshot(session)
        again = _snapshot(session)
    finally:
        engine_session.jpeg_thumb = real
    image = next(item for item in state["iconic"] if item["name"] == "Image")
    assert held == [True]
    assert arrays == [0]
    assert image["thumb"].startswith("data:image/jpeg")
    assert again["iconic"][0]["thumb"] == image["thumb"]
    assert session.vars["Image"].width == 900


def test_inspect_code_template_reads_the_ean():
    session = run_source((ROOT / "samples" / "inspect_code.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    assert int(session.vars["Count"]) == 1
    assert session.vars["Text"] == "4006381333931"
    assert session.vars["QrText"] == "VISDEV-3"
    assert session.display.base is not None


def test_inspect_stereo_template_shows_the_baseline():
    session = run_source((ROOT / "samples" / "inspect_stereo.viz").read_text(encoding="utf-8"))
    assert session.error is None, session.error
    pose0 = [float(item) for item in session.vars["Pose0"]]
    pose1 = [float(item) for item in session.vars["Pose1"]]
    assert pose0 == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert abs(float(session.vars["BaselineX"]) - 0.04) < 0.008
    assert abs(pose1[1] + 0.01) < 0.008
    assert abs(pose1[2] - 0.005) < 0.008
    assert "相機1相對" in session.vars["CalibDataID"].summary()


def test_second_camera_pose_is_the_baseline():
    import cv2
    import numpy as np

    from app.engine.chapters.calib_ops import (
        calibrate_cameras,
        create_calib_data,
        get_calib_data,
        set_calib_data_calib_object,
        set_calib_data_observ_points,
    )

    class Ctx:
        pass

    ctx = Ctx()
    rows, cols, square = 4, 5, 0.01
    obj = np.zeros((rows * cols, 3), np.float64)
    obj[:, 0] = np.tile(np.arange(cols), rows) * square
    obj[:, 1] = np.repeat(np.arange(rows), cols) * square
    matrix = np.array([[900.0, 0, 320.0], [0, 900.0, 240.0], [0, 0, 1]])
    views = [
        (np.array([[0.15], [-0.08], [0.05]]), np.array([[-0.02], [0.01], [0.45]])),
        (np.array([[-0.20], [0.12], [-0.04]]), np.array([[0.015], [-0.02], [0.55]])),
        (np.array([[0.05], [0.18], [0.10]]), np.array([[0.0], [0.02], [0.62]])),
    ]
    baseline = np.array([0.04, -0.01, 0.005])
    relative_rotation = cv2.Rodrigues(np.array([[0.02], [-0.03], [0.04]]))[0]
    model = create_calib_data("calibration_object", 2, 1, ctx=ctx).control[0]
    set_calib_data_calib_object(model, 0, "4,5,0.01", ctx=ctx)

    def project(rvec, tvec):
        points, _ = cv2.projectPoints(obj, rvec, tvec, matrix, np.zeros(5))
        return points.reshape(-1, 2)

    for rvec, tvec in views:
        image = project(rvec, tvec)
        set_calib_data_observ_points(model, 0, 0, image[:, 1].tolist(), image[:, 0].tolist(), ctx=ctx)
    for rvec, tvec in views:
        rotation, _ = cv2.Rodrigues(rvec)
        moved = relative_rotation @ rotation
        shifted = relative_rotation @ tvec.reshape(3) + baseline
        rvec_cam, _ = cv2.Rodrigues(moved)
        image = project(rvec_cam, shifted.reshape(3, 1))
        set_calib_data_observ_points(model, 1, 0, image[:, 1].tolist(), image[:, 0].tolist(), ctx=ctx)
    error = float(calibrate_cameras(model, ctx=ctx).control[0])
    assert error < 1.0, error
    pose0 = get_calib_data(model, "camera", 0, "pose", ctx=ctx).control[0]
    pose1 = get_calib_data(model, "camera", 1, "pose", ctx=ctx).control[0]
    assert [float(item) for item in pose0] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert abs(float(pose1[0]) - baseline[0]) < 0.008
    assert abs(float(pose1[1]) - baseline[1]) < 0.008
    assert abs(float(pose1[2]) - baseline[2]) < 0.008
    assert max(abs(float(item)) for item in pose1[3:6]) < 0.08
    assert "相機1相對" in model.summary()
    params0 = get_calib_data(model, "camera", 0, "params", ctx=ctx).control[0]
    params1 = get_calib_data(model, "camera", 1, "params", ctx=ctx).control[0]
    assert len(params0) == 9 and len(params1) == 9

    alone = create_calib_data("calibration_object", 1, 1, ctx=ctx).control[0]
    set_calib_data_calib_object(alone, 0, "4,5,0.01", ctx=ctx)
    for rvec, tvec in views:
        image = project(rvec, tvec)
        set_calib_data_observ_points(alone, 0, 0, image[:, 1].tolist(), image[:, 0].tolist(), ctx=ctx)
    alone_error = float(calibrate_cameras(alone, ctx=ctx).control[0])
    alone_pose = get_calib_data(alone, "camera", 0, "pose", ctx=ctx).control[0]
    assert alone_error < 0.05
    assert [float(item) for item in alone_pose] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_calibration_draws_points_and_errors():
    found = run_source(
        "Vizion.read_image (Front, 'samples/calib_board_front.png')\n"
        "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)\n"
        "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,36')\n"
        "Vizion.find_calib_object (Front, CalibDataID, 0, 0, '', '')\n"
    )
    assert found.error is None, found.error
    first = found.public_state()["display"]
    assert first and first["width"] == 428 and first["height"] == 356
    texts = [shape["text"] for shape in first["shapes"] if shape["type"] == "text"]
    assert any("35 點" in text for text in texts)
    assert sum(1 for shape in first["shapes"] if shape["type"] == "polyline") == 70

    calibrated = run_source(
        "Vizion.read_image (Front, 'samples/calib_board_front.png')\n"
        "Vizion.read_image (Left, 'samples/calib_board_left.png')\n"
        "Vizion.read_image (Right, 'samples/calib_board_right.png')\n"
        "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)\n"
        "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,36')\n"
        "Vizion.find_calib_object (Front, CalibDataID, 0, 0, '', '')\n"
        "Vizion.find_calib_object (Left, CalibDataID, 0, 0, '', '')\n"
        "Vizion.find_calib_object (Right, CalibDataID, 0, 0, '', '')\n"
        "Vizion.calibrate_cameras (CalibDataID, Error)\n"
    )
    assert calibrated.error is None, calibrated.error
    calib = calibrated.vars["CalibDataID"]
    assert float(calibrated.vars["Error"]) < 5
    assert len(calib.observation_errors) == 3
    assert all(item is not None and item < 5 for item in calib.observation_errors)
    assert "誤差" in calib.summary()
    assert calib.observations[0][2:] == (428, 356)
    assert any(line.startswith("標定誤差") for line in calibrated.log)
    drawn = calibrated.public_state()["display"]
    assert drawn["height"] > 356
    labels = [shape["text"] for shape in drawn["shapes"] if shape["type"] == "text"]
    assert any(text.startswith("平均誤差") for text in labels)
    assert sum("誤差" in text for text in labels) >= 3
    assert sum(1 for shape in drawn["shapes"] if shape["type"] == "polyline") > 200


def test_chessboard_5_7_10_is_thirty_five_corners():
    from app.engine.chapters.calib_ops import CalibData, _object_points, _parse_descr

    class Ctx:
        root = ROOT

    plate = _parse_descr("5,7,10", Ctx())
    assert plate.kind == "chessboard"
    assert plate.rows == 5 and plate.cols == 7 and plate.square == 10
    same = _parse_descr("5x7x10", Ctx())
    assert (same.rows, same.cols, same.square) == (5, 7, 10)
    wide = _parse_descr("5，7，10", Ctx())
    assert (wide.rows, wide.cols, wide.square) == (5, 7, 10)
    model = CalibData()
    model.rows, model.cols, model.square = plate.rows, plate.cols, plate.square
    points = _object_points(model)
    assert len(points) == 35
    assert points[0].tolist() == [0.0, 0.0, 0.0]
    assert points[1].tolist() == [10.0, 0.0, 0.0]
    assert points[7].tolist() == [0.0, 10.0, 0.0]
    assert points[-1].tolist() == [60.0, 40.0, 0.0]


def test_plate_file_and_two_cameras():
    import cv2
    import numpy as np

    from app.engine.chapters.calib_ops import (
        calibrate_cameras,
        create_calib_data,
        find_calib_object,
        get_calib_data,
        parse_plate_description,
        set_calib_data_calib_object,
        set_calib_data_observ_points,
    )
    from app.engine.objects import Image

    ensure_samples(ROOT)
    plate = parse_plate_description((ROOT / "samples" / "calplate_4x5.cpd").read_text(encoding="utf-8"))
    assert plate.kind == "circles"
    assert plate.rows == 4 and plate.cols == 5
    assert plate.polarity == "light"
    assert plate.marks is not None and len(plate.marks) == 20
    assert abs(float(plate.marks[0, 1]) - 0.03) < 1e-9
    assert abs(float(plate.square) - 0.01) < 1e-9

    class Ctx:
        root = ROOT

    def plate_image() -> np.ndarray:
        image = np.zeros((240, 300), np.uint8)
        for x_value, y_value, _zee in plate.marks:
            cx = int(round(50 + float(x_value) * 5000))
            cy = int(round(40 + (0.03 - float(y_value)) * 5000))
            cv2.circle(image, (cx, cy), 10, 255, -1)
        return image

    model = create_calib_data("calibration_object", 1, 1, ctx=Ctx()).control[0]
    set_calib_data_calib_object(model, 0, "samples/calplate_4x5.cpd", ctx=Ctx())
    find_calib_object(Image(plate_image()), model, 0, 0, "", "", ctx=Ctx())
    found = model.observations[0][1]
    expected = np.array([[50 + float(x_value) * 5000, 40 + (0.03 - float(y_value)) * 5000] for x_value, y_value, _zee in plate.marks])
    assert np.linalg.norm(found - expected, axis=1).max() < 1.0
    set_calib_data_calib_object(model, 0, "samples/calplate_4x5.cpd", ctx=Ctx())
    focal = 900.0
    principal = (160.0, 120.0)
    for distance, slide in ((0.35, -0.01), (0.5, 0.0), (0.7, 0.015)):
        rows = []
        cols = []
        for x_value, y_value, _zee in plate.marks:
            cols.append(focal * (float(x_value) + slide) / distance + principal[0])
            rows.append(focal * (float(y_value) - 0.015) / distance + principal[1])
        set_calib_data_observ_points(model, 0, 0, rows, cols, ctx=Ctx())
    error = float(calibrate_cameras(model, ctx=Ctx()).control[0])
    assert error < 0.05, error
    params = get_calib_data(model, "camera", 0, "params", ctx=Ctx()).control[0]
    assert len(params) == 9 and float(params[0]) > 0

    both = create_calib_data("calibration_object", 2, 1, ctx=Ctx()).control[0]
    set_calib_data_calib_object(both, 0, "samples/calplate_4x5.cpd", ctx=Ctx())
    for camera in (0, 1):
        for distance, slide in ((0.35, -0.01), (0.5, 0.0), (0.7, 0.015)):
            rows = []
            cols = []
            for x_value, y_value, _zee in plate.marks:
                cols.append(focal * (float(x_value) + slide + camera * 0.004) / distance + principal[0])
                rows.append(focal * (float(y_value) - 0.015) / distance + principal[1])
            set_calib_data_observ_points(both, camera, 0, rows, cols, ctx=Ctx())
    both_error = float(calibrate_cameras(both, ctx=Ctx()).control[0])
    assert both_error < 1.0, both_error
    first = get_calib_data(both, "camera", 0, "params", ctx=Ctx()).control[0]
    second = get_calib_data(both, "camera", 1, "params", ctx=Ctx()).control[0]
    assert len(first) == 9 and len(second) == 9
    assert len(both.observations) == 6
    refused = run_source(
        "Vizion.gen_image_const (Image, 'byte', 80, 60)\n"
        "Vizion.create_calib_data ('calibration_object', 1, 1, CalibDataID)\n"
        "Vizion.set_calib_data_calib_object (CalibDataID, 0, '5,7,10')\n"
        "Vizion.find_calib_object (Image, CalibDataID, 1, 0, '', '')\n"
    )
    assert refused.error is not None and "相機編號" in refused.error


def test_code93_lowercase_reads_as_uppercase():
    import numpy as np

    from app.engine.barcode1d import render_code93

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_code93_lowercase_reads_VISDEV.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Text"] == "VISDEV"
    assert sample.vars["Still"] == "VISDEV"
    assert np.array_equal(render_code93("VISDEV"), render_code93("visdev"))
    rejected = run_source("Vizion.gen_code93 (Code, 'visdev!')\n")
    assert rejected.error is not None and "Code 93" in rejected.error


def test_code2of5_odd_length_pads_a_zero():
    import numpy as np

    from app.engine.barcode1d import render_code2of5

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_code2of5_odd_length_reads_012345.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["OddText"] == "012345"
    assert sample.vars["EvenText"] == "123456"
    assert np.array_equal(render_code2of5("12345"), render_code2of5("012345"))
    rejected = run_source("Vizion.gen_code2of5 (Code, '12a')\n")
    assert rejected.error is not None and "只能是數字" in rejected.error


def test_code128_digits_use_code_set_c():
    from app.engine.chapters.code_ops import _code128_values, render_code128

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_code128_digits_use_code_set_c.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["DigitsText"] == "123456"
    assert sample.vars["PaddedText"] == "0123"
    assert sample.vars["WordText"] == "Hi"
    assert sample.vars["OddText"] == "12345"
    assert sample.vars["MixedText"] == "A12"
    assert _code128_values("Hi")[0] == 104
    assert _code128_values("A12") == [104, ord("A") - 32, ord("1") - 32, ord("2") - 32]
    assert _code128_values("123456")[0] == 105
    assert _code128_values("123456")[1:] == [12, 34, 56]
    assert render_code128("ABCDEF").shape[1] - render_code128("123456").shape[1] == 66
    assert render_code128("Hi").shape[1] == render_code128("AB").shape[1]
    try:
        render_code128("Hi\n")
    except ValueError as exc:
        assert "Code 128" in str(exc)
    else:
        raise AssertionError("控制字元應該被拒絕")


def test_get_text_result_num_counts_words():
    from app.engine.chapters.code_ops import TextResult
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_get_text_result_blank_image_num_is_0.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert int(sample.vars["Count"]) == 0
    assert sample.vars["Words"] == []
    read = REGISTRY["get_text_result"].fn
    one = TextResult(10, 10, [{"text": "A", "score": 0.5, "row": 1.0, "column": 2.0}])
    assert read(one, "num", ctx=None).control[0] == 1
    assert read(one, "class", ctx=None).control[0] == ["A"]
    assert read(TextResult(10, 10, []), "num", ctx=None).control[0] == 0


def test_hdevelop_batch102():
    import numpy as np

    from app.engine.ncc_model import _keys_peak

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gauss_image_smoothed_shift_row_is_39.23.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert float(sample.vars["Still"]) == 40
    assert abs(float(sample.vars["SRow"]) - 39.23) < 1e-9
    assert abs(float(sample.vars["SCol"]) - 47.86) < 1e-9
    assert abs(39.23 - 39.25) < abs(40 - 39.25)
    assert _keys_peak(np.array([1.0, 1.0, 1.0, 1.0, 1.0])) == 0.0
    assert _keys_peak(np.array([0.97, 0.9999, 1.0, 0.99, 0.97])) == 0.0


def test_hdevelop_batch103():
    from app.engine.datamatrix import decode_image, encode_modules
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_datamatrix_print_datamatrix_read_VISDEV.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Text"] == "VISDEV"
    assert len(sample.vars["Symbol"].contours) == 1
    assert "'Data Matrix'" in REGISTRY["create_data_code_2d_model"].control_in[0].choices
    modules = encode_modules("12AB")
    assert decode_image(modules)[0][0] == "12AB"


def test_hdevelop_batch105():
    from app.engine.operators import REGISTRY
    from app.engine.pdf417 import decode_image, encode_modules

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_pdf417_print_pdf417_read_VISDEV.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Text"] == "VISDEV"
    assert len(sample.vars["Symbol"].contours) == 1
    assert "'PDF417'" in REGISTRY["create_data_code_2d_model"].control_in[0].choices
    assert decode_image(encode_modules("A"))[0][0] == "A"
    assert decode_image(encode_modules("Hi-9"))[0][0] == "Hi-9"


def test_hdevelop_batch106():
    from app.engine.aztec import decode_image, encode_modules
    from app.engine.operators import REGISTRY

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_aztec_print_aztec_read_VISDEV.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Text"] == "VISDEV"
    assert len(sample.vars["Symbol"].contours) == 1
    assert "'Aztec'" in REGISTRY["create_data_code_2d_model"].control_in[0].choices
    assert decode_image(encode_modules("A"))[0][0] == "A"
    assert encode_modules("1234567890123456").shape[0] > 15
    assert decode_image(encode_modules("1234567890123456"))[0][0] == "1234567890123456"


def test_hdevelop_batch107():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_open_serial_serial_write_read_VISDEV.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Text"] == "VISDEV"
    missing = run_source("Vizion.open_serial ('COM199', Handle)\n")
    assert missing.error is not None and "打不開 COM199" in missing.error


def test_hdevelop_batch108():
    import socket
    import threading

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_open_socket_connect_tcp_send_read_VISDEV.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Text"] == "VISDEV"
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def serve():
        conn, _addr = server.accept()
        conn.settimeout(2)
        data = b""
        while len(data) < 6:
            part = conn.recv(6 - len(data))
            if not part:
                break
            data += part
        conn.sendall(data)
        conn.close()

    threading.Thread(target=serve, daemon=True).start()
    live = run_source(
        f"Vizion.open_socket_connect ('127.0.0.1', {port}, Socket)\n"
        "Vizion.send_data (Socket, 'VISDEV')\n"
        "Vizion.receive_data (Socket, 6, Text)\n"
        "Vizion.close_socket (Socket)\n"
    )
    server.close()
    assert live.error is None, live.error
    assert live.vars["Text"] == "VISDEV"
    closed = socket.socket()
    closed.bind(("127.0.0.1", 0))
    missing_port = closed.getsockname()[1]
    closed.close()
    missing = run_source(f"Vizion.open_socket_connect ('127.0.0.1', {missing_port}, Socket)\n")
    assert missing.error is not None and f"連不上 127.0.0.1:{missing_port}" in missing.error


def test_hdevelop_batch109():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_split_skeleton_cross_into_four.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Count"] == 4
    assert sample.vars["Parts"].count() == 4


def test_live_camera_reads_a_new_frame_each_grab():
    from app.engine.chapters import camera_ops

    values = [10, 80, 160, 200]

    def grab(_index):
        return np.full((3, 4, 3), values.pop(0), np.uint8)

    handle = camera_ops.AcqHandle("opencv", "0", "gray", 0, 0, cap=camera_ops._LiveCamera(grab, 0, grab(0)))
    first = camera_ops._load_camera(handle, False)
    second = camera_ops._load_camera(handle, False)
    latest = camera_ops._load_camera(handle, True)
    assert int(first.mean()) == 10
    assert int(second.mean()) == 80
    assert int(latest.mean()) == 200
    handle.release()
    try:
        camera_ops._load_camera(handle, False)
    except RuntimeError as exc:
        assert "相機沒有打開" in str(exc)
    else:
        raise AssertionError("a closed camera should fail")


def test_camera_failures_say_what_to_do_next():
    missing = run_source("Vizion.open_framegrabber ('File', 'samples/no-such-camera.png', Acq)\n")
    assert missing.error and "找不到" in missing.error and "選檔" in missing.error
    folder = ROOT / "workspace" / "empty_camera_dir"
    folder.mkdir(parents=True, exist_ok=True)
    for child in folder.iterdir():
        if child.is_file():
            child.unlink()
    empty = run_source("Vizion.open_framegrabber ('File', 'workspace/empty_camera_dir', Acq)\n")
    assert empty.error and "沒有影像" in empty.error and "png" in empty.error
    done = run_source(
        "Vizion.open_framegrabber ('File', 'samples/code39.png', Acq)\n"
        "Vizion.grab_image (Image, Acq)\n"
        "Vizion.grab_image (Image2, Acq)\n"
    )
    assert done.error and "重新 open_framegrabber" in done.error


def test_unopened_camera_error_stays_the_same():
    from app.engine.chapters import camera_ops
    import app.engine.dshow as dshow

    original_grab = dshow.grab_bgr
    original_capture = camera_ops.cv2.VideoCapture

    def boom(_index):
        raise RuntimeError("sample grabber failed")

    class Dead:
        def isOpened(self):
            return False

        def release(self):
            return None

    dshow.grab_bgr = boom
    camera_ops.cv2.VideoCapture = lambda *_args, **_kwargs: Dead()
    try:
        try:
            camera_ops._open_camera(0, 0, 0)
        except RuntimeError as exc:
            assert str(exc) == (
                "打不開相機 0。請確認相機有接上、沒有被別的程式占用。編號 0 是第一台。"
                "沒有相機時，請改跑檔案、範本裡的產線，用 samples/line_batch。"
            )
        else:
            raise AssertionError("camera should not open")
    finally:
        dshow.grab_bgr = original_grab
        camera_ops.cv2.VideoCapture = original_capture


def test_open_framegrabber_directshow():
    session = run_source(
        "Vizion.open_framegrabber ('OpenCV', '0', 'gray', 320, 240, Handle)\n"
        "Vizion.grab_image (Image, Handle)\n"
        "Vizion.close_framegrabber (Handle)\n"
    )
    if session.error:
        assert "打不開相機" in session.error or "讀不到畫面" in session.error
        return
    image = session.vars["Image"]
    assert image.height == 240 and image.width == 320


def test_hdevelop_batch110():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_set_serial_param_serial_baud_19200.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Baud"] == 19200
    assert sample.vars["Bits"] == 8
    assert sample.vars["Flow"] == "none"
    assert sample.vars["Total"] == 500
    assert sample.vars["Gap"] == 20
    bad = run_source(
        "Vizion.open_serial ('loopback', Handle)\n"
        "Vizion.set_serial_param (Handle, 9600, 8, 'none', 'mark', 1, 200, 50)\n"
    )
    assert bad.error is not None and "Parity" in bad.error


def test_hdevelop_batch111():
    import socket

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_open_socket_accept_listen_read_VISDEV.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Text"] == "VISDEV"
    assert sample.vars["Wait"] == 1
    assert int(sample.vars["Port"]) > 0
    held = socket.socket()
    held.bind(("127.0.0.1", 0))
    busy = held.getsockname()[1]
    blocked = run_source(f"Vizion.open_socket_accept ({busy}, Listen, Port)\n")
    held.close()
    assert blocked.error is not None and f"聽不了 127.0.0.1:{busy}" in blocked.error


def test_hdevelop_batch112():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_ean8_print_ean8_read_55123453.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Text"] == "55123453"
    cases = [
        ("Vizion.gen_upce (Code, '123456')\n", "UPC-E", "123456"),
        ("Vizion.gen_codabar (Code, 'A1234B')\n", "Codabar", "A1234B"),
        ("Vizion.gen_code2of5 (Code, '123456')\n", "2/5 Interleaved", "123456"),
        ("Vizion.gen_industrial2of5 (Code, '12345')\n", "2/5 Industrial", "12345"),
        ("Vizion.gen_code93 (Code, 'VISDEV')\n", "Code 93", "VISDEV"),
        ("Vizion.gen_msi (Code, '123456')\n", "MSI", "123456"),
    ]
    for head, kind, expected in cases:
        session = run_source(
            head
            + "Vizion.create_bar_code_model (Handle)\n"
            + f"Vizion.find_bar_code (Code, Symbol, Handle, '{kind}', Decoded)\n"
            + "Text := Decoded[1]\n"
        )
        assert session.error is None, (kind, session.error)
        assert session.vars["Text"] == expected


def test_hdevelop_batch113():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_rectangularity_bar_rectangularity_and_direction.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(sample.vars["Angle"] - 0) < 1e-6
    assert abs(sample.vars["LinePhi"] - 0) < 1e-9
    assert abs(sample.vars["Axis"] - 0) < 1e-9
    assert sample.vars["Rect1"] == sample.vars["FeatRect"][0]
    assert sample.vars["Circ"][0] == sample.vars["FeatCirc"][0]
    assert sample.vars["Ecc"][0] == sample.vars["FeatEcc"][0]
    assert sample.vars["Rect1"] > 0.9
    folded = run_source(
        "Vizion.line_orientation (0, 10, 0, 0, Phi)\n"
        "Vizion.angle_lx (0, 10, 0, 0, Raw)\n"
    )
    assert folded.error is None, folded.error
    assert abs(folded.vars["Phi"]) < 1e-9
    assert abs(abs(folded.vars["Raw"]) - 3.14159265) < 1e-5


def test_hdevelop_batch114():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_circularity_xld_square_length_is_120.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Len"] == 120
    assert sample.vars["FeatLen"][0] == 120
    assert sample.vars["FirstRow"] == 10
    assert sample.vars["FirstCol"] == 10
    assert sample.vars["StartGray"] == 180
    assert sample.vars["Pixel"] == 180
    assert sample.vars["Circ"][0] == sample.vars["FeatCirc"][0]
    assert sample.vars["Comp"][0] == sample.vars["FeatComp"][0]
    assert sample.vars["Conv"][0] == sample.vars["FeatConv"][0]
    assert sample.vars["Ecc"][0] == sample.vars["FeatEcc"][0]


def test_hdevelop_batch115():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_remove_obj_two_blocks_drop_second.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Count"] == 2
    assert sample.vars["Left"] == 1
    assert sample.vars["Added"] == 3
    assert sample.vars["Same"] == 2
    assert sample.vars["DilateArea"] == sample.vars["RankArea"]
    eroded = run_source(
        "Vizion.gen_image_const (Image, 'byte', 80, 60)\n"
        "Vizion.gen_rectangle1 (Top, 10, 10, 20, 25)\n"
        "Vizion.erosion_rectangle1 (Top, Eroded, 3, 3)\n"
        "Vizion.rank_region (Top, Ranked, 3, 3, 9)\n"
        "Vizion.area_center (Eroded, AreaE, RowE, ColE)\n"
        "Vizion.area_center (Ranked, AreaR, RowR, ColR)\n"
        "ErodeArea := AreaE[1]\n"
        "RankArea := AreaR[1]\n"
    )
    assert eroded.error is None, eroded.error
    assert eroded.vars["ErodeArea"] == eroded.vars["RankArea"]


def test_hdevelop_batch116():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_bit_mask_40_and_240_is_32.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["MaskGray"] == 32
    assert sample.vars["AndGray"] == 32
    assert sample.vars["SliceGray"] == 255
    assert sample.vars["DevGray"] == sample.vars["SigmaGray"]


def test_hdevelop_batch117():
    import math
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gen_sphere_object_model_3d_center_123_radius_4.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Cx"] == 1
    assert sample.vars["Cy"] == 2
    assert sample.vars["Cz"] == 3
    assert sample.vars["Radius"] == 4
    assert sample.vars["MinX"] == -3
    assert sample.vars["MaxX"] == 5
    assert sample.vars["Diameter"] == 8
    assert sample.vars["Center"][0] == 1
    box = run_source(
        "Vizion.gen_box_object_model_3d (0, 0, 0, 2, 2, 2, Box)\n"
        "Vizion.smallest_sphere_object_model_3d (Box, Cx, Cy, Cz, Radius)\n"
        "Vizion.get_object_model_3d_params (Box, 'diameter', Diameter)\n"
        "Vizion.get_object_model_3d_params (Box, 'center', Center)\n"
    )
    assert box.error is None, box.error
    assert abs(box.vars["Cx"] - 1) < 1e-6
    assert abs(box.vars["Radius"] - math.sqrt(3)) < 1e-6
    assert abs(box.vars["Diameter"] - 2 * math.sqrt(3)) < 1e-6
    assert abs(box.vars["Center"][0] - box.vars["Cx"]) < 1e-6
    triangle = run_source(
        "Vizion.gen_object_model_3d_from_points ([0, 2, 1], [0, 0, 1.73205080757], [0, 0, 0], Tri)\n"
        "Vizion.smallest_sphere_object_model_3d (Tri, Cx, Cy, Cz, Radius)\n"
    )
    assert triangle.error is None, triangle.error
    assert abs(triangle.vars["Cx"] - 1) < 1e-5
    assert abs(triangle.vars["Cy"] - 0.57735026919) < 1e-5
    assert abs(triangle.vars["Radius"] - 1.15470053838) < 1e-5


def test_hdevelop_batch118():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_frangi_dark_line_brighter_than_block.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["LineGray"] > sample.vars["BlockGray"]
    assert sample.vars["LineGray"] > 80
    assert sample.vars["BlockGray"] < 20
    assert sample.vars["FlatGray"] < 20
    assert sample.vars["Caught"] > 100


def test_hdevelop_batch119():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_sauvola_bright_side_scratch_stays.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["ScratchGray"] == 78
    assert sample.vars["Scratch"] == 1
    assert sample.vars["Block"] == 0
    assert sample.vars["LeftFlat"] == 0
    assert sample.vars["RightFlat"] == 0
    assert sample.vars["GlobalScratch"] == 0
    assert sample.vars["GlobalBlock"] == 1
    even = run_source("Vizion.read_image (Scene, 'samples/hard_scene.png')\nVizion.sauvola (Scene, Ink, 14, 0.5)\n")
    assert "奇數" in even.error
    low = run_source("Vizion.read_image (Scene, 'samples/hard_scene.png')\nVizion.sauvola (Scene, Ink, 15, 0)\n")
    assert "K" in low.error


def test_hdevelop_batch120():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_guided_filter_step_sides_stay.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Left"] < 55
    assert sample.vars["Right"] > 180
    assert sample.vars["BlurLeft"] > sample.vars["Left"] + 10
    assert sample.vars["BlurRight"] < sample.vars["Right"] - 5
    assert sample.vars["NoiseAfter"] < sample.vars["NoiseBefore"] * 0.7
    bad = run_source("Vizion.read_image (Scene, 'samples/hard_edge.png')\nVizion.guided_filter (Scene, Kept, 0, 800)\n")
    assert "Radius" in bad.error
    low = run_source("Vizion.read_image (Scene, 'samples/hard_edge.png')\nVizion.guided_filter (Scene, Kept, 8, 0)\n")
    assert "Eps" in low.error


def test_hdevelop_batch121():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_emphasize_clahe_dark_corner_and_bright_field.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["ClaheDark"] > sample.vars["RawDark"] + 15
    assert sample.vars["ClaheBright"] > sample.vars["RawBright"] + 15
    assert sample.vars["HistDark"] < sample.vars["RawDark"]
    assert sample.vars["HistBright"] > sample.vars["ClaheBright"]
    bad = run_source("Vizion.read_image (Scene, 'samples/hard_contrast.png')\nVizion.emphasize_clahe (Scene, Local, 0, 8)\n")
    assert "ClipLimit" in bad.error
    tiles = run_source("Vizion.read_image (Scene, 'samples/hard_contrast.png')\nVizion.emphasize_clahe (Scene, Local, 8, 1)\n")
    assert "Tiles" in tiles.error


def test_hdevelop_batch122():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_edges_zernike_edge_matches_measure.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(sample.vars["ZCol"] - sample.vars["Edge"]) < 0.05
    assert abs(sample.vars["ZCol"] - 40.37) < 0.15
    assert abs(sample.vars["Edge"] - 40.37) < 0.15
    assert len(sample.vars["Cols"]) > 80
    bad = run_source("Vizion.read_image (Scene, 'samples/hard_zernike.png')\nVizion.edges_zernike (Scene, Edges, 6, 80, Rows, Cols)\n")
    assert "奇數" in bad.error
    low = run_source("Vizion.read_image (Scene, 'samples/hard_zernike.png')\nVizion.edges_zernike (Scene, Edges, 7, 0, Rows, Cols)\n")
    assert "MinAmp" in low.error


def test_hdevelop_batch123():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_tuple_is_handle_matrix_stays_after_clear.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Head"] == "H" and sample.vars["Third"] == "C"
    assert sample.vars["OsName"] == "Windows_NT"
    assert [sample.vars[name] for name in ("C1", "C2", "C3", "C4")] == [19, 22, 43, 50]
    assert sample.vars["IsHandle"] == 1 and sample.vars["StillHandle"] == 1
    assert sample.vars["Valid"] == 1 and sample.vars["ValidAfter"] == 0
    assert sample.vars["Serial"] == 0 and sample.vars["NumberHandle"] == 0
    assert sample.vars["Sem"] == "matrix"
    assert sample.vars["IntOne"] == 1 and sample.vars["RealOne"] == 0
    missing = run_source("Vizion.tuple_environment ('VISDEV_NO_SUCH_ENV', Value)\n")
    assert "環境變數" in missing.error
    over = run_source("Vizion.tuple_str_bit_select ('HALCON', 9, Selected)\n")
    assert "超出" in over.error


def test_hdevelop_batch124():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_sum_matrix_copy_cholesky_and_file.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Total"] == 10
    assert sample.vars["D1"] == 2 and sample.vars["O1"] == 1
    assert sample.vars["Low"] == -1 and sample.vars["High"] == 1
    assert sample.vars["R1"] == 0 and sample.vars["R2"] == 0
    assert [sample.vars[name] for name in ("L1", "L2", "L3", "L4")] == [5, 0, 3, 3]
    assert [sample.vars[name] for name in ("U1", "U2", "U3", "U4")] == [5, 3, 0, 3]
    assert sample.vars["B1"] == 5
    factored = run_source("""
Vizion.create_matrix (2, 2, [4, 3, 6, 3], M)
Vizion.decompose_matrix (M, 'general', Lower, Upper)
Vizion.get_full_matrix (Lower, L)
Vizion.get_full_matrix (Upper, U)
""")
    assert factored.error is None, factored.error
    assert factored.vars["L"][0] == 1
    assert abs(factored.vars["L"][2] - 2 / 3) < 1e-9
    assert factored.vars["U"][0] == 6 and factored.vars["U"][3] == 1
    missing = run_source("Vizion.read_matrix ('workspace/no_such_matrix.mtx', M)\n")
    assert "讀不到" in missing.error


def test_hdevelop_batch131():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_equ_histo_image_rect_lower_right_is_255.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Eq00"] == 0 and sample.vars["Eq12"] == 85 and sample.vars["Eq21"] == 170 and sample.vars["Eq22"] == 255
    assert sample.vars["CropW"] == 4 and sample.vars["CropH"] == 3 and sample.vars["CropGray"] == 9
    assert sample.vars["PartW"] == 5 and sample.vars["PartH"] == 4 and sample.vars["PartGray"] == 4
    assert sample.vars["ExpW"] == 6 and sample.vars["ExpH"] == 6 and sample.vars["Edge9"] == 9 and sample.vars["Inner5"] == 5
    assert sample.vars["InsideCenter"] == 40 and sample.vars["InsideEdge"] == 0
    assert sample.vars["SkelMid"] == 9 and sample.vars["SkelSide"] == 0
    assert sample.vars["PickedRed"] == 10 and sample.vars["MovedGray"] == 7
    assert sample.vars["GrowCount"] == 2 and sample.vars["Grow1"] == 4 and sample.vars["Grow2"] == 4
    assert sample.vars["Bar1"] == 4 and sample.vars["RandCount"] == 2 and sample.vars["Rand1"] == 4 and sample.vars["Rand2"] == 4
    assert sample.vars["Ham1"] == 13
    assert sample.vars["StartRow"] == 0 and sample.vars["StartCol"] == 0
    assert sample.vars["Code0"] == 2 and sample.vars["Code1"] == 0 and sample.vars["Code2"] == 6 and sample.vars["Code3"] == 4
    assert sample.vars["Len1"] == 10 and sample.vars["PhiLine"] == 0 and sample.vars["ColEnd"] == 10
    assert sample.vars["LenReg"] == 10 and sample.vars["PhiReg"] == 0 and sample.vars["RowReg"] == 0 and sample.vars["ColReg"] == 5
    assert abs(sample.vars["DevReg"]) < 1e-6
    assert sample.vars["Ell0"] == 4 and sample.vars["Ell1"] == 0 and sample.vars["Rect0"] == 2 and sample.vars["Rect1"] == 0
    assert sample.vars["M20a"] == 8 and sample.vars["M02a"] == 8 and sample.vars["M11a"] == 4 and sample.vars["Any20"] == 8
    assert sample.vars["KeptCount"] == 1


def test_hdevelop_batch130():
    import math

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_polar_trans_image_radius3_is_200.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["PolarGray"] == 200
    assert sample.vars["Red1"] == 10 and sample.vars["Green1"] == 20 and sample.vars["Blue1"] == 30
    assert abs(sample.vars["Ra1"] - math.sqrt(8)) < 1e-6 and sample.vars["Rb1"] == 0 and sample.vars["Phi1"] == 0
    assert sample.vars["Grow1"] == 4 and sample.vars["KeptCount"] == 1
    assert sample.vars["Clip0"] == 2 and sample.vars["Clip1"] == 8 and sample.vars["Dist1"] == 3
    assert sample.vars["Ori1"] == 0 and sample.vars["Era1"] == 6 and sample.vars["Erb1"] == 2
    assert sample.vars["Ephi1"] == 0 and sample.vars["Ecc1"] == 3
    assert sample.vars["Abs11"] == 16 and sample.vars["MeanCenter"] == 1
    assert sample.vars["TileGray"] == 7 and sample.vars["TileZero"] == 0
    assert sample.vars["Line1"] == 4 and sample.vars["Rand1"] == 25
    assert sample.vars["Thick1"] == 3 and sample.vars["I1a"] == 2 and sample.vars["I2a"] == 0
    assert sample.vars["Len1"] == 10 and sample.vars["PhiLine"] == 0
    assert sample.vars["RowA"] == 1 and sample.vars["ColA"] == 2
    assert sample.vars["RowE0"] == 10 and sample.vars["ColE0"] == 24
    assert abs(sample.vars["RowE90"] - 12) < 1e-6 and abs(sample.vars["ColE90"] - 20) < 1e-6


def test_hdevelop_batch129():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_gray_dilation_shape_neighbor_is_9.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["DilateNear"] == 9 and sample.vars["ShapeNear"] == 9
    assert sample.vars["OpenCenter"] == 0 and sample.vars["OpenShape"] == 0
    assert sample.vars["ErodeNear"] == 0 and sample.vars["ErodeCorner"] == 9 and sample.vars["ErodeShape"] == 0
    assert sample.vars["CloseCenter"] == 9 and sample.vars["CloseShape"] == 9
    assert sample.vars["M20a"] == 8 and sample.vars["M02a"] == 8 and sample.vars["M11a"] == 4
    assert sample.vars["Area1"] == 4 and sample.vars["Row1"] == 1 and sample.vars["Col1"] == 1 and sample.vars["Rect1"] == 1
    assert sample.vars["Clip0"] == 2 and sample.vars["Clip1"] == 8
    assert sample.vars["In1"] == 1 and sample.vars["In2"] == 0 and sample.vars["Fill1"] == 25
    assert sample.vars["HullCount"] == 4 and sample.vars["Clean1"] == 4
    assert sample.vars["Mu20a"] == 1 and sample.vars["Mu02a"] == 1 and sample.vars["Mu11a"] == 0
    assert sample.vars["Bin3"] == 16 and sample.vars["BinSize"] == 5 and sample.vars["Dark1"] == 1
    assert sample.vars["MovedGray"] == 7


def test_hdevelop_batch128():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_sin_image_gray0_sine_is_128.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Sin0"] == 128 and sample.vars["Sin1"] == 235
    assert sample.vars["Cos0"] == 255 and sample.vars["Cos1"] == 196
    assert sample.vars["Tan0"] == 128 and sample.vars["Tan1"] == 140
    assert sample.vars["Asin0"] == 128 and sample.vars["Acos0"] == 128 and sample.vars["Acos255"] == 0
    assert sample.vars["Atan0"] == 128 and sample.vars["Atan160"] == 191
    assert sample.vars["Plane11"] == 15 and sample.vars["Bowl20"] == 8 and sample.vars["RedGray"] == 30
    assert sample.vars["Ent"] == 29 and sample.vars["Mid22"] == 45
    assert sample.vars["FrameW"] == 6 and sample.vars["FrameH"] == 5 and sample.vars["Edge"] == 9
    assert sample.vars["Fast1"] == 3 and sample.vars["Bin1"] == 8 and sample.vars["Soft11"] == 1
    assert sample.vars["Kind"] == "byte"
    assert sample.vars["Poly1"] == 20 and sample.vars["In1"] == 1 and sample.vars["In2"] == 0
    assert sample.vars["Dist1"] == 5 and abs(sample.vars["Ratio1"] - 4 / 3) < 1e-9


def np_degrees_diag():
    import math

    return math.degrees(math.atan2(2 ** 0.5, 1.0))


def test_language():
    source = """
Sum := 0
for Index := 1 to 4 by 1
  Sum := Sum + Index
endfor
T := [10, 20, 30]
A := T[2]
L := |T|
if (Sum = 10 and A = 20)
  Ok := 1
else
  Ok := 0
endif
while (L < 5)
  L := L + 1
endwhile
"""
    session = run_source(source)
    assert session.error is None, session.error
    assert session.vars["Sum"] == 10
    assert session.vars["A"] == 20
    assert session.vars["Ok"] == 1
    rounded = run_source("A := round(14.2364, 3)\nB := round(deg(rad(15)), 3)\n")
    assert rounded.error is None, rounded.error
    assert rounded.vars["A"] == 14.236
    assert rounded.vars["B"] == 15.0
    assert session.vars["L"] == 5
    assert session.vars["Index"] == 5


def test_blob():
    ensure_samples(ROOT)
    source = (ROOT / "samples" / "blob_inspect.hdv").read_text(encoding="utf-8")
    session = run_source(source)
    print("blob", session.error, "count", session.vars.get("Count"), "area", session.vars.get("Area"))
    assert session.error is None, session.error
    assert session.vars["Count"] == 3, session.vars["Count"]
    assert len(session.vars["Area"]) == 3
    display = session.public_state()["display"]
    assert display and display["image"]
    kinds = {shape["type"] for shape in display["shapes"]}
    assert "polyline" in kinds and "text" in kinds


def test_other_samples():
    from app.engine.samples_gen import SAMPLE_ORDER

    ensure_samples(ROOT)
    titles = []
    for name in SAMPLE_ORDER:
        source = (ROOT / "samples" / name).read_text(encoding="utf-8")
        title = next((line.strip().lstrip("*").strip() for line in source.splitlines() if line.strip().startswith("*")), "")
        assert title, name
        titles.append(title)
        session = run_source(source)
        assert session.error is None, (name, session.error)
        if name == "ocr_read.hdv":
            found = " ".join(str(item) for item in session.vars["Text"])
            assert "VISDEV" in found and "168" in found, found
    assert len(titles) == len(set(titles))


def test_shape_match():
    import math

    import cv2
    import numpy as np

    from app.engine.objects import Image
    from app.engine.render import read_cv_image
    from app.engine.samples_gen import SHAPE_POSES
    from app.engine.shape_model import MATCH_COLORS, create_shape_model, find_shape_model, transform_contours

    ensure_samples(ROOT)
    template = Image(read_cv_image(ROOT / "samples" / "shape_template.png"))
    scene = Image(read_cv_image(ROOT / "samples" / "shapes_scene.png"))
    model = create_shape_model(template, 0, 0, "use_polarity", 40, 10)
    assert model.contours, "模型應該有次像素輪廓"
    rows, cols, angles, scores = find_shape_model(
        scene, model, math.radians(-180), math.radians(360), 0.7, 0, 0.4, True, 0, 0.4
    )
    print("shape", list(zip(rows, cols, [math.degrees(a) for a in angles], scores)))
    assert len(rows) == 4, len(rows)
    used = set()
    for row, col, degrees in SHAPE_POSES:
        best = None
        for index, (found_row, found_col, found_angle, score) in enumerate(zip(rows, cols, angles, scores)):
            if index in used:
                continue
            distance = math.hypot(found_row - row, found_col - col)
            turn = abs((found_angle - math.radians(degrees) + math.pi) % (2 * math.pi) - math.pi)
            if best is None or distance < best[0]:
                best = (distance, turn, score, index)
        assert best[0] <= 1.5, best
        assert best[1] <= math.radians(2.0), best
        assert best[2] >= 0.65, best
        used.add(best[3])

    gray = scene.gray().astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edge = (np.hypot(gx, gy) >= 40).astype(np.uint8)
    distance = cv2.distanceTransform(1 - edge, cv2.DIST_L2, 3)
    for row, col, angle in zip(rows, cols, angles):
        points = [point for chain in transform_contours(model, row, col, angle) for point in chain]
        assert len(points) >= 8
        samples = []
        for x, y in points[::2]:
            iy = int(round(y))
            ix = int(round(x))
            if 0 <= iy < distance.shape[0] and 0 <= ix < distance.shape[1]:
                samples.append(float(distance[iy, ix]))
        assert samples and float(np.mean(samples)) < 1.5, float(np.mean(samples))

    source = (ROOT / "samples" / "shape_match.hdv").read_text(encoding="utf-8")
    session = run_source(source)
    assert session.error is None, session.error
    assert session.vars["Count"] == 4
    colors = []
    for shape in session.public_state()["display"]["shapes"]:
        if shape["type"] == "polyline":
            colors.append(shape["color"].lower())
    distinct = set(colors)
    assert len(distinct) >= 4, distinct
    assert distinct <= {item.lower() for item in MATCH_COLORS}
    degree_labels = [
        shape["text"]
        for shape in session.public_state()["display"]["shapes"]
        if shape["type"] == "text" and shape["text"].endswith("°")
    ]
    assert len(degree_labels) == 4
    assert all(len(item.split(".")[-1]) == 4 for item in degree_labels)


def test_ncc_locate():
    import math

    from app.engine.objects import Image
    from app.engine.operators import teach_ncc_model
    from app.engine.render import read_cv_image
    from app.engine.samples_gen import SHAPE_POSES
    from app.engine.shape_model import MATCH_COLORS

    class Ctx:
        pass

    ensure_samples(ROOT)
    template = Image(read_cv_image(ROOT / "samples" / "shape_template.png"))
    scene = Image(read_cv_image(ROOT / "samples" / "shapes_scene.png"))
    taught = teach_ncc_model(template, 15, 15, 164, 164, 0, math.radians(-180), math.radians(360), 0, "use_polarity", ctx=Ctx())
    model = taught.control[0]
    assert "已教導" in model.summary()
    from app.engine.ncc_model import find_ncc_model

    rows, cols, angles, scores = find_ncc_model(
        scene, model, math.radians(-180), math.radians(360), 0.75, 0, 0.4, True, 0
    )
    print("ncc", list(zip(rows, cols, [round(math.degrees(a), 2) for a in angles], [round(s, 3) for s in scores])))
    assert len(rows) == 4, len(rows)
    used = set()
    for row, col, degrees in SHAPE_POSES:
        best = None
        for index, (found_row, found_col, found_angle, score) in enumerate(zip(rows, cols, angles, scores)):
            if index in used:
                continue
            distance = math.hypot(found_row - row, found_col - col)
            turn = abs((found_angle - math.radians(degrees) + math.pi) % (2 * math.pi) - math.pi)
            if best is None or distance < best[0]:
                best = (distance, turn, score, index)
        assert best[0] <= 1.5, best
        assert best[1] <= math.radians(2.0), best
        assert best[2] >= 0.75, best
        used.add(best[3])

    source = (ROOT / "samples" / "ncc_locate.hdv").read_text(encoding="utf-8")
    session = run_source(source)
    assert session.error is None, session.error
    assert session.vars["Count"] == 4
    assert "已教導" in session.vars["ModelID"].summary()
    polylines = []
    labels = []
    for shape in session.public_state()["display"]["shapes"]:
        if shape["type"] == "polyline":
            polylines.append(shape)
        if shape["type"] == "text" and "°" in shape["text"]:
            labels.append(shape["text"])
    assert polylines and max(len(item["points"]) for item in polylines) <= 4
    assert len(labels) == 4
    assert all(len(item.split(".")[-1]) == 4 and item.endswith("°") for item in labels)
    assert len({item["color"].lower() for item in polylines}) >= 4
    assert {item["color"].lower() for item in polylines} <= {item.lower() for item in MATCH_COLORS}


def test_threshold_histogram_and_morphology():
    import cv2

    ensure_samples(ROOT)
    source = """
Vizion.read_image (Image, 'samples/parts.png')
Vizion.threshold (Image, Region, 60, 255)
Vizion.dev_set_color ('#ff3b30')
Vizion.dev_display (Image)
Vizion.dev_display (Region)
Vizion.erosion_rectangle1 (Region, Eroded, 5, 5)
Vizion.dilation_rectangle1 (Region, Dilated, 5, 5)
Vizion.opening_rectangle1 (Region, Opened, 5, 5)
Vizion.closing_rectangle1 (Region, Closed, 5, 5)
Vizion.opening_circle (Region, CircleOpen, 3.5)
Vizion.boundary (Region, Border, 'inner')
Vizion.connection (Region, Parts)
Vizion.dev_set_colored (6)
Vizion.dev_display (Parts)
"""
    session = run_source(source)
    assert session.error is None, session.error
    histo = session.public_state()["histogram"]
    assert histo["kind"] == "range"
    assert histo["literal"] is True
    assert histo["min"] == 60 and histo["max"] == 255
    assert len(histo["bins"]) == 256
    assert sum(histo["bins"][60:256]) == histo["area"]
    assert histo["color"] == "#ff3b30"
    original = int(cv2.countNonZero(session.vars["Region"].union_mask()))
    eroded = int(cv2.countNonZero(session.vars["Eroded"].union_mask()))
    dilated = int(cv2.countNonZero(session.vars["Dilated"].union_mask()))
    border = int(cv2.countNonZero(session.vars["Border"].union_mask()))
    assert eroded < original < dilated
    assert 0 < border < original
    assert session.vars["Opened"].count() == 1
    assert session.vars["Closed"].count() == 1
    assert session.vars["CircleOpen"].count() == 1
    colors = {shape["color"].lower() for shape in session.public_state()["display"]["shapes"] if shape["type"] == "polyline"}
    assert len(colors) >= 2

    binary = run_source("""
Vizion.read_image (Image, 'samples/parts.png')
Vizion.binary_threshold (Image, Region, 'max_separability', 'light', Used)
""")
    assert binary.error is None, binary.error
    otsu = binary.public_state()["histogram"]
    assert otsu["kind"] == "otsu"
    assert otsu["literal"] is False
    assert otsu["threshold"] == binary.vars["Used"]


def test_preview_mask_while_dragging():
    import numpy as np

    from app.engine.operators import preview_mask

    gray = np.array([[10, 60], [80, 200]], np.uint8)
    ranged = preview_mask(gray, None, "range", 60, 255, 0, "binary")
    assert ranged.tolist() == [[0, 255], [255, 255]]
    opened = preview_mask(gray, None, "opencv", 0, 255, 60, "binary")
    assert opened.tolist() == [[0, 0], [255, 255]]
    inverted = preview_mask(gray, None, "opencv", 0, 255, 60, "binary_inv")
    assert inverted.tolist() == [[255, 255], [0, 0]]


def test_opencv_link_list():
    import numpy as np

    from app.engine.objects import Image
    from app.engine.operators import opencv_connection, opencv_threshold, select_obj
    from app.engine.render import compose_display

    class Ctx:
        pass

    gray = np.zeros((40, 40), np.uint8)
    gray[5:25, 5:25] = 200
    gray[10:16, 10:16] = 0
    gray[6:14, 30:37] = 180
    region = opencv_threshold(Image(gray), 100, "binary", ctx=Ctx()).iconic[0]
    assert region.count() == 1
    link = region.links[0]
    outers = [index for index, parent in enumerate(link["parent"]) if parent < 0]
    holes = [index for index, parent in enumerate(link["parent"]) if parent >= 0]
    assert len(outers) == 2
    assert len(holes) == 1
    owner = link["parent"][holes[0]]
    assert owner in outers
    chain = []
    cursor = link["child"][owner]
    while cursor >= 0 and cursor not in chain:
        chain.append(cursor)
        cursor = link["next"][cursor]
    assert holes[0] in chain
    assert any(link["next"][index] in outers or link["prev"][index] in outers for index in outers)
    assert all(len(chain) >= 4 for chain in link["points"])

    parts = opencv_connection(region, 8, ctx=Ctx()).iconic[0]
    assert parts.count() == 2
    assert len(parts.links) == 2
    assert any(any(parent >= 0 for parent in item["parent"]) for item in parts.links)
    picked = select_obj(parts, 1, ctx=Ctx()).iconic[0]
    assert picked.count() == 1 and picked.links and len(picked.links) == 1

    shapes = compose_display(Image(gray), [{"kind": "region", "region": region, "color": "#00e05a", "draw": "margin", "width": 2}])
    assert sum(1 for shape in shapes["shapes"] if shape["type"] == "polyline") >= 3


def test_features_and_auto_threshold():
    import numpy as np

    from app.engine.objects import Image
    from app.engine.operators import binary_threshold, region_measures, smooth_histo_level, subpixel_boundary

    class Ctx:
        pass

    gray = np.zeros((80, 80), np.uint8)
    gray[:40] = 20
    gray[40:] = 200
    level = smooth_histo_level(gray)
    assert 21 <= level <= 199
    mask = np.zeros((40, 40), np.uint8)
    mask[5:30, 5:30] = 255
    mask[12:20, 12:20] = 0
    values = region_measures(mask)
    assert values["area"] == 25 * 25 - 8 * 8
    from app.engine.operators import gray_measures
    bright = np.zeros((40, 40), np.uint8)
    bright[5:30, 5:30] = 180
    bright[12:20, 12:20] = 40
    gray_values = gray_measures(mask, bright)
    assert gray_values["gray_min"] == 180
    assert gray_values["gray_max"] == 180
    assert gray_values["gray_mean"] == 180
    assert gray_values["gray_median"] == 180
    assert values["holes_num"] == 1
    assert values["area_holes"] > 20
    assert values["circularity"] > 0
    ramp = np.tile(np.arange(40, dtype=np.uint8), (30, 1))
    solid = np.zeros((30, 40), np.uint8)
    solid[:, 21:] = 255
    lines = subpixel_boundary(solid, ramp, 20.5)
    flat = [point for chain in lines for point in chain]
    assert any(abs(point[0] - round(point[0])) > 0.02 for point in flat)
    ctx = Ctx()
    binary_threshold(Image(gray), "triangle", "light", ctx=ctx)
    assert ctx.histogram["method"] == "triangle"
    binary_threshold(Image(np.full((64, 64), 128, np.uint8)), "local_mean", "dark", ctx=ctx)
    assert ctx.histogram["kind"] == "local"


def test_threshold_sub_pix():
    import numpy as np

    from app.engine.objects import Image
    from app.engine.operators import threshold_sub_pix

    class Ctx:
        pass

    ramp = np.tile(np.arange(40, dtype=np.uint8), (30, 1))
    ctx = Ctx()
    result = threshold_sub_pix(Image(ramp), 20.5, ctx=ctx)
    border = result.iconic[0]
    assert border.multi_color
    assert border.contours
    flat = [point for line in border.contours for point in line]
    assert any(abs(point[0] - round(point[0])) > 0.02 for point in flat)
    assert ctx.histogram["kind"] == "subpix"
    assert ctx.histogram["threshold"] == 20.5
    blobs = np.zeros((40, 80), np.uint8)
    blobs[5:16, 5:16] = 255
    blobs[5:16, 50:66] = 255
    split = threshold_sub_pix(Image(blobs), 128, ctx=ctx).iconic[0]
    assert set(split.groups) == {0, 1}
    assert ctx.threshold_region.count() == 2
    hole = np.zeros((40, 40), np.uint8)
    hole[4:36, 4:36] = 255
    hole[14:26, 14:26] = 0
    holed = threshold_sub_pix(Image(hole), 128, ctx=ctx).iconic[0]
    assert set(holed.groups) == {0}
    assert len(holed.contours) >= 2


def test_binary_region_color():
    import base64

    import cv2
    import numpy as np

    from app.engine.objects import Image
    from app.engine.operators import threshold
    from app.engine.session import Session

    gray = np.zeros((20, 20), np.uint8)
    gray[5:15, 5:15] = 200
    session = Session("* x\n", ROOT)
    session.display.base = Image(gray)
    threshold(Image(gray), 100, 255, ctx=session)
    assert session.region_color == "#ff3b30"

    def fill_pixels(color=None):
        if color:
            session.region_color = color
        state = session.public_state()
        fills = [shape for shape in state["display"]["shapes"] if shape["type"] == "fill"]
        assert fills
        raw = base64.b64decode(fills[0]["image"].split(",", 1)[1])
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
        return image[image[:, :, 3] > 0]

    red = fill_pixels()
    assert int(red[:, 2].max()) > 200
    assert int(red[:, 0].max()) < 80
    blue = fill_pixels("#3d8bff")
    assert int(blue[:, 0].max()) > 200
    assert int(blue[:, 2].max()) < 80


def test_connection_distance():
    import numpy as np

    from app.engine.objects import Region
    from app.engine.operators import connection
    from app.engine.parser import parse_program

    class Ctx:
        histogram = None

    mask = np.zeros((7, 12), np.uint8)
    mask[3, 1] = 255
    mask[3, 4] = 255
    mask[3, 10] = 255
    region = Region(12, 7, [mask])
    assert connection(region, 1, ctx=Ctx()).iconic[0].count() == 3
    assert connection(region, ctx=Ctx()).iconic[0].count() == 3
    near = connection(region, 3, ctx=Ctx()).iconic[0]
    assert near.count() == 2
    assert int(np.count_nonzero(near.union_mask())) == 3
    assert connection(region, 6, ctx=Ctx()).iconic[0].count() == 1
    diagonal = np.zeros((3, 3), np.uint8)
    diagonal[0, 1] = 255
    diagonal[1, 0] = 255
    assert connection(Region(3, 3, [diagonal]), 1, ctx=Ctx()).iconic[0].count() == 1
    omitted = parse_program("Vizion.connection (Region, Parts)\n")
    given = parse_program("Vizion.connection (Region, Parts, 4.5)\n")
    assert omitted.stmts[0].control_in == ["1"]
    assert given.stmts[0].control_in == ["4.5"]


def test_connection_halcon_and_opencv():
    import numpy as np

    from app.engine.objects import Region
    from app.engine.operators import connection, opencv_connection, opencv_threshold
    from app.engine.shape_model import MATCH_COLORS

    class Ctx:
        histogram = None

    mask = np.zeros((3, 3), np.uint8)
    mask[0, 1] = 255
    mask[1, 0] = 255
    region = Region(3, 3, [mask])
    ctx = Ctx()
    halcon = connection(region, ctx=ctx).iconic[0]
    opencv8 = opencv_connection(region, 8, ctx=ctx).iconic[0]
    opencv4 = opencv_connection(region, 4, ctx=ctx).iconic[0]
    assert halcon.count() == 1 and halcon.multi_color
    assert opencv8.count() == 1 and opencv8.multi_color
    assert opencv4.count() == 2

    ensure_samples(ROOT)
    source = """
Vizion.read_image (Image, 'samples/parts.png')
Vizion.threshold (Image, Region, 60, 255)
Vizion.connection (Region, Parts)
Vizion.dev_display (Image)
Vizion.dev_set_draw ('margin')
Vizion.dev_display (Parts)
Vizion.opencv_threshold (Image, OpenCVRegion, 60, 'binary')
Vizion.opencv_connection (OpenCVRegion, OpenCVParts, 4)
Vizion.count_obj (Parts, HalconCount)
Vizion.count_obj (OpenCVParts, OpenCVCount)
"""
    session = run_source(source)
    assert session.error is None, session.error
    assert session.vars["Parts"].multi_color
    assert session.vars["HalconCount"] >= 3
    assert session.vars["OpenCVCount"] >= 1
    histo = session.public_state()["histogram"]
    assert histo["kind"] == "opencv"
    assert histo["threshold"] == 60
    assert histo["literal"] is True
    colors = [shape["color"].lower() for shape in session.public_state()["display"]["shapes"] if shape["type"] == "polyline"]
    assert len(set(colors)) >= 3
    assert set(colors) <= {item.lower() for item in MATCH_COLORS}


def test_arrow_into_procedure_stops_at_the_header():
    source = "\n".join([
        "procedure add (: : Value : Result)",
        "    Result := Value + 1",
        "endprocedure",
        "Value := 4",
        "add (Value, Out)",
    ])
    session = Session(source, ROOT)
    session.set_line(2)
    assert session.current_line() == 1
    assert session.log[-1] == "程序還沒呼叫，箭頭停在程序標題"
    session.step()
    assert session.current_line() == 4
    assert "Result" not in session.vars
    session.step()
    assert session.vars["Value"] == 4
    session.step()
    assert session.current_line() == 2
    assert session.public_state()["procedure"] == "add"
    session.set_line(2)
    assert session.current_line() == 2
    assert session.vars["Value"] == 4
    session.set_line(5)
    assert session.current_line() == 5
    assert session.public_state()["procedure"] is None
    assert session.vars["Value"] == 4
    assert "Out" not in session.vars
    session.step()
    assert session.current_line() == 2
    session.step()
    assert session.vars["Out"] == 5

    stopped = Session(source, ROOT)
    stopped.breakpoints = {2}
    stopped.run()
    assert stopped.paused == "breakpoint"
    assert stopped.current_line() == 2
    stopped.set_line(4)
    assert stopped.current_line() == 4
    assert stopped.paused is None
    stopped.run()
    assert stopped.paused == "breakpoint"
    assert stopped.current_line() == 2
    assert "Result" not in stopped.vars
    assert stopped.vars["Value"] == 4


def test_failed_run_keeps_results_and_names_the_line():
    source = "\n".join([
        "Vizion.gen_image_const (Image, 'byte', 32, 24)",
        "Vizion.threshold (Image, Region, 0, 255)",
        "Count := 7",
        "Count := Missing + 1",
    ])
    session = run_source(source)
    assert session.error_line == 4
    assert session.current_line() == 4
    assert "第 4 行" in session.error
    assert "Missing" in session.error
    assert sum(1 for line in session.log if str(line).startswith("錯誤")) == 1
    assert "第 4 行" in session.log[-1]
    assert session.vars["Count"] == 7
    assert session.display.base is not None
    assert session.display.base.width == 32
    state = session.public_state()
    assert state["line"] == 4
    assert state["errorLine"] == 4
    assert state["display"] and state["display"].get("image")
    iconic = {item["name"] for item in state["iconic"]}
    assert iconic == {"Image", "Region"}
    assert any(item["name"] == "Count" and "7" in item["text"] for item in state["control"])


def test_failed_procedure_keeps_local_results():
    source = "\n".join([
        "Ready := 3",
        "procedure work (: : : )",
        "    Vizion.gen_image_const (Image, 'byte', 16, 16)",
        "    Vizion.threshold (Image, Region, 0, 255)",
        "    Area := 12",
        "    Area := Missing",
        "endprocedure",
        "work ()",
        "After := 1",
    ])
    session = run_source(source)
    assert session.error_line == 6
    assert session.current_line() == 6
    assert "第 6 行" in session.error
    assert "Missing" in session.error
    assert sum(1 for line in session.log if str(line).startswith("錯誤")) == 1
    assert "Image" not in session.vars
    assert "After" not in session.vars
    assert session.vars["Ready"] == 3
    assert session.display.base is not None
    assert session.display.base.width == 16
    state = session.public_state()
    assert state["procedure"] == "work"
    assert state["line"] == 6
    iconic = {item["name"]: item["scope"] for item in state["iconic"]}
    assert iconic["Image"] == "程序"
    assert iconic["Region"] == "程序"
    control = {item["name"]: item for item in state["control"]}
    assert control["Area"]["scope"] == "程序"
    assert "12" in control["Area"]["text"]
    assert control["Ready"]["scope"] == "呼叫端"
    assert "After" not in control


def test_path_fields_list_project_files():
    from fastapi import HTTPException

    from app.desktop import DesktopApi
    from app.engine.operators import REGISTRY
    from app.main import data_files

    read = next(param for param in REGISTRY["read_image"].control_in if param.name == "FileName")
    plate = next(param for param in REGISTRY["set_calib_data_calib_object"].control_in if param.name == "CalibObjDescr")
    assert read.browse == "image"
    assert plate.browse == "plate"
    written = next(param for param in REGISTRY["write_image"].control_in if param.name == "FileName")
    device = next(param for param in REGISTRY["open_framegrabber"].control_in if param.name == "Device")
    assert written.browse == "image"
    assert device.browse == "image"
    listed = data_files("plate")
    assert any(item["path"] == "samples/calplate_4x5.cpd" for item in listed)
    images = data_files("image")
    assert any(item["path"] == "samples/parts.png" for item in images)
    assert all(not item["name"].startswith(".") for item in images + listed)
    try:
        data_files("nope")
    except HTTPException as exc:
        assert exc.status_code == 400
    else:
        raise AssertionError("不認識的種類應該拒絕")
    api = DesktopApi()
    assert api._portable_path(str(ROOT / "samples" / "calplate_4x5.cpd")) == "samples/calplate_4x5.cpd"
    assert api._portable_path("/tmp/board.cpd") == "/tmp/board.cpd"


def test_calib_fields_say_how_to_fill_them():
    from app.engine.operators import REGISTRY

    plate = REGISTRY["set_calib_data_calib_object"].setup_text()
    assert "'5,7,10'" in plate
    assert "calplate_4x5.cpd" in plate
    camera = REGISTRY["find_calib_object"].setup_text()
    assert "只有一台就填 0" in camera
    created = REGISTRY["create_calib_data"].setup_text()
    assert "兩台填 2" in created
    readout = REGISTRY["get_calib_data"].setup_text()
    assert "第幾台相機" in readout


def test_procedure_error_marks_the_inner_line():
    source = "\n".join([
        "procedure add (: : Value : Result)",
        "    Result := Value + Missing",
        "endprocedure",
        "add (4, Out)",
    ])
    session = run_source(source)
    assert session.error_line == 2
    assert "Missing" in session.error
    assert "Out" not in session.vars
    stepped = Session(source, ROOT)
    stepped.step()
    stepped.step()
    assert stepped.current_line() == 2
    stepped.step()
    assert stepped.error_line == 2
    assert "Missing" in stepped.error


def test_procedure_step_keeps_caller_variables_visible():
    source = "\n".join([
        "Ready := 1",
        "procedure add (: : Value : Result)",
        "    Result := Value + 1",
        "endprocedure",
        "add (4, Out)",
    ])
    session = Session(source, ROOT)
    session.step()
    session.step()
    session.step()
    assert session.current_line() == 3
    assert session.public_state()["procedure"] == "add"
    names = [item["name"] for item in session.public_state()["control"]]
    assert names.index("Value") < names.index("Ready")
    assert "Out" not in names
    scopes = {item["name"]: item["scope"] for item in session.public_state()["control"]}
    assert scopes["Value"] == "程序"
    assert scopes["Ready"] == "呼叫端"
    session.step()
    assert session.vars["Out"] == 5
    assert session.public_state()["procedure"] is None
    after = [item["name"] for item in session.public_state()["control"]]
    assert "Ready" in after and "Out" in after
    assert "Value" not in after
    assert all(item.get("scope") in (None,) for item in session.public_state()["control"])


def test_move_program_arrow():
    session = Session("a := 1\nb := a + 1\nc := b + 1\n", ROOT)
    session.step()
    session.step()
    assert session.vars["b"] == 2
    assert session.current_line() == 3
    session.error = "先停在這裡"
    session.set_line(2)
    assert session.error is None
    assert session.current_line() == 2
    session.step()
    assert session.vars["b"] == 2
    assert session.current_line() == 3
    session.set_line(99)
    assert session.done
    assert session.current_line() is None


def test_xld_contour_tools():
    session = run_source("""
Vizion.gen_contour_polygon_xld (Diag, [0, 20], [0, 20])
Vizion.fit_line_contour_xld (Diag, Line, Row1, Col1, Row2, Col2)
Vizion.gen_contour_polygon_xld (Box, [10, 10, 40, 40, 10], [10, 40, 40, 10, 10])
Vizion.gen_region_contour_xld (Box, Region, 'filled')
Vizion.area_center (Region, Area, CenterRow, CenterCol)
Vizion.segment_contours_xld (Box, Sides, 'lines', 1)
Vizion.count_obj (Sides, SideCount)
Vizion.select_obj (Sides, FirstSide, 1)
Vizion.get_contour_xld (FirstSide, SideRow, SideCol)
Vizion.select_contours_xld (Sides, LongSides, 'contlength', 20, 40)
Vizion.segment_contours_xld (Box, Corners, 'angle', 0.5)
Vizion.gen_contour_polygon_xld (Arc, [60, 100, 140, 100], [100, 140, 100, 60])
Vizion.fit_circle_contour_xld (Arc, Circle, CRow, CCol, Radius)
Vizion.fit_rectangle2_contour_xld (Box, Rect, RRow, RCol, Phi, Len1, Len2)
Vizion.close_contours_xld (Diag, Closed)
Vizion.smooth_contours_xld (Box, SmoothBox, 5)
Vizion.length_xld (Diag, DiagLength)
""")
    assert session.error is None, session.error
    ends = {(round(session.vars["Row1"][0], 1), round(session.vars["Col1"][0], 1)), (round(session.vars["Row2"][0], 1), round(session.vars["Col2"][0], 1))}
    assert ends == {(0.0, 0.0), (20.0, 20.0)}
    assert abs(session.vars["DiagLength"][0] - (20 * 2 ** 0.5)) < 0.2
    assert 800 < session.vars["Area"][0] < 1100
    assert session.vars["SideCount"] == 4
    assert len(session.vars["SideRow"]) == 2
    assert len(session.vars["LongSides"].contours) == 4
    assert len(session.vars["Corners"].contours) == 4
    assert abs(session.vars["CRow"][0] - 100) < 1
    assert abs(session.vars["CCol"][0] - 100) < 1
    assert abs(session.vars["Radius"][0] - 40) < 1
    assert abs(session.vars["Len1"][0] - 15) < 1.5
    assert session.vars["Closed"].contours[0][0] == session.vars["Closed"].contours[0][-1]
    assert len(session.vars["SmoothBox"].contours) == 1

    from app.engine.objects import XLD
    from app.engine.operators import union_adjacent_contours_xld, union_collinear_contours_xld

    gap = XLD(80, 40, [[[0, 0], [10, 0]], [[14, 0], [24, 0]]])
    joined = union_adjacent_contours_xld(gap, 8, 0.5, ctx=None).iconic[0]
    assert len(joined.contours) == 1
    assert abs(joined.contours[0][0][0] - 0) < 0.1
    assert abs(joined.contours[0][-1][0] - 24) < 0.1
    bent = XLD(40, 40, [[[0, 0], [10, 0]], [[12, 0], [12, 10]]])
    stayed = union_collinear_contours_xld(bent, 8, 0.3, ctx=None).iconic[0]
    assert len(stayed.contours) == 2


def test_ocr_catalog():
    from app.engine.objects import Image
    from app.engine.ocr import catalog, method_choices, normalize_method, read_text

    info = catalog()
    keys = [item.key for item in info.engines]
    assert keys[0] == "auto"
    assert len(keys) == len(set(keys))
    for name in (
        "rapid", "paddle", "easyocr", "windows", "tesseract",
        "doctr", "surya", "cnocr", "trocr", "got", "manga", "ddddocr", "kerasocr", "mmocr",
        "azure", "google", "textract", "baidu", "tencent", "aliyun", "huawei", "xfyun", "youdao",
        "mathpix", "ocrspace", "abbyy",
    ):
        assert name in keys, name
    assert info.best() == "rapid"
    assert all(item.purpose for item in info.engines)
    assert normalize_method("PP-OCR") == "rapid"
    assert normalize_method("iflytek") == "xfyun"
    choices = method_choices()
    assert choices[0] == "'auto'" and "'azure'" in choices and "'paddle'" in choices
    image = Image(np.zeros((16, 16, 3), np.uint8))
    try:
        read_text(image, "azure")
        raise AssertionError("沒有金鑰時不該連到雲端")
    except RuntimeError as exc:
        assert "還沒裝好" in str(exc)


def test_ocr_page():
    image = np.full((160, 640, 3), 255, np.uint8)
    cv2.putText(image, "VISDEV OCR 168", (24, 96), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)
    path = ROOT / "workspace" / "_ocr_page.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)
    session = run_source("Vizion.read_image (Image, 'workspace/_ocr_page.png')\nVizion.ocr_find (Image, Words, 'auto', Text, Row, Column, Score)\n")
    assert session.error is None, session.error
    found = " ".join(str(item) for item in session.vars["Text"])
    assert "VISDEV" in found and "168" in found, found
    assert len(session.vars["Words"].contours) == len(session.vars["Text"])
    outside = run_source("Vizion.read_image (Image, 'workspace/_ocr_page.png')\nVizion.ocr_find (Image, Words, 'auto', 0, 0, 20, 20, Text, Row, Column, Score)\n")
    path.unlink(missing_ok=True)
    assert outside.error is None, outside.error
    missed = " ".join(str(item) for item in outside.vars["Text"])
    assert "VISDEV" not in missed and "168" not in missed, missed


def test_gauge():
    from app.engine.gauge import measure_point
    from app.engine.objects import XLD
    from app.engine.samples_gen import GAUGE_PROGRAM, write_gauge

    gray = np.full((30, 180), 20, np.uint8)
    gray[:, 100:] = 200
    hits = measure_point(gray, None, 15, 40, 15, 160, 1, 8, "all", "best")
    assert len(hits) == 1
    assert abs(hits[0].column - 99.5) < 0.45, hits[0]
    try:
        measure_point(gray, None, 15, 40, 15, 160, 1, 8, "negative", "best")
        raise AssertionError("由亮到暗不該出現在這段剖面上")
    except ValueError as exc:
        assert "找不到邊緣" in str(exc)

    write_gauge(ROOT / "samples" / "gauge.png")
    session = run_source(GAUGE_PROGRAM)
    assert session.error is None, session.error
    problems = []

    def num(value):
        if isinstance(value, (list, tuple)):
            assert len(value) == 1, value
            value = value[0]
        return float(value)

    def near(name, value, expected, tol):
        actual = num(value)
        if abs(actual - expected) > tol:
            problems.append(f"{name}: {actual}，預期 {expected} ± {tol}")

    near("點的行", session.vars["PCol"], 107.5, 0.6)
    near("圓心列", session.vars["CRow"], 110, 0.8)
    near("圓心行", session.vars["CCol"], 150, 0.8)
    near("半徑", session.vars["CRad"], 42.25, 0.8)
    near("直線列1", session.vars["LRow1"], 199.5, 0.6)
    near("直線列2", session.vars["LRow2"], 199.5, 0.6)
    near("直線角", session.vars["LAngle"], 0, 0.04)
    near("矩形列", session.vars["RRow"], 244.5, 0.8)
    near("矩形行", session.vars["RCol"], 429.5, 0.8)
    near("矩形角", session.vars["RPhi"], 0, 0.04)
    near("矩形半長", session.vars["RL1"], 70, 0.8)
    near("矩形半寬", session.vars["RL2"], 45, 0.8)
    near("扇形列", session.vars["WRow"], 340, 1.5)
    near("扇形行", session.vars["WCol"], 170, 1.5)
    near("內半徑", session.vars["WR1"], 24, 1.2)
    near("外半徑", session.vars["WR2"], 58, 1.2)
    near("扇形起角", session.vars["WStart"], 0.35, 0.08)
    near("扇形張角", session.vars["WExtent"], 1.8, 0.1)
    rows = [float(item) for item in session.vars["GRows"]]
    cols = [float(item) for item in session.vars["GCols"]]
    expect = [(199.5, 359.5), (199.5, 499.5), (289.5, 499.5), (289.5, 359.5)]
    if len(rows) != 4 or len(cols) != 4:
        problems.append(f"多邊形頂點數 {len(rows)}")
    else:
        for index, (er, ec) in enumerate(expect):
            if abs(rows[index] - er) > 1.2 or abs(cols[index] - ec) > 1.2:
                problems.append(f"頂點{index + 1}: ({rows[index]}, {cols[index]})，預期 ({er}, {ec})")
    near("兩點距離", session.vars["Dist"], 5, 1e-6)
    near("點線距離", session.vars["DistLine"], 10, 1e-6)
    near("點圓距離", session.vars["DistCircle"], 0, 1e-6)
    near("直角", session.vars["RightAngle"], np.pi / 2, 1e-6)
    near("交點列", session.vars["IRow"], 20, 1e-6)
    near("交點行", session.vars["ICol"], 10, 1e-6)
    for name in ("Mark", "Line", "Circle", "Rect", "Wedge", "Poly"):
        shape = session.vars[name]
        if not isinstance(shape, XLD) or not shape.contours:
            problems.append(f"{name} 沒有輪廓")
    assert session.vars["CUsed"] >= 12
    assert session.vars["LUsed"] >= 8
    assert not problems, "；".join(problems)
    short = run_source(
        "Vizion.read_image (Image, 'samples/gauge.png')\n"
        "Vizion.gauge_point (Image, Mark2, 110, 80, 110, 220, PRow2, PCol2, Amp2, Dist2)\n"
    )
    assert short.error is None, short.error
    picked = float(short.vars["PCol2"][0] if isinstance(short.vars["PCol2"], list) else short.vars["PCol2"])
    assert abs(picked - 107.5) < 0.6 or abs(picked - 192.5) < 0.6, picked
    parallel = run_source("Vizion.intersection_ll (0, 0, 0, 10, 2, 0, 2, 10, Row, Column)\n")
    assert parallel.error and "平行" in parallel.error


def test_breakpoint_added_during_run_stops_later():
    import threading
    import time

    session = Session("for Index := 1 to 100000 by 1\nK := Index\nendfor\n", ROOT)

    def add_break():
        while session.step_count < 3 and not session.done and session.paused != "breakpoint":
            time.sleep(0.002)
        with session.lock:
            session.breakpoints.add(2)

    threading.Thread(target=add_break, daemon=True).start()
    session.run()
    assert session.paused == "breakpoint"
    assert session.current_line() == 2
    assert session.vars["K"] >= 1
    assert session.step_count < 100000


def test_breakpoint_stops_and_times_lines():
    source = "a := 1\n* note\nb := 2\nc := a + b\n"
    session = Session(source, ROOT)
    session.breakpoints = {3}
    session.run()
    assert session.paused == "breakpoint"
    assert session.current_line() == 3
    assert session.vars["a"] == 1
    assert "b" not in session.vars
    first = {item["line"]: item for item in session.public_state()["lineTimes"]}
    assert first[1]["count"] == 1
    assert first[1]["ms"] >= 0
    assert 3 not in first
    session.run()
    assert session.paused is None
    assert session.done
    assert session.vars["c"] == 3
    assert session.current_line() == 4
    timed = {item["line"]: item for item in session.public_state()["lineTimes"]}
    assert timed[3]["count"] == 1
    assert timed[4]["count"] == 1
    assert 2 not in timed
    commented = Session(source, ROOT)
    commented.breakpoints = {2}
    commented.run()
    assert commented.paused == "breakpoint"
    assert commented.current_line() == 3
    assert "b" not in commented.vars
    plain = Session("a := 1\nb := 2\n", ROOT)
    plain.run()
    assert plain.done
    assert plain.current_line() == 2
    loop = Session("for i := 1 to 4 by 1\nk := i\nendfor\n", ROOT)
    loop.run()
    counts = {item["line"]: item["count"] for item in loop.public_state()["lineTimes"]}
    assert counts[2] == 4
    plain.reset()
    assert plain.public_state()["lineTimes"] == []
    assert plain.current_line() == 1


def test_file_roundtrip_and_reset_closes_handle():
    import shutil

    folder = ROOT / "workspace" / "_file_ops"
    shutil.rmtree(folder, ignore_errors=True)
    folder.mkdir(parents=True)
    held = None
    try:
        session = run_source(
            "Vizion.make_dir ('workspace/_file_ops/nested')\n"
            "Vizion.open_file ('workspace/_file_ops/note.txt', 'output', Handle)\n"
            "Vizion.fwrite_string (Handle, 'VISDEV line')\n"
            "Vizion.fnew_line (Handle)\n"
            "Vizion.fwrite_string (Handle, 'second')\n"
            "Vizion.close_file (Handle)\n"
            "Vizion.open_file ('workspace/_file_ops/note.txt', 'input', Handle)\n"
            "Vizion.fread_line (Handle, Line, Eof1)\n"
            "Vizion.fread_string (Handle, Word, Eof2)\n"
            "Vizion.close_file (Handle)\n"
            "Vizion.copy_file ('workspace/_file_ops/note.txt', 'workspace/_file_ops/nested/copy.txt')\n"
            "Vizion.file_exists ('workspace/_file_ops/nested/copy.txt', Exists)\n"
            "Vizion.list_files ('workspace/_file_ops', ['files', 'directories', 'recursive'], Listed)\n"
            "Vizion.delete_file ('workspace/_file_ops/nested/copy.txt')\n"
            "Vizion.remove_dir ('workspace/_file_ops/nested')\n"
        )
        assert session.error is None, session.error
        assert session.vars["Line"] == "VISDEV line"
        assert session.vars["Eof1"] == 0
        assert session.vars["Word"] == "second"
        assert session.vars["Exists"] == 1
        listed = session.vars["Listed"]
        assert any(str(item).endswith("note.txt") for item in listed)
        assert any(str(item).endswith("nested") for item in listed)
        assert not (folder / "nested").exists()

        held = run_source("Vizion.open_file ('workspace/_file_ops/note.txt', 'input', Handle)\n")
        assert held.error is None, held.error
        handle = held.vars["Handle"]
        assert handle.closed is False
        held.reset()
        assert handle.closed is True
        (folder / "note.txt").unlink()
    finally:
        if held is not None:
            held.reset()
        shutil.rmtree(folder, ignore_errors=True)


def test_socket_available_bytes():
    import socket

    from app.engine.chapters.socket_ops import SocketLink, get_socket_param

    left, right = socket.socketpair()
    link = SocketLink(right, "pair")
    try:
        left.sendall(b"VISDEV")
        pending = get_socket_param(link, "available", ctx=None)
        assert pending.control[0] == 6
        assert right.recv(6) == b"VISDEV"
        assert get_socket_param(link, "available", ctx=None).control[0] == 0
    finally:
        link.release()
        left.close()


def test_relative_path_stays_inside_and_wait_does_not_block():
    import threading
    import time

    from app.engine.chapters.file_ops import wait_seconds

    ensure_samples(ROOT)
    escaped = run_source("Vizion.file_exists ('../outside.txt', Exists)\n")
    assert escaped.error and "離開" in escaped.error
    nested = run_source("Vizion.file_exists ('samples/../samples/parts.png', Exists)\n")
    assert nested.error is None, nested.error
    assert nested.vars["Exists"] == 1
    outside = run_source(f"Vizion.file_exists ('{ROOT}', Exists)\n")
    assert outside.error is None, outside.error
    assert outside.vars["Exists"] == 1

    session = Session("Vizion.wait_seconds (2)\nA := 1\n", ROOT)
    session.defer_wait = True
    started = time.perf_counter()
    session.run()
    assert time.perf_counter() - started < 0.3
    assert session.paused == "wait"
    assert "A" not in session.vars
    session.wait_until = time.monotonic()
    session.run(resume=True)
    assert session.error is None, session.error
    assert session.vars["A"] == 1
    assert session.done

    class Ctx:
        cancel = False
        proc_depth = 0
        defer_wait = False

    ctx = Ctx()

    def flip():
        time.sleep(0.12)
        ctx.cancel = True

    threading.Thread(target=flip, daemon=True).start()
    started = time.perf_counter()
    wait_seconds(5, ctx=ctx)
    assert time.perf_counter() - started < 1.0


def test_step_lands_inside_the_procedure():
    source = "\n".join([
        "procedure add (: : Value : Result)",
        "    Result := Value + 1",
        "    Result := Result + 1",
        "endprocedure",
        "add (4, Out)",
        "After := Out",
    ])
    session = Session(source, ROOT)
    session.step()
    assert session.current_line() == 5
    session.step()
    assert session.current_line() == 2
    assert session.public_state()["procedure"] == "add"
    assert session.vars["Value"] == 4
    assert "Out" not in session.vars
    session.step()
    assert session.current_line() == 3
    assert session.vars["Result"] == 5
    session.step()
    assert session.current_line() == 6
    assert session.public_state()["procedure"] is None
    assert session.vars["Out"] == 6
    assert "Value" not in session.vars
    session.step()
    assert session.done
    assert session.vars["After"] == 6

    nested = "\n".join([
        "procedure outer (: : Value : Result)",
        "    inner (Value, Mid)",
        "    Result := Mid + 3",
        "endprocedure",
        "procedure inner (: : Value : Result)",
        "    Result := Value + 2",
        "endprocedure",
        "outer (4, Out)",
    ])
    entered = Session(nested, ROOT)
    entered.step()
    entered.step()
    assert entered.current_line() == 8
    entered.step()
    assert entered.current_line() == 2
    assert entered.public_state()["procedure"] == "outer"
    entered.step()
    assert entered.current_line() == 6
    assert entered.public_state()["procedure"] == "inner"
    assert "Out" not in entered.vars
    entered.run()
    assert entered.error is None, entered.error
    assert entered.vars["Out"] == 9
    assert entered.done

    whole = run_source(nested)
    assert whole.error is None, whole.error
    assert whole.vars["Out"] == 9


def test_run_stops_inside_a_procedure_breakpoint():
    source = "\n".join([
        "procedure add (: : Value : Result)",
        "    Result := Value + 1",
        "    Result := Result + 1",
        "endprocedure",
        "add (4, Out)",
        "After := Out",
    ])
    session = Session(source, ROOT)
    session.breakpoints = {3}
    session.run()
    assert session.paused == "breakpoint"
    assert session.current_line() == 3
    assert session.public_state()["procedure"] == "add"
    assert session.vars["Result"] == 5
    assert "Out" not in session.vars
    session.run()
    assert session.error is None, session.error
    assert session.vars["Out"] == 6
    assert session.vars["After"] == 6
    assert session.done


def test_trace_names_each_line_inside_a_procedure():
    source = "\n".join([
        "procedure add (: : Value : Result)",
        "    Result := Value + 1",
        "    Result := Result + 1",
        "endprocedure",
        "Ready := 3",
        "add (4, Out)",
    ])
    session = Session(source, ROOT)
    session.trace = True
    session.run()
    assert session.error is None, session.error
    assert session.vars["Out"] == 6
    body = [line for line in session.log if "add:" in line]
    assert any(line.endswith("add: Result := 5") for line in body)
    assert any(line.endswith("add: Result := 6") for line in body)
    assert any(line.endswith("add → Out := 6") for line in session.log)
    assert any(line.endswith("Ready := 3") and "add:" not in line for line in session.log)
    nested = "\n".join([
        "procedure inner (: : Value : Result)",
        "    Result := Value + 2",
        "endprocedure",
        "procedure outer (: : Value : Result)",
        "    inner (Value, Mid)",
        "    Result := Mid + 3",
        "endprocedure",
        "outer (4, Out)",
    ])
    deep = Session(nested, ROOT)
    deep.trace = True
    deep.run()
    assert deep.vars["Out"] == 9
    assert any(line.endswith("inner: Result := 6") for line in deep.log)
    assert any(line.endswith("outer: inner → Mid := 6") for line in deep.log)
    assert any(line.endswith("outer: Result := 9") for line in deep.log)
    stepped = Session(source, ROOT)
    stepped.trace = True
    stepped.step()
    stepped.step()
    stepped.step()
    assert stepped.public_state()["procedure"] == "add"
    stepped.step()
    assert any(line.endswith("add: Result := 5") for line in stepped.log)


def test_trace_shows_recursive_procedure_depth():
    source = "\n".join([
        "procedure down (: : N : Last)",
        "    if (N > 0)",
        "        down (N - 1, Last)",
        "    else",
        "        Last := 7",
        "    endif",
        "endprocedure",
        "down (2, Out)",
    ])
    session = Session(source, ROOT)
    session.trace = True
    session.run()
    assert session.error is None, session.error
    assert session.vars["Out"] == 7
    assert any(line.endswith("down×3: Last := 7") for line in session.log)
    assert any(line.endswith("down×2: down → Last := 7") for line in session.log)
    assert not any(line.endswith("down: Last := 7") for line in session.log)


def test_trace_log_sends_only_new_lines():
    from app.engine.session import omit_known_state

    session = Session("Ready := 1\nAfter := Ready + 1\n", ROOT)
    session.trace = True
    session.step()
    first = session.public_state()
    session.step()
    second = session.public_state()
    assert second["logRev"] != first["logRev"]
    assert second["logSize"] == first["logSize"] + 1
    tail = omit_known_state(second, None, None, first["logRev"], first["logSize"])
    assert tail["logFrom"] == first["logSize"]
    assert tail["log"] == [second["log"][-1]]
    same = omit_known_state(second, None, None, second["logRev"], second["logSize"])
    assert same["log"] is None and same["logKept"] is True


def test_wait_inside_procedure_resumes():
    import time

    source = """
procedure pause (: : : Result)
    Seen := 3
    Vizion.wait_seconds (2)
    Result := Seen + 1
endprocedure
pause (Out)
Outer := 2
"""
    session = Session(source, ROOT)
    session.defer_wait = True
    started = time.perf_counter()
    session.run()
    assert time.perf_counter() - started < 0.3
    assert session.paused == "wait"
    assert "Out" not in session.vars
    session.wait_until = time.monotonic()
    session.run(resume=True)
    assert session.error is None, session.error
    assert session.vars["Out"] == 4
    assert session.vars["Outer"] == 2
    assert session.done


def test_wait_inside_nested_procedure_resumes():
    import time

    source = """
procedure inner (: : : Result)
    Seen := 3
    Vizion.wait_seconds (2)
    Result := Seen + 1
endprocedure
procedure outer (: : : Out)
    inner (Mid)
    Out := Mid + 5
endprocedure
outer (Value)
After := 7
"""
    session = Session(source, ROOT)
    session.defer_wait = True
    started = time.perf_counter()
    session.run()
    assert time.perf_counter() - started < 0.3
    assert session.paused == "wait"
    assert "Value" not in session.vars
    session.wait_until = time.monotonic()
    session.run(resume=True)
    assert session.error is None, session.error
    assert session.vars["Value"] == 9
    assert session.vars["After"] == 7
    assert session.done


def test_directshow_import_does_not_require_windows():
    import sys

    from app.engine.dshow import grab_bgr

    if sys.platform == "win32":
        return
    try:
        grab_bgr(0)
    except RuntimeError as exc:
        assert "打不開相機" in str(exc)
    else:
        raise AssertionError("沒有 DirectShow 時應該失敗")


def test_parse_error():
    try:
        parse_program("not_an_operator (A, B)\n")
    except ParseError as exc:
        assert exc.line == 1
    else:
        raise AssertionError("應該要拒絕未知算子")
    try:
        parse_program("gen_rectangle1 (A, 0, 0, 1, 1)\n")
    except ParseError as exc:
        assert "Vizion.gen_rectangle1" in str(exc)
    else:
        raise AssertionError("沒有前綴的算子應該被拒絕")
    parse_program("Vizion.gen_rectangle1 (A, 0, 0, 1, 1)\n")


def test_score_shift_matches_and_cancel_stops():
    import threading
    import time

    import numpy as np

    from app.engine.shape_model import _add_shifted, _shift_to_origin

    source = np.arange(12, dtype=np.float32).reshape(3, 4)
    for row_offset, col_offset in ((0, 0), (1, -1), (-2, 2), (5, 0)):
        folded = np.zeros_like(source)
        _add_shifted(folded, source, row_offset, col_offset)
        assert np.array_equal(folded, _shift_to_origin(source, row_offset, col_offset))

    session = Session(
        "for Index := 1 to 2000000 by 1\nendfor\n",
        ROOT,
    )

    def stop_soon():
        while session.step_count < 30:
            time.sleep(0.002)
        session.cancel = True

    threading.Thread(target=stop_soon, daemon=True).start()
    session.run()
    assert session.paused == "stopped"
    assert session.step_count < 2000000


def test_pick_area_and_center_use_existing_measures():
    import numpy as np

    from app.engine.operators import region_measures

    mask = np.zeros((10, 20), np.uint8)
    mask[2:6, 4:10] = 255
    values = region_measures(mask)
    assert values["area"] == 24
    assert abs(values["row"] - 3.5) < 1e-6
    assert abs(values["column"] - 6.5) < 1e-6


def test_unchanged_display_is_not_reencoded():
    from app.engine import render
    from app.engine.session import omit_known_state

    session = run_source(
        "Vizion.gen_image_const (Image, 'byte', 64, 48)\n"
        "Vizion.threshold (Image, Region, 0, 255)\n"
    )
    assert session.error is None, session.error
    seen = {"n": 0}
    original = render.png_data_url

    def counting(array):
        seen["n"] += 1
        return original(array)

    render.png_data_url = counting
    try:
        first = session.public_state()
        encoded = seen["n"]
        assert encoded >= 2
        assert first["display"]["image"]
        second = session.public_state()
        assert seen["n"] == encoded
        assert second["display"] is first["display"]
        assert second["displayRev"] == first["displayRev"]
        kept = omit_known_state(second, second["displayRev"], second["varRev"], second["logRev"])
        assert kept["display"] is None and kept["displayKept"] is True
        assert kept["iconic"] is None and kept["iconicKept"] is True
        assert kept["control"] is None and kept["controlKept"] is True
        assert kept["log"] is None and kept["logKept"] is True
        assert second["display"]["image"] == first["display"]["image"]
        assert second["iconic"][0]["name"]
        assert omit_known_state(second, None, None, None)["display"] is first["display"]
        session.vars["Mark"] = 7
        marked = session.public_state()
        assert marked["varRev"] != second["varRev"]
        sent = omit_known_state(marked, marked["displayRev"], second["varRev"], marked["logRev"])
        assert sent.get("controlKept") is not True
        assert any(item["name"] == "Mark" for item in sent["control"])
        session.display.overlays.append(
            {"kind": "text", "text": "ok", "row": 4, "col": 4, "color": "#ffffff"}
        )
        third = session.public_state()
        assert seen["n"] > encoded
        assert third["displayRev"] != first["displayRev"]
        assert third["display"]["image"] == first["display"]["image"]
        assert any(shape.get("text") == "ok" for shape in third["display"]["shapes"])
        session.region_color = "#3d8bff"
        colored = session.public_state()
        assert colored["displayRev"] != third["displayRev"]
        assert colored["display"] is not third["display"]
    finally:
        render.png_data_url = original


def test_unchanged_variable_thumbs_are_not_reencoded():
    from app.engine import session as session_mod

    session = run_source("Vizion.gen_image_const (Image, 'byte', 80, 60)\n")
    assert session.error is None, session.error
    seen = {"n": 0}
    original = session_mod.jpeg_thumb

    def counting(array, max_edge=112):
        seen["n"] += 1
        return original(array, max_edge)

    session_mod.jpeg_thumb = counting
    try:
        first = session.public_state()
        assert seen["n"] >= 1
        encoded = seen["n"]
        thumb = first["iconic"][0]["thumb"]
        assert thumb.startswith("data:image/jpeg")
        second = session.public_state()
        assert seen["n"] == encoded
        assert second["iconic"][0]["thumb"] == thumb
        assert second["iconic"][0]["info"] == first["iconic"][0]["info"]
    finally:
        session_mod.jpeg_thumb = original


def test_display_updates_before_the_program_finishes():
    import threading
    import time

    session = Session(
        "Vizion.gen_image_const (Image, 'byte', 8, 6)\nVizion.dev_display (Image)\nfor Index := 1 to 100000 by 1\nendfor\n",
        ROOT,
    )

    def watch():
        while session.display_stamp == 0 and session.paused != "stopped" and not session.done:
            time.sleep(0.002)
        if session.display_stamp:
            session.cancel = True

    threading.Thread(target=watch, daemon=True).start()
    session.run()
    assert session.display_stamp >= 1
    assert session.paused == "stopped"


def test_csharp_export_matches_python_threshold_and_connection():
    import base64
    import hashlib
    import json
    import subprocess

    from app.engine.export_cs import ExportError, export_csharp
    from app.engine.objects import Image
    from app.engine.operators import _foreground_runs, area_center, connection, halcon_components, threshold
    from app.engine.render import read_cv_image, to_uint8

    ensure_samples(ROOT)
    sample = (ROOT / "samples" / "export_parts.viz").read_bytes()
    assert sample.count(b"\r\n") >= 5
    assert sample.count(b"\n") == sample.count(b"\r\n")
    source = sample.decode("utf-8")
    session = run_source(source)
    assert session.error is None, session.error
    assert session.vars["Parts"].count() >= 3
    code = export_csharp(source)
    assert 'Operators.ReadImage("samples/parts.png")' in code
    assert "Operators.Threshold(Image, 60, 255)" in code
    assert "Operators.Connection(Region)" in code
    assert "Operators.Overlays(Parts)" in code
    assert "view(Image.Width, Image.Height, Image.ForDisplay(), Image.Layout," in code
    assert "ExportedProgram.Run(window.Show)" in code
    counted = export_csharp(
        "Vizion.read_image (Image, 'samples/parts.png')\n"
        "Vizion.threshold (Image, Region, 60, 255)\n"
        "Vizion.connection (Region, Parts)\n"
        "Vizion.count_obj (Parts, Number)\n"
        "Vizion.area_center (Parts, Area, Row, Column)\n"
    )
    assert "Operators.Count(Parts)" in counted
    assert "Operators.AreaCenterOf(Parts)" in counted
    assert "Area[exportIndex] = exportMeasured1[exportIndex].Area" in counted
    from fastapi import HTTPException

    from app.main import ExportCsIn, export_csharp_program

    payload = export_csharp_program(ExportCsIn(source=source, name="export_parts.viz"))
    assert payload["name"] == "export_parts.cs"
    assert "export_parts.Run(window.Show)" in payload["source"]
    try:
        export_csharp_program(
            ExportCsIn(
                source="Vizion.read_image (Image, 'samples/parts.png')\nVizion.write_image (Image, 'workspace/out.png')\n",
                name="out",
            )
        )
    except HTTPException as exc:
        assert exc.status_code == 400
        assert "write_image" in str(exc.detail)
    else:
        raise AssertionError("寫檔不該匯出")
    page = (ROOT / "app" / "static" / "index.html").read_text(encoding="utf-8")
    script = (ROOT / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'data-cmd="exportCsharp"' in page
    assert "/api/export/csharp" in script
    try:
        export_csharp("Vizion.read_image (Image, 'samples/parts.png')\nVizion.write_image (Image, 'workspace/out.png')\n")
    except ExportError as exc:
        assert "write_image" in str(exc)
    else:
        raise AssertionError("不該匯出寫檔")
    far = export_csharp(
        "Vizion.read_image (Image, 'samples/parts.png')\n"
        "Vizion.threshold (Image, Region, 60, 255)\n"
        "Vizion.connection (Region, Parts, 5)\n"
        "Vizion.select_shape (Parts, Big, 'area', 'and', 80, 99999)\n"
        "Vizion.dilation_rectangle1 (Big, Grown, 3, 3)\n"
        "Vizion.get_image_size (Image, Width, Height)\n"
        "Vizion.get_grayval (Image, 10, 20, Grayval)\n"
    )
    assert "Operators.Connection(Region, 5)" in far
    assert 'SelectShape(Parts, new[] { "area" }, "and"' in far
    assert "Operators.DilateRectangle(Big, 3, 3)" in far
    assert "Image.Width" in far and "Image.Height" in far
    assert "Operators.GrayAt(Image, 10, 20)" in far
    distance_bytes = (ROOT / "samples" / "export_distance.viz").read_bytes()
    assert distance_bytes.count(b"\r\n") >= 8
    assert distance_bytes.count(b"\n") == distance_bytes.count(b"\r\n")
    distance_program = distance_bytes.decode("utf-8")
    distance_session = run_source(distance_program)
    assert distance_session.error is None, distance_session.error
    assert distance_session.vars["Number"] >= 1
    assert "Operators.DilateRectangle(Big, 3, 3)" in export_csharp(distance_program)
    circle_program = """
Vizion.read_image (Image, 'samples/parts.png')
Vizion.threshold (Image, Region, 60, 255)
Vizion.opening_circle (Region, Opened, 2.5)
Vizion.dilation_circle (Opened, Grown, 1)
Vizion.boundary (Grown, Edge, 'outer')
Vizion.connection (Opened, Parts)
"""
    circle_code = export_csharp(circle_program)
    assert "Operators.OpenCircle(Region, 2.5)" in circle_code
    assert "Operators.DilateCircle(Opened, 1.0)" in circle_code
    assert 'Operators.Boundary(Grown, "outer")' in circle_code
    domain_program = """
Vizion.read_image (Image, 'samples/parts.png')
Vizion.gen_rectangle1 (Window, 40, 80, 200, 300)
Vizion.reduce_domain (Image, Window, Limited)
Vizion.threshold (Limited, Region, 60, 255)
Vizion.get_domain (Limited, Domain)
Vizion.change_domain (Image, Window, Replaced)
Vizion.crop_rectangle1 (Limited, Part, 40, 80, 80, 120)
"""
    domain_code = export_csharp(domain_program)
    assert "Operators.ReduceDomain(Image, Window)" in domain_code
    assert "Operators.Threshold(Limited, 60, 255)" in domain_code
    assert "Operators.DomainOf(Limited)" in domain_code
    assert "Operators.ChangeDomain(Image, Window)" in domain_code
    assert "Operators.Crop(Limited, 40, 80, 80, 120)" in domain_code
    assert "ForDisplay()" in domain_code

    def digest(masks):
        hasher = hashlib.sha256()
        hasher.update(str(len(masks)).encode())
        for mask in masks:
            hasher.update(np.ascontiguousarray(mask).tobytes())
        return hasher.hexdigest()

    rng = np.random.default_rng(0)
    masks = []
    for shape in ((3, 3), (7, 12), (64, 80), (1, 20), (20, 1), (30, 30)):
        masks.append((rng.random(shape) > 0.7).astype(np.uint8) * 255)
    diagonal = np.zeros((3, 3), np.uint8)
    diagonal[0, 1] = 255
    diagonal[1, 0] = 255
    masks.append(diagonal)
    dots = np.zeros((7, 12), np.uint8)
    dots[3, 1] = dots[3, 4] = dots[3, 10] = 255
    masks.append(dots)
    class Ctx:
        histogram = None

    picture = read_cv_image(ROOT / "samples" / "parts.png")
    masks.append(threshold(Image(picture), 60, 255, ctx=Ctx()).iconic[0].union_mask())
    expected = [
        "a4cbd7c11acafefff07566b6fbb6d83aac6d0c586a93440747b7f38344856115",
        "ba865d9ea48e10a4150efc567f294e3b24770c0c949862874f61f3325ce7ce41",
        "e50d3bfef884d992e2e157b74d382d8ec495e7a143a9b23dbc3862dabb76c268",
        "b9b9267090f80c2d8bcd8c0465b4b6260f7a9c6f5034b778f707409db1cf0e94",
        "8a1bd6423831d4121cb23bd291988e9aba22133147dbf0134b389458dc9a2eab",
        "9a33e84004d5374109ea1d5d02f63909f51210f00f57d678e0b7e0525c681611",
        "bb0434c93d431b1b3848ebc75e2377f3ffe5eb62ca8a8616dd6fa6e75a9b9806",
        "21c98421d5e9a66451bece8647cb2cb56283aca3ea3cfd32ab463321c65f7ad8",
        "30de60404ae5783c63cf752f0da867e5bfc49c0b5f5871a55624d5ab80905369",
    ]
    for mask, digest_hex in zip(masks, expected):
        assert digest(halcon_components(mask)) == digest_hex

    folder = ROOT / "workspace" / "csharp-check"
    folder.mkdir(parents=True, exist_ok=True)
    color = np.zeros((2, 3, 3), np.uint8)
    color[0, 0] = (10, 20, 30)
    color[1, 2] = (0, 0, 255)
    wide = np.array([[0, 257, 514, 65535]], np.uint16)
    gray = np.zeros((8, 8), np.uint8)
    gray[1:4, 2:6] = 200
    gray[6, 6] = 180
    paths = {
        "color": folder / "color.png",
        "wide": folder / "wide.png",
        "gray": folder / "gray.png",
        "parts": ROOT / "samples" / "parts.png",
    }
    cv2.imwrite(str(paths["color"]), color)
    cv2.imwrite(str(paths["wide"]), wide)
    cv2.imwrite(str(paths["gray"]), gray)

    def decoded(path):
        data = to_uint8(read_cv_image(path))
        if data.ndim == 3 and data.shape[2] == 4:
            data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
        return data

    def run_list(mask):
        found, _rows = _foreground_runs(mask)
        return [[row, start, end - start + 1] for row, start, end in found]

    cases = []
    for name, path, low, high in (
        ("color", paths["color"], 22, 22),
        ("wide", paths["wide"], 2, 255),
        ("gray", paths["gray"], 100, 255),
        ("parts", paths["parts"], 60, 255),
    ):
        image = decoded(path)
        region = threshold(Image(image), low, high, ctx=Ctx()).iconic[0]
        parts = connection(region, 1, ctx=Ctx()).iconic[0]
        areas, rows, cols = area_center(parts, ctx=Ctx()).control
        cases.append(
            {
                "path": str(path),
                "width": int(image.shape[1]),
                "height": int(image.shape[0]),
                "channels": 1 if image.ndim == 2 else int(image.shape[2]),
                "pixels": base64.b64encode(np.ascontiguousarray(image).tobytes()).decode("ascii"),
                "min": low,
                "max": high,
                "threshold": run_list(region.union_mask()),
                "components": [run_list(mask) for mask in parts.objects],
                "count": parts.count(),
                "centers": [[int(area), float(row), float(col)] for area, row, col in zip(areas, rows, cols)],
            }
        )
    from app.engine.objects import Region
    from app.engine.operators import (
        closing_rectangle1,
        dilation_rectangle1,
        erosion_rectangle1,
        intensity,
        opening_rectangle1,
        select_shape,
        union2,
        difference,
        gen_rectangle1,
        fill_up,
        complement,
        reduce_domain,
        crop_rectangle1,
        get_domain,
        dilation_circle,
        erosion_circle,
        opening_circle,
        closing_circle,
        boundary,
    )
    from app.engine.chapters.image_ops import change_domain, overpaint_region, paint_region
    from app.engine.render import apply_domain_dim

    class RectCtx(Ctx):
        ref_shape = (8, 10)

        def canvas_size(self):
            return self.ref_shape

    def objects_of(region):
        return [run_list(mask) for mask in region.objects]

    def one_runs(result):
        region = result.iconic[0]
        if region.count() == 0:
            return []
        return run_list(region.union_mask())

    ops = []
    dots = np.zeros((7, 12), np.uint8)
    dots[3, 1] = dots[3, 4] = dots[3, 10] = 255
    dot_region = Region(12, 7, [dots])
    for distance in (2.5, 3, 6):
        parts = connection(dot_region, distance, ctx=Ctx()).iconic[0]
        ops.append(
            {
                "op": "connection",
                "name": f"dots-{distance}",
                "width": 12,
                "height": 7,
                "distance": distance,
                "runs": run_list(dots),
                "components": objects_of(parts),
            }
        )
    rng = np.random.default_rng(7)
    blob = (rng.random((24, 28)) > 0.82).astype(np.uint8) * 255
    blob_region = Region(28, 24, [blob])
    parts = connection(blob_region, 4.5, ctx=Ctx()).iconic[0]
    ops.append(
        {
            "op": "connection",
            "name": "random-4.5",
            "width": 28,
            "height": 24,
            "distance": 4.5,
            "runs": run_list(blob),
            "components": objects_of(parts),
        }
    )
    block = np.zeros((12, 16), np.uint8)
    block[4:7, 5:9] = 255
    block_region = Region(16, 12, [block])
    for mode, fn in (
        ("dilate", dilation_rectangle1),
        ("erode", erosion_rectangle1),
        ("open", opening_rectangle1),
        ("close", closing_rectangle1),
    ):
        ops.append(
            {
                "op": "morph",
                "name": mode,
                "mode": mode,
                "width": 16,
                "height": 12,
                "kernelWidth": 4,
                "kernelHeight": 3,
                "runs": run_list(block),
                "out": one_runs(fn(block_region, 4, 3, ctx=Ctx())),
            }
        )
    for mode, fn, radius in (
        ("dilateCircle", dilation_circle, 1),
        ("erodeCircle", erosion_circle, 1.5),
        ("openCircle", opening_circle, 2.5),
        ("closeCircle", closing_circle, 3.5),
    ):
        ops.append(
            {
                "op": "morph",
                "name": mode,
                "mode": mode,
                "width": 16,
                "height": 12,
                "distance": radius,
                "runs": run_list(block),
                "out": one_runs(fn(block_region, radius, ctx=Ctx())),
            }
        )
    ops.append(
        {
            "op": "morph",
            "name": "inner",
            "mode": "inner",
            "width": 16,
            "height": 12,
            "runs": run_list(block),
            "out": one_runs(boundary(block_region, "inner", ctx=Ctx())),
        }
    )
    ops.append(
        {
            "op": "morph",
            "name": "outer",
            "mode": "outer",
            "width": 16,
            "height": 12,
            "runs": run_list(block),
            "out": one_runs(boundary(block_region, "outer", ctx=Ctx())),
        }
    )
    other = np.zeros_like(block)
    other[5:9, 7:12] = 255
    other_region = Region(16, 12, [other])
    ops.append(
        {
            "op": "bool",
            "name": "difference",
            "mode": "difference",
            "width": 16,
            "height": 12,
            "left": run_list(block),
            "right": run_list(other),
            "out": one_runs(difference(block_region, other_region, ctx=Ctx())),
        }
    )
    ops.append(
        {
            "op": "bool",
            "name": "union",
            "mode": "union",
            "width": 16,
            "height": 12,
            "left": run_list(block),
            "right": run_list(other),
            "out": one_runs(union2(block_region, other_region, ctx=Ctx())),
        }
    )
    hole = block.copy()
    hole[5, 6:8] = 0
    hole_region = Region(16, 12, [hole])
    ops.append(
        {
            "op": "morph",
            "name": "fill",
            "mode": "fill",
            "width": 16,
            "height": 12,
            "runs": run_list(hole),
            "out": one_runs(fill_up(hole_region, ctx=Ctx())),
        }
    )
    crack = np.array(
        [
            [255, 255, 255, 255, 255],
            [255, 255, 255, 255, 255],
            [255, 255, 0, 255, 255],
            [255, 0, 255, 255, 255],
            [0, 255, 255, 255, 255],
        ],
        np.uint8,
    )
    crack_region = Region(5, 5, [crack])
    ops.append(
        {
            "op": "morph",
            "name": "fill-diagonal",
            "mode": "fill",
            "width": 5,
            "height": 5,
            "runs": run_list(crack),
            "out": one_runs(fill_up(crack_region, ctx=Ctx())),
        }
    )
    ops.append(
        {
            "op": "morph",
            "name": "complement",
            "mode": "complement",
            "width": 16,
            "height": 12,
            "runs": run_list(block),
            "out": one_runs(complement(block_region, ctx=Ctx())),
        }
    )
    pieces = connection(block_region, 1, ctx=Ctx()).iconic[0]
    kept = select_shape(pieces, "area", "and", 1, 99999, ctx=Ctx()).iconic[0]
    ops.append(
        {
            "op": "shape",
            "name": "area",
            "width": 16,
            "height": 12,
            "feature": "area",
            "minimum": 1,
            "maximum": 99999,
            "objects": objects_of(pieces),
            "kept": list(range(kept.count())),
        }
    )
    tone_image = np.zeros((12, 16), np.uint8)
    tone_image[4:7, 5:9] = 40
    tone_image[4, 5] = 80
    means, deviations = intensity(pieces, Image(tone_image), ctx=Ctx()).control
    ops.append(
        {
            "op": "tone",
            "name": "block",
            "width": 16,
            "height": 12,
            "pixels": base64.b64encode(np.ascontiguousarray(tone_image).tobytes()).decode("ascii"),
            "objects": objects_of(pieces),
            "means": [float(item) for item in means],
            "deviations": [float(item) for item in deviations],
        }
    )
    ops.append(
        {
            "op": "bounds",
            "name": "block",
            "width": 16,
            "height": 12,
            "objects": objects_of(pieces),
            "rects": [[4, 5, 6, 8]],
        }
    )
    ops.append(
        {
            "op": "shape",
            "name": "area-too-small",
            "width": 16,
            "height": 12,
            "feature": "area",
            "minimum": 100,
            "maximum": 99999,
            "objects": objects_of(pieces),
            "kept": [],
        }
    )
    ops.append(
        {
            "op": "rect",
            "name": "clipped",
            "width": 10,
            "height": 8,
            "row1": 1,
            "column1": 2,
            "row2": 4,
            "column2": 6,
            "out": one_runs(gen_rectangle1(1, 2, 4, 6, ctx=RectCtx())),
        }
    )
    color_pixels = np.zeros((2, 3, 3), np.uint8)
    color_pixels[0, 0] = (10, 20, 30)
    ops.append(
        {
            "op": "gray",
            "name": "bgr",
            "width": 3,
            "height": 2,
            "channels": 3,
            "pixels": base64.b64encode(np.ascontiguousarray(color_pixels).tobytes()).decode("ascii"),
            "row": 0,
            "column": 0,
            "values": [30, 20, 10],
        }
    )
    field = np.full((6, 8), 200, np.uint8)
    field[0, 0] = 10
    window = np.zeros((6, 8), np.uint8)
    window[1:4, 2:6] = 255
    window_region = Region(8, 6, [window])
    limited = reduce_domain(Image(field), window_region, ctx=Ctx()).iconic[0]
    kept = threshold(limited, 100, 255, ctx=Ctx()).iconic[0]
    ops.append(
        {
            "op": "reduce",
            "name": "threshold-inside",
            "width": 8,
            "height": 6,
            "channels": 1,
            "pixels": base64.b64encode(np.ascontiguousarray(field).tobytes()).decode("ascii"),
            "runs": run_list(window),
            "min": 100,
            "max": 255,
            "out": [] if kept.count() == 0 else run_list(kept.union_mask()),
            "display": base64.b64encode(np.ascontiguousarray(apply_domain_dim(limited)).tobytes()).decode("ascii"),
        }
    )
    inner = np.zeros((6, 8), np.uint8)
    inner[2:3, 3:5] = 255
    limited_again = reduce_domain(limited, Region(8, 6, [inner]), ctx=Ctx()).iconic[0]
    kept_again = threshold(limited_again, 100, 255, ctx=Ctx()).iconic[0]
    ops.append(
        {
            "op": "reduce",
            "name": "threshold-intersect",
            "width": 8,
            "height": 6,
            "channels": 1,
            "pixels": base64.b64encode(np.ascontiguousarray(field).tobytes()).decode("ascii"),
            "runs": run_list(window),
            "second": run_list(inner),
            "min": 100,
            "max": 255,
            "out": [] if kept_again.count() == 0 else run_list(kept_again.union_mask()),
        }
    )
    replaced = change_domain(limited, Region(8, 6, [inner]), ctx=Ctx()).iconic[0]
    ops.append(
        {
            "op": "reduce",
            "name": "change",
            "mode": "change",
            "width": 8,
            "height": 6,
            "channels": 1,
            "pixels": base64.b64encode(np.ascontiguousarray(field).tobytes()).decode("ascii"),
            "runs": run_list(inner),
            "min": 100,
            "max": 255,
            "out": one_runs(threshold(replaced, 100, 255, ctx=Ctx())),
        }
    )
    cropped = crop_rectangle1(limited, 1, 2, 3, 5, ctx=Ctx()).iconic[0]
    ops.append(
        {
            "op": "crop",
            "name": "domain",
            "width": 8,
            "height": 6,
            "pixels": base64.b64encode(np.ascontiguousarray(field).tobytes()).decode("ascii"),
            "runs": run_list(window),
            "row1": 1,
            "column1": 2,
            "row2": 3,
            "column2": 5,
            "kernelWidth": cropped.width,
            "kernelHeight": cropped.height,
            "display": base64.b64encode(np.ascontiguousarray(cropped.data).tobytes()).decode("ascii"),
            "out": one_runs(get_domain(cropped, ctx=Ctx())),
        }
    )
    painted = overpaint_region(window_region, limited, 40, "fill", ctx=Ctx()).iconic[0]
    ops.append(
        {
            "op": "paint",
            "name": "over",
            "mode": "over",
            "feature": "fill",
            "gray": 40,
            "width": 8,
            "height": 6,
            "channels": 1,
            "pixels": base64.b64encode(np.ascontiguousarray(field).tobytes()).decode("ascii"),
            "runs": run_list(window),
            "left": run_list(window),
            "display": base64.b64encode(np.ascontiguousarray(painted.data).tobytes()).decode("ascii"),
        }
    )
    edged = paint_region(window_region, Image(field), 255, "margin", ctx=Ctx()).iconic[0]
    ops.append(
        {
            "op": "paint",
            "name": "margin",
            "mode": "fill",
            "feature": "margin",
            "gray": 255,
            "width": 8,
            "height": 6,
            "channels": 1,
            "pixels": base64.b64encode(np.ascontiguousarray(field).tobytes()).decode("ascii"),
            "left": run_list(window),
            "display": base64.b64encode(np.ascontiguousarray(edged.data).tobytes()).decode("ascii"),
        }
    )
    spec = folder / "cases.json"
    spec.write_text(json.dumps({"cases": cases, "ops": ops}), encoding="utf-8")
    check = ROOT / "dotnet" / "Vizion.Operators.Check" / "Vizion.Operators.Check.csproj"
    completed = subprocess.run(
        ["dotnet", "run", "-f", "net8.0", "--project", str(check), "--", "--compare", str(spec)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == "__main__":
    test_language()
    test_blob()
    test_other_samples()
    test_shape_match()
    test_ncc_locate()
    test_threshold_histogram_and_morphology()
    test_preview_mask_while_dragging()
    test_opencv_link_list()
    test_features_and_auto_threshold()
    test_threshold_sub_pix()
    test_binary_region_color()
    test_connection_distance()
    test_connection_halcon_and_opencv()
    test_move_program_arrow()
    test_xld_contour_tools()
    test_ocr_page()
    test_gauge()
    test_breakpoint_stops_and_times_lines()
    test_parse_error()
    print("all passed")
