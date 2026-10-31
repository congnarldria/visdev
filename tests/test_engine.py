import sys
from pathlib import Path

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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch1.viz").read_text(encoding="utf-8"))
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
    rounded = run_source("Vizion.tuple_round ([-1.5, 1.5], Round)\ntuple_split ('a,b', ',', Parts)\n")
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch2.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch3.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch4.viz").read_text(encoding="utf-8"))
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

    exhausted = run_source("Vizion.open_framegrabber ('File', 'samples/code39.png', Acq)\ngrab_image (Image, Acq)\ngrab_image (Image2, Acq)\n")
    assert exhausted.error and "沒有下一張" in exhausted.error

    folder = ROOT / "workspace" / "batch4_frames"
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / "a.png"), np.full((20, 20), 10, np.uint8))
    cv2.imwrite(str(folder / "b.png"), np.full((20, 20), 200, np.uint8))
    frames = run_source(
        "Vizion.open_framegrabber ('File', 'workspace/batch4_frames', 'gray', 0, 0, Acq)\n"
        "Vizion.grab_image (A, Acq)\ngrab_image (B, Acq)\nclose_framegrabber (Acq)\n"
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch5.viz").read_text(encoding="utf-8"))
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
    opened = REGISTRY["gray_opening_rect"].fn(Image(plain), 5, 5, ctx=None).iconic[0]
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch6.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch7.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch8.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch9.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch10.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch11.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch12.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch13.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch14.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch15.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch16.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch17.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch18.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch19.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch20.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch21.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch22.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch23.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch24.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch25.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch26.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch27.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch28.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch29.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch30.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch31.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch32.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch33.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch34.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch35.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch36.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch37.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch38.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch39.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch40.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch41.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch42.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch43.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch44.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch45.viz").read_text(encoding="utf-8"))
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
    empty = run_source("Vizion.gen_rectangle1 (Box, 0, 0, 1, 1)\ndifference (Box, Box, Empty)\nthickness_max (Empty, Row, Column, Distance)\n")
    assert empty.error and "區域是空的" in empty.error


def test_hdevelop_batch46():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch46.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch47.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch48.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch49.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch50.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch51.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch52.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch53.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch54.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch55.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch56.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch57.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch58.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch59.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch60.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch61.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch62.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch63.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch64.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch65.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch66.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch67.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch68.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch69.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch104.viz").read_text(encoding="utf-8"))
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


def test_hdevelop_batch70():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch70.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch71.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Area"] == [100]
    assert abs(float(sample.vars["Row"][0]) - 14.5) < 1e-6
    assert abs(float(sample.vars["Column"][0]) - 14.5) < 1e-6
    assert sample.vars["Maximum"] == 80
    assert abs(float(sample.vars["Mean"]) - (8010 / 4800)) < 1e-9
    negative = run_source(
        "Vizion.gen_image_const (A, 'byte', 8, 8)\ngen_image_proto (A, B, 1)\ndiff_region (A, B, D, -1)\n"
    )
    assert "不能是負的" in (negative.error or "")


def test_hdevelop_batch72():
    import cv2

    from app.engine.chapters.code_ops import _decode_qr

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch72.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch73.viz").read_text(encoding="utf-8"))
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
    negative = run_source("Vizion.gen_image_const (A, 'byte', 16, 16)\nridge_region (A, R, 1, -1, 'dark')\n")
    assert "不能是負的" in (negative.error or "")


def test_hdevelop_batch74():
    import math

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch74.viz").read_text(encoding="utf-8"))
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
    negative = run_source("Vizion.gen_image_const (A, 'byte', 16, 16)\nedge_region (A, R, 1, -1)\n")
    assert "不能是負的" in (negative.error or "")


def test_hdevelop_batch75():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch75.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch76.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch77.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch78.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch79.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["RowMin"] == 10 and sample.vars["RowMax"] == 10
    assert sample.vars["ColMin"] == 0 and sample.vars["ColMax"] == 30
    assert abs(float(sample.vars["PlainMean"]) - 12.4) < 1e-6
    assert sample.vars["CleanMin"] == 10 and sample.vars["CleanMax"] == 10


def test_hdevelop_batch80():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch80.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch81.viz").read_text(encoding="utf-8"))
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
    bad = run_source("Vizion.gen_image_const (A, 'byte', 16, 16)\nget_grayval_interpolated (A, 1, 1, 'cubic', G)\n")
    assert "只能是" in (bad.error or "")


def test_hdevelop_batch82():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch82.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch83.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch84.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert float(sample.vars["ThinBegin"][0]) == 45.5
    assert float(sample.vars["ThinEnd"][0]) == 45.5
    assert float(sample.vars["WideBegin"][0]) == 39.5
    assert float(sample.vars["WideEnd"][0]) == 39.5


def test_hdevelop_batch85():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch85.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch86.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Affine"] == [1, 0, 2, 0, 1, 5]
    assert sample.vars["Row0"] == 2 and sample.vars["Col0"] == 5
    assert sample.vars["Row1"] == 12 and sample.vars["Col1"] == 20


def test_hdevelop_batch87():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch87.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Row0"] == 3 and sample.vars["Col0"] == 4
    assert sample.vars["Row1"] == 13 and sample.vars["Col1"] == 14


def test_hdevelop_batch88():
    import math

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch88.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["BRow"][0]) - 80) < 0.02
    assert abs(float(sample.vars["BCol"][0]) - 90) < 0.02
    assert abs(float(sample.vars["BR1"][0]) - 40) < 0.02
    assert abs(float(sample.vars["BR2"][0]) - 18) < 0.02
    delta = abs(float(sample.vars["BPhi"][0]) - 0.4) % math.pi
    assert min(delta, math.pi - delta) < 0.02


def test_hdevelop_batch89():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch89.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["FRow"][0] == 50 and sample.vars["FCol"][0] == 80
    assert sample.vars["FPhi"][0] == 0
    assert sample.vars["FL1"][0] == 36 and sample.vars["FL2"][0] == 18
    assert sample.vars["SRow"][0] == 43 and sample.vars["SL2"][0] == 25


def test_hdevelop_batch90():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch90.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Alpha"])) < 1e-9
    assert abs(float(sample.vars["Beta"])) < 1e-9
    assert abs(float(sample.vars["Gamma"])) < 1e-9
    assert abs(float(sample.vars["RawA"]) + 20.0 / 3.0) < 1e-6
    assert abs(float(sample.vars["RawB"]) + 20.0 / 3.0) < 1e-6
    assert abs(float(sample.vars["RawG"]) - 460.0 / 9.0) < 1e-6


def test_hdevelop_batch91():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch91.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Nx"])) < 1e-9
    assert abs(float(sample.vars["Ny"])) < 1e-9
    assert abs(float(sample.vars["Nz"]) - 1) < 1e-9
    assert abs(float(sample.vars["Dist"])) < 1e-9


def test_hdevelop_batch92():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch92.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["Ax"]) - 1) < 1e-9
    assert abs(float(sample.vars["Ay"])) < 1e-9
    assert abs(float(sample.vars["Az"])) < 1e-9


def test_hdevelop_batch93():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch93.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["LCol"][1]) - 28.85060845773462) < 1e-6
    assert abs(float(sample.vars["LCol"][3]) - 71.14939154226538) < 1e-6
    assert sample.vars["RCol"][1] == 30 and sample.vars["RCol"][3] == 70
    assert sample.vars["RRow"][1] == 44 and sample.vars["RRow"][2] == 56


def test_hdevelop_batch94():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch94.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch95.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert all(abs(float(item) - 10) < 1e-6 for item in sample.vars["Row"])
    assert [abs(float(item) - expected) < 1e-6 for item, expected in zip(sample.vars["Col"], [0, 10, 20, 30, 40])] == [True] * 5


def test_hdevelop_batch96():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch96.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Mid"] == 112
    assert sample.vars["Average"] == 100
    assert sample.vars["Kept"] == 200


def test_hdevelop_batch97():
    import math

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch97.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["WideP"] == 0 and sample.vars["WideA"] == 0
    assert abs(float(sample.vars["TallP"]) - math.pi / 2) < 1e-12
    assert abs(float(sample.vars["TallA"]) - math.pi / 2) < 1e-12
    assert abs(float(sample.vars["WRa"]) - math.sqrt(133)) < 1e-9
    assert abs(float(sample.vars["WRb"]) - math.sqrt(8)) < 1e-9


def test_hdevelop_batch98():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch98.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Mid"] == 112
    assert sample.vars["Average"] == 100
    assert sample.vars["Pulled"] == 112
    assert sample.vars["Row"] == 0
    assert abs(float(sample.vars["Column"]) - 0.5) < 1e-9


def test_hdevelop_batch99():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch99.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    center = 30 + 1 / 3
    assert sample.vars["FRow"] == 20 and sample.vars["GRow"] == 20
    assert abs(float(sample.vars["FCol"]) - (30 + 1 / 6)) < 1e-9
    assert abs(float(sample.vars["GCol"]) - 30.33298435807228) < 1e-9
    assert abs(float(sample.vars["GCol"]) - center) < abs(float(sample.vars["FCol"]) - center)


def test_hdevelop_batch100():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch100.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert abs(float(sample.vars["BRow"]) - 24.094262617339446) < 1e-9
    assert abs(float(sample.vars["BCol"]) - 36.094262648985456) < 1e-9


def test_hdevelop_batch101():
    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch101.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    refined = float(sample.vars["S"])
    still = float(sample.vars["Still"])
    assert still == 1.0
    assert abs(refined - 1.07505) < 1e-5
    assert abs(refined - 1.1) < abs(still - 1.1)


def test_hdevelop_batch102():
    import numpy as np

    from app.engine.ncc_model import _keys_peak

    ensure_samples(ROOT)
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch102.viz").read_text(encoding="utf-8"))
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
    sample = run_source((ROOT / "samples" / "ViZion_hdevelop_batch103.viz").read_text(encoding="utf-8"))
    assert sample.error is None, sample.error
    assert sample.vars["Text"] == "VISDEV"
    assert len(sample.vars["Symbol"].contours) == 1
    assert "'Data Matrix'" in REGISTRY["create_data_code_2d_model"].control_in[0].choices
    modules = encode_modules("12AB")
    assert decode_image(modules)[0][0] == "12AB"


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
    session = run_source("Vizion.read_image (Image, 'workspace/_ocr_page.png')\nocr_find (Image, Words, 'auto', Text, Row, Column, Score)\n")
    assert session.error is None, session.error
    found = " ".join(str(item) for item in session.vars["Text"])
    assert "VISDEV" in found and "168" in found, found
    assert len(session.vars["Words"].contours) == len(session.vars["Text"])
    outside = run_source("Vizion.read_image (Image, 'workspace/_ocr_page.png')\nocr_find (Image, Words, 'auto', 0, 0, 20, 20, Text, Row, Column, Score)\n")
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
