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
tuple_select (T, 0, B)
tuple_select (T, 1, C)
tuple_gen_sequence (1, 5, 2, Seq)
tuple_add (Seq, 10, Plus)
tuple_sub (Plus, 1, Minus)
tuple_less_elem (Seq, 4, Less)
tuple_equal (Seq, [1, 3, 5], Same)
tuple_find ([4, 8, 4], 4, Found)
tuple_find ([1, 2], 9, Missing)
tuple_sort ([3, 1, 2], Sorted)
tuple_string (12.5, '.1f', Text)
tuple_regexp_match ('ab12', '[0-9]+', Digits)
tuple_substr ('HALCON', 0, 2, Head)
tuple_gen_const (3, 7, Const)
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
    sample = run_source((ROOT / "samples" / "hdevelop_batch1.hdv").read_text(encoding="utf-8"))
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
    rounded = run_source("tuple_round ([-1.5, 1.5], Round)\ntuple_split ('a,b', ',', Parts)\n")
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
    sample = run_source((ROOT / "samples" / "hdevelop_batch2.hdv").read_text(encoding="utf-8"))
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
read_image (Image, 'samples/parts.png')
threshold (Image, Region, 60, 255)
dev_set_color ('#ff3b30')
dev_display (Image)
dev_display (Region)
erosion_rectangle1 (Region, Eroded, 5, 5)
dilation_rectangle1 (Region, Dilated, 5, 5)
opening_rectangle1 (Region, Opened, 5, 5)
closing_rectangle1 (Region, Closed, 5, 5)
opening_circle (Region, CircleOpen, 3.5)
boundary (Region, Border, 'inner')
connection (Region, Parts)
dev_set_colored (6)
dev_display (Parts)
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
read_image (Image, 'samples/parts.png')
binary_threshold (Image, Region, 'max_separability', 'light', Used)
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
    omitted = parse_program("connection (Region, Parts)\n")
    given = parse_program("connection (Region, Parts, 4.5)\n")
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
read_image (Image, 'samples/parts.png')
threshold (Image, Region, 60, 255)
connection (Region, Parts)
dev_display (Image)
dev_set_draw ('margin')
dev_display (Parts)
opencv_threshold (Image, OpenCVRegion, 60, 'binary')
opencv_connection (OpenCVRegion, OpenCVParts, 4)
count_obj (Parts, HalconCount)
count_obj (OpenCVParts, OpenCVCount)
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
gen_contour_polygon_xld (Diag, [0, 20], [0, 20])
fit_line_contour_xld (Diag, Line, Row1, Col1, Row2, Col2)
gen_contour_polygon_xld (Box, [10, 10, 40, 40, 10], [10, 40, 40, 10, 10])
gen_region_contour_xld (Box, Region, 'filled')
area_center (Region, Area, CenterRow, CenterCol)
segment_contours_xld (Box, Sides, 'lines', 1)
count_obj (Sides, SideCount)
select_obj (Sides, FirstSide, 1)
get_contour_xld (FirstSide, SideRow, SideCol)
select_contours_xld (Sides, LongSides, 'contlength', 20, 40)
segment_contours_xld (Box, Corners, 'angle', 0.5)
gen_contour_polygon_xld (Arc, [60, 100, 140, 100], [100, 140, 100, 60])
fit_circle_contour_xld (Arc, Circle, CRow, CCol, Radius)
fit_rectangle2_contour_xld (Box, Rect, RRow, RCol, Phi, Len1, Len2)
close_contours_xld (Diag, Closed)
smooth_contours_xld (Box, SmoothBox, 5)
length_xld (Diag, DiagLength)
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
    session = run_source("read_image (Image, 'workspace/_ocr_page.png')\nocr_find (Image, Words, 'auto', Text, Row, Column, Score)\n")
    assert session.error is None, session.error
    found = " ".join(str(item) for item in session.vars["Text"])
    assert "VISDEV" in found and "168" in found, found
    assert len(session.vars["Words"].contours) == len(session.vars["Text"])
    outside = run_source("read_image (Image, 'workspace/_ocr_page.png')\nocr_find (Image, Words, 'auto', 0, 0, 20, 20, Text, Row, Column, Score)\n")
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
        "read_image (Image, 'samples/gauge.png')\n"
        "gauge_point (Image, Mark2, 110, 80, 110, 220, PRow2, PCol2, Amp2, Dist2)\n"
    )
    assert short.error is None, short.error
    picked = float(short.vars["PCol2"][0] if isinstance(short.vars["PCol2"], list) else short.vars["PCol2"])
    assert abs(picked - 107.5) < 0.6 or abs(picked - 192.5) < 0.6, picked
    parallel = run_source("intersection_ll (0, 0, 0, 10, 2, 0, 2, 10, Row, Column)\n")
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
        return
    raise AssertionError("應該要拒絕未知算子")


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
