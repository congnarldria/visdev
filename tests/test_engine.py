import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.engine.parser import ParseError, parse_program
from app.engine.samples_gen import ensure_samples
from app.engine.session import Session


def run_source(source: str) -> Session:
    session = Session(source, ROOT)
    session.run()
    return session


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
    ensure_samples(ROOT)
    for name in ("roi_check.hdv", "edge_amp.hdv"):
        source = (ROOT / "samples" / name).read_text(encoding="utf-8")
        session = run_source(source)
        print(name, session.error, "count", session.vars.get("Count"))
        assert session.error is None, (name, session.error)


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
    test_parse_error()
    print("all passed")
