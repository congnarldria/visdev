"""OCR engines.

auto uses the strongest general local engine that is actually installed.
It never sends an image to a cloud service and never switches to a captcha,
formula, or single-line model. Cloud and research engines run only when the
program explicitly names them. Licensed vendor SDKs (HALCON Deep OCR, Cognex
OCRMax, Keyence, Omron, Euresys EasyOCR2, ABBYY FineReader desktop) need the
vendor runtime and are not embedded.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .objects import Image


@dataclass
class OcrHit:
    text: str
    score: float
    box: list[list[float]]


@dataclass
class EngineInfo:
    key: str
    title: str
    summary: str
    installed: bool = False
    note: str = ""
    group: str = ""
    purpose: str = ""
    auto_ok: bool = False
    rank: int = 100


@dataclass(frozen=True)
class EngineSpec:
    key: str
    title: str
    group: str
    purpose: str
    note: str
    ready: Callable[[], bool]
    rank: int = 100
    auto: bool = False
    remote: bool = False
    aliases: tuple[str, ...] = ()


@dataclass
class OcrCatalog:
    engines: list[EngineInfo] = field(default_factory=list)

    def ready(self, key: str) -> bool:
        return any(item.key == key and item.installed for item in self.engines)

    def best(self) -> str | None:
        ranked = [item for item in self.engines if item.installed and item.auto_ok]
        if not ranked:
            return None
        ranked.sort(key=lambda item: item.rank)
        return ranked[0].key


_MODELS: dict[str, object] = {}
_SPEC_CACHE: dict[str, bool] = {}

_LOCAL = "本機通用"
_DOC = "文件、手寫與特殊場景"
_CLOUD = "雲端服務"


def _module(name: str) -> bool:
    cached = _SPEC_CACHE.get(name)
    if cached is not None:
        return cached
    try:
        ok = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        ok = False
    _SPEC_CACHE[name] = ok
    return ok


def _keys(*names: str) -> bool:
    return all(os.environ.get(name, "").strip() for name in names)


def _tesseract_exe() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found
    default = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if default.is_file():
        return str(default)
    return None


def _spec(
    key: str,
    title: str,
    group: str,
    purpose: str,
    note: str,
    ready: Callable[[], bool],
    rank: int = 100,
    auto: bool = False,
    remote: bool = False,
    aliases: tuple[str, ...] = (),
) -> EngineSpec:
    return EngineSpec(key, title, group, purpose, note, ready, rank, auto, remote, aliases)


# 每一條 purpose 是選引擎時的專業用途。auto 只看 auto=True 且已安裝的本機引擎。
_SPECS: tuple[EngineSpec, ...] = (
    _spec("rapid", "RapidOCR", _LOCAL, "PP-OCR 的 ONNX Runtime 版，面向中文與英文場景文字。不載入 Paddle 推理框架，是這台電腦預設的本機引擎。", "未安裝時執行：pip install rapidocr-onnxruntime", lambda: _module("rapidocr_onnxruntime"), 10, True, aliases=("rapidocr", "ppocr", "pp-ocr")),
    _spec("paddle", "PaddleOCR", _LOCAL, "百度 PP-OCRv6，含文字行方向分類。偵測、識別與方向比 ONNX 輕量版完整，適合歪斜與拍照中文；模型較大。", "未安裝時執行：pip install paddlepaddle==3.3.1 paddleocr==3.7.0。不要安裝 opencv-contrib-python，否則會換掉現有的 OpenCV。", lambda: _module("paddleocr"), 20, True, aliases=("paddleocr",)),
    _spec("easyocr", "EasyOCR", _LOCAL, "CRAFT 偵測加 CRNN 識別的多語場景文字。語系覆蓋廣，依賴 PyTorch，CPU 上比 ONNX 慢。", "未安裝時執行：pip install easyocr。會一併裝上很大的 PyTorch。", lambda: _module("easyocr"), 30, True, aliases=("easy",)),
    _spec("cnocr", "CnOCR", _LOCAL, "針對中文印刷體的輕量識別，適合文件與螢幕中文。不是通用場景多語引擎。", "未安裝時執行：pip install cnocr", lambda: _module("cnocr"), 40, True, aliases=("chineseocr",)),
    _spec("windows", "Windows OCR", _LOCAL, "Windows.Media.Ocr。用系統已安裝的語言包讀螢幕與掃描文字，不另下載模型；歪斜場景字較弱。", "未安裝時執行：pip install winocr", lambda: _module("winocr"), 50, True, aliases=("win", "winrt")),
    _spec("tesseract", "Tesseract", _LOCAL, "LSTM 文件 OCR。乾淨掃描件與固定排版較穩，拍照彎曲與藝術字較弱。繁體要 chi_tra 語言包。", "未安裝時請安裝 Tesseract，並加上 chi_tra 語言包。", lambda: _tesseract_exe() is not None, 80, True, aliases=("tess",)),
    _spec("doctr", "docTR", _DOC, "Mindee 的文件 OCR：DBNet 偵測加 CRNN 或 ViT 識別。適合拉丁文掃描文件與表單，預設模型不是繁體中文。", "未安裝時執行：pip install python-doctr。第一次執行會下載模型。", lambda: _module("doctr"), 60, True, aliases=("python-doctr",)),
    _spec("surya", "Surya", _DOC, "版面感知的文件 OCR，含行偵測。適合多欄文件、掃描件與部分手寫；模型大，CPU 較慢。", "未安裝時執行：pip install surya-ocr。第一次執行會下載模型。", lambda: _module("surya"), 70, True, aliases=("suryaocr",)),
    _spec("mmocr", "MMOCR", _DOC, "OpenMMLab 文字偵測與識別工具箱。研究與自訂偵測、識別骨幹時用，預設 DBNet 加 CRNN 偏拉丁文。", "未安裝時執行：pip install mmocr。相依很多，不建議和現有環境一起裝。", lambda: _module("mmocr"), 90, True, aliases=("openmmlab",)),
    _spec("kerasocr", "Keras OCR", _DOC, "Keras 版 CRAFT 偵測加 CRNN。適合英文場景文字的對照實驗，依賴 TensorFlow。", "未安裝時執行：pip install keras-ocr", lambda: _module("keras_ocr"), 100, True, aliases=("keras-ocr", "keras")),
    _spec("trocr", "TrOCR", _DOC, "微軟 Transformer 單行識別。印刷用 trocr-base-printed，手寫可設 TROCR_MODEL=microsoft/trocr-base-handwritten。沒有偵測器，整頁會被當成一行。", "未安裝時執行：pip install transformers。第一次執行會下載模型。", lambda: _module("transformers"), aliases=("microsoft",)),
    _spec("got", "GOT-OCR 2.0", _DOC, "端到端文件模型，整頁讀成文字或 markdown，適合複雜版面、表格與公式的整頁轉寫。不逐行給外框。", "未安裝時執行：pip install transformers。模型約數 GB，第一次執行才下載。可用 GOT_OCR_TYPE=format 輸出版面。", lambda: _module("transformers"), aliases=("gotocr", "got-ocr", "gotocr2")),
    _spec("manga", "MangaOCR", _DOC, "日文漫畫對白。輸入應是已經裁好的一個對話框，不負責整頁分格。", "未安裝時執行：pip install manga-ocr", lambda: _module("manga_ocr"), aliases=("mangaocr",)),
    _spec("ddddocr", "ddddocr", _DOC, "驗證碼與簡單字元。先偵測字塊再逐塊分類，不適合合同、書籍或場景長文。", "未安裝時執行：pip install ddddocr", lambda: _module("ddddocr"), aliases=("dddd", "captcha")),
    _spec("azure", "Azure AI Vision", _CLOUD, "微軟 Read API。適合多語文件、掃描件與手寫的雲端產線；有行級外框。影像會離開這台電腦。", "設定 AZURE_OCR_ENDPOINT 與 AZURE_OCR_KEY。", lambda: _keys("AZURE_OCR_ENDPOINT", "AZURE_OCR_KEY"), remote=True),
    _spec("google", "Google Cloud Vision", _CLOUD, "DOCUMENT_TEXT_DETECTION。適合多語文件與自然場景，字級外框完整。影像會上傳到 Google。", "設定 GOOGLE_VISION_API_KEY。", lambda: _keys("GOOGLE_VISION_API_KEY"), remote=True, aliases=("gcv", "vision")),
    _spec("textract", "Amazon Textract", _CLOUD, "AWS 文件文字偵測。適合表單、發票與掃描件的雲端批次，回傳行級外框。", "安裝 boto3，並設定 AWS_ACCESS_KEY_ID 與 AWS_SECRET_ACCESS_KEY。", lambda: _module("boto3") and _keys("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"), remote=True, aliases=("aws", "amazon")),
    _spec("baidu", "百度文字識別", _CLOUD, "通用文字識別（含位置）。中文印刷、拍照與螢幕截圖的商用雲端引擎。", "設定 BAIDU_OCR_API_KEY 與 BAIDU_OCR_SECRET_KEY。", lambda: _keys("BAIDU_OCR_API_KEY", "BAIDU_OCR_SECRET_KEY"), remote=True, aliases=("baiduocr",)),
    _spec("tencent", "騰訊雲 OCR", _CLOUD, "通用印刷體高精度。適合中文票據、文件與拍照印刷字，回傳多邊形外框。", "安裝 tencentcloud-sdk-python，並設定 TENCENT_OCR_SECRET_ID 與 TENCENT_OCR_SECRET_KEY。", lambda: _module("tencentcloud") and _keys("TENCENT_OCR_SECRET_ID", "TENCENT_OCR_SECRET_KEY"), remote=True, aliases=("tencentocr",)),
    _spec("aliyun", "阿里雲 OCR", _CLOUD, "通用文字識別。適合中文文件、卡證與拍照印刷字的企業雲端流程。", "安裝 alibabacloud_ocr_api20210707，並設定 ALIBABA_CLOUD_ACCESS_KEY_ID 與 ALIBABA_CLOUD_ACCESS_KEY_SECRET。", lambda: _module("alibabacloud_ocr_api20210707") and _keys("ALIBABA_CLOUD_ACCESS_KEY_ID", "ALIBABA_CLOUD_ACCESS_KEY_SECRET"), remote=True, aliases=("alibaba", "ali")),
    _spec("huawei", "華為雲 OCR", _CLOUD, "通用文字識別。適合已在華為雲上的中文文件與表格前處理。", "安裝 huaweicloudsdkocr，並設定 HUAWEI_OCR_AK 與 HUAWEI_OCR_SK。", lambda: _module("huaweicloudsdkocr") and _keys("HUAWEI_OCR_AK", "HUAWEI_OCR_SK"), remote=True, aliases=("huaweicloud",)),
    _spec("xfyun", "訊飛 OCR", _CLOUD, "訊飛通用文字識別，中英印刷與拍照件。適合要接訊飛帳號的中文產線。", "設定 XFYUN_APP_ID 與 XFYUN_API_KEY。", lambda: _keys("XFYUN_APP_ID", "XFYUN_API_KEY"), remote=True, aliases=("iflytek", "xunfei")),
    _spec("youdao", "有道 OCR", _CLOUD, "網易有道通用文字識別。中英文件與自然場景的另一路中文雲端引擎。", "設定 YOUDAO_APP_KEY 與 YOUDAO_APP_SECRET。", lambda: _keys("YOUDAO_APP_KEY", "YOUDAO_APP_SECRET"), remote=True, aliases=("youdaoocr",)),
    _spec("mathpix", "Mathpix", _CLOUD, "數學、物理與化學式。適合公式、教材與科學 PDF，不是一般場景文字。", "設定 MATHPIX_APP_ID 與 MATHPIX_APP_KEY。", lambda: _keys("MATHPIX_APP_ID", "MATHPIX_APP_KEY"), remote=True, aliases=("math",)),
    _spec("ocrspace", "OCR.space", _CLOUD, "OCR.space 引擎。適合少量多語文件的外部備援，繁體用 cht。", "設定 OCR_SPACE_API_KEY。", lambda: _keys("OCR_SPACE_API_KEY"), remote=True, aliases=("ocr.space",)),
    _spec("abbyy", "ABBYY Cloud OCR", _CLOUD, "ABBYY 雲端文件擷取。適合發票、表單與高品質掃描件的正式產線，不是場景藝術字。", "設定 ABBYY_APPLICATION_ID 與 ABBYY_PASSWORD。伺服器可選 ABBYY_SERVER_URL。", lambda: _keys("ABBYY_APPLICATION_ID", "ABBYY_PASSWORD"), remote=True, aliases=("finereader",)),
)


def _aliases() -> dict[str, str]:
    table = {"auto": "auto"}
    for spec in _SPECS:
        table[spec.key] = spec.key
        for alias in spec.aliases:
            table[alias] = spec.key
    return table


def normalize_method(method: str) -> str:
    key = str(method or "auto").strip().lower().replace(" ", "")
    table = _aliases()
    if key not in table:
        raise ValueError(f"OCR 方法不在目錄裡，得到 {method}。請改用 auto，或算子說明中的引擎名稱。")
    return table[key]


def method_choices() -> list[str]:
    return ["'auto'"] + [f"'{spec.key}'" for spec in _SPECS]


def catalog() -> OcrCatalog:
    engines = [
        EngineInfo(
            "auto",
            "自動",
            "在已安裝的本機通用引擎裡挑最強的一個。不把影像送到雲端，也不改用驗證碼、公式或單行模型。",
            True,
            group=_LOCAL,
            purpose="在已安裝的本機通用引擎裡挑最強的一個。不把影像送到雲端，也不改用驗證碼、公式或單行模型。",
            auto_ok=False,
        )
    ]
    for spec in _SPECS:
        engines.append(
            EngineInfo(
                spec.key,
                spec.title,
                spec.purpose,
                spec.ready(),
                spec.note,
                spec.group,
                spec.purpose,
                spec.auto and not spec.remote,
                spec.rank,
            )
        )
    return OcrCatalog(engines)


def describe_catalog() -> str:
    info = catalog()
    lines = [
        "讀出影像上的文字，每個字塊一個外框。",
        "auto 只在已經裝好的本機通用引擎裡挑，不會把影像送到雲端，也不會改用驗證碼、公式或單行模型。",
        "原廠授權引擎（HALCON Deep OCR、Cognex OCRMax、Keyence、Omron、Euresys EasyOCR2、ABBYY FineReader 桌面版）要原廠執行環境，不能內嵌。下面每一套都有實際呼叫程式，選到且這台已啟用才會跑。",
        "",
    ]
    for item in info.engines:
        if item.key == "auto":
            continue
        state = "已可使用" if item.installed else "這台還沒啟用"
        extra = "" if item.installed else item.note
        lines.append(f"{item.key}  {item.title}：{item.purpose}{state}。{extra}")
    lines.append(f"這台電腦 auto 會用 {info.best() or '（還沒有可用引擎）'}。")
    return "\n".join(lines)


def _prepare(image: Image) -> np.ndarray:
    data = image.data
    if data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    elif data.shape[2] == 4:
        data = cv2.cvtColor(data, cv2.COLOR_BGRA2BGR)
    if image.domain is not None:
        data = data.copy()
        data[image.domain == 0] = 255
    return np.ascontiguousarray(data)


def _quad(points) -> list[list[float]]:
    box = [[float(point[0]), float(point[1])] for point in points]
    if len(box) >= 2 and box[0] != box[-1]:
        box.append(box[0])
    return box


def _rect(x: float, y: float, width: float, height: float) -> list[list[float]]:
    x2 = x + width
    y2 = y + height
    return _quad([[x, y], [x2, y], [x2, y2], [x, y2]])


def _clean(hits: list[OcrHit]) -> list[OcrHit]:
    kept = []
    for hit in hits:
        text = " ".join(str(hit.text).split())
        if not text:
            continue
        score = float(hit.score)
        if score > 1.5:
            score = score / 100.0
        score = min(1.0, max(0.0, score))
        kept.append(OcrHit(text, score, hit.box))
    return kept


def _require(key: str) -> None:
    info = catalog()
    item = next(entry for entry in info.engines if entry.key == key)
    if item.installed:
        return
    raise RuntimeError(f"{item.title} 還沒裝好。{item.note} 或改用 auto。")


def _cached(name: str, factory):
    if name not in _MODELS:
        _MODELS[name] = factory()
    return _MODELS[name]


def _rapid(image: np.ndarray) -> list[OcrHit]:
    _require("rapid")
    engine = _cached("rapid", lambda: __import__("rapidocr_onnxruntime", fromlist=["RapidOCR"]).RapidOCR())
    result, _elapsed = engine(image)
    hits = []
    for item in result or []:
        box, text, score = item[0], item[1], item[2]
        hits.append(OcrHit(str(text), float(score), _quad(box)))
    return _clean(hits)


def _paddle(image: np.ndarray) -> list[OcrHit]:
    _require("paddle")

    def build():
        from paddleocr import PaddleOCR

        # oneDNN on this Paddle build cannot run the PP-OCRv6 graph.
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        return PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
            enable_mkldnn=False,
        )

    engine = _cached("paddle", build)
    hits = []
    for page in engine.predict(image) or []:
        texts = list(page["rec_texts"])
        scores = list(page["rec_scores"])
        polys = list(page["rec_polys"])
        for text, score, poly in zip(texts, scores, polys):
            hits.append(OcrHit(str(text), float(score), _quad(poly)))
    return _clean(hits)


def _easyocr(image: np.ndarray) -> list[OcrHit]:
    _require("easyocr")

    def build():
        import easyocr

        return easyocr.Reader(["ch_tra", "en"], gpu=False)

    reader = _cached("easyocr", build)
    hits = []
    for box, text, score in reader.readtext(image):
        hits.append(OcrHit(str(text), float(score), _quad(box)))
    return _clean(hits)


def _box_of(rect) -> list[list[float]] | None:
    if rect is None:
        return None
    if isinstance(rect, dict):
        x = float(rect.get("x", 0))
        y = float(rect.get("y", 0))
        width = float(rect.get("width", 0))
        height = float(rect.get("height", 0))
    else:
        x = float(getattr(rect, "x", 0))
        y = float(getattr(rect, "y", 0))
        width = float(getattr(rect, "width", 0))
        height = float(getattr(rect, "height", 0))
    if width <= 0 or height <= 0:
        return None
    return _rect(x, y, width, height)


def _windows(image: np.ndarray) -> list[OcrHit]:
    _require("windows")
    import winocr

    result = None
    error = None
    for lang in ("zh-Hant", "en", "zh-Hans"):
        try:
            result = winocr.recognize_cv2_sync(image, lang)
            break
        except Exception as exc:
            error = exc
            result = None
    if result is None:
        raise RuntimeError(f"Windows OCR 讀取失敗：{error}") from error
    lines = result.get("lines", []) if isinstance(result, dict) else (getattr(result, "lines", None) or [])
    whole = result.get("text", "") if isinstance(result, dict) else getattr(result, "text", "")
    hits = []
    if not lines and whole:
        height, width = image.shape[:2]
        hits.append(OcrHit(str(whole), 1.0, _rect(0, 0, width, height)))
        return _clean(hits)
    for line in lines:
        words = line.get("words", []) if isinstance(line, dict) else (getattr(line, "words", None) or [])
        pieces = []
        rects = []
        for word in words:
            if isinstance(word, dict):
                pieces.append(str(word.get("text", "")))
                rects.append(word.get("bounding_rect"))
            else:
                pieces.append(str(getattr(word, "text", "")))
                rects.append(getattr(word, "bounding_rect", None))
        text = "".join(pieces).strip()
        if not text:
            text = str(line.get("text", "") if isinstance(line, dict) else getattr(line, "text", ""))
        boxes = [box for box in (_box_of(rect) for rect in rects) if box]
        if boxes:
            xs = [point[0] for box in boxes for point in box]
            ys = [point[1] for box in boxes for point in box]
            box = _rect(min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys))
        else:
            box = _rect(0, 0, image.shape[1], max(1.0, image.shape[0] * 0.05))
        hits.append(OcrHit(text, 1.0, box))
    return _clean(hits)


def _tesseract_langs(exe: str) -> str:
    import subprocess

    try:
        listed = subprocess.run([exe, "--list-langs"], capture_output=True, text=True, timeout=20)
        names = {line.strip() for line in (listed.stdout or "").splitlines()}
    except (OSError, subprocess.TimeoutExpired):
        names = set()
    chosen = [name for name in ("chi_tra", "chi_sim", "eng") if name in names]
    return "+".join(chosen) if chosen else "eng"


def _tesseract(image: np.ndarray) -> list[OcrHit]:
    exe = _tesseract_exe()
    if not exe:
        _require("tesseract")
    import subprocess

    langs = _tesseract_langs(exe)
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "page.png"
        if not cv2.imwrite(str(path), image):
            raise RuntimeError("無法把影像交給 Tesseract")
        done = subprocess.run(
            [exe, str(path), "stdout", "-l", langs, "--psm", "6", "tsv"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    if done.returncode != 0:
        detail = (done.stderr or done.stdout or "Tesseract 失敗").strip()
        raise RuntimeError(detail)
    hits = []
    rows = (done.stdout or "").splitlines()
    if not rows:
        return []
    header = rows[0].split("\t")
    index = {name: pos for pos, name in enumerate(header)}
    for line in rows[1:]:
        cells = line.split("\t")
        if len(cells) <= index.get("text", 11):
            continue
        if cells[index.get("level", 0)] != "5":
            continue
        text = cells[index["text"]].strip()
        if not text:
            continue
        left = float(cells[index["left"]])
        top = float(cells[index["top"]])
        width = float(cells[index["width"]])
        height = float(cells[index["height"]])
        score = float(cells[index["conf"]]) if "conf" in index else 0.0
        hits.append(OcrHit(text, score, _rect(left, top, width, height)))
    return _clean(hits)


def _png_bytes(image: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("無法把影像編成 PNG")
    return buf.tobytes()


def _b64(image: np.ndarray) -> str:
    import base64

    return base64.b64encode(_png_bytes(image)).decode("ascii")


def _pil(image: np.ndarray):
    from PIL import Image as PILImage

    return PILImage.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def _full(image: np.ndarray) -> list[list[float]]:
    height, width = image.shape[:2]
    return _rect(0, 0, width, height)


def _box_or_full(points, image: np.ndarray) -> list[list[float]]:
    try:
        box = _quad(points)
    except (TypeError, ValueError):
        box = []
    if len(box) < 3:
        return _full(image)
    return box


def _page(text, image: np.ndarray) -> list[OcrHit]:
    body = str(text or "").strip()
    if not body:
        return []
    return [OcrHit(body, 1.0, _full(image))]


def _request(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 60, method: str | None = None):
    import urllib.error
    import urllib.request

    req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, {key.lower(): value for key, value in resp.headers.items()}, resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"OCR 服務回應 {exc.code}：{body[:400]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"連不到 OCR 服務：{exc.reason}") from exc


def _request_json(url: str, data: bytes | None = None, headers: dict | None = None, timeout: int = 60, method: str | None = None):
    import json

    status, headers_out, raw = _request(url, data, headers, timeout, method)
    if not raw:
        return status, headers_out, {}
    return status, headers_out, json.loads(raw.decode("utf-8"))


def _doctr(image: np.ndarray) -> list[OcrHit]:
    """文件 OCR。DBNet 偵測，識別預設偏拉丁文掃描件。"""
    _require("doctr")

    def build():
        from doctr.models import ocr_predictor

        return ocr_predictor(pretrained=True)

    predictor = _cached("doctr", build)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    exported = predictor([rgb]).export()
    height, width = image.shape[:2]
    hits = []
    for page in exported.get("pages") or []:
        for block in page.get("blocks") or []:
            for line in block.get("lines") or []:
                for word in line.get("words") or []:
                    geometry = word.get("geometry") or []
                    if len(geometry) >= 2:
                        (x0, y0), (x1, y1) = geometry[0], geometry[1]
                        box = _rect(float(x0) * width, float(y0) * height, max(1.0, (float(x1) - float(x0)) * width), max(1.0, (float(y1) - float(y0)) * height))
                    else:
                        box = _full(image)
                    hits.append(OcrHit(str(word.get("value") or ""), float(word.get("confidence") or 1), box))
    return _clean(hits)


def _surya(image: np.ndarray) -> list[OcrHit]:
    """版面感知的文件 OCR，一行一個外框。"""
    _require("surya")

    def build():
        from surya.detection import DetectionPredictor
        from surya.recognition import RecognitionPredictor

        return DetectionPredictor(), RecognitionPredictor()

    detector, recognizer = _cached("surya", build)
    predictions = recognizer([_pil(image)], det_predictor=detector)
    page = predictions[0]
    hits = []
    for line in getattr(page, "text_lines", None) or []:
        hits.append(OcrHit(str(getattr(line, "text", "")), float(getattr(line, "confidence", 1) or 1), _box_or_full(getattr(line, "polygon", None), image)))
    return _clean(hits)


def _cnocr(image: np.ndarray) -> list[OcrHit]:
    """中文印刷體識別。"""
    _require("cnocr")

    def build():
        from cnocr import CnOcr

        return CnOcr()

    engine = _cached("cnocr", build)
    hits = []
    for item in engine.ocr(image) or []:
        hits.append(OcrHit(str(item.get("text") or ""), float(item.get("score") or 1), _box_or_full(item.get("position"), image)))
    return _clean(hits)


def _mmocr(image: np.ndarray) -> list[OcrHit]:
    """OpenMMLab 的 DBNet 偵測與 CRNN 識別。"""
    _require("mmocr")

    def build():
        from mmocr.apis import MMOCRInferencer

        return MMOCRInferencer(det="DBNet", rec="CRNN")

    infer = _cached("mmocr", build)
    result = infer(image, return_vis=False)
    pred = (result.get("predictions") or [{}])[0]
    texts = list(pred.get("rec_texts") or [])
    scores = list(pred.get("rec_scores") or [])
    polys = list(pred.get("det_polygons") or [])
    hits = []
    for index, text in enumerate(texts):
        score = float(scores[index]) if index < len(scores) else 1.0
        poly = polys[index] if index < len(polys) else None
        if poly is not None and len(poly) == 8 and not isinstance(poly[0], (list, tuple)):
            poly = [[poly[i], poly[i + 1]] for i in range(0, 8, 2)]
        hits.append(OcrHit(str(text), score, _box_or_full(poly, image)))
    return _clean(hits)


def _kerasocr(image: np.ndarray) -> list[OcrHit]:
    """Keras CRAFT 加 CRNN，適合英文場景對照。"""
    _require("kerasocr")

    def build():
        import keras_ocr

        return keras_ocr.pipeline.Pipeline()

    pipeline = _cached("kerasocr", build)
    groups = pipeline.recognize([cv2.cvtColor(image, cv2.COLOR_BGR2RGB)])
    hits = []
    for text, box in groups[0] if groups else []:
        hits.append(OcrHit(str(text), 1.0, _box_or_full(box, image)))
    return _clean(hits)


def _trocr(image: np.ndarray) -> list[OcrHit]:
    """單行 Transformer。整張圖只產生一段文字。"""
    _require("trocr")

    def build():
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        name = os.environ.get("TROCR_MODEL", "microsoft/trocr-base-printed")
        processor = TrOCRProcessor.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name)
        model.eval()
        return processor, model, torch

    processor, model, torch = _cached("trocr", build)
    pixels = processor(images=_pil(image), return_tensors="pt").pixel_values
    with torch.no_grad():
        ids = model.generate(pixels)
    text = processor.batch_decode(ids, skip_special_tokens=True)[0]
    return _page(text, image)


def _got(image: np.ndarray) -> list[OcrHit]:
    """整頁端到端轉寫，不拆字塊外框。"""
    _require("got")

    def build():
        from transformers import AutoModel, AutoTokenizer

        name = os.environ.get("GOT_OCR_MODEL", "stepfun-ai/GOT-OCR2_0")
        tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        model = AutoModel.from_pretrained(name, trust_remote_code=True, device_map="cpu")
        model.eval()
        return tokenizer, model

    tokenizer, model = _cached("got", build)
    kind = os.environ.get("GOT_OCR_TYPE", "ocr")
    with tempfile.TemporaryDirectory() as folder:
        path = str(Path(folder) / "page.png")
        if not cv2.imwrite(path, image):
            raise RuntimeError("無法把影像交給 GOT-OCR")
        text = model.chat(tokenizer, path, ocr_type=kind)
    return _page(text, image)


def _manga(image: np.ndarray) -> list[OcrHit]:
    """一個日文漫畫對話框。"""
    _require("manga")

    def build():
        from manga_ocr import MangaOcr

        return MangaOcr()

    return _page(_cached("manga", build)(_pil(image)), image)


def _ddddocr(image: np.ndarray) -> list[OcrHit]:
    """驗證碼：先找字塊，再逐塊分類。"""
    _require("ddddocr")

    def build():
        import ddddocr

        return ddddocr.DdddOcr(det=True, show_ad=False), ddddocr.DdddOcr(show_ad=False)

    detector, reader = _cached("ddddocr", build)
    raw = _png_bytes(image)
    boxes = detector.detection(raw) or []
    if not boxes:
        return _page(reader.classification(raw), image)
    hits = []
    height, width = image.shape[:2]
    for box in boxes:
        x1, y1, x2, y2 = [int(float(value)) for value in box[:4]]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, max(x1 + 1, x2)), min(height, max(y1 + 1, y2))
        text = reader.classification(_png_bytes(image[y1:y2, x1:x2]))
        hits.append(OcrHit(str(text), 1.0, _rect(x1, y1, x2 - x1, y2 - y1)))
    return _clean(hits)


def _azure(image: np.ndarray) -> list[OcrHit]:
    """Azure Read。影像上傳到呼叫端設定的端點。"""
    _require("azure")
    import time

    endpoint = os.environ["AZURE_OCR_ENDPOINT"].rstrip("/")
    key = os.environ["AZURE_OCR_KEY"]
    headers = {"Ocp-Apim-Subscription-Key": key, "Content-Type": "application/octet-stream"}
    _status, response_headers, _raw = _request(endpoint + "/vision/v3.2/read/analyze", _png_bytes(image), headers, method="POST")
    operation = response_headers.get("operation-location")
    if not operation:
        raise RuntimeError("Azure OCR 沒有回傳作業位址")
    payload = {}
    for _ in range(40):
        time.sleep(0.5)
        _status, _headers, payload = _request_json(operation, headers={"Ocp-Apim-Subscription-Key": key})
        state = str(payload.get("status") or "").lower()
        if state == "succeeded":
            break
        if state == "failed":
            raise RuntimeError("Azure OCR 分析失敗")
    else:
        raise RuntimeError("Azure OCR 逾時")
    hits = []
    for page in ((payload.get("analyzeResult") or {}).get("readResults") or []):
        for line in page.get("lines") or []:
            values = line.get("boundingBox") or []
            points = [[values[i], values[i + 1]] for i in range(0, len(values) - 1, 2)]
            hits.append(OcrHit(str(line.get("text") or ""), 1.0, _box_or_full(points, image)))
    return _clean(hits)


def _google(image: np.ndarray) -> list[OcrHit]:
    """Google 文件文字偵測，一字一個外框。影像上傳到 Google。"""
    _require("google")
    import json
    import urllib.parse

    url = "https://vision.googleapis.com/v1/images:annotate?key=" + urllib.parse.quote(os.environ["GOOGLE_VISION_API_KEY"])
    body = json.dumps({"requests": [{"image": {"content": _b64(image)}, "features": [{"type": "DOCUMENT_TEXT_DETECTION"}]}]}).encode()
    _status, _headers, payload = _request_json(url, body, {"Content-Type": "application/json"})
    response = (payload.get("responses") or [{}])[0]
    if response.get("error"):
        raise RuntimeError(str(response["error"]))
    annotation = response.get("fullTextAnnotation") or {}
    hits = []
    for page in annotation.get("pages") or []:
        for block in page.get("blocks") or []:
            for para in block.get("paragraphs") or []:
                for word in para.get("words") or []:
                    text = "".join(symbol.get("text", "") for symbol in word.get("symbols") or [])
                    vertices = ((word.get("boundingBox") or {}).get("vertices") or [])
                    points = [[vert.get("x", 0), vert.get("y", 0)] for vert in vertices]
                    hits.append(OcrHit(text, float(word.get("confidence") or 1), _box_or_full(points, image)))
    if not hits and annotation.get("text"):
        return _page(annotation["text"], image)
    return _clean(hits)


def _textract(image: np.ndarray) -> list[OcrHit]:
    """Amazon Textract 的行級文件文字。影像上傳到 AWS。"""
    _require("textract")
    import boto3

    client = boto3.client("textract", region_name=os.environ.get("AWS_DEFAULT_REGION") or os.environ.get("AWS_REGION") or "us-east-1")
    response = client.detect_document_text(Document={"Bytes": _png_bytes(image)})
    height, width = image.shape[:2]
    hits = []
    for block in response.get("Blocks") or []:
        if block.get("BlockType") != "LINE":
            continue
        box = (block.get("Geometry") or {}).get("BoundingBox") or {}
        left = float(box.get("Left") or 0) * width
        top = float(box.get("Top") or 0) * height
        hits.append(OcrHit(str(block.get("Text") or ""), float(block.get("Confidence") or 100), _rect(left, top, max(1.0, float(box.get("Width") or 0) * width), max(1.0, float(box.get("Height") or 0) * height))))
    return _clean(hits)


def _baidu(image: np.ndarray) -> list[OcrHit]:
    """百度通用文字識別，含位置。影像上傳到百度。"""
    _require("baidu")
    import urllib.parse

    query = urllib.parse.urlencode({"grant_type": "client_credentials", "client_id": os.environ["BAIDU_OCR_API_KEY"], "client_secret": os.environ["BAIDU_OCR_SECRET_KEY"]})
    _status, _headers, token = _request_json("https://aip.baidubce.com/oauth/2.0/token?" + query, b"", method="POST")
    access = token.get("access_token")
    if not access:
        raise RuntimeError("百度 OCR 沒有拿到 access_token：" + str(token.get("error_description") or token.get("error_msg") or "未授權"))
    body = urllib.parse.urlencode({"image": _b64(image)}).encode()
    _status, _headers, payload = _request_json(
        "https://aip.baidubce.com/rest/2.0/ocr/v1/general?access_token=" + urllib.parse.quote(access),
        body,
        {"Content-Type": "application/x-www-form-urlencoded"},
    )
    if payload.get("error_code"):
        raise RuntimeError(f"百度 OCR {payload.get('error_code')}：{payload.get('error_msg')}")
    hits = []
    for item in payload.get("words_result") or []:
        loc = item.get("location") or {}
        hits.append(OcrHit(str(item.get("words") or ""), 1.0, _rect(float(loc.get("left") or 0), float(loc.get("top") or 0), max(1.0, float(loc.get("width") or 1)), max(1.0, float(loc.get("height") or 1)))))
    return _clean(hits)


def _tencent(image: np.ndarray) -> list[OcrHit]:
    """騰訊雲通用印刷體高精度。影像上傳到騰訊雲。"""
    _require("tencent")
    import json

    from tencentcloud.common import credential
    from tencentcloud.ocr.v20181119 import models, ocr_client

    cred = credential.Credential(os.environ["TENCENT_OCR_SECRET_ID"], os.environ["TENCENT_OCR_SECRET_KEY"])
    client = ocr_client.OcrClient(cred, os.environ.get("TENCENT_OCR_REGION", "ap-guangzhou"))
    request = models.GeneralAccurateOCRRequest()
    request.ImageBase64 = _b64(image)
    payload = json.loads(client.GeneralAccurateOCR(request).to_json_string())
    hits = []
    for item in payload.get("TextDetections") or []:
        polygon = item.get("Polygon") or []
        if polygon:
            box = _box_or_full([[point.get("X", 0), point.get("Y", 0)] for point in polygon], image)
        else:
            item_box = item.get("ItemPolygon") or {}
            box = _rect(float(item_box.get("X") or 0), float(item_box.get("Y") or 0), max(1.0, float(item_box.get("Width") or 1)), max(1.0, float(item_box.get("Height") or 1)))
        hits.append(OcrHit(str(item.get("DetectedText") or ""), float(item.get("Confidence") or 100), box))
    return _clean(hits)


def _aliyun(image: np.ndarray) -> list[OcrHit]:
    """阿里雲通用文字識別。影像上傳到阿里雲。"""
    _require("aliyun")
    import io
    import json

    from alibabacloud_ocr_api20210707 import models as ocr_models
    from alibabacloud_ocr_api20210707.client import Client
    from alibabacloud_tea_openapi import models as open_api_models

    config = open_api_models.Config(
        access_key_id=os.environ["ALIBABA_CLOUD_ACCESS_KEY_ID"],
        access_key_secret=os.environ["ALIBABA_CLOUD_ACCESS_KEY_SECRET"],
    )
    config.endpoint = os.environ.get("ALIBABA_CLOUD_OCR_ENDPOINT", "ocr-api.cn-hangzhou.aliyuncs.com")
    request = ocr_models.RecognizeGeneralRequest(body=io.BytesIO(_png_bytes(image)))
    data = json.loads(Client(config).recognize_general(request).body.data or "{}")
    hits = []
    for item in data.get("prism_wordsInfo") or []:
        points = [[point.get("x", 0), point.get("y", 0)] for point in item.get("pos") or []]
        hits.append(OcrHit(str(item.get("word") or ""), 1.0, _box_or_full(points, image)))
    if not hits and data.get("content"):
        return _page(data["content"], image)
    return _clean(hits)


def _huawei(image: np.ndarray) -> list[OcrHit]:
    """華為雲通用文字識別。影像上傳到華為雲。"""
    _require("huawei")
    from huaweicloudsdkcore.auth.credentials import BasicCredentials
    from huaweicloudsdkocr.v1 import GeneralTextRequestBody, OcrClient, RecognizeGeneralTextRequest
    from huaweicloudsdkocr.v1.region.ocr_region import OcrRegion

    credentials = BasicCredentials(os.environ["HUAWEI_OCR_AK"], os.environ["HUAWEI_OCR_SK"])
    region = OcrRegion.value_of(os.environ.get("HUAWEI_OCR_REGION", "cn-north-4"))
    client = OcrClient.new_builder().with_credentials(credentials).with_region(region).build()
    request = RecognizeGeneralTextRequest()
    request.body = GeneralTextRequestBody(image=_b64(image))
    result = client.recognize_general_text(request).result
    hits = []
    for block in getattr(result, "words_block_list", None) or []:
        loc = getattr(block, "location", None)
        if loc is not None:
            box = _rect(float(getattr(loc, "left", 0) or 0), float(getattr(loc, "top", 0) or 0), max(1.0, float(getattr(loc, "width", 1) or 1)), max(1.0, float(getattr(loc, "height", 1) or 1)))
        else:
            box = _full(image)
        hits.append(OcrHit(str(getattr(block, "words", "") or ""), 1.0, box))
    return _clean(hits)


def _xfyun(image: np.ndarray) -> list[OcrHit]:
    """訊飛通用文字識別。影像上傳到訊飛。"""
    _require("xfyun")
    import base64
    import hashlib
    import json
    import time
    import urllib.parse

    param = base64.b64encode(json.dumps({"language": "cn|en"}).encode()).decode()
    now = str(int(time.time()))
    checksum = hashlib.md5((os.environ["XFYUN_API_KEY"] + now + param).encode()).hexdigest()
    headers = {
        "X-Appid": os.environ["XFYUN_APP_ID"],
        "X-CurTime": now,
        "X-Param": param,
        "X-CheckSum": checksum,
        "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
    }
    body = urllib.parse.urlencode({"image": _b64(image)}).encode()
    _status, _headers, payload = _request_json("https://webapi.xfyun.cn/v1/service/v1/ocr/general", body, headers)
    if str(payload.get("code")) not in ("0", "00000"):
        raise RuntimeError(f"訊飛 OCR {payload.get('code')}：{payload.get('desc') or payload.get('message') or '失敗'}")
    hits = []
    data = payload.get("data") or {}
    for block in data.get("block") or []:
        for line in block.get("line") or []:
            words = []
            points = []
            for word in line.get("word") or []:
                words.append(str(word.get("content") or ""))
                location = word.get("location") or {}
                for name in ("top_left", "top_right", "right_bottom", "left_bottom"):
                    point = location.get(name) or {}
                    if point:
                        points.append([point.get("x", 0), point.get("y", 0)])
            hits.append(OcrHit("".join(words), float(line.get("confidence") or 1), _box_or_full(points, image)))
    return _clean(hits)


def _youdao(image: np.ndarray) -> list[OcrHit]:
    """有道通用文字識別。影像上傳到有道。"""
    _require("youdao")
    import hashlib
    import time
    import urllib.parse
    import uuid

    image_b64 = _b64(image)
    salt = uuid.uuid4().hex
    now = str(int(time.time()))
    folded = image_b64 if len(image_b64) <= 20 else image_b64[:10] + str(len(image_b64)) + image_b64[-10:]
    sign = hashlib.sha256((os.environ["YOUDAO_APP_KEY"] + folded + salt + now + os.environ["YOUDAO_APP_SECRET"]).encode()).hexdigest()
    body = urllib.parse.urlencode({
        "img": image_b64,
        "langType": "zh-CHS",
        "detectType": "10012",
        "imageType": "1",
        "appKey": os.environ["YOUDAO_APP_KEY"],
        "salt": salt,
        "sign": sign,
        "signType": "v3",
        "curtime": now,
        "docType": "json",
    }).encode()
    _status, _headers, payload = _request_json("https://openapi.youdao.com/ocrapi", body, {"Content-Type": "application/x-www-form-urlencoded"})
    if str(payload.get("errorCode")) != "0":
        raise RuntimeError(f"有道 OCR {payload.get('errorCode')}：{payload.get('msg') or '失敗'}")
    hits = []
    for region in ((payload.get("Result") or {}).get("regions") or []):
        for line in region.get("lines") or []:
            raw = str(line.get("boundingBox") or "")
            nums = [float(part) for part in raw.split(",") if part.strip()]
            if len(nums) >= 4:
                box = _rect(nums[0], nums[1], max(1.0, nums[2]), max(1.0, nums[3]))
            else:
                box = _full(image)
            hits.append(OcrHit(str(line.get("text") or ""), 1.0, box))
    return _clean(hits)


def _mathpix(image: np.ndarray) -> list[OcrHit]:
    """公式與科學排版。影像上傳到 Mathpix。"""
    _require("mathpix")
    import json

    body = json.dumps({"src": "data:image/png;base64," + _b64(image), "formats": ["text"], "include_line_data": True}).encode()
    headers = {"app_id": os.environ["MATHPIX_APP_ID"], "app_key": os.environ["MATHPIX_APP_KEY"], "Content-Type": "application/json"}
    _status, _headers, payload = _request_json("https://api.mathpix.com/v3/text", body, headers)
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    hits = []
    for line in payload.get("line_data") or []:
        hits.append(OcrHit(str(line.get("text") or ""), float(line.get("confidence") or 1), _box_or_full(line.get("cnt"), image)))
    if not hits:
        return _page(payload.get("text") or "", image)
    return _clean(hits)


def _ocrspace(image: np.ndarray) -> list[OcrHit]:
    """OCR.space 文件備援。影像上傳到 OCR.space。"""
    _require("ocrspace")
    import urllib.parse

    body = urllib.parse.urlencode({
        "apikey": os.environ["OCR_SPACE_API_KEY"],
        "base64Image": "data:image/png;base64," + _b64(image),
        "language": os.environ.get("OCR_SPACE_LANGUAGE", "cht"),
        "isOverlayRequired": "true",
    }).encode()
    _status, _headers, payload = _request_json("https://api.ocr.space/parse/image", body, {"Content-Type": "application/x-www-form-urlencoded"})
    if payload.get("IsErroredOnProcessing"):
        message = payload.get("ErrorMessage") or payload.get("ErrorDetails") or "OCR.space 失敗"
        raise RuntimeError(str(message))
    parsed = (payload.get("ParsedResults") or [{}])[0]
    hits = []
    for line in ((parsed.get("TextOverlay") or {}).get("Lines") or []):
        for word in line.get("Words") or []:
            hits.append(OcrHit(str(word.get("WordText") or ""), 1.0, _rect(float(word.get("Left") or 0), float(word.get("Top") or 0), max(1.0, float(word.get("Width") or 1)), max(1.0, float(word.get("Height") or 1)))))
    if not hits:
        return _page(parsed.get("ParsedText") or "", image)
    return _clean(hits)


def _abbyy_task(raw: bytes) -> dict[str, str]:
    import xml.etree.ElementTree as ET

    root = ET.fromstring(raw)
    node = root if root.tag.endswith("task") else root.find(".//task")
    if node is None:
        raise RuntimeError(raw.decode("utf-8", "replace")[:300])
    return dict(node.attrib)


def _abbyy(image: np.ndarray) -> list[OcrHit]:
    """ABBYY 雲端文件擷取。影像上傳到 ABBYY。回傳整頁文字。"""
    _require("abbyy")
    import base64
    import time

    server = os.environ.get("ABBYY_SERVER_URL", "https://cloud-westus.ocrsdk.com").rstrip("/")
    token = base64.b64encode(f"{os.environ['ABBYY_APPLICATION_ID']}:{os.environ['ABBYY_PASSWORD']}".encode()).decode()
    headers = {"Authorization": "Basic " + token}
    _status, _response_headers, raw = _request(server + "/processImage?exportFormat=txt&language=ChinesePRC,ChineseTaiwan,English", _png_bytes(image), headers, method="POST")
    task = _abbyy_task(raw)
    for _ in range(40):
        status = task.get("status")
        if status == "Completed" and task.get("resultUrl"):
            _done, _headers, text = _request(task["resultUrl"], headers=headers)
            return _page(text.decode("utf-8", "replace"), image)
        if status in ("ProcessingFailed", "NotEnoughCredits"):
            raise RuntimeError(f"ABBYY OCR {status}：{task.get('error') or '失敗'}")
        time.sleep(0.5)
        _status, _headers, raw = _request(server + "/getTaskStatus?taskId=" + task.get("id", ""), headers=headers)
        task = _abbyy_task(raw)
    raise RuntimeError("ABBYY OCR 逾時")


_RUNNERS = {
    "rapid": _rapid,
    "paddle": _paddle,
    "easyocr": _easyocr,
    "cnocr": _cnocr,
    "windows": _windows,
    "tesseract": _tesseract,
    "doctr": _doctr,
    "surya": _surya,
    "mmocr": _mmocr,
    "kerasocr": _kerasocr,
    "trocr": _trocr,
    "got": _got,
    "manga": _manga,
    "ddddocr": _ddddocr,
    "azure": _azure,
    "google": _google,
    "textract": _textract,
    "baidu": _baidu,
    "tencent": _tencent,
    "aliyun": _aliyun,
    "huawei": _huawei,
    "xfyun": _xfyun,
    "youdao": _youdao,
    "mathpix": _mathpix,
    "ocrspace": _ocrspace,
    "abbyy": _abbyy,
}

_MISSING = [spec.key for spec in _SPECS if spec.key not in _RUNNERS]
if _MISSING:
    raise RuntimeError("OCR 目錄缺少呼叫程式：" + ", ".join(_MISSING))


def read_text(image: Image, method: str = "auto") -> tuple[str, list[OcrHit]]:
    chosen = normalize_method(method)
    if chosen == "auto":
        chosen = catalog().best()
        if not chosen:
            raise RuntimeError("還沒有可用的 OCR 引擎。請安裝 rapidocr-onnxruntime 或 winocr。")
    hits = _RUNNERS[chosen](_prepare(image))
    return chosen, hits
