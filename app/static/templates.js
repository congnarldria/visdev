(function () {
  const TEMPLATES = [
    { cmd: "inspectTemplate", title: "零件檢測", file: "inspect_parts.viz", expect: "8 個零件" },
    { cmd: "gaugeTemplate", title: "邊緣量測", file: "inspect_gauge.viz", expect: "邊緣點在行 107.5" },
    { cmd: "caliperTemplate", title: "卡尺", file: "inspect_caliper.viz", expect: "直徑約 83.93，寬 140，高 90" },
    { cmd: "defectTemplate", title: "瑕疵", file: "inspect_defect.viz", expect: "暗塊 1 個、面積 3550；亮點 1 個、面積 144" },
    { cmd: "colorTemplate", title: "色彩", file: "inspect_color.viz", expect: "綠塊面積 1200、色調 60" },
    { cmd: "charsTemplate", title: "字元", file: "inspect_chars.viz", expect: "讀出 VISDEV 168" },
    { cmd: "stripeTemplate", title: "條紋寬度", file: "inspect_stripe.viz", expect: "最寬亮條 22" },
    { cmd: "lineTemplate", title: "產線", file: "inspect_line.viz", expect: "合格 3、不合格 2" },
    { cmd: "lineCameraTemplate", title: "產線相機", file: "inspect_line_camera.viz", expect: "相機編號 0。沒有相機時改跑檔案產線" },
    { cmd: "lineOutTemplate", title: "產線輸出", file: "inspect_line_out.viz", expect: "TCP loopback，合格 3、不合格 2" },
    { cmd: "missingTemplate", title: "缺件", file: "inspect_missing.viz", expect: "應有 8 個零件" },
    { cmd: "holesTemplate", title: "孔數", file: "inspect_holes.viz", expect: "4 個暗孔，直徑約 28，偏差 0" },
    { cmd: "lineHolesTemplate", title: "孔數產線", file: "inspect_line_holes.viz", expect: "合格 2、不合格 3" },
    { cmd: "locateTemplate", title: "形狀定位", file: "inspect_locate.viz", expect: "場景裡找到 4 個" },
    { cmd: "stereoTemplate", title: "相對姿態", file: "inspect_stereo.viz", expect: "Pose1 的第一個數約 0.04 公尺" },
    { cmd: "codeTemplate", title: "條碼", file: "inspect_code.viz", expect: "Text 是 4006381333931，QrText 是 VISDEV-3" },
    { cmd: "fitTemplate", title: "找線找圓", file: "inspect_fit.viz", expect: "直線到行 30，圓心列 50 行 80，半徑 20" },
  ];

  const api = { templates: TEMPLATES };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof window !== "undefined") window.VisdevTemplates = api;
})();
