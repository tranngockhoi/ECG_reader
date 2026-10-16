/* engine.js – chạy mô hình ECG ngay trong trình duyệt (onnxruntime-web).
   Port từ ecgmodel.py: tiền xử lý đúng như lúc huấn luyện, chấm 28 nhãn, bản đồ nhiệt (Grad-CAM), chất lượng ảnh, kết luận.
   Chỉ để NGHIÊN CỨU / GIẢI TRÍ. */
(function (global) {
  'use strict';
  const MEAN = [0.485, 0.456, 0.406], STD = [0.229, 0.224, 0.225];

  const GROUPS = {
    NORM: { vi: 'Bình thường', icon: '😊', tone: 'ok', desc: 'Hình dạng sóng và nhịp trông giống ECG bình thường.' },
    MI: { vi: 'Nhồi máu cơ tim', icon: '💔', tone: 'bad', desc: "Dấu hiệu giống nhồi máu cơ tim (theo nhãn PTB-XL, không phải 'tắc động mạch hoàn toàn')." },
    STTC: { vi: 'Thay đổi ST-T', icon: '〰️', tone: 'warn', desc: 'Đoạn ST và sóng T khác thường (thiếu máu cục bộ, thay đổi không đặc hiệu…).' },
    CD: { vi: 'Rối loạn dẫn truyền', icon: '⚡', tone: 'warn', desc: 'Xung điện đi qua tim chậm hoặc bất thường (block nhánh, block nhĩ thất, WPW…).' },
    HYP: { vi: 'Phì đại / giãn buồng tim', icon: '🫀', tone: 'warn', desc: 'Dấu hiệu dày thất hoặc giãn nhĩ.' },
  };
  const SUBS = {
    AMI: ['NMCT thành trước', 'MI', 'Nhồi máu cơ tim vùng trước (V1–V4…).'],
    IMI: ['NMCT thành dưới', 'MI', 'Nhồi máu cơ tim vùng dưới (II, III, aVF).'],
    LMI: ['NMCT thành bên', 'MI', 'Nhồi máu cơ tim vùng bên (I, aVL, V5–V6).'],
    PMI: ['NMCT thành sau', 'MI', "Nhồi máu thành sau thường kín đáo: ST chênh xuống V1–V3 có thể là hình 'soi gương'."],
    ISCA: ['Thiếu máu cục bộ – vùng trước', 'STTC', 'Thay đổi ST-T gợi ý thiếu máu vùng trước.'],
    ISCI: ['Thiếu máu cục bộ – vùng dưới', 'STTC', 'Thay đổi ST-T gợi ý thiếu máu vùng dưới.'],
    ISC_: ['Thay đổi thiếu máu cục bộ (khác)', 'STTC', 'Thay đổi ST-T gợi ý thiếu máu, không rõ vùng.'],
    NST_: ['Thay đổi ST-T không đặc hiệu', 'STTC', 'ST-T khác thường nhưng không gợi ý bệnh cụ thể.'],
    STTC: ['Thay đổi ST-T', 'STTC', 'Đoạn ST và sóng T khác thường.'],
    CLBBB: ['Block nhánh trái hoàn toàn', 'CD', 'QRS rộng, dạng điển hình của block nhánh trái.'],
    ILBBB: ['Block nhánh trái không hoàn toàn', 'CD', 'Dạng block nhánh trái nhưng QRS chưa rộng hẳn.'],
    CRBBB: ['Block nhánh phải hoàn toàn', 'CD', "QRS rộng, dạng rSR' ở V1."],
    IRBBB: ['Block nhánh phải không hoàn toàn', 'CD', 'Dạng block nhánh phải nhưng QRS chưa rộng hẳn.'],
    IVCD: ['Rối loạn dẫn truyền trong thất', 'CD', 'QRS rộng nhưng không phải block nhánh điển hình.'],
    'LAFB/LPFB': ['Block phân nhánh trái trước/sau', 'CD', 'Lệch trục do block một phân nhánh của nhánh trái.'],
    _AVB: ['Block nhĩ thất', 'CD', 'Dẫn truyền từ nhĩ xuống thất bị chậm hoặc ngắt quãng.'],
    WPW: ['Hội chứng WPW', 'CD', 'PR ngắn, sóng delta (đường dẫn truyền phụ).'],
    LVH: ['Phì đại thất trái', 'HYP', 'Biên độ QRS cao, có thể kèm thay đổi ST-T.'],
    RVH: ['Phì đại thất phải', 'HYP', 'Ưu thế sóng R ở V1, trục lệch phải.'],
    SEHYP: ['Phì đại vách liên thất', 'HYP', 'Dày vách liên thất.'],
    'LAO/LAE': ['Giãn/phì đại nhĩ trái', 'HYP', 'Sóng P rộng, hai đỉnh ở II hoặc pha âm ở V1.'],
    'RAO/RAE': ['Giãn/phì đại nhĩ phải', 'HYP', 'Sóng P cao nhọn ở II.'],
  };
  const HIDDEN = new Set(['sub_NORM']);

  function labelMeta(key) {
    const i = key.indexOf('_'), kind = key.slice(0, i), code = key.slice(i + 1);
    if (kind === 'sup') { const g = GROUPS[code]; return { key, code, kind: 'sup', group: code, vi: g.vi, icon: g.icon, tone: g.tone, desc: g.desc }; }
    const [vi, grp, desc] = SUBS[code];
    return { key, code, kind: 'sub', group: grp, vi, icon: GROUPS[grp].icon, tone: GROUPS[grp].tone, desc };
  }
  const level = a => a == null || isNaN(a) ? 'chưa rõ' : a >= 0.95 ? 'rất tốt' : a >= 0.90 ? 'tốt' : a >= 0.85 ? 'tạm được' : 'còn yếu';

  /* ---------- resize song tuyến tính có khử răng cưa, giống PIL.Image.resize(BILINEAR) ---------- */
  function coeffs(inSize, outSize) {
    const scale = inSize / outSize, fs = Math.max(scale, 1), support = fs;
    const out = [];
    for (let x = 0; x < outSize; x++) {
      const c = (x + 0.5) * scale;
      const xmin = Math.max(Math.trunc(c - support + 0.5), 0), xmax = Math.min(Math.trunc(c + support + 0.5), inSize);
      const w = []; let s = 0;
      for (let k = xmin; k < xmax; k++) { let t = (k - c + 0.5) / fs; t = Math.abs(t); const v = t < 1 ? 1 - t : 0; w.push(v); s += v; }
      for (let k = 0; k < w.length; k++) w[k] /= (s || 1);
      out.push({ xmin, w });
    }
    return out;
  }
  /* src: Uint8 mảng có `ch` kênh xen kẽ; trả về Uint8 cùng số kênh */
  function resize(src, sw, sh, dw, dh, ch) {
    let cur = src, cw = sw;
    if (dw !== sw) {
      const cf = coeffs(sw, dw), tmp = new Uint8ClampedArray(dw * sh * ch);
      for (let y = 0; y < sh; y++) {
        const rowS = y * sw * ch, rowD = y * dw * ch;
        for (let x = 0; x < dw; x++) {
          const { xmin, w } = cf[x];
          for (let c = 0; c < ch; c++) { let a = 0; for (let k = 0; k < w.length; k++) a += w[k] * cur[rowS + (xmin + k) * ch + c]; tmp[rowD + x * ch + c] = a + 0.5; }
        }
      }
      cur = tmp; cw = dw;
    }
    if (dh !== sh) {
      const cf = coeffs(sh, dh), out = new Uint8ClampedArray(cw * dh * ch);
      for (let y = 0; y < dh; y++) {
        const { xmin, w } = cf[y];
        for (let x = 0; x < cw; x++) for (let c = 0; c < ch; c++) {
          let a = 0; for (let k = 0; k < w.length; k++) a += w[k] * cur[((xmin + k) * cw + x) * ch + c];
          out[(y * cw + x) * ch + c] = a + 0.5;
        }
      }
      cur = out;
    }
    return cur;
  }
  const rgbaToRgb = d => { const n = d.length >> 2, o = new Uint8ClampedArray(n * 3); for (let i = 0, j = 0; i < n; i++) { o[j++] = d[4 * i]; o[j++] = d[4 * i + 1]; o[j++] = d[4 * i + 2]; } return o; };
  const toGray = (rgb, n) => { const g = new Uint8ClampedArray(n); for (let i = 0; i < n; i++) g[i] = (299 * rgb[3 * i] + 587 * rgb[3 * i + 1] + 114 * rgb[3 * i + 2] + 500) / 1000; return g; };
  function percentile(sorted, p) { const i = p / 100 * (sorted.length - 1), lo = Math.floor(i), hi = Math.min(lo + 1, sorted.length - 1); return sorted[lo] + (sorted[hi] - sorted[lo]) * (i - lo); }

  /* ---------- chất lượng ảnh ---------- */
  function qualityReport(rgb, w, h) {
    const gw = 1024, gh = Math.max(1, Math.trunc(1024 * h / w));
    const gSrc = toGray(rgb, w * h);
    const g8 = resize(gSrc, w, h, gw, gh, 1), g = Float32Array.from(g8);
    let sum = 0, clip = 0, dark = 0; for (let i = 0; i < g.length; i++) { sum += g[i]; if (g[i] >= 252) clip++; if (g[i] < 90) dark++; }
    const bright = sum / g.length, clipped = clip / g.length, darkF = dark / g.length;
    let ls = 0, ls2 = 0, n = 0;
    for (let y = 1; y < gh - 1; y++) for (let x = 1; x < gw - 1; x++) {
      const i = y * gw + x, v = 4 * g[i] - g[i - gw] - g[i + gw] - g[i - 1] - g[i + 1]; ls += v; ls2 += v * v; n++;
    }
    const sharp = n ? ls2 / n - (ls / n) * (ls / n) : 0;
    const sorted = Float32Array.from(g).sort(), contrast = percentile(sorted, 95) - percentile(sorted, 5);
    const aw = 256, ah = Math.max(1, Math.trunc(256 * h / w)), a = resize(rgb, w, h, aw, ah, 3);
    let col = 0; for (let i = 0; i < aw * ah; i++) { const r = a[3 * i], gg = a[3 * i + 1], b = a[3 * i + 2], mx = Math.max(r, gg, b), mn = Math.min(r, gg, b); if (mx - mn > 18 && mx > 140) col++; }
    const colored = col / (aw * ah);
    const notes = []; let score = 100;
    const add = (lv, text) => notes.push({ level: lv, text });
    if (sharp < 25) { add('bad', 'Ảnh khá mờ – thử giữ chắc tay, chạm màn hình để lấy nét.'); score -= 30; }
    else if (sharp < 60) { add('warn', 'Ảnh hơi mềm, chụp rõ nét hơn sẽ tốt hơn.'); score -= 12; }
    else add('ok', 'Nét ECG sắc nét.');
    if (bright < 90) { add('bad', 'Ảnh quá tối – hãy chụp nơi sáng hơn.'); score -= 25; }
    else if (bright < 130) { add('warn', 'Ảnh hơi tối.'); score -= 10; }
    else if (clipped > 0.7) { add('warn', 'Ảnh có thể bị cháy sáng – giảm sáng hoặc tránh ánh đèn chiếu thẳng.'); score -= 10; }
    else add('ok', 'Độ sáng ổn.');
    if (contrast < 60) { add('warn', 'Độ tương phản thấp, nét ECG có thể bị chìm.'); score -= 12; }
    if (h > w * 1.1) { add('warn', 'Ảnh chụp dọc. ECG thường nằm ngang – hãy xoay ảnh hoặc chụp ngang.'); score -= 15; }
    if (darkF < 0.002) { add('bad', 'Hầu như không thấy nét ECG – đây có phải tờ ECG không?'); score -= 30; }
    else if (colored > 0.01) add('ok', 'Thấy lưới ô vuông có màu.');
    else add('info', 'Không thấy lưới màu (có thể là bản in trắng đen).');
    if (Math.min(w, h) < 500) { add('warn', 'Ảnh có độ phân giải thấp.'); score -= 12; }
    score = Math.max(0, Math.min(100, Math.round(score)));
    return { score, notes, sharpness: +sharp.toFixed(1), brightness: +bright.toFixed(1), width: w, height: h };
  }

  /* ---------- kết luận ---------- */
  function buildResult(cols, probs, reliab) {
    const scores = [];
    for (const key of cols) {
      if (HIDDEN.has(key)) continue;
      const m = labelMeta(key), a = reliab[key] == null ? null : reliab[key];
      Object.assign(m, { score: probs[key], auroc: a, level: level(a) }); scores.push(m);
    }
    const sup = {}; scores.filter(s => s.kind === 'sup').forEach(s => sup[s.code] = s);
    const subs = scores.filter(s => s.kind === 'sub').sort((x, y) => y.score - x.score);
    const abn = ['MI', 'STTC', 'CD', 'HYP'].filter(c => sup[c]).map(c => sup[c]).sort((x, y) => y.score - x.score);
    const flagged = abn.filter(s => s.score >= 0.5), nOk = sup.NORM ? sup.NORM.score : 0;
    let verdict;
    if (flagged.length) {
      const top = flagged[0], names = flagged.slice(0, 3).map(s => s.vi).join(' · ');
      verdict = { tone: top.tone, icon: top.icon, title: flagged.length === 1 ? 'Nghi ngờ: ' + top.vi : 'Nhiều dấu hiệu cần để ý', text: flagged.length > 1 ? names : top.desc, level: 'phát hiện' };
      if (flagged.some(s => s.code === 'MI')) verdict.extra = "Lưu ý: 'nhồi máu cơ tim' ở đây theo nhãn PTB-XL, không phải kết luận 'tắc động mạch cấp'.";
    } else if (nOk >= 0.5) verdict = { tone: 'ok', icon: '😊', title: 'Trông khá ổn', text: GROUPS.NORM.desc, level: 'bình thường' };
    else verdict = { tone: 'mute', icon: '🤔', title: 'Mô hình chưa chắc chắn', text: 'Không nhãn nào vượt ngưỡng rõ ràng. Gần nhất: ' + abn.slice(0, 2).map(s => `${s.vi} (${Math.round(s.score * 100)}%)`).join(' · '), level: 'chưa rõ' };
    return { verdict, groups: ['NORM', 'MI', 'STTC', 'CD', 'HYP'].filter(c => sup[c]).map(c => sup[c]), subs };
  }

  /* ---------- engine ---------- */
  const E = { cfg: null, session: null, backend: null, rt: new Map(), KEEP: 12, GROUPS, SUBS, labelMeta, level, resize, qualityReport, buildResult };

  E.setConfig = cfg => { E.cfg = cfg; };

  E.loadModel = async function (bytes, useGpu) {
    ort.env.wasm.wasmPaths = new URL('ort/', location.href).href;
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.max(1, Math.min(4, navigator.hardwareConcurrency || 2)) : 1;
    const base = { graphOptimizationLevel: 'all' };
    const tries = useGpu && navigator.gpu ? [['webgpu'], ['wasm']] : [['wasm']];
    let lastErr = null;
    for (const ep of tries) {
      try {
        E.session = await ort.InferenceSession.create(bytes, { ...base, executionProviders: ep });
        E.backend = ep[0] + (ep[0] === 'wasm' ? ` · ${ort.env.wasm.numThreads} luồng` : '');
        if (useGpu && ep[0] === 'wasm' && navigator.gpu) E.gpuFailed = true;
        return E.backend;
      } catch (e) { lastErr = e; console.warn('Không tạo được phiên với', ep, e); }
    }
    throw lastErr;
  };

  E.analyze = async function (blob) {
    if (!E.session) throw new Error('Mô hình chưa sẵn sàng.');
    const t0 = performance.now(), { H, W, cols } = E.cfg;
    const bmp = await createImageBitmap(blob, { imageOrientation: 'from-image' });
    const sw = bmp.width, sh = bmp.height;
    const cv = document.createElement('canvas'); cv.width = sw; cv.height = sh;
    const cx = cv.getContext('2d', { willReadFrequently: true });
    cx.fillStyle = '#fff'; cx.fillRect(0, 0, sw, sh); cx.drawImage(bmp, 0, 0);
    const rgb = rgbaToRgb(cx.getImageData(0, 0, sw, sh).data);
    // letterbox nền trắng
    const s = Math.min(W / sw, H / sh), nw = Math.max(1, Math.round(sw * s)), nh = Math.max(1, Math.round(sh * s));
    const small = resize(rgb, sw, sh, nw, nh, 3), view = new Uint8ClampedArray(W * H * 3).fill(255);
    const ox = Math.floor((W - nw) / 2), oy = Math.floor((H - nh) / 2);
    for (let y = 0; y < nh; y++) view.set(small.subarray(y * nw * 3, (y + 1) * nw * 3), ((y + oy) * W + ox) * 3);
    // tensor CHW chuẩn hoá ImageNet
    const plane = W * H, x = new Float32Array(3 * plane);
    for (let i = 0; i < plane; i++) for (let c = 0; c < 3; c++) x[c * plane + i] = (view[3 * i + c] / 255 - MEAN[c]) / STD[c];
    const out = await E.session.run({ x: new ort.Tensor('float32', x, [1, 3, H, W]) });
    const z = out.logits.data, feat = Float32Array.from(out.feat.data), fd = out.feat.dims;
    const probs = {}; cols.forEach((c, i) => probs[c] = 1 / (1 + Math.exp(-z[i])));
    const res = buildResult(cols, probs, E.cfg.reliab);
    const quality = qualityReport(rgb, sw, sh);
    // ảnh hiển thị
    const jid = Math.random().toString(36).slice(2, 10) + Date.now().toString(36);
    const vc = document.createElement('canvas'); vc.width = W; vc.height = H;
    const vx = vc.getContext('2d'), id = vx.createImageData(W, H);
    for (let i = 0; i < plane; i++) { id.data[4 * i] = view[3 * i]; id.data[4 * i + 1] = view[3 * i + 1]; id.data[4 * i + 2] = view[3 * i + 2]; id.data[4 * i + 3] = 255; }
    vx.putImageData(id, 0, 0);
    const viewBlob = await new Promise(r => vc.toBlob(r, 'image/jpeg', 0.9));
    const rt = { feat, fd, urls: { orig: URL.createObjectURL(blob), view: URL.createObjectURL(viewBlob) }, cam: new Map() };
    E.rt.set(jid, rt);
    while (E.rt.size > E.KEEP) { const k = E.rt.keys().next().value, o = E.rt.get(k); Object.values(o.urls).forEach(URL.revokeObjectURL); o.cam.forEach(URL.revokeObjectURL); E.rt.delete(k); }
    const abn = res.groups.filter(g => g.code !== 'NORM').reduce((a, b) => b.score > a.score ? b : a), top = res.groups.reduce((a, b) => b.score > a.score ? b : a);
    return Object.assign({ id: jid }, res, { quality, cam_default: abn.score >= 0.5 ? abn.key : top.key, seconds: +((performance.now() - t0) / 1000).toFixed(2) });
  };

  /* Grad-CAM của nhãn `key` -> URL ảnh PNG RGBA 512×256 (giống heat_rgba của bản Python) */
  E.cam = async function (jid, key) {
    const rt = E.rt.get(jid); if (!rt) return null;
    if (rt.cam.has(key)) return rt.cam.get(key);
    const j = E.cfg.cols.indexOf(key); if (j < 0) return null;
    const [, C, fh, fw] = rt.fd, N = fh * fw, Wt = E.cfg.cls_w[j], A = rt.feat, cam = new Float32Array(N);
    for (let c = 0; c < C; c++) {
      let pos = 0; const o = c * N; for (let p = 0; p < N; p++) if (A[o + p] > 0) pos++;
      const w = Wt[c] * (pos / N) / N; if (!w) continue;
      for (let p = 0; p < N; p++) cam[p] += w * A[o + p];
    }
    let mx = 0; for (let p = 0; p < N; p++) { if (cam[p] < 0) cam[p] = 0; if (cam[p] > mx) mx = cam[p]; }
    if (mx > 0) for (let p = 0; p < N; p++) cam[p] /= mx;
    const OW = 512, OH = 256, cv = document.createElement('canvas'); cv.width = OW; cv.height = OH;
    const cx = cv.getContext('2d'), img = cx.createImageData(OW, OH);
    for (let y = 0; y < OH; y++) {
      const fy = Math.min(Math.max((y + 0.5) * fh / OH - 0.5, 0), fh - 1), y0 = Math.floor(fy), y1 = Math.min(y0 + 1, fh - 1), ty = fy - y0;
      for (let x = 0; x < OW; x++) {
        const fx = Math.min(Math.max((x + 0.5) * fw / OW - 0.5, 0), fw - 1), x0 = Math.floor(fx), x1 = Math.min(x0 + 1, fw - 1), tx = fx - x0;
        const c = cam[y0 * fw + x0] * (1 - tx) * (1 - ty) + cam[y0 * fw + x1] * tx * (1 - ty) + cam[y1 * fw + x0] * (1 - tx) * ty + cam[y1 * fw + x1] * tx * ty;
        const r = Math.min(Math.max(2.2 * c, 0), 1), g = Math.min(Math.max(2.2 * c * (2 - 2 * c), 0), 1), b = c < 0.5 ? Math.min(Math.max(1.4 - 3 * c, 0), 1) : 0;
        const al = Math.pow(Math.min(Math.max((c - 0.15) / 0.85, 0), 1), 0.8), i = 4 * (y * OW + x);
        img.data[i] = 255 * r; img.data[i + 1] = 255 * g * 0.85; img.data[i + 2] = 255 * b; img.data[i + 3] = 255 * 0.85 * al;
      }
    }
    cx.putImageData(img, 0, 0);
    const blob = await new Promise(r => cv.toBlob(r, 'image/png')), url = URL.createObjectURL(blob);
    rt.cam.set(key, url); return url;
  };

  global.Engine = E;
})(self);
