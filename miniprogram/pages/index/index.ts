import { char_dict } from './char_dict';

// 模型链接
const DET_MODEL_URL = 'https://fastly.jsdelivr.net/gh/Le0n-Mo/PP-OCRv3@main/det_ppocr_v3.onnx';
const REC_MODEL_URL = 'https://fastly.jsdelivr.net/gh/Le0n-Mo/PP-OCRv3@main/rec_ppocr_v3.onnx';

Page({
  data: {
    imageList: [] as string[],
    currentIndex: 0,
    canvases: [] as any[],
    ctxs: [] as any[],
    isModelReady: false,
    isManualMode: false,
    detSession: null as any,
    recSession: null as any,
  },

  onLoad() { this.prepareModels(); },

  async prepareModels() {
    wx.showLoading({ title: 'AI引擎准备中...' });
    try {
      const detPath = await this.downloadModel(DET_MODEL_URL, 'det_v3.onnx');
      const recPath = await this.downloadModel(REC_MODEL_URL, 'rec_v3.onnx');
      const detSession = (wx as any).createInferenceSession({ model: detPath, typicalShape: { 'x': [1, 3, 320, 320] } });
      const recSession = (wx as any).createInferenceSession({ model: recPath, typicalShape: { 'x': [1, 3, 48, 320] } });
      detSession.onLoad(() => {
        this.setData({ isModelReady: true, detSession, recSession });
        wx.hideLoading();
      });
    } catch (e) { wx.hideLoading(); }
  },

  preprocessImage(imgData: any) {
  const { data, width, height } = imgData;
  const size = width * height;
  const floatArray = new Float32Array(1 * 3 * size);
  
  // NCHW 内存布局索引
  let rIdx = 0;
  let gIdx = size;
  let bIdx = size * 2;

  for (let i = 0; i < data.length; i += 4) {
    // 模式 0 逻辑：像素值直接除以 255.0
    floatArray[rIdx++] = data[i] / 255.0;     // R
    floatArray[gIdx++] = data[i + 1] / 255.0; // G
    floatArray[bIdx++] = data[i + 2] / 255.0; // B
  }
  return floatArray;
  },

  async processSingleImage(index: number) {
    const canvas = this.data.canvases[index];
    const ctx = this.data.ctxs[index];
    if (!canvas || !ctx) return;

    console.log(`%c >>> 开始处理第 ${index + 1} 张图 <<< `, 'background: #222; color: #bada55');

    try {
      const inputSize = 320;
      const offCanvas = (wx as any).createOffscreenCanvas({ type: '2d', width: inputSize, height: inputSize });
      const offCtx = offCanvas.getContext('2d');
      offCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, inputSize, inputSize);
      const imgData = offCtx.getImageData(0, 0, inputSize, inputSize);

      const floatArray = this.preprocessImage(imgData);

      const detRes = await this.data.detSession.run({ 
        'x': { shape: [1, 3, inputSize, inputSize], data: floatArray.buffer, type: 'float32' } 
      });
      
      const boxes = this.getBoxes(detRes, canvas.width, canvas.height);
      // --- 调试：在画布上画出检测到的原始蓝色框 ---
      ctx.strokeStyle = 'blue';
      ctx.lineWidth = 1;
      boxes.forEach(b => ctx.strokeRect(b.x, b.y, b.w, b.h));

      console.log(`[DET] 检测到候选框数量: ${boxes.length}`);

      for (let j = 0; j < boxes.length; j++) {
        const text = await this.recognizeText(boxes[j], index);
        
        // --- vConsole 调试输出 ---
        console.log(`  [REC] 框-${j} 内容: 「${text}」`);

        const res = this.isSensitive(text);
        if (res.sensitive) {
          console.warn(`    ⚠️ 判定为敏感信息: "${text}"，准备打码`);
          this.maskBox(boxes[j], text, res.splitIdx, index);
        }
      }
    } catch (e) { console.error("推理异常:", e); }
  },

  async recognizeText(box: any, index: number) {
    const canvas = this.data.canvases[index];
    const recSession = this.data.recSession;
    const recH = 48, recW = 320;

    const offCanvas = (wx as any).createOffscreenCanvas({ type: '2d', width: recW, height: recH });
    const offCtx = offCanvas.getContext('2d');
    
    // 填充纯白背景
    offCtx.fillStyle = 'white';
    offCtx.fillRect(0, 0, recW, recH);

    // 再次微调切片坐标：确保不越界
    const sx = Math.max(0, box.x);
    const sy = Math.max(0, box.y);
    const sw = Math.min(canvas.width - sx, box.w);
    const sh = Math.min(canvas.height - sy, box.h);

    // 计算缩放：保持文字纵横比，不拉伸
    const ratio = sw / sh;
    const targetW = Math.min(recW, recH * ratio);
    
    // 居中绘制：留出一点左边距
    offCtx.drawImage(canvas, sx, sy, sw, sh, 5, 0, targetW, recH);
    
    const imgData = offCtx.getImageData(0, 0, recW, recH);
    const floatArray = this.preprocessForREC(imgData);

    const res = await recSession.run({ 
      'x': { shape: [1, 3, recH, recW], data: floatArray.buffer, type: 'float32' } 
    });
    const outputName = Object.keys(res)[0];
    return this.ctcDecode(new Float32Array(res[outputName].data));
  },

  preprocessForREC(imgData: any) {
    const { data, width, height } = imgData;
    const floatArray = new Float32Array(1 * 3 * height * width);
    
    // PaddleOCR 标准均值和方差
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];
    
    const size = width * height;
    // 按照 B-G-R 的顺序填充 NCHW
    for (let i = 0; i < size; i++) {
      // 这里的索引 i*4 分别是 R, G, B, A
      // 存入 floatArray 的顺序是 B(Idx), G(Idx), R(Idx)
      floatArray[i + size * 0] = ((data[i * 4 + 2] / 255.0) - mean[2]) / std[2]; // B
      floatArray[i + size * 1] = ((data[i * 4 + 1] / 255.0) - mean[1]) / std[1]; // G
      floatArray[i + size * 2] = ((data[i * 4 + 0] / 255.0) - mean[0]) / std[0]; // R
    }
    return floatArray;
  },

  chooseImage() {
    wx.chooseMedia({
      count: 9,
      mediaType: ['image'],
      success: (res) => {
        const paths = res.tempFiles.map(f => f.tempFilePath);
        this.setData({ imageList: paths, currentIndex: 0, canvases: [], ctxs: [] });
        setTimeout(() => { this.initAllCanvases(); }, 500);
      }
    });
  },

  initAllCanvases() {
    const { imageList } = this.data;
    imageList.forEach((path, index) => {
      const query = wx.createSelectorQuery();
      query.select(`#canvas-${index}`).fields({ node: true, size: true }).exec((res) => {
        const canvas = res[0].node;
        const ctx = canvas.getContext('2d');
        const img = canvas.createImage();
        img.src = path;
        img.onload = () => {
          canvas.width = img.width; canvas.height = img.height;
          ctx.drawImage(img, 0, 0, img.width, img.height);
          const { canvases, ctxs } = this.data;
          canvases[index] = canvas; ctxs[index] = ctx;
          this.setData({ canvases, ctxs });
        };
      });
    });
  },

  onSwiperChange(e: any) { this.setData({ currentIndex: e.detail.current }); },

  async runInferenceAll() {
    if (!this.data.isModelReady) return;
    const { imageList } = this.data;
    wx.showLoading({ title: 'AI实验处理中...' });
    for (let i = 0; i < imageList.length; i++) {
      this.setData({ currentIndex: i });
      await this.processSingleImage(i);
    }
    wx.hideLoading();
  },

  getBoxes(inferenceRes: any, w: number, h: number) {
    const outputName = Object.keys(inferenceRes)[0];
    const mapData = new Float32Array(inferenceRes[outputName].data);
    const mapSize = 320; 
    let rawBoxes: any[] = [];

    for (let y = 0; y < mapSize; y += 4) {
      for (let x = 0; x < mapSize; x += 4) {
        if (mapData[y * mapSize + x] > 0.3) { 
          let merged = false;
          for (let b of rawBoxes) {
            if (x >= b.l - 30 && x <= b.r + 30 && y >= b.t - 20 && y <= b.b + 20) {
              b.l = Math.min(b.l, x); b.r = Math.max(b.r, x);
              b.t = Math.min(b.t, y); b.b = Math.max(b.b, y);
              merged = true; break;
            }
          }
          if (!merged) rawBoxes.push({ l: x, r: x, t: y, b: y });
        }
      }
    }

    let filteredBoxes = rawBoxes.filter((boxA, idxA) => {
      return !rawBoxes.some((boxB, idxB) => {
        if (idxA === idxB) return false;
        return boxA.l >= boxB.l - 5 && boxA.r <= boxB.r + 5 &&
               boxA.t >= boxB.t - 5 && boxA.b <= boxB.b + 5;
      });
    });

    return filteredBoxes.map(b => {
      const rawW = b.r - b.l;
      const rawH = b.b - b.t;
      
      let adjustedH = rawH;
      const aspectRatio = rawW / rawH;
      const standardCharH = 18; 
      if (aspectRatio > 4 && adjustedH < standardCharH) { adjustedH = standardCharH; }

      // 宽度左右各扩 12% 左右
      const pW = (rawW * 0.12) + 5; 
      const pH = (adjustedH * 0.4) + 5;

      // 宽度系数设为 1.8，左侧偏移设为 0.8
      return {
        x: ((b.l - pW * 0.8) / mapSize) * w,
        y: (((b.t + b.b) / 2 - (adjustedH / 2 + pH)) / mapSize) * h,
        w: ((rawW + pW * 1.8) / mapSize) * w, 
        h: ((adjustedH + pH * 2) / mapSize) * h
      };
    });
  },

  maskBox(box: any, text: string, splitIdx: number, index: number) {
    const ctx = this.data.ctxs[index];
    if (!ctx) return;

    ctx.fillStyle = 'black';

    // --- 校准比例 (基于 1.8 倍扩张后的框) ---
    const textStartRate = 0.08; 
    const textContentRate = 0.84; 

    const perCharWidth = (box.w * textContentRate) / text.length;

    let safeSplitIdx = splitIdx;
    if (safeSplitIdx < 0) safeSplitIdx = 0;

    // 计算起始位置：起点 = 框左边界 + 预留空白 + (跳过的字数 * 单字宽)
    let finalX = box.x + (box.w * textStartRate) + (safeSplitIdx * perCharWidth);
    
    // 计算打码宽度
    let sensitiveTextLen = text.length - safeSplitIdx;
    let finalW = (sensitiveTextLen * perCharWidth) + 6; // 稍微多给 6 像素保护

    if (safeSplitIdx === 0) {
      finalX = box.x + (box.w * textStartRate) - 5;
      finalW = (box.w * textContentRate) + 10;
    }

    // 宽度安全检查
    if (finalW < 0) finalW = 0;

    // 执行绘制
    ctx.fillRect(finalX, box.y, finalW, box.h);
  },

  toggleManualMode() { this.setData({ isManualMode: !this.data.isManualMode }); },
  touchStart(e: any) { if (this.data.isManualMode) this.drawAtPoint(e.touches[0].x, e.touches[0].y); },
  touchMove(e: any) { if (this.data.isManualMode) this.drawAtPoint(e.touches[0].x, e.touches[0].y); },
  drawAtPoint(viewX: number, viewY: number) {
    const index = this.data.currentIndex;
    const canvas = this.data.canvases[index];
    const ctx = this.data.ctxs[index];
    const query = wx.createSelectorQuery();
    query.select(`#canvas-${index}`).boundingClientRect(rect => {
    const x = (viewX / rect.width) * canvas.width;
    const y = (viewY / rect.height) * canvas.height;
    ctx.fillStyle = 'black';
    const brush = canvas.width * 0.06;
    ctx.fillRect(x - brush/2, y - brush/2, brush, brush);
    }).exec();
  },

  showExportMenu() {
    if (this.data.imageList.length === 0) return;

    wx.showActionSheet({
      itemList: ['保存当前这张', `全部保存 (${this.data.imageList.length}张)`],
      itemColor: '#07c160',
      success: (res) => {
        if (res.tapIndex === 0) {
          this.saveImage(); // 调用单张保存
        } else if (res.tapIndex === 1) {
          this.batchSave(); // 调用批量保存
        }
      }
    });
  },

  async batchSave() {
    const { canvases, imageList } = this.data;
    
    wx.showLoading({ title: '准备批量导出...', mask: true });

    // 检查权限 (批量保存容易触发权限拦截，先检查一下)
    try {
      for (let i = 0; i < imageList.length; i++) {
        wx.showLoading({ title: `正在保存第 ${i + 1}/${imageList.length} 张`, mask: true });
        
        // 核心：等待当前这张保存成功后再跑下一张，防止并发冲突
        await this.saveToAlbumSync(canvases[i]);
      }
      
      wx.hideLoading();
      wx.showModal({
        title: '导出成功',
        content: `已成功保存 ${imageList.length} 张图片到系统相册`,
        showCancel: false
      });
    } catch (err) {
      wx.hideLoading();
      wx.showModal({
        title: '保存中断',
        content: '请确保已授权相册权限，或检查手机存储空间。',
        showCancel: false
      });
    }
  },

  saveToAlbumSync(canvas: any): Promise<void> {
    return new Promise((resolve, reject) => {
      wx.canvasToTempFilePath({
        canvas: canvas,
        destWidth: canvas.width,
        destHeight: canvas.height,
        success: (res) => {
          wx.saveImageToPhotosAlbum({
            filePath: res.tempFilePath,
            success: () => resolve(),
            fail: (err) => reject(err)
          });
        },
        fail: (err) => reject(err)
      });
    });
  },

  saveImage() {
    const index = this.data.currentIndex;
    const canvas = this.data.canvases[index];
    wx.canvasToTempFilePath({
      canvas: canvas,
      destWidth: canvas.width, destHeight: canvas.height,
      success: (res) => { wx.saveImageToPhotosAlbum({ filePath: res.tempFilePath, success: () => wx.showToast({ title: '保存成功' }) }); }
    });
  },

  downloadModel(url: string, fileName: string): Promise<string> {
    return new Promise((resolve, reject) => {
      const fs = wx.getFileSystemManager();
      const targetPath = `${wx.env.USER_DATA_PATH}/${fileName}`;
      try { fs.accessSync(targetPath); resolve(targetPath); } 
      catch (e) { wx.downloadFile({ url, success: (res) => { if (res.statusCode === 200) { fs.saveFileSync(res.tempFilePath, targetPath); resolve(targetPath); } }, fail: reject }); }
    });
  },

  ctcDecode(data: Float32Array) {
    let str = ""; let lastIdx = -1;
    for (let i = 0; i < 80; i++) {
      let maxVal = 0, maxIdx = 0;
      for (let j = 0; j < 6625; j++) { if (data[i * 6625 + j] > maxVal) { maxVal = data[i * 6625 + j]; maxIdx = j; } }
      if (maxIdx > 0 && maxIdx !== lastIdx) { str += char_dict[maxIdx - 1] || ""; }
      lastIdx = maxIdx;
    }
    return str;
  },

  isSensitive(text: string) {
    // --- 1. 获取用户自定义配置 (如果没有则使用默认全开) ---
    const config = wx.getStorageSync('mask_config') || {
      maskName: true, maskId: true, maskPhone: true, maskAddress: true
    };

    // --- 2. 模式定义 ---
    const idCardReg = /(^\d{15}$)|(^\d{17}(\d|X|x)$)/;
    const phoneReg = /1[3-9]\d{9}/;
    const bankReg = /\d{16,19}/;

    let entityLabel = ''; 
    let splitIdx = -1;

    // --- 3. 联动判定 ---
    // A. 判定身份证/银行卡 (配置项: maskId)
    if (config.maskId && (idCardReg.test(text) || (text.length >= 15 && /\d{10,}/.test(text)) || bankReg.test(text))) {
      entityLabel = 'ID';
    } 
    // B. 判定手机号 (配置项: maskPhone)
    else if (config.maskPhone && phoneReg.test(text)) {
      entityLabel = 'PHONE';
    } 
    // C. 判定姓名 (配置项: maskName)
    else if (config.maskName) {
      const nameLabels = ['姓名', '名字', '联系人', '收货人', 'Name'];
      if (nameLabels.some(l => text.includes(l))) entityLabel = 'NAME';
    }
    
    // D. 判定地址 (配置项: maskAddress)
    if (config.maskAddress && !entityLabel) {
      const addrLabels = ['地址', '住址', '所在地', 'Address'];
      if (addrLabels.some(l => text.includes(l))) entityLabel = 'ADDRESS';
    }

    // --- 4. 确定打码起始点 ---
    if (entityLabel) {
      const separators = ['：', ':', ' '];
      for (let sep of separators) {
        const idx = text.indexOf(sep);
        if (idx !== -1) { splitIdx = idx + 1; break; }
      }
      if (splitIdx === -1) {
        const labels = ['姓名', '名字', '联系人', '地址', '证件号', '身份证'];
        for (let l of labels) { 
          if (text.startsWith(l)) { 
            splitIdx = l.length; break; 
          } 
        }
      }
      if (splitIdx === -1) splitIdx = 0;
      return { sensitive: true, splitIdx };
    }

    return { sensitive: false, splitIdx: 0 };
  },
});