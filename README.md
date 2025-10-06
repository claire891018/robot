
# 🤖 Robot Car System — 模組簡介

本系統透過電腦端作為「大腦」，負責語音理解、影像感知、情緒判斷與決策，再透過 Wi-Fi 控制小車行動。小車本身保有基本避障與安全反射能力（急停、看門狗、近距離防撞）。

---

## 模組清單

### 1. `listener.py` （聆聽 / 語音轉文字）

* **功能** ：將麥克風輸入的聲音轉換成文字。
* **模型/技術** ：ASR 模型 （預設 openai/whisper-large-v3）。
* **輸出** ：`{text, confidence, lang, timestamp}`

---

### 2. `speaker.py` （說話 / 文字轉語音）

* **功能** ：將文字轉成語音播放，支援基本語氣。
* **模型/技術** ：TTS 模型 （預設 MediaTek-Research/BreezyVoice）。
* **輸入** ：`{text, tone, speed}`
* **備註** ：支援中斷與佇列。

---

### 3. `vision.py` （影像 / 場景感知）

* **功能** ：處理相機影像，支援「尋找指定目標物件」。
* **模型/技術** ：預設 Ultralytics/YOLO11
* **輸出** ：`{found, bbox, center, distance_m, confidence}`

---

### 4. `navigator.py` （導航 / 控制產生）

* **功能** ：根據目標與影像資訊，計算速度命令 (`v,w`)，直到抵達或受阻。
* **模型/技術** ：
  * facebook/nwm (CVPR 2025)
* **輸出** ：`{status, v, w}`，狀態包含 `moving|blocked|arrived|lost_target`

---

### 5. `emotion.py` （情緒 / 人類情緒辨識）

* **功能** ：透過臉部或聲音分析人的情緒，僅影響對話語氣，不影響物理控制。
* **模型/技術** ：
  * SER: emotion2vec/emotion2vec_plus_large
  * FER: trpakov/vit-face-expression
* **輸出** ：`{state: happy|sad|anxious|angry|neutral, confidence}`

---

### 6. `brain.py` （大腦 / 中央決策）

* **功能** ：整合所有模組，負責意圖判斷、狀態機管理與任務仲裁。
* **特點** ：
  * 所有決策由 brain 完成
  * 負責「任務分流」：導航任務 vs 對話任務
* 維護狀態：`idle, chatting, navigating, blocked, arrived, estop`

---

## `brain.py` 的逐步任務流程

1. **聆聽**
   * 呼叫 `listen`er，取得語音文字（含信心分數）。
2. **判斷**
   * 在 brain 內解析文字：
     * **導航任務** （例如「我想去告示牌右邊」）
     * **對話任務** （例如「你在做什麼？」）
     * **控制任務** （例如「停一下」）
3. **澄清**
   * 若資訊不足（例如沒說左/右） → 呼叫 `speaker `問「要停在左邊還是右邊？」
   * 等待 `listener` 回覆 → 更新任務資訊。
4. **情緒判斷**
   * 呼叫 `vision`（臉部 ROI）或 `listen`er（聲音片段）送給 `emotion` → 得到情緒。
   * 根據情緒決定說話 tone（calm, warm, happy…）。
5. **任務執行**
   * **導航任務** ：
   * 呼叫 `vision` 偵測目標 → 呼叫 `navigate` 計算 `v,w` → 經網路送給小車。
   * 持續監控小車回傳狀態（moving/blocked/arrived）。
   * **對話任務** ：
   * 不送導航指令（或保持 0 速），直接呼叫 `speaking` 回覆。
   * 允許「邊走邊聊」：導航迴圈不會因為 speaking 停止。
6. **異常處理**
   * 若小車回 `blocked` → 呼叫 `speaking` 詢問：「前方被擋住，要繞開嗎？」
   * 若小車超時（300ms 無回應） → 視為斷線，speaking 提示「通訊中斷，小車已停下」。
   * 若情緒偵測為 anxious/angry → 回應更溫和，提供更清楚的選項。
7. **完成/結束**
   * 到點：`navigate` 狀態=arrived → speaking：「已抵達告示牌右側」。
   * 重設狀態回 `idle`，等待下一個任務。
