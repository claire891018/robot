import numpy as np
import torch
import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional
from collections import deque

# 忽略 SpeechBrain 的 deprecation 警告
warnings.filterwarnings("ignore", message="Module 'speechbrain.pretrained'")

@dataclass
class SpeakerSegment:
    """說話人片段"""
    speaker_id: int
    start_ts: float
    end_ts: float
    text: str
    confidence: float
    embedding: np.ndarray

class Diarizer:
    """
    即時說話人辨識模組
    
    實作論文提出的:
    1. X-vector embedding extraction (via SpeechBrain)
    2. Incremental clustering (cosine similarity threshold)
    3. Embedding update (EWMA)
    4. Temporal smoothing (5-second window)
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.75,
        ewma_alpha: float = 0.9,
        smoothing_window: float = 5.0,
        device: str = "cuda"
    ):
        print(f"[Diarizer] 初始化 (device={device})")
        
        self.threshold = similarity_threshold
        self.alpha = ewma_alpha
        self.window_sec = smoothing_window
        self.device = device
        
        # 說話人資料庫
        self.speakers: Dict[int, np.ndarray] = {}
        self.next_id = 1
        
        # 時間平滑用的歷史記錄
        self.history: deque = deque(maxlen=100)
        
        # 載入 SpeechBrain x-vector 模型
        try:
            # 使用新的 API (speechbrain.inference)
            try:
                from speechbrain.inference.speaker import EncoderClassifier
            except ImportError:
                # Fallback to old API
                from speechbrain.pretrained import EncoderClassifier
            
            print("[Diarizer] 載入 SpeechBrain x-vector 模型...")
            
            self.embedding_model = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-xvect-voxceleb",
                savedir="pretrained_models/spkrec-xvect-voxceleb",
                run_opts={"device": self.device}
            )
            
            print("[Diarizer] ✓ X-vector model loaded (SpeechBrain)")
            
        except Exception as e:
            print(f"[Diarizer] ✗ Embedding model 載入失敗: {e}")
            print("[Diarizer] 系統將無法正常運作")
            self.embedding_model = None
            raise
    
    def extract_embedding(self, audio_segment: np.ndarray, sr: int = 16000) -> np.ndarray:
        """
        提取音訊片段的 x-vector embedding
        
        Args:
            audio_segment: 音訊片段 (numpy array, int16 or float32)
            sr: 採樣率
            
        Returns:
            embedding vector (512-dim)
        """
        try:
            # 轉換為 float32 並正規化
            if audio_segment.dtype == np.int16:
                audio_f32 = audio_segment.astype(np.float32) / 32768.0
            else:
                audio_f32 = audio_segment.astype(np.float32)
            
            # 確保是 1D array
            if audio_f32.ndim > 1:
                audio_f32 = audio_f32.squeeze()
            
            # SpeechBrain 需要 torch tensor
            audio_tensor = torch.from_numpy(audio_f32).unsqueeze(0).to(self.device)
            
            # 提取 embedding
            with torch.no_grad():
                embedding = self.embedding_model.encode_batch(audio_tensor)
                embedding = embedding.squeeze().cpu().numpy()
            
            return embedding
            
        except Exception as e:
            print(f"[Diarizer] Embedding extraction 失敗: {e}")
            import traceback
            traceback.print_exc()
            # 返回 dummy embedding
            return np.random.randn(512).astype(np.float32)
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """計算兩個 embedding 的 cosine similarity"""
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(emb1, emb2) / (norm1 * norm2))
    
    def assign_speaker(
        self,
        embedding: np.ndarray,
        timestamp: float
    ) -> int:
        """
        分配說話人 ID (Incremental Clustering)
        
        Args:
            embedding: 當前片段的 embedding
            timestamp: 時間戳記
            
        Returns:
            speaker_id
        """
        if not self.speakers:
            # 第一個說話人
            self.speakers[self.next_id] = embedding
            speaker_id = self.next_id
            self.next_id += 1
            return speaker_id
        
        # 計算與所有已知說話人的相似度
        similarities = {}
        for spk_id, spk_emb in self.speakers.items():
            sim = self.cosine_similarity(embedding, spk_emb)
            similarities[spk_id] = sim
        
        # 找最相似的
        best_id = max(similarities, key=similarities.get)
        best_sim = similarities[best_id]
        
        # 判斷是否為同一人
        if best_sim >= self.threshold:
            # 更新 embedding (EWMA)
            self.speakers[best_id] = (
                self.alpha * self.speakers[best_id] + 
                (1 - self.alpha) * embedding
            )
            return best_id
        else:
            # 新說話人
            self.speakers[self.next_id] = embedding
            speaker_id = self.next_id
            self.next_id += 1
            return speaker_id
    
    def temporal_smoothing(
        self,
        speaker_id: int,
        timestamp: float
    ) -> int:
        """
        時間平滑 (防止短時間內頻繁切換)
        
        檢查最近 5 秒內的說話人，如果主要是另一個人，則修正
        """
        # 取得最近 window_sec 內的片段
        recent = [
            seg for seg in self.history
            if timestamp - seg.end_ts <= self.window_sec
        ]
        
        if len(recent) < 3:
            # 資料不足，不修正
            return speaker_id
        
        # 統計最近的說話人分布
        speaker_counts = {}
        for seg in recent:
            speaker_counts[seg.speaker_id] = speaker_counts.get(seg.speaker_id, 0) + 1
        
        # 如果當前 speaker_id 在最近很少出現，可能是誤判
        if speaker_id not in speaker_counts:
            # 取最常出現的
            most_common = max(speaker_counts, key=speaker_counts.get)
            if speaker_counts[most_common] >= len(recent) * 0.6:
                return most_common
        
        return speaker_id
    
    def process_utterance(
        self,
        audio_segment: np.ndarray,
        text: str,
        start_ts: float,
        end_ts: float,
        confidence: float
    ) -> Dict:
        """
        處理單一 utterance，返回帶有 speaker_id 的結果
        
        Args:
            audio_segment: 音訊片段 (int16 numpy array)
            text: 辨識文字
            start_ts: 開始時間
            end_ts: 結束時間
            confidence: ASR 信心度
            
        Returns:
            包含 speaker_id 的字典
        """
        # 1. 提取 embedding
        embedding = self.extract_embedding(audio_segment)
        
        # 2. 分配說話人
        speaker_id = self.assign_speaker(embedding, end_ts)
        
        # 3. 時間平滑
        speaker_id = self.temporal_smoothing(speaker_id, end_ts)
        
        # 4. 記錄到歷史
        segment = SpeakerSegment(
            speaker_id=speaker_id,
            start_ts=start_ts,
            end_ts=end_ts,
            text=text,
            confidence=confidence,
            embedding=embedding
        )
        self.history.append(segment)
        
        return {
            "type": "utterance_with_speaker",
            "speaker_id": speaker_id,
            "text": text,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "confidence": confidence
        }
    
    def reset(self):
        """重置說話人資料庫"""
        self.speakers = {}
        self.history.clear()
        self.next_id = 1
        print("[Diarizer] 說話人資料已重置")
        