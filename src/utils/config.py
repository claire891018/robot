from dotenv import load_dotenv
import os  
load_dotenv()

ASR_LANG = "zh"
CAMERA_INDEX = 0
FOV_DEG = 90
TARGET_REAL_HEIGHT_M = 0.10
FRAME_WIDTH = 1280
KP_V = 0.35
KP_W = 0.6
GOAL_TOL = 0.35
MAX_V = 0.6
MAX_W = 1.2

# OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_URL = os.getenv("OLLAMA_URL")
OLLAMA_MODEL = "gemma3:27b"
DEPTH_MODE = "heuristic"
MIDAS_WEIGHTS = "weights/midas_small.onnx"
MIDAS_INPUT_SIZE = (256, 256)
TTS_IP=os.getenv("TTS_IP")

SYSPROMPT = """
你來自台灣，是一個會講中文的服務機器人（可以臺灣國語、中英夾雜），回答要簡短、中二黑色幽默、口語化，適合唸出來聽。
注意不要加任何 emoji，不要一次講太長，盡量在三句話以內。
如果聽到的是移動或導航相關指令，可以簡短回覆「好，我來試試看」之類的話，
同時把指令內容留給大腦做視覺與導航判斷。
"""

