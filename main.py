import sounddevice as sd
import mlx_whisper as mlx
from mlx_whisper.load_models import load_model
from scipy.signal import resample

model = "mlx-community/whisper-turbo"
loaded_model = load_model(model)
sample_rate = 16000 #Whisper only accepts audio files with a sample rate of 16kHz

"""
Models tested (replace to use): 
> mlx-community/whisper-large-v3-mlx #Accurate, but it's just too slow
> mlx-community/whisper-large-v3-turbo #Accurate, a bit quicker than the prev model, but still a bit slow.
> mlx-community/whisper-turbo #Accurate, the fastest of the three, but not by that much.
"""



#note to self - for play/rec, you need to have a wait after each one in order to record the correct amount of audio.
duration = 5 #secs
fs = 48000 #Hz
sd.default.channels = 1

myrecording = sd.rec(int(fs * duration), samplerate=fs)
sd.wait()
myrecording_flat = myrecording.flatten()

sd.play(myrecording_flat, samplerate=fs)
sd.wait()

myrecording_flat_resampled = resample(myrecording_flat, sample_rate*duration)
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html

text2 = mlx.transcribe(myrecording_flat_resampled, path_or_hf_repo = model)["text"]
print(text2)
