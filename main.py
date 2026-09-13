import sounddevice as sd
import mlx_whisper as mlx
"""
#note to self - for play/rec, you need to have a wait after each one in order to record the correct amount of audio.
duration = 5 #secs
fs = 48000 #Hz
sd.default.channels = 1
myrecording = sd.rec(int(fs * duration), samplerate=fs)
sd.wait()
sd.play(myrecording, samplerate=fs)
sd.wait()
print(myrecording.shape)
print(fs * duration)
for i in range(0,192001,48000):
    print(myrecording[i:i+48000])
"""


text = mlx.transcribe("TestAudio.m4a", path_or_hf_repo="models/large")['text']
print(text)