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

model = "mlx-community/whisper-turbo"

text1 = mlx.transcribe("TestAudio1.m4a", path_or_hf_repo = model)['text']
print(text1)


text2 = mlx.transcribe("TestAudio2.m4a", path_or_hf_repo = model)['text']
print(text2)

"""
Models tested (replace to use): 
> mlx-community/whisper-large-v3-mlx #Accurate, but it's just too slow
> mlx-community/whisper-large-v3-turbo #Accurate, a bit quicker than the prev model, but still a bit slow.
> mlx-community/whisper-turbo #Accurate, the fastest of the three, but not by that much.
"""