import sounddevice as sd
import mlx_whisper as mlx
from mlx_whisper.load_models import load_model
from scipy.signal import resample
from pynput.keyboard import Key, Controller
from pynput import keyboard
from time import sleep

model = "mlx-community/whisper-turbo"
loaded_model = load_model(model)
sample_rate = 16000 #Whisper only accepts audio files with a sample rate of 16kHz

"press, release are a pair. type to use variables"


"""
Models tested (replace to use): 
> mlx-community/whisper-large-v3-mlx #Accurate, but it's just too slow
> mlx-community/whisper-large-v3-turbo #Accurate, a bit quicker than the prev model, but still a bit slow.
> mlx-community/whisper-turbo #Accurate, the fastest of the three, but not by that much.
"""


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

"""

cmd_held = False #base state

def on_press(key):
    global cmd_held
    print(key)
    if key == Key.cmd_r:
        cmd_held = True
        print(cmd_held)
    if cmd_held == True:
        try:
            if key.char == "]":
                print("Both CMD and ] are pressed")
             
            elif key.char == "§":
                print("Both CMD and § are pressed. Killing program.")
                return False

        except:
            pass



def on_release(key):
    global cmd_held
    if key == Key.cmd_r:
        cmd_held = False

with keyboard.Listener(on_press = on_press, on_release = on_release) as listener: #Starts and stops listener in one line - Context Managing
    listener.join()
    #^^keeps the listener object active until conditions are met.