import sounddevice as sd
import mlx_whisper as mlx
import numpy as np
from mlx_whisper.load_models import load_model
from scipy.signal import resample
from pynput.keyboard import Key, Controller
from pynput import keyboard
from time import sleep

"""
Models tested (replace to use): 
> mlx-community/whisper-large-v3-mlx #Accurate, but it's just too slow
> mlx-community/whisper-large-v3-turbo #Accurate, a bit quicker than the prev model, but still a bit slow.
> mlx-community/whisper-turbo #Accurate, the fastest of the three, but not by that much.
"""

model = "mlx-community/whisper-turbo"
loaded_model = load_model(model)
sample_rate = 16000 #Whisper only accepts audio files with a sample rate of 16kHz

"press, release are a pair. type to use variables"


"""
#note to self - for play/rec, you need to have a wait after each one in order to record the correct amount of audio.
duration = 5 #secs

myrecording = sd.rec(int(fs * duration), samplerate=fs)
sd.wait()
myrecording_flat = myrecording.flatten()

sd.play(myrecording_flat, samplerate=fs)
sd.wait()

myrecording_flat_resampled = resample(myrecording_flat, sample_rate*duration)
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.resample.html
text2 = mlx.transcribe(myrecording_flat_resampled, path_or_hf_repo = model)["text"]
#Remember that mlx.transcribe() needs a 1D NumpyArray, so that's why we flattened it. We need (x,) rather than (x,1) which is 2D
print(text2)

#Basically yeah, this only works for a set length of audio, not variable. If I want to do a push-to-talk style, we have to use an audio stream that constantly appends
# to an array.
"""

#<<< START of Audio input functions >>>
audio_blocks = []
fs = 48000 #Hz
sd.default.channels = 1

recording = False

def callback(indata, frames, time, status): #the function that is called by the InputStream object every time it records a block
    global audio_blocks
    audio_blocks.append(indata)

stream = sd.InputStream(samplerate=fs, channels=sd.default.channels, callback=callback)
#<<< END of Audio input functions >>>

#<<< START of Hotkey functions >>>
cmd_held = False #base state
brack_held = False #base state

def on_press(key):
    global cmd_held
    global brack_held
    try:

        if key == Key.cmd_r:
            cmd_held = True

        if key.char == "]":
            brack_held = True

        if cmd_held == True:
            if key.char == "§":
                print("Both CMD and § are pressed. Killing program.")
                return False

        update_recording()

    except:
        pass


def on_release(key):
    global cmd_held
    try: 
        if key == Key.cmd_r:
            cmd_held = False

        if key.char == ']':
            brack_held = False

        update_recording()
    except:
        pass

#<<< END of Hotkey functions >>>


def update_recording():
    global recording
    global audio_blocks

    should_be_recording = cmd_held and brack_held

    #if recording and should_be_recording: Just done for understanding
        #recording = True

    if recording and not should_be_recording:
        recording = False
        print("Recording stopped...")
        stream.stop()
        OneD_audio_blocks = np.concatenate(audio_blocks, axis = 0)
        #for future reference, np.concatenate takes a list of lists and turns it into one 2D numpy array.
        OneD_audio_blocks_flattened = OneD_audio_blocks.flatten()
        n = len(OneD_audio_blocks_flattened)
        duration = n/fs
        OneD_audio_blocks_flattened_resampled = resample(OneD_audio_blocks_flattened, int(duration*16000))
        speech_to_text(OneD_audio_blocks_flattened_resampled)


    elif not recording and should_be_recording:
        recording = True
        print("Recording started...")
        audio_blocks = []
        stream.start()

def speech_to_text(audio_to_transcribe):
    print("Transcribing...")
    text = mlx.transcribe(audio_to_transcribe, path_or_hf_repo = model)["text"]
    print(text)


with keyboard.Listener(on_press = on_press, on_release = on_release) as listener: #Starts and stops listener in one line - Context Managing
    listener.join()
    #^^keeps the listener object active until conditions are met.