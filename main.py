import sounddevice as sd
import mlx_whisper as mlx
import numpy as np
from scipy.signal import resample
from pynput.keyboard import Key, Controller
from pynput import keyboard

"""
Models tested (replace to use): 
> mlx-community/whisper-large-v3-mlx #Accurate, but it's just too slow
> mlx-community/whisper-large-v3-turbo #Accurate, a bit quicker than the prev model, but still a bit slow.
> mlx-community/whisper-turbo #Accurate, the fastest of the three, but not by that much.
"""

#*** Variables ***#

model = "mlx-community/whisper-turbo"
sample_rate = 16000 #Whisper only accepts audio files with a sample rate of 16kHz

fs = 48000 #Hz

recording = False

cmd_held = False #base state
brack_held = False #base state

min_duration = 0.5

audio_blocks = []

#*** END Variables ***#

#Loading model into cache to speed up transcription
def model_loader():
    cache_loader = np.zeros((sample_rate, ), dtype = np.float32)
    mlx.transcribe(cache_loader, path_or_hf_repo = model)
    print("Model ready...")


#<<< START of Audio input functions >>>
def callback(indata, frames, time, status): #the function that is called by the InputStream object every time it records a block
    global audio_blocks
    audio_blocks.append(indata.copy())
#<<< END of Audio input functions >>>



#<<< START of Hotkey functions >>>

def on_press(key):
    global cmd_held
    global brack_held

    char = getattr(key, "char", None)

    if key == Key.cmd_r:
        cmd_held = True

    if char == "]":
        brack_held = True

    if cmd_held == True:
        if char == "§":
            print("Both CMD and § are pressed. Killing program.")
            return False

    update_recording()



def on_release(key):
    global cmd_held
    global brack_held
    char = getattr(key, "char", None)

    if key == Key.cmd_r:
        cmd_held = False
        update_recording()

    if char == ']':
        brack_held = False
        update_recording()

#<<< END of Hotkey functions >>>

#<<< START of recording state checker >>>
def update_recording():
    global recording
    global audio_blocks
    should_be_recording = cmd_held and brack_held

    if recording and not should_be_recording:
        recording = False
        print("Recording stopped...")
        stream.stop()

        valid_audio, duration, processed_audio = check_audio_length(audio_blocks)

        if valid_audio:
            resampled_OneD_audio_blocks = resample(processed_audio, int(duration * sample_rate))
            #^^ Resampling 48kHz audio from user to 16kHz audio for Whisper to work.
            speech_to_text(resampled_OneD_audio_blocks)
        
        return



    elif should_be_recording and not recording:
        recording = True
        print("Recording started...")
        audio_blocks = []
        stream.start()

def check_audio_length(audio_to_check):


    if len(audio_to_check) > 0:

        duration, joined_flattened_audio = joining_audio(audio_to_check)

        if duration >= min_duration:
            return True, duration, joined_flattened_audio

        
        return False, None, None


def joining_audio(audio_to_join):
        concatenated_audio = np.concatenate(audio_to_join, axis = 0)
        flattened_audio = concatenated_audio.flatten()
        length_flattened = len(flattened_audio)
        duration = length_flattened/fs

        return duration, flattened_audio

#<<< END of processing and recording state checker>>>


#Transcription Function
def speech_to_text(audio_to_transcribe):
    print("Transcribing...")
    text = mlx.transcribe(audio_to_transcribe, path_or_hf_repo = model)["text"]
    print("Text: ")
    print(text)
    user_keyboard.type(text)


if __name__ == "__main__":
    user_keyboard = Controller()
    model_loader()

    stream = sd.InputStream(samplerate=fs, channels = 1, callback=callback)

    with keyboard.Listener(on_press = on_press, on_release = on_release) as listener: #Starts and stops listener in one line - Context Managing
        listener.join()
        #^^keeps the listener object active until conditions are met.

                                                                                                         