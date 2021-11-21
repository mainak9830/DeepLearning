import numpy as np
from tensorflow import keras
import librosa
MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050
class _Keyword_Spotting_Service:
    model = None
    _mappings = [
        "down",
        "go",
        "left",
        "no",
        "off",
        "on",
        "right",
        "stop",
        "up",
        "yes"
    ]

    _instance = None

    def predict(self, file_path):

        #extract MFCCs
        MFCCs = self.preprocess(file_path)

        #convert 2d into 4d
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]

        #make_prediction

        predictions = self.model.predict(MFCCs) #[[0.1,...,0.9]]
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mappings[predicted_index]

        return predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, hop_length=512, n_fft=2048):
        #load audio file
        signal, sr = librosa.load(file_path)

        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)


        return MFCCs.T




def Keyword_Spotting_Service():

    if(_Keyword_Spotting_Service._instance is None):
        _Keyword_Spotting_Service._instance = _Keyword_Spotting_Service()
        _Keyword_Spotting_Service.model = keras.models.load_model(MODEL_PATH)

    return _Keyword_Spotting_Service._instance


if __name__ == "__main__":

    kss = Keyword_Spotting_Service()
    print(kss.predict("test/up.wav"))