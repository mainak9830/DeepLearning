import librosa
import json
import os

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):

    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    #loop through all the sub_dirs

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        print(i)

        if dirpath is not dataset_path:
            #update mappings
            category = dirpath.split("\\")[-1]
            data["mappings"].append(category)

            #extract MFCCs

            for f in filenames:
                #get file
                file_path = os.path.join(dirpath, f)

                #load audio_file
                signal, sr = librosa.load(file_path)

                if len(signal) >= SAMPLES_TO_CONSIDER:

                    signal = signal[:SAMPLES_TO_CONSIDER]

                    MFCC = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)

                    #store data
                    data["labels"].append(i-1)
                    data["MFCCs"].append(MFCC.T.tolist())
                    data["files"].append(file_path)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

if __name__ == "__main__":
    prepare_dataset(DATASET_PATH, JSON_PATH)
