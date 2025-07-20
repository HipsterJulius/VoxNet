# -*- coding: utf-8 -*-

import os
import time
from nets.voxNet import VoxNet
import tensorflow as tf
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


SHARED_DIR = "/shared"
VOXEL_SRC = os.path.join(SHARED_DIR, "voxel.npy")
PREDICTION_PATH_DST = os.path.join(SHARED_DIR, "prediction.txt")
DONE_PATH = os.path.join(SHARED_DIR, "done.txt")

MODEL_DIR = "/workspace/VoxNet/logs"
model = VoxNet()
classifier = tf.estimator.Estimator(model_fn=model.core, model_dir=MODEL_DIR)

print("Modell geladen...")
print("VoxNet eval runner gestartet...")
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')
while True:
    if os.path.exists(DONE_PATH):
        print("Stoppsignal empfangen.")
        os.remove(DONE_PATH)
        break 

    if os.path.exists(VOXEL_SRC):
        try:
            #print("Voxel-Datei entdeckt starte Inferenz...")

            # Eingabe laden
            voxel = np.load(VOXEL_SRC).astype(np.float32)
            voxel = np.expand_dims(voxel, axis=0)  # (1, 32, 32, 32)

            input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"OccuGrid_input": voxel},
                num_epochs=1,
                shuffle=False
            )

            # Vorhersage
            predictions = classifier.predict(input_fn=input_fn)
            pred = next(predictions)
            label_id = int(pred['pred_cls'])

            # Ergebnis speichern
            with open(PREDICTION_PATH_DST, "w") as f:
                f.write(str(label_id))
            #print({label_id})

            # Eingabedatei entfernen für nächsten Zyklus
            os.remove(VOXEL_SRC)

        except Exception as e:
            print("Fehler beim Verarbeiten:", e)
    else:
        time.sleep(0.1)
