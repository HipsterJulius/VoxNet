# -*- coding: utf-8 -*-

import os
import time
import traceback
from nets.voxNet import VoxNet
import tensorflow as tf
import numpy as np
import absl.logging

absl.logging.set_verbosity(absl.logging.ERROR)
absl.logging.set_stderrthreshold('error')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

SHARED_DIR = "/shared"
VOXEL_SRC = os.path.join(SHARED_DIR, "voxel.npy")
PREDICTION_PATH_DST = os.path.join(SHARED_DIR, "prediction_voxnet.txt")
DONE_PATH = os.path.join(SHARED_DIR, "done_voxnet.txt")
READY_PATH = os.path.join(SHARED_DIR, "model_ready_voxnet.txt")
MODEL_DIR = "/workspace/VoxNet/logs"

# Vorherige Prediction entfernen
if os.path.exists(PREDICTION_PATH_DST):
    os.remove(PREDICTION_PATH_DST)

model = VoxNet()
classifier = tf.estimator.Estimator(model_fn=model.core, model_dir=MODEL_DIR)
print("VoxNet-Modell geladen.")

print("VoxNet eval runner gestartet...")
ready_flag_set = False
while True:
    if os.path.exists(DONE_PATH):
        print("Stoppsignal empfangen.")
        os.remove(DONE_PATH)
        break

    if os.path.exists(VOXEL_SRC) and not os.path.exists(PREDICTION_PATH_DST):
        # → Signal "bereit" erst setzen, wenn erste Anfrage eingeht
        if not ready_flag_set:
            with open(READY_PATH, "w") as f:
                f.write("ready")
            ready_flag_set = True
            print("Modell ist jetzt bereit (model_ready_voxnet.txt gesetzt).")

        try:
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

            if os.path.exists(PREDICTION_PATH_DST):
                os.remove(PREDICTION_PATH_DST)

            with open(PREDICTION_PATH_DST, "w") as f:
                f.write(str(label_id))

            os.remove(VOXEL_SRC)

        except Exception as e:
            print("Fehler beim Verarbeiten:")
            traceback.print_exc()
    else:
        time.sleep(0.05)
