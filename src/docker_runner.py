# -*- coding: utf-8 -*-

import os
import time
import shutil

SHARED_DIR = "/shared"
VOXEL_SRC = os.path.join(SHARED_DIR, "voxel.npy")
VOXEL_DEST = "/workspace/VoxNet/voxel.npy"

PREDICTION_PATH_SRC = "/workspace/VoxNet/src/prediction.txt"
PREDICTION_PATH_DST = os.path.join(SHARED_DIR, "prediction.txt")

print("VoxNet eval runner gestartet...")

while True:
    if os.path.exists(VOXEL_SRC):
        try:
            print("Voxel-Datei entdeckt – starte Inferenz...")

            # Voxel-Datei verschieben
            shutil.move(VOXEL_SRC, VOXEL_DEST)

            os.system("/workspace/VoxNet/venv/bin/python /workspace/VoxNet/src/eval_custom.py")

            # Ergebnis übertragen
            if os.path.exists(PREDICTION_PATH_SRC):
                shutil.move(PREDICTION_PATH_SRC, PREDICTION_PATH_DST)
                print("Prediction gespeichert.")
            else:
                print("Keine prediction.txt gefunden.")

        except Exception as e:
            print("Fehler beim Verarbeiten:", e)

    time.sleep(0.1)
