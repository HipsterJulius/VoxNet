import sys
import tensorflow as tf
import numpy as np
import os

from nets.voxNet import VoxNet

INPUT_FILE = "/shared/voxel.npy"
OUTPUT_FILE = "/shared/prediction.txt"
MODEL_DIR = "/workspace/VoxNet/logs"

def main(argv):
    
    print("Es geht los!!")
    
    # VoxelNet-Model
    model = VoxNet()
    classifier = tf.estimator.Estimator(model_fn=model.core, model_dir=MODEL_DIR)

    # Eingabe laden
    voxel = np.load(INPUT_FILE).astype(np.float32)
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

    # Speichern
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(str(label_id))

    print("Vorhersage:", label_id)


if __name__ == "__main__":
    tf.app.run(main=main, argv=sys.argv)
