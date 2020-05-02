# ...continuing from above
import cv2
import numpy as np
import onnxruntime as ort

if __name__ == '__main__':
    onnx_path = "/Users/tony/PycharmProjects/pytorch-train/checkpoints/landscape/DCGAN-wikiart-0.0002-Adam-default-#02-24-14/export.onnx"
    ort_session = ort.InferenceSession(onnx_path)

    outputs = ort_session.run(None,
                              {'input_z': np.random.randn(10, 100, 1, 1).astype(np.float32)})
    image_rgb = ((outputs[0] + 1) / 2) * 255
    image_rgb = image_rgb.astype(np.uint8)
    image_rgb = image_rgb[:, ::-1, :, :]
    for i in range(image_rgb.shape[0]):
        cv2.imwrite("{}.png".format(i), image_rgb[i].transpose([1, 2, 0]))
