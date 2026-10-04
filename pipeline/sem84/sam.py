import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps


class MobileSam:
    """MobileSAM ONNX wrapper. Scores are model confidences, not annotation accuracy."""

    def __init__(self, models_dir, threads=4):
        o = ort.SessionOptions()
        o.intra_op_num_threads = threads
        o.inter_op_num_threads = 1
        p = ['CPUExecutionProvider']
        self.enc = ort.InferenceSession(str(models_dir / 'mobile_sam_image_encoder.onnx'), o, providers=p)
        self.dec = ort.InferenceSession(str(models_dir / 'sam_mask_decoder_multi.onnx'), o, providers=p)

    def embed(self, crop):
        h, w = crop.shape
        s = 1024 / max(h, w)
        nw, nh = round(w * s), round(h * s)
        im = ImageOps.autocontrast(Image.fromarray(crop).convert('RGB'), cutoff=0.25)
        im = im.resize((nw, nh), Image.Resampling.BILINEAR)
        emb = self.enc.run(None, {'input_image': np.asarray(im, dtype=np.float32)})[0]
        return {'emb': emb, 'h': h, 'w': w, 'sx': nw / w, 'sy': nh / h}

    def decode(self, e, x, y):
        feeds = {
            'image_embeddings': e['emb'],
            'point_coords': np.array([[[x * e['sx'], y * e['sy']], [0, 0]]], dtype=np.float32),
            'point_labels': np.array([[1, -1]], dtype=np.float32),
            'mask_input': np.zeros((1, 1, 256, 256), dtype=np.float32),
            'has_mask_input': np.zeros(1, dtype=np.float32),
            'orig_im_size': np.array([e['h'], e['w']], dtype=np.float32),
        }
        logits, scores, _ = self.dec.run(None, feeds)
        return logits[0], scores[0]
