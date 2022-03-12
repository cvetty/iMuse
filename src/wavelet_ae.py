import tensorflow as tf
from tensorflow.keras import Model
from wavelet_encoder import WaveletEncoder
from wavelet_decoder import WaveletDecoder

from utils import wct, preprocess_feat, get_style_correlation_transform

class WaveletAE(Model):
    def __init__(self):
        super().__init__()
        self._name = 'WaveletAE'
        self.encoder = WaveletEncoder()
        self.decoder = WaveletDecoder()
        self.load_weights('../weights/wavelet_autoencoder')

    def call(self, inputs, trainable=False):
        x, skips, _ = self.encoder(inputs)
        output, _ = self.decoder(x, skips)

        return output

    def transfer(self, content_img, style_img, encoder_transfer=True, skips_transfer=True, decoder_transfer=True, stylization_coeffs={'encoder': 1, 'decoder': 1, 'skips': 1}):
        style_features, style_skips = self.get_features(style_img)
        x, content_skips, _ = self.encoder(content_img, style_features['encoder'] if encoder_transfer else None, stylization_coeffs['encoder'])

        if skips_transfer:
            for key in content_skips.keys():
                for i in range(3):
                    content_skips[key][0] = tf.map_fn(
                        lambda x: wct(x[0], x[1], stylization_coeffs['skips']),
                        (content_skips[key][0], style_skips[key][0]),
                        dtype=tf.float32
                    )

        out, _ = self.decoder(x, content_skips, style_features['decoder'] if decoder_transfer else None, stylization_coeffs['decoder'])
        out = tf.clip_by_value(out, 0, 1)

        return out

    def get_features(self, inputs):
        encoder_out, skips, encoder_feat = self.encoder(inputs)
        _, decoder_feat = self.decoder(encoder_out, skips)

        features = {
            'encoder': encoder_feat,
            'decoder': decoder_feat,
        }

        return features, skips

    def get_style_correlations(self, inputs, blocks=['block1', 'block2', 'block3', 'block4'], ede=True, normalize=True):
        _, _, encoder_feat = self.encoder(inputs)
        correlations = []
        means = []

        def process_correlation(feature_map, normalize=normalize):
            feat, _ = preprocess_feat(feature_map, center=True)
            feat = tf.matmul(feat, feat, transpose_b=True) / (feat.shape[1] - 1)

            if normalize:
                feat = feat - tf.reduce_mean(feat, 0)
                feat = feat / tf.reduce_max(tf.abs(feat))

            return feat

        def process_feat(feat):
            if ede:
                return get_style_correlation_transform(feat, normalize=normalize)
            else:
                return process_correlation(feat)

        for block in blocks:
            corr = tf.map_fn(process_feat, encoder_feat[block])
            mean = tf.map_fn(lambda feat: preprocess_feat(feat, center=False)[1], encoder_feat[block])
            correlations.append(corr)
            means.append(mean)

        return correlations, means