import tensorflow as tf
from tensorflow.keras import backend as K


class AdaptiveLesionFocalLoss(tf.keras.losses.Loss):
    """
    Adaptive Lesion Focal Loss (ALFL)

    Designed for medical image segmentation with small or sparse lesions
    (e.g., Breast Ultrasound images).

    Combines:
    - Tversky index (FP/FN control)
    - Focal modulation (hard example emphasis)
    - Adaptive lesion-aware boosting (small lesion sensitivity)
    """

    def __init__(
        self,
        alpha=0.3,
        beta=0.7,
        gamma=0.75,
        boost_scale=3.0,
        boost_decay=10.0,
        smooth=1e-6,
        reduction=tf.keras.losses.Reduction.AUTO,
        name="AdaptiveLesionFocalLoss",
    ):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.boost_scale = boost_scale
        self.boost_decay = boost_decay
        self.smooth = smooth

    def call(self, y_true, y_pred):
        # Ensure numerical stability
        y_pred = K.clip(y_pred, self.smooth, 1.0 - self.smooth)

        # Compute per-sample loss
        y_true = K.reshape(y_true, (K.shape(y_true)[0], -1))
        y_pred = K.reshape(y_pred, (K.shape(y_pred)[0], -1))

        tp = K.sum(y_true * y_pred, axis=1)
        fp = K.sum((1.0 - y_true) * y_pred, axis=1)
        fn = K.sum(y_true * (1.0 - y_pred), axis=1)

        # Tversky index
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        # Focal Tversky
        focal_tversky = K.pow(
            K.clip(1.0 - tversky, 1e-7, 1.0), self.gamma
        )

        # Lesion ratio (per-sample)
        lesion_pixels = K.sum(y_true, axis=1)
        total_pixels = K.cast(K.shape(y_true)[1], tf.float32)
        lesion_ratio = lesion_pixels / K.maximum(total_pixels, 1.0)

        boost = 1.0 + self.boost_scale * K.exp(
            -self.boost_decay * K.maximum(lesion_ratio, 1e-7)
        )

        return focal_tversky * boost

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
                "boost_scale": self.boost_scale,
                "boost_decay": self.boost_decay,
                "smooth": self.smooth,
            }
        )
        return config
