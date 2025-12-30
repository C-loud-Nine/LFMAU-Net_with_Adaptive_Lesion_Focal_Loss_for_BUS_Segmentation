"""
LFMAUNET.py

Enhanced LFMAU-Net model for Breast Ultrasound Segmentation (BUS).

Features:
- EfficientNetB4 encoder with CBAM attention
- Refined attention gates in decoder
- Bilinear upsampling + SpatialDropout2D
- Reproducible seed utility
"""

import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, backend as K
from tensorflow.keras.applications import EfficientNetB4

# =============================
# Reproducibility utility
# =============================
def set_global_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# =============================
# CBAM Attention Block
# =============================
def cbam_block(x, ratio=16, kernel_size=3):
    """CBAM attention (channel + spatial)"""
    ch = K.int_shape(x)[-1]
    # Channel Attention
    avg_pool = layers.GlobalAveragePooling2D(keepdims=True)(x)
    max_pool = layers.GlobalMaxPooling2D(keepdims=True)(x)
    shared_mlp = tf.keras.Sequential([
        layers.Conv2D(ch // ratio, 1, activation='relu', use_bias=True),
        layers.Conv2D(ch, 1, use_bias=True)
    ])
    avg_out = shared_mlp(avg_pool)
    max_out = shared_mlp(max_pool)
    ca = layers.Activation('sigmoid')(avg_out + max_out)
    x = layers.Multiply()([x, ca])
    # Spatial Attention
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    sa = layers.Conv2D(1, kernel_size, padding='same', activation='sigmoid')(concat)
    return layers.Multiply()([x, sa])

# =============================
# Refined Attention Gate
# =============================
def refined_attention_gate(x, g, inter_channels):
    """Refined attention gate for skip connections"""
    x_refined = layers.Conv2D(inter_channels, 3, padding='same', activation='relu')(x)
    g_refined = layers.Conv2D(inter_channels, 3, padding='same', activation='relu')(g)
    theta_x = layers.Conv2D(inter_channels, 1, padding='same')(x_refined)
    phi_g = layers.Conv2D(inter_channels, 1, padding='same')(g_refined)
    add = layers.Add()([theta_x, phi_g])
    relu = layers.Activation('relu')(add)
    psi = layers.Conv2D(1, 1, padding='same', activation='sigmoid')(relu)
    return layers.Multiply()([x, psi])

# =============================
# Upsample Block
# =============================
def upsample_block(x, filters, size=2, kernel_size=3):
    x = layers.UpSampling2D(size=(size, size), interpolation='bilinear')(x)
    return layers.Conv2D(filters, kernel_size, activation='relu', padding='same')(x)

# =============================
# LFMAU-Net Segmentation Model
# =============================
def enhanced_segmentation_model(
    input_size=(224, 224, 3),
    num_seg_classes=1,
    dropout_rate=0.4,
    l2_lambda=1e-4
):
    """Build LFMAU-Net segmentation model"""
    base_model = EfficientNetB4(input_shape=input_size, include_top=False, weights='imagenet')

    # Freeze early layers
    for layer in base_model.layers[:30]:
        layer.trainable = False

    # Encoder skip connections
    s1 = cbam_block(base_model.get_layer('block2a_expand_activation').output, ratio=8, kernel_size=7)
    s2 = cbam_block(base_model.get_layer('block3a_expand_activation').output, ratio=16, kernel_size=7)
    s3 = cbam_block(base_model.get_layer('block4a_expand_activation').output, ratio=16, kernel_size=3)
    s4 = cbam_block(base_model.get_layer('block6a_expand_activation').output, ratio=16, kernel_size=3)
    bridge = cbam_block(base_model.get_layer('top_activation').output, ratio=16, kernel_size=3)

    # Decoder
    d1 = upsample_block(bridge, 512)
    att1 = refined_attention_gate(s4, d1, 256)
    d1 = layers.Concatenate()([d1, att1])
    d1 = layers.Conv2D(512, 3, activation='relu', padding='same')(d1)
    d1 = layers.BatchNormalization()(d1)
    d1 = layers.SpatialDropout2D(dropout_rate)(d1)

    d2 = upsample_block(d1, 256)
    att2 = refined_attention_gate(s3, d2, 128)
    d2 = layers.Concatenate()([d2, att2])
    d2 = layers.Conv2D(256, 3, activation='relu', padding='same')(d2)
    d2 = layers.BatchNormalization()(d2)
    d2 = layers.SpatialDropout2D(dropout_rate)(d2)

    d3 = upsample_block(d2, 128)
    att3 = refined_attention_gate(s2, d3, 64)
    d3 = layers.Concatenate()([d3, att3])
    d3 = layers.Conv2D(128, 3, activation='relu', padding='same')(d3)
    d3 = layers.BatchNormalization()(d3)
    d3 = layers.SpatialDropout2D(dropout_rate)(d3)

    d4 = upsample_block(d3, 64)
    att4 = refined_attention_gate(s1, d4, 32)
    d4 = layers.Concatenate()([d4, att4])
    d4 = layers.Conv2D(64, 3, activation='relu', padding='same')(d4)
    d4 = layers.BatchNormalization()(d4)
    d4 = layers.SpatialDropout2D(dropout_rate)(d4)

    seg_output = upsample_block(d4, 32)
    seg_output = layers.Conv2D(32, 3, activation='relu', padding='same')(seg_output)
    seg_output = layers.Conv2D(num_seg_classes, 1, activation='sigmoid', name='segmentation_output')(seg_output)

    return models.Model(inputs=base_model.input, outputs=seg_output)

# =============================
# Optional: Add weight decay
# =============================
def add_weight_decay(model, l2_lambda=1e-4):
    """Apply L2 weight decay to BatchNorm layers"""
    for layer in model.layers:
        if isinstance(layer, layers.BatchNormalization):
            layer.gamma_regularizer = regularizers.l2(l2_lambda)
            layer.beta_regularizer = regularizers.l2(l2_lambda)
    return model

# =============================
# Example usage
# =============================
if __name__ == "__main__":
    set_global_seed(42)
    model = enhanced_segmentation_model()
    model = add_weight_decay(model)
    model.summary()
