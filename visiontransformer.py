import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Flatten

# Transformer Block
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"),
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    
    def call(self, inputs, training):
        # Self-attention block
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)  # Residual connection + norm
        
        # Feed-forward block
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection + norm
        
        return out2


# Patch Embedding for Vision Transformer
class PatchEmbedding(Layer):
    def __init__(self, patch_size, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = Dense(embed_dim)
    
    def call(self, images):
        # images shape: (batch_size, height, width, channels)
        batch_size = tf.shape(images)[0]
        
        # Extract patches
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1,1,1,1],
            padding='VALID'
        )  # shape: (batch_size, num_patches_h, num_patches_w, patch_size*patch_size*channels)
        
        # Flatten patches
        patches = tf.reshape(patches, [batch_size, -1, patches.shape[-1]])  # (batch_size, num_patches, patch_flattened_dim)
        
        # Linear projection to embedding dim
        embeddings = self.proj(patches)  # (batch_size, num_patches, embed_dim)
        
        return embeddings


# Vision Transformer model skeleton
class VisionTransformer(tf.keras.Model):
    def __init__(self, num_patches, embed_dim, num_heads, ff_dim, num_layers, num_classes, patch_size):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, embed_dim)
        
        self.pos_embedding = tf.Variable(tf.random.normal([1, num_patches, embed_dim]))
        
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        
        self.flatten = Flatten()
        self.dropout = Dropout(0.1)
        self.classifier = Dense(num_classes, activation='softmax')
    
    def call(self, images, training=False):
        x = self.patch_embedding(images)
        x = x + self.pos_embedding
        
        for block in self.transformer_blocks:
            x = block(x, training=training)
        
        x = self.flatten(x)
        x = self.dropout(x, training=training)
        logits = self.classifier(x)
        return logits
