lass SelfAttention(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.query_dense = tf.keras.layers.Dense(d_model)
        self.key_dense = tf.keras.layers.Dense(d_model)
        self.value_dense = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
        # Linear layers
        Q = self.query_dense(inputs)
        K = self.key_dense(inputs)
        V = self.value_dense(inputs)

        # Attention scores: Q · K^T / sqrt(d_k)
        d_k = tf.cast(tf.shape(K)[-1], tf.float32)
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(d_k)

        # Softmax to get attention weights
        attention_weights = tf.nn.softmax(scores, axis=-1)

        # Weighted sum of values
        output = tf.matmul(attention_weights, V)

        return output
