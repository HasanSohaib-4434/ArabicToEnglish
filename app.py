import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        batch_size = tf.shape(q)[0]
        q = self.split_heads(self.wq(q), batch_size)
        k = self.split_heads(self.wk(k), batch_size)
        v = self.split_heads(self.wv(v), batch_size)
        attn_output = tf.nn.softmax(tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.depth, tf.float32)))
        attn_output = tf.matmul(attn_output, v)
        attn_output = tf.transpose(attn_output, perm=[0, 2, 1, 3])
        return self.dense(tf.reshape(attn_output, (batch_size, -1, self.d_model)))

class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x):
        attn_output = self.attention(x, x, x)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        return self.layernorm2(out1 + ffn_output)

class TransformerDecoder(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff):
        super(TransformerDecoder, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.attention1 = MultiHeadAttention(d_model, num_heads)
        self.attention2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, enc_output):
        attn1 = self.attention1(x, x, x)
        out1 = self.layernorm1(x + attn1)
        attn2 = self.attention2(out1, enc_output, enc_output)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        return self.layernorm3(out2 + ffn_output)

@tf.keras.utils.register_keras_serializable()
class Transformer(tf.keras.Model):
    def __init__(self, vocab_size, d_model, num_heads, dff, max_len):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.max_len = max_len

        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, input_length=max_len)
        self.encoder = TransformerEncoder(d_model, num_heads, dff)
        self.decoder = TransformerDecoder(d_model, num_heads, dff)
        self.final_layer = tf.keras.layers.Dense(vocab_size, activation='softmax')

    def call(self, inputs):
        enc_output = self.encoder(self.embedding(inputs))
        dec_output = self.decoder(self.embedding(inputs), enc_output)
        return self.final_layer(dec_output)

model = load_model("transformer_model", custom_objects={"Transformer": Transformer, "TransformerEncoder": TransformerEncoder, "TransformerDecoder": TransformerDecoder, "MultiHeadAttention": MultiHeadAttention})

with open('arabic_tokenizer.pkl', 'rb') as f:
    arabic_tokenizer = pickle.load(f)
with open('english_tokenizer.pkl', 'rb') as f:
    english_tokenizer = pickle.load(f)

def generate_translation(input_text):
    sequence = arabic_tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(sequence, maxlen=50, padding='post')
    prediction = model.predict(padded_sequence)
    predicted_sequence = np.argmax(prediction, axis=-1)
    translated_text = ' '.join(english_tokenizer.index_word.get(idx, '') for idx in predicted_sequence[0] if idx > 0)
    return translated_text

st.title("Arabic to English Translation")
st.write("Enter Arabic text below, and the model will translate it to English.")

input_text = st.text_area("Arabic Input", "")

if st.button("Translate"):
    if input_text.strip():
        translation = generate_translation(input_text)
        st.subheader("Translated English Text:")
        st.write(translation)
    else:
        st.warning("Please enter Arabic text to translate.")

st.markdown("---")
st.markdown("👨‍💻 Built with TensorFlow & Streamlit")
