import streamlit as st
import torch
import sys

# Add emotions module to path
sys.path.append("emotions")
from emotions.config import *

sys.path.append(".")
from transformerNew import Transformer
from emotions.train import predict_sentiment, load_model, load_vocabulary


@st.cache_resource
def load_binary_model():
    """Load the binary sentiment model and vocabulary (cached)."""
    # Initialize transformer architecture
    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=NUM_CLASSES,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_length=MAX_SEQ_LENGTH,
        dropout=DROPOUT,
        pad_token_id=PAD_IDX,
        mask=False,
        encoder_only=True,
    ).to(DEVICE)

    # Load trained weights
    model = load_model(model, f"emotions/{MODEL_SAVE_PATH}", DEVICE)
    vocab = load_vocabulary(f"emotions/{VOCAB_SAVE_PATH}")

    return model, vocab

def render_home() -> None:
    """Render the main page."""
    st.header("Main Page")
    st.write("Authors: Volodya Aleksanyan, Paul Gautier")
    st.info("Welcome! Use the sidebar to switch between sections.")


def render_binary_emotion() -> None:
    """Render the binary emotion recognition section."""
    st.header("Binary Emotion Recognition")
    st.write("Task: negative / positive")

    st.markdown(
        """
        This section is for **binary sentiment/emotion recognition** on English text.
        The target model was trained on **IMDb movie reviews** from Hugging Face,
        and will output one of two labels: **positive** or **negative**.
        """
    )

    with st.expander("What this section is for"):
        st.write(
            "Use this page to test short or long English text snippets and get "
            "a binary sentiment prediction."
        )
        st.write("Expected labels: **positive** / **negative**")

    user_text = st.text_area(
        "Enter English text",
        placeholder="Example: I really enjoyed this movie and would watch it again.",
        height=160,
    )
    st.caption(f"Character count: {len(user_text)}")

    if st.button("Predict", type="primary"):
        if not user_text.strip():
            st.error("Please enter some text before clicking Predict.")
            return

        # Load model and vocabulary (cached)
        with st.spinner("Loading model..."):
            model, vocab = load_binary_model()

        # Run prediction
        with st.spinner("Analyzing text..."):
            label, confidence, probs = predict_sentiment(
                user_text, model, vocab, DEVICE
            )

        # Display results
        st.subheader("Prediction")
        if label == "Positive":
            st.success(f"**{label}** (confidence: {confidence:.1%})")
        else:
            st.error(f"**{label}** (confidence: {confidence:.1%})")

        # Show probability breakdown
        st.caption(
            f"Probabilities: Negative {probs[0]:.1%} | Positive {probs[1]:.1%}"
        )


def render_multiclass_emotion() -> None:
    """Render the multiclass emotion prediction section."""
    st.header("Multiclass Emotion Prediction")
    st.write("Task: sad, joy, fear, ...")
    st.write("Sorry, coming soon... ://")


def render_third_model() -> None:
    """Render the third-model placeholder section."""
    st.header("3rd Model")
    st.write("Sorry, coming soon... ://")


def main() -> None:
    """Render the app with basic section navigation."""
    st.set_page_config(page_title="Attention", layout="centered")
    st.title("Attention")

    app_mode = st.sidebar.selectbox(
        "Choose the app mode",
        (
            "Main Page",
            "Binary Emotion Recognition",
            "Multiclass Emotion Prediction",
            "3rd Model",
        ),
    )

    if app_mode == "Main Page":
        render_home()
    elif app_mode == "Binary Emotion Recognition":
        render_binary_emotion()
    elif app_mode == "Multiclass Emotion Prediction":
        render_multiclass_emotion()
    else:
        render_third_model()



if __name__ == "__main__":
    main()
