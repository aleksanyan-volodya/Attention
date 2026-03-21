import streamlit as st
import sys

# Add emotions module to path
sys.path.append("emotions")
from emotions.config import *

sys.path.append(".")
from emotions.app_binary_helpers import build_pretrained_binary_model, validate_transformer_dimensions, train_custom_binary_model
from emotions.train import predict_sentiment, explain_prediction, load_model, load_vocabulary


@st.cache_resource
def load_binary_model():
    """Load the binary sentiment model and vocabulary (cached)."""
    return build_pretrained_binary_model()

def render_home() -> None:
    """Render the main page."""
    st.header("Main Page")
    st.write("Authors: Volodya Aleksanyan, Paul Gautier")
    st.info("Welcome! Use the sidebar to switch between sections.")


def render_binary_emotion() -> None:
    """Render the binary emotion recognition section with 2 workflows.
    
    Workflow 1 uses the pretrained model.
    Workflow 2 lets the user train a smaller custom model on CPU
    """
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

        # Run prediction + token importance in one pass
        with st.spinner("Analyzing text..."):
            label, confidence, top_tokens = explain_prediction(
                user_text, model, vocab, DEVICE,
                max_length=MAX_SEQ_LENGTH,
                top_k=5,
            )
            # Also get class probabilities for the breakdown line
            _, _, probs = predict_sentiment(
                user_text, model, vocab, DEVICE, max_length=MAX_SEQ_LENGTH
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

        # Show which tokens contributed most
        st.subheader("Most influential tokens")
        st.caption(
            "Importance is computed with gradient x input: how much each token "
            "pulled the model toward this prediction (normalized, 1.0 = most important)."
        )
        for token, score in top_tokens:
            bar = int(score * 20)  # scale to 20-char bar
            st.text(f"{token:<20} {'█' * bar} {score:.2f}")


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
