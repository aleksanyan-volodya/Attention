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

    workflow_mode = st.radio(
        "Choose workflow",
        (
            "Use pretrained model",
            "Train a new model (CPU-friendly)",
        ),
    )
    
    if workflow_mode == "Use pretrained model":
        model_source = "pretrained"
    else:
        st.warning(
            "Training runs on CPU in your setup, so it can be slow"
            "Start with small values (for example 1-5 epochs). "
            "For 0.7 accuracy you need at least 9 epochs "
        )

        # Basic settings that are safe to edit for CPU training
        col1, col2 = st.columns(2)
        with col1:
            epochs = st.slider("Epochs", min_value=2, max_value=20, value=5)
            learning_rate = st.number_input(
                "Learning rate",
                min_value=1e-6,
                max_value=1e-2,
                value=5e-5,
                format="%.6f",
            )
            batch_size = st.select_slider(
                "Batch size",
                options=[8, 16, 32, 64],
                value=16,
            )
            max_seq_length = st.select_slider(
                "Max sequence length",
                options=[64, 128, 256, 384],
                value=128,
            )
        with col2:
            train_samples = st.slider(
                "Train subset size",
                min_value=500,
                max_value=10000,
                value=3000,
                step=500,
            )
            test_samples = st.slider(
                "Test subset size",
                min_value=500,
                max_value=5000,
                value=1000,
                step=500,
            )
            vocab_build_size = st.slider(
                "Vocabulary build samples",
                min_value=2000,
                max_value=25000,
                value=8000,
                step=1000,
            )
        
        # advaced parameters might be changed by the user later on.
        d_model, num_heads, num_layers = 128, 4, 2
        d_ff = max(d_model, d_model * num_layers)
        dropout = 0.2

        if not validate_transformer_dimensions(int(d_model), int(num_heads)):
            st.error("Invalid model settings: d_model must be divisible by num_heads.")

        elif st.button("Start training", type="secondary"):
            # TODO : add possibility to show time left or epochs OR EVEN A GAME HAHAAHAH
            with st.spinner("Training model on CPU. This may take several minutes..."):
                trained_model, trained_vocab, metrics = train_custom_binary_model(
                    epochs=epochs,
                    learning_rate=float(learning_rate),
                    batch_size=int(batch_size),
                    max_seq_length=int(max_seq_length),
                    train_samples=int(train_samples),
                    test_samples=int(test_samples),
                    vocab_build_size=int(vocab_build_size),
                    d_model=int(d_model),
                    num_heads=int(num_heads),
                    num_layers=int(num_layers),
                    d_ff=int(d_ff),
                    dropout=float(dropout),
                )
            st.session_state["custom_binary_model"] = trained_model
            st.session_state["custom_binary_vocab"] = trained_vocab
            st.session_state["custom_binary_max_seq_len"] = int(max_seq_length)
            st.session_state["custom_binary_metrics"] = metrics
            st.success(f"Training finished. Final test accuracy: {metrics['final_test_accuracy']:.2%}")

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
