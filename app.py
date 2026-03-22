import streamlit as st
import sys

# Add emotions module to path
sys.path.append("emotions")
from emotions.config import *

sys.path.append(".")
from emotions.app_binary_helpers import (
    build_pretrained_binary_model, 
    validate_transformer_dimensions, 
    train_custom_binary_model,
)
from emotions.train import predict_sentiment, explain_prediction
from multi_emotions.app_multilabel_helpers import (
    predict_multilabel,
    train_custom_multilabel_model,
    validate_transformer_dimensions as validate_transformer_dimensions_multilabel,
)
from multi_emotions.config import EMOTION_LABELS


def inject_global_styles() -> None:
    """Apply a custom visual theme for a cleaner and more appealing UI."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@500;700;800&family=IBM+Plex+Sans:wght@400;500;600&display=swap');

        :root {
            --accent: #bf3f17;
            --accent-soft: #fde9dc;
            --ink: #16181b;
            --muted: #444c57;
            --card: #ffffff;
            --border: #ddcfc4;
        }

        .stApp {
            font-family: 'IBM Plex Sans', sans-serif;
            background:
                radial-gradient(circle at 5% 0%, #ffeede 0%, transparent 45%),
                radial-gradient(circle at 100% 12%, #edf8f2 0%, transparent 35%),
                linear-gradient(180deg, #fffaf7 0%, #f6f7f5 100%);
            color: var(--ink);
        }

        .stApp,
        .stApp p,
        .stApp li,
        .stApp label,
        .stApp span,
        .stApp div {
            color: var(--ink);
        }

        .stCaption,
        .soft-card-body,
        .hero-subtitle {
            color: var(--muted) !important;
        }

        h1, h2, h3, .stMarkdown strong {
            font-family: 'Manrope', sans-serif;
            letter-spacing: -0.01em;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fff7f1 0%, #ffffff 100%);
            border-right: 1px solid var(--border);
        }

        [data-testid="stSidebar"] * {
            color: #22272e;
        }

        [data-testid="stMetric"] {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 8px 12px;
            box-shadow: 0 10px 30px rgba(43, 38, 35, 0.05);
        }

        .hero {
            background: linear-gradient(130deg, #fff5eb 0%, #ffffff 65%);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 24px 24px 20px 24px;
            margin-bottom: 14px;
            box-shadow: 0 14px 30px rgba(43, 38, 35, 0.07);
        }

        .hero-kicker {
            color: var(--accent);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            margin-bottom: 8px;
        }

        .hero-title {
            font-family: 'Manrope', sans-serif;
            font-size: 2rem;
            font-weight: 800;
            line-height: 1.15;
            color: var(--ink);
            margin: 0 0 6px 0;
        }

        .hero-subtitle {
            color: var(--muted);
            font-size: 1rem;
            margin: 0;
            max-width: 760px;
        }

        .soft-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 14px 16px;
            box-shadow: 0 8px 24px rgba(20, 20, 20, 0.04);
            margin-bottom: 10px;
        }

        .soft-card-title {
            font-family: 'Manrope', sans-serif;
            font-weight: 700;
            margin-bottom: 6px;
            color: var(--ink);
        }

        .soft-card-body {
            color: var(--muted);
            font-size: 0.95rem;
            line-height: 1.45;
        }

        .pill-row {
            margin: 8px 0 10px 0;
        }

        .pill {
            display: inline-block;
            background: var(--accent-soft);
            border: 1px solid #e7bca4;
            color: #6f2b12;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 0.8rem;
            font-weight: 600;
            margin: 0 6px 6px 0;
        }

        textarea,
        input,
        [data-baseweb="select"] > div,
        [data-baseweb="input"] > div {
            background: #ffffff !important;
            color: var(--ink) !important;
            border-color: #c8ccd3 !important;
        }

        [data-baseweb="select"] *,
        [data-baseweb="input"] * {
            color: var(--ink) !important;
        }

        [data-testid="stTextArea"] label,
        [data-testid="stTextInput"] label,
        [data-testid="stNumberInput"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label,
        [data-testid="stRadio"] label {
            color: #1f2329 !important;
            font-weight: 600;
        }

        div[data-testid="stProgressBar"] > div > div > div {
            background: linear-gradient(90deg, #d9501e 0%, #ff7d45 100%);
        }

        @media (max-width: 768px) {
            .hero {
                padding: 18px 16px;
            }

            .hero-title {
                font-size: 1.6rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero(title: str, subtitle: str, kicker: str) -> None:
    """Render a reusable hero block for each section."""
    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-kicker">{kicker}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(title: str, body: str) -> None:
    """Render a compact white card with border and shadow."""
    st.markdown(
        f"""
        <div class="soft-card">
            <div class="soft-card-title">{title}</div>
            <div class="soft-card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_binary_model():
    """Load the binary sentiment model and vocabulary (cached)."""
    return build_pretrained_binary_model()

def render_home() -> None:
    """Render the main page."""
    render_hero(
        title="Attention: Emotion Intelligence Playground",
        subtitle=(
            "Explore sentiment and emotion models in an interactive space. "
            "Train faster experiments on CPU, inspect predictions, and compare workflows."
        ),
        kicker="Transformer NLP",
    )

    left, right = st.columns([1.2, 1], gap="large")

    with left:
        render_info_card(
            "What you can do",
            "Run binary sentiment inference instantly with a pretrained model, "
            "or train your own compact model and use it for custom predictions.",
        )
        render_info_card(
            "Multi-label workflow",
            "Train a model on GoEmotions and predict several emotions for one text. "
            "Great for mixed feelings where one label is not enough.",
        )
    with right:
        st.metric("Models in app", "2", "Binary + Multi-label")
        st.metric("Inference mode", "Interactive", "Session-based")
        st.metric("Authors", "2", "Volodya Aleksanyan, Paul Gautier")

    st.info("Use the sidebar to switch sections and start experimenting.")


def render_binary_emotion() -> None:
    """Render the binary emotion recognition section with 2 workflows.
    
    Workflow 1 uses the pretrained model.
    Workflow 2 lets the user train a smaller custom model on CPU
    """
    render_hero(
        title="Binary Emotion Recognition",
        subtitle="Classify text as positive or negative and inspect the confidence and influential tokens.",
        kicker="Workflow 1",
    )

    st.markdown(
        """
        <div class="pill-row">
            <span class="pill">Labels: positive / negative</span>
            <span class="pill">Dataset: IMDb</span>
            <span class="pill">Runtime: CPU friendly</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This section is for **binary sentiment/emotion recognition** on English text.
        The target model was trained on **IMDb movie reviews** from Hugging Face,
        and will output one of two labels: **positive** or **negative**.
        """
    )

    with st.expander("What this section is for", expanded=False):
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
        horizontal=True,
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
        
        # Optional advaced parameters might be changed by the user
        with st.expander("Advanced model hyperparameters"):
            adv1, adv2 = st.columns(2)
            with adv1:
                d_model = st.select_slider(
                    "d_model",
                    options=[64, 128, 256, 384],
                    value=128,
                )
                num_layers = st.slider("num_layers", min_value=1, max_value=6, value=2)
            with adv2:
                num_heads = st.select_slider(
                    "num_heads",
                    options=[2, 4, 8],
                    value=4,
                )
                dropout = st.slider(
                    "dropout",
                    min_value=0.0,
                    max_value=0.6,
                    value=0.2,
                    step=0.05,
                )

            d_ff_default = max(128, d_model * 2)
            d_ff = st.number_input(
                "d_ff",
                min_value=64,
                max_value=2048,
                value=d_ff_default,
                step=64,
            )

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

        if "custom_binary_metrics" in st.session_state:
            m = st.session_state["custom_binary_metrics"]
            st.caption(
                f"Last trained model: epochs={int(m['epochs'])}, train={int(m['train_samples'])}, "
                f"test={int(m['test_samples'])}, max_seq_length={int(m['max_seq_length'])}, "
                f"final_test_accuracy={m['final_test_accuracy']:.2%}"
            )
            c1, c2 = st.columns(2)
            c1.metric("Final test accuracy", f"{m['final_test_accuracy']:.2%}")
            c2.metric("Epochs", f"{int(m['epochs'])}")

        # final sanity check
        has_custom_model = (
            "custom_binary_model" in st.session_state
            and "custom_binary_vocab" in st.session_state
        )
        if has_custom_model:
            model_source = "custom"
            st.info("Using your trained model for prediction.")
        else:
            model_source = "pretrained"
            st.info("No custom model trained yet. Predictions will use the pretrained model")


    user_text = st.text_area(
        "Enter English text",
        placeholder="Example: I really enjoyed this movie and would watch it again.",
        height=160,
    )
    
    st.progress(min(len(user_text) / 600.0, 1.0))
    st.caption(f"Character count: {len(user_text)}")

    if st.button("Predict", type="primary"):
        if not user_text.strip():
            st.error("Please enter some text before clicking Predict.")
            return

        # Select model source (custom if user trained one in this session).
        if model_source == "custom":
            model = st.session_state["custom_binary_model"]
            vocab = st.session_state["custom_binary_vocab"]
            predict_max_length = st.session_state.get(
                "custom_binary_max_seq_len", MAX_SEQ_LENGTH
            )
        else:
            with st.spinner("Loading model..."):
                model, vocab = load_binary_model()
            predict_max_length = MAX_SEQ_LENGTH

        # Run prediction + token importance in one pass
        with st.spinner("Analyzing text..."):
            label, confidence, top_tokens = explain_prediction(
                user_text, model, vocab, DEVICE,
                max_length=predict_max_length,
                top_k=5,
            )
            # Also get class probabilities for the breakdown line
            _, _, probs = predict_sentiment(
                user_text, model, vocab, DEVICE, max_length=predict_max_length
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
        p1, p2 = st.columns(2)
        p1.progress(min(max(float(probs[0]), 0.0), 1.0))
        p1.caption(f"Negative: {probs[0]:.1%}")
        p2.progress(min(max(float(probs[1]), 0.0), 1.0))
        p2.caption(f"Positive: {probs[1]:.1%}")

        # Show which tokens contributed most
        st.subheader("Most influential tokens")
        st.caption(
            "Importance is computed with gradient x input: how much each token "
            "pulled the model toward this prediction (normalized, 1.0 = most important)."
        )
        for token, score in top_tokens:
            left, right = st.columns([3, 2])
            with left:
                st.write(f"**{token}**")
            with right:
                st.progress(min(max(float(score), 0.0), 1.0))
                st.caption(f"importance: {score:.2f}")


def render_multiclass_emotion() -> None:
    """Render the multi-label emotion section.

    This workflow is train-only. No pretrained model is provided.
    """
    render_hero(
        title="Multi-label Emotion Prediction",
        subtitle="Train your own GoEmotions model and predict multiple emotions from one text.",
        kicker="Workflow 2",
    )

    st.markdown(
        """
        <div class="pill-row">
            <span class="pill">Task: multi-label</span>
            <span class="pill">Dataset: GoEmotions</span>
            <span class="pill">Train first, then predict</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        This section trains a **multi-label emotion model** on GoEmotions.
        Unlike binary sentiment, there is **no pretrained model** here.
        You need to train your own model first, then run predictions.
        """
    )

    st.warning(
        "Training runs on CPU in your setup, so it can be slow. "
        "Start with small values first."
    )

    col1, col2 = st.columns(2)
    with col1:
        epochs = st.slider("Epochs", min_value=1, max_value=15, value=3)
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
        threshold = st.slider(
            "Prediction threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
        )
    with col2:
        train_samples = st.slider(
            "Train subset size",
            min_value=1000,
            max_value=12000,
            value=4000,
            step=500,
        )
        test_samples = st.slider(
            "Test subset size",
            min_value=500,
            max_value=5000,
            value=1200,
            step=250,
        )
        vocab_build_size = st.slider(
            "Vocabulary build samples",
            min_value=2000,
            max_value=25000,
            value=8000,
            step=1000,
        )

    with st.expander("Advanced model hyperparameters"):
        adv1, adv2 = st.columns(2)
        with adv1:
            d_model = st.select_slider(
                "d_model",
                options=[64, 128, 256, 384],
                value=128,
            )
            num_layers = st.slider("num_layers", min_value=1, max_value=6, value=2)
        with adv2:
            num_heads = st.select_slider(
                "num_heads",
                options=[2, 4, 8],
                value=4,
            )
            dropout = st.slider(
                "dropout",
                min_value=0.0,
                max_value=0.6,
                value=0.2,
                step=0.05,
            )

        d_ff_default = max(128, d_model * 2)
        d_ff = st.number_input(
            "d_ff",
            min_value=64,
            max_value=2048,
            value=d_ff_default,
            step=64,
        )

    if not validate_transformer_dimensions_multilabel(int(d_model), int(num_heads)):
        st.error("Invalid model settings: d_model must be divisible by num_heads.")

    elif st.button("Start multi-label training", type="secondary"):
        with st.spinner("Training multi-label model on CPU. This may take several minutes..."):
            trained_model, trained_vocab, metrics = train_custom_multilabel_model(
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
                threshold=float(threshold),
            )

        st.session_state["custom_multilabel_model"] = trained_model
        st.session_state["custom_multilabel_vocab"] = trained_vocab
        st.session_state["custom_multilabel_max_seq_len"] = int(max_seq_length)
        st.session_state["custom_multilabel_threshold"] = float(threshold)
        st.session_state["custom_multilabel_metrics"] = metrics

        st.success(
            "Training finished. "
            f"Final test micro-F1: {metrics['final_test_f1']:.2%}"
        )

    if "custom_multilabel_metrics" in st.session_state:
        m = st.session_state["custom_multilabel_metrics"]
        st.caption(
            f"Last trained model: epochs={int(m['epochs'])}, train={int(m['train_samples'])}, "
            f"test={int(m['test_samples'])}, max_seq_length={int(m['max_seq_length'])}, "
            f"threshold={m['threshold']:.2f}, final_test_f1={m['final_test_f1']:.2%}"
        )
        m1, m2 = st.columns(2)
        m1.metric("Final test micro-F1", f"{m['final_test_f1']:.2%}")
        m2.metric("Threshold", f"{m['threshold']:.2f}")

    has_custom_model = (
        "custom_multilabel_model" in st.session_state
        and "custom_multilabel_vocab" in st.session_state
    )
    if not has_custom_model:
        st.info("Train a model first. Prediction is disabled until training is finished.")
        return

    st.info(
        "Using your trained multi-label model for prediction. "
        f"Labels: {', '.join(EMOTION_LABELS)}"
    )

    user_text = st.text_area(
        "Enter English text",
        placeholder="Example: I am excited and nervous about tomorrow.",
        height=160,
    )
    st.progress(min(len(user_text) / 600.0, 1.0))
    st.caption(f"Character count: {len(user_text)}")

    if st.button("Predict multi-label emotions", type="primary"):
        if not user_text.strip():
            st.error("Please enter some text before clicking Predict.")
            return

        model = st.session_state["custom_multilabel_model"]
        vocab = st.session_state["custom_multilabel_vocab"]
        max_len = st.session_state.get("custom_multilabel_max_seq_len", 128)
        pred_threshold = st.session_state.get("custom_multilabel_threshold", 0.5)

        with st.spinner("Analyzing text..."):
            predicted_labels, prob_dict = predict_multilabel(
                text=user_text,
                model=model,
                vocab=vocab,
                threshold=float(pred_threshold),
                max_length=int(max_len),
            )

        st.subheader("Prediction")
        st.success(f"Predicted labels: {', '.join(predicted_labels)}")
        st.caption(f"Threshold used: {float(pred_threshold):.2f}")

        st.subheader("Per-label probabilities")
        for label in EMOTION_LABELS:
            prob = float(prob_dict[label])
            lcol, pcol = st.columns([1.5, 3])
            with lcol:
                st.write(f"**{label}**")
            with pcol:
                st.progress(min(max(prob, 0.0), 1.0))
                st.caption(f"{prob:.1%}")


def render_third_model() -> None:
    """Render the third-model placeholder section."""
    render_hero(
        title="Third Model",
        subtitle="This section is reserved for the next experiment and will be available soon.",
        kicker="Coming soon",
    )
    render_info_card(
        "Planned direction",
        "This slot can host an explainability module, a multilingual model, "
        "or a comparison benchmark between architectures.",
    )


def main() -> None:
    """Render the app with basic section navigation."""
    st.set_page_config(page_title="Attention", layout="wide")
    inject_global_styles()

    st.markdown("## Attention")
    st.caption("Interactive NLP dashboard for emotion and sentiment workflows")

    app_mode = st.sidebar.selectbox(
        "Choose section",
        (
            "Main Page",
            "Binary Emotion Recognition",
            "Multi-label Emotion Prediction",
            "3rd Model",
        ),
    )

    if app_mode == "Main Page":
        render_home()
    elif app_mode == "Binary Emotion Recognition":
        render_binary_emotion()
    elif app_mode == "Multi-label Emotion Prediction":
        render_multiclass_emotion()
    else:
        render_third_model()



if __name__ == "__main__":
    main()
