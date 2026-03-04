import streamlit as st


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
    st.info("Model integration status: UI is ready, model wiring is coming next.")

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

        st.subheader("Prediction")
        st.warning("Sorry, coming soon... ://")
        st.caption(
            "Once the model is connected, this will return either 'positive' or 'negative'."
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
