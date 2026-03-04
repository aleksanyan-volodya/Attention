"""Minimal Streamlit app entrypoint."""

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
    st.write("Sorry, coming soon... ://")


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

    section = st.sidebar.radio(
        "Go to section",
        (
            "Main Page",
            "Binary Emotion Recognition",
            "Multiclass Emotion Prediction",
            "3rd Model",
        ),
    )

    if section == "Main Page":
        render_home()
    elif section == "Binary Emotion Recognition":
        render_binary_emotion()
    elif section == "Multiclass Emotion Prediction":
        render_multiclass_emotion()
    else:
        render_third_model()


if __name__ == "__main__":
    main()
