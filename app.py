"""Minimal Streamlit app entrypoint."""

import streamlit as st


def main() -> None:
    """Render a minimal page that always loads."""
    st.set_page_config(page_title="Attention", layout="centered")
    st.title("Attention")


if __name__ == "__main__":
    main()
