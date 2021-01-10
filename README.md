# Streamlit Neural Style

This repository holds an example application using [Streamlit](https://streamlit.io) to execute a basic neural style transfer via [pystiche](https://github.com/pmeier/pystiche).
The primary purpose is as a learning exercise to compare Streamlit to Jupyter notebooks.  However, it's also handy as a quick UI for executing transfers.

After installing prerequisites, start the UI with this command:
```unix
streamlit run --server.headless True --server.address "0.0.0.0" transfer.py
```