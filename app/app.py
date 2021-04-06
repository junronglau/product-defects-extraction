import defect_app
import summary_app
from multiapp import MultiApp
import streamlit as st

app = MultiApp()
app.add_app("Defect Extraction", defect_app.app)
app.add_app("Defect Summarization ", summary_app.app)
app.run()