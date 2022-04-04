import streamlit as st

from Net import create_model, predictimg
from preprocessingImg import preprocessimg

st.markdown('''# Simple image recognition''')

img = st.file_uploader("Update file", type=['png', 'jpg'])
show_file = st.empty()

if not img:
    show_file.info("Please upadate a file {}".format(' '.join(['png', 'jpg'])))
else:
    st.image(img, use_column_width=True)
    input_model = preprocessimg(img)
    model = create_model()
    results = predictimg(model, input_model)

    for i in range(len(results)):
        st.text(f"Net prediction {i} , label : {' '.join(results[i][0].split()[1:])} \n\t\t probability : {results[i][1]}%")
