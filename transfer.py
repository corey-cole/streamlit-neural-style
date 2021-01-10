#
# Streamlit sample for using Pystiche neural style transfer
# Execute with:
# streamlit run --server.headless True --server.address "0.0.0.0" transfer.py
#
import streamlit as st
import io 
import pystiche

from PIL import Image
from pystiche import demo, enc, loss, ops, optim
from pystiche.image import read_image, show_image, write_image
from pystiche.misc import get_device
from tempfile import NamedTemporaryFile

IMAGE_EXTENSIONS = ['png','jpg','jpeg']

def apply_style(criterion, style_tensor, content_tensor, step_count):
    criterion.set_content_image(content_tensor)
    criterion.set_style_image(style_tensor)
    input_image = content_tensor.clone()
    output_image = optim.image_optimization(
        input_image, criterion, num_steps=xfer_step_count
    )
    # There's probably a better way to do this, but this works...
    output_image_bytes = None
    with NamedTemporaryFile(prefix='nst', suffix='.jpg', delete=False) as f3:
        output_file_name = f3.name
        write_image(output_image, output_file_name)
        f3.close()
        f3 = open(output_file_name, 'rb')
        output_image_bytes = f3.read()
    
    return output_image_bytes

def uploaded_file_to_image(file_bytes, device='cpu', size=500):
    """Convert Streamlit file_uploader widget output to Tensor"""
    with NamedTemporaryFile(prefix='nst') as f:
        f.write(file_bytes.read())
        f.flush()
        return read_image(f.name, size=size, device=device)

# Set up pystiche variables
style_image = None
content_image = None
output_image = None
xfer_step_count = 500
size = 500 # This is small but works with a 4GB card and 500 steps

device = get_device()
st.title(f"Neural Style Transfer ({device})")
st.markdown("""
    This Streamlit application is a port of the getting started Jupyter notebook
    from [pystiche](https://pystiche.readthedocs.io/en/stable/galleries/examples/beginner/example_nst_with_pystiche.html)
""")

# Initialize pystiche
multi_layer_encoder = enc.vgg19_multi_layer_encoder()
content_layer = "relu4_2"
content_encoder = multi_layer_encoder.extract_encoder(content_layer)
content_weight = 1e0
content_loss = ops.FeatureReconstructionOperator(
    content_encoder, score_weight=content_weight
)

style_layers = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")
style_weight = 1e3

def get_style_op(encoder, layer_weight):
    return ops.GramOperator(encoder, score_weight=layer_weight)


style_loss = ops.MultiLayerEncodingOperator(
    multi_layer_encoder, style_layers, get_style_op, score_weight=style_weight,
)

criterion = loss.PerceptualLoss(content_loss, style_loss).to(device)

# Configure the UI using a streamlit sidebar
xfer_step_count = st.sidebar.number_input("Number of steps", step=1, min_value=0, max_value=500)
size = st.sidebar.number_input("Image size", step=10, min_value=50, max_value=500)
style_image = st.sidebar.file_uploader("Style image", type=IMAGE_EXTENSIONS)
content_image = st.sidebar.file_uploader("Content image", type=IMAGE_EXTENSIONS)
btn_clicked = st.sidebar.button("Go")

# Changing any of the inputs will reset btn_clicked
if btn_clicked:
    ps_style_image = None   
    if style_image is not None:
        ps_style_image = uploaded_file_to_image(style_image, size=size, device=device)
    
    ps_content_image = None
    if content_image is not None:
        ps_content_image = uploaded_file_to_image(content_image, size=size, device=device)

    output_image_bytes = None
    # This takes a while on my GTX 1050 Ti, so I'm assuming it will take a while anywhere.
    with st.spinner("Applying style..."):
        output_image_bytes = apply_style(criterion, ps_style_image, ps_content_image, xfer_step_count)

    if output_image_bytes is not None:
        st.image(output_image_bytes)

