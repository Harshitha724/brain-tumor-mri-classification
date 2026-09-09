import os
import torch
import torch.nn.functional as F
import streamlit as st
import numpy as np

from PIL import Image
from torchvision import transforms

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.models import get_resnet34_finetuned


# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = (
    "checkpoints/resnet34_lr5e-5_best.pth"
)

CLASSES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]

CLASS_DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary"
}


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Brain Tumor MRI Classifier",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_model():

    model = get_resnet34_finetuned(
        num_classes=4
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)
    model.eval()

    return model, checkpoint


model, checkpoint = load_model()


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


display_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# ============================================================
# HEADER
# ============================================================

st.title("🧠 Brain Tumor MRI Classifier")

st.markdown(
    """
    ### ResNet34-based MRI Image Classification

    Upload a brain MRI image to classify it into one of four
    categories and visualize the regions that influenced the
    model's prediction using Grad-CAM.
    """
)

st.divider()


# ============================================================
# MODEL INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Model",
        "ResNet34 Fine-tuned"
    )

with col2:
    st.metric(
        "Best Validation Accuracy",
        f"{checkpoint['val_accuracy']:.2f}%"
    )

with col3:
    st.metric(
        "Final Test Accuracy",
        "94.94%"
    )


st.divider()


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png"],
    help="Upload a brain MRI image in JPG, JPEG, or PNG format."
)


# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # --------------------------------------------------------
    # Display original image
    # --------------------------------------------------------

    st.subheader("Input MRI")

    input_col1, input_col2 = st.columns(2)

    with input_col1:

        st.image(
            image,
            caption="Uploaded MRI",
            use_container_width=True
        )

    # --------------------------------------------------------
    # Preprocess
    # --------------------------------------------------------

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Model prediction
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )

        confidence, prediction = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = prediction.item()

    predicted_class = CLASSES[
        predicted_index
    ]

    confidence_value = (
        confidence.item() * 100
    )


    # --------------------------------------------------------
    # Prediction result
    # --------------------------------------------------------

    with input_col2:

        st.subheader("Prediction")

        st.success(
            f"Predicted Class: "
            f"{CLASS_DISPLAY_NAMES[predicted_class]}"
        )

        st.metric(
            "Confidence",
            f"{confidence_value:.2f}%"
        )

        st.write(
            f"Model prediction: "
            f"**{CLASS_DISPLAY_NAMES[predicted_class]}**"
        )


    st.divider()


    # ========================================================
    # CLASS PROBABILITIES
    # ========================================================

    st.subheader("Class Probabilities")

    probability_columns = st.columns(4)

    for i, class_name in enumerate(CLASSES):

        probability = (
            probabilities[0][i].item() * 100
        )

        with probability_columns[i]:

            st.metric(
                CLASS_DISPLAY_NAMES[class_name],
                f"{probability:.2f}%"
            )

            st.progress(
                int(probability)
            )


    st.divider()


    # ========================================================
    # GRAD-CAM
    # ========================================================

    st.subheader("Grad-CAM Visualization")

    st.write(
        "Grad-CAM highlights image regions that contributed "
        "to the model's predicted class."
    )


    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    target_layers = [
        model.layer4[-1]
    ]


    # --------------------------------------------------------
    # Prepare image for Grad-CAM
    # --------------------------------------------------------

    original_image = image.resize(
        (224, 224)
    )

    rgb_image = np.array(
        original_image
    ).astype(
        np.float32
    ) / 255.0


    # --------------------------------------------------------
    # Generate Grad-CAM
    # --------------------------------------------------------

    cam = GradCAM(
        model=model,
        target_layers=target_layers
    )

    targets = [
        ClassifierOutputTarget(
            predicted_index
        )
    ]

    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=targets
    )[0]


    visualization = show_cam_on_image(
        rgb_image,
        grayscale_cam,
        use_rgb=True
    )


    # --------------------------------------------------------
    # Display Grad-CAM
    # --------------------------------------------------------

    cam_col1, cam_col2 = st.columns(2)

    with cam_col1:

        st.image(
            original_image,
            caption="Original MRI",
            use_container_width=True
        )

    with cam_col2:

        st.image(
            visualization,
            caption="Grad-CAM",
            use_container_width=True
        )


    st.divider()


    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader("Prediction Summary")

    st.write(
        f"""
        The model classified this image as
        **{CLASS_DISPLAY_NAMES[predicted_class]}**
        with a confidence of **{confidence_value:.2f}%**.
        """
    )

    st.info(
        "Grad-CAM is an interpretability visualization. "
        "It shows regions that contributed to the model's "
        "prediction and should not be interpreted as a "
        "medical diagnosis or tumor segmentation."
    )


# ============================================================
# NO IMAGE UPLOADED
# ============================================================

else:

    st.info(
        "👆 Upload an MRI image above to begin classification."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Brain Tumor MRI Classification Project | "
    "ResNet34 Fine-tuned | Research/Educational Use Only"
)

st.caption(
    "This application is not intended for medical diagnosis."
)
