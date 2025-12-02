import streamlit as st
import os
from src.data_loader import get_image_files, save_labels
from src.labeler import label_image
from src.splitter import split_dataset, organize_dataset

st.set_page_config(page_title="Image Labeler", layout="wide")

st.title("Image Labeler & Splitter")

# Sidebar Configuration
st.sidebar.header("Configuration")
input_dir = st.sidebar.text_input("Input Directory", value="./data/raw")
output_dir = st.sidebar.text_input("Output Directory", value="./data/processed")
split_ratio = st.sidebar.slider("Train/Test Split Ratio", 0.0, 1.0, 0.8)

# Main Content
tab1, tab2 = st.tabs(["Labeling", "Splitting"])

with tab1:
    st.header("Image Labeling")

    if st.button("Load Images"):
        if os.path.exists(input_dir):
            files = get_image_files(input_dir)
            st.session_state["files"] = files
            st.success(f"Found {len(files)} images.")
        else:
            st.error("Input directory does not exist.")

    if "files" in st.session_state and st.session_state["files"]:
        if st.button("Start Labeling"):
            # Progress Bars
            st.write("Overall Progress")
            overall_bar = st.progress(0)
            st.write("Current Image Progress")
            current_bar = st.progress(0)
            
            status_text = st.empty()
            labeled_data = []

            files = st.session_state["files"]
            total_files = len(files)

            for i, file_path in enumerate(files):
                status_text.text(f"Processing {os.path.basename(file_path)}...")

                # Define callback
                def update_progress(percent, message):
                    current_bar.progress(percent)
                    status_text.text(f"Processing {os.path.basename(file_path)}: {message}")

                # Label image
                result = label_image(file_path, progress_callback=update_progress)
                result["filename"] = os.path.basename(file_path)
                labeled_data.append(result)

                # Reset current bar and update overall
                current_bar.progress(1.0)
                overall_bar.progress((i + 1) / total_files)

            st.session_state["labeled_data"] = labeled_data
            status_text.text("Labeling Complete!")

            # Save results
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "labels.json")
            save_labels(labeled_data, save_path)
            st.success(f"Labels saved to {save_path}")

    if "labeled_data" in st.session_state:
        st.subheader("Results")
        st.json(st.session_state["labeled_data"])

with tab2:
    st.header("Dataset Splitting")

    if st.button("Split Dataset"):
        if os.path.exists(input_dir):
            files = get_image_files(input_dir)
            if not files:
                st.error("No files found to split.")
            else:
                try:
                    train_files, test_files = split_dataset(files, split_ratio)
                    train_dir, test_dir = organize_dataset(
                        train_files, test_files, output_dir
                    )

                    st.success("Dataset split successfully!")
                    col1, col2 = st.columns(2)
                    col1.metric("Train Set", len(train_files))
                    col2.metric("Test Set", len(test_files))

                    st.info(f"Train data: {train_dir}")
                    st.info(f"Test data: {test_dir}")

                except Exception as e:
                    st.error(f"Error splitting dataset: {e}")
        else:
            st.error("Input directory does not exist.")
