import streamlit as st
import os
import json
from src.data_loader import get_image_files, save_labels
from src.labeler import label_image
from src.splitter import split_dataset, organize_dataset

st.set_page_config(page_title="Image Labeler", layout="wide")

st.title("Image Labeler & Splitter")

# Sidebar Configuration
st.sidebar.header("Configuration")
input_dir = st.sidebar.text_input("Input Directory", value="./data/raw")
output_dir = st.sidebar.text_input("Output Directory", value="./data/processed")
max_resolution = st.sidebar.slider(
    "Max Image Resolution",
    min_value=256,
    max_value=1024,
    value=1024,
    help="Lower resolution increases processing speed but may reduce label quality.",
)
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
            stop_placeholder = st.empty()
            
            # Initialize or retrieve existing data
            if "labeled_data" not in st.session_state:
                st.session_state["labeled_data"] = []
            
            labeled_data = st.session_state["labeled_data"]
            # Filter out already labeled files if you wanted to support resuming, 
            # but for now we'll just append or start over. 
            # Let's start fresh for "Start Labeling".
            labeled_data = [] 

            # Prepare save path
            # Ensure the output directory exists before saving
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, "labels.json")

            files = st.session_state["files"]
            total_files = len(files)

            for i, file_path in enumerate(files):
                # Check for stop
                # The stop button allows the user to interrupt the process safely
                if stop_placeholder.button("Stop Labeling", key=f"stop_{i}"):
                    st.warning("Labeling stopped by user.")
                    break

                status_text.text(f"Processing {os.path.basename(file_path)}...")

                # Define callback
                def update_progress(percent, message):
                    """
                    Callback function to update the progress bar and status text.
                    """
                    current_bar.progress(percent)
                    status_text.text(f"Processing {os.path.basename(file_path)}: {message}")

                # Label image
                try:
                    result = label_image(
                        file_path,
                        progress_callback=update_progress,
                        max_size=max_resolution,
                    )
                    result["filename"] = os.path.basename(file_path)
                    result["original_path"] = file_path
                    labeled_data.append(result)
                    
                    # Update session state incrementally
                    st.session_state["labeled_data"] = labeled_data
                    
                    # Save incrementally
                    save_labels(labeled_data, save_path)
                    
                except Exception as e:
                    st.error(f"Error processing {file_path}: {e}")

                # Reset current bar and update overall
                current_bar.progress(1.0)
                overall_bar.progress((i + 1) / total_files)

            status_text.text("Labeling Complete!" if len(labeled_data) == total_files else "Labeling Interrupted.")
            stop_placeholder.empty() # Remove stop button
            
            if len(labeled_data) == total_files:
                st.success(f"All labels saved to {save_path}")
            else:
                st.info(f"Progress saved to {save_path}")

    if "labeled_data" in st.session_state:
        st.subheader("Results")
        st.write(f"Labeled {len(st.session_state['labeled_data'])} images.")
        st.json(st.session_state["labeled_data"])

with tab2:
    st.header("Dataset Splitting")

    if st.button("Split Dataset"):
        if os.path.exists(output_dir):
            labels_path = os.path.join(output_dir, "labels.json")
            files = []
            
            # Try to load from labels.json first
            if os.path.exists(labels_path):
                try:
                    with open(labels_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        # Extract original paths
                        for item in data:
                            if "original_path" in item and os.path.exists(item["original_path"]):
                                files.append(item["original_path"])
                            elif "filename" in item:
                                # Fallback: try to construct path from input_dir if original_path is missing or invalid
                                potential_path = os.path.join(input_dir, item["filename"])
                                if os.path.exists(potential_path):
                                    files.append(potential_path)
                    
                    if files:
                        st.success(f"Loaded {len(files)} labeled images from {labels_path}")
                    else:
                        st.warning("Found labels.json but could not extract valid image paths.")
                except Exception as e:
                    st.error(f"Error loading labels.json: {e}")

            # Fallback to input directory if no labeled files found
            if not files:
                if os.path.exists(input_dir):
                    st.info("No labeled data found. Using all images from input directory.")
                    files = get_image_files(input_dir)
                else:
                    st.error("Input directory does not exist.")
            
            if not files:
                st.error("No files found to split.")
            else:
                try:
                    train_files, test_files = split_dataset(files, split_ratio)
                    
                    # Load labels if available to pass to organize_dataset
                    labeled_data = []
                    if os.path.exists(labels_path):
                        try:
                            with open(labels_path, "r", encoding="utf-8") as f:
                                labeled_data = json.load(f)
                        except Exception:
                            pass # Ignore errors here, just won't split labels

                    train_dir, test_dir = organize_dataset(
                        train_files, test_files, output_dir, labeled_data=labeled_data
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
