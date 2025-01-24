import streamlit as st
import json
import os
from PIL import Image

st.set_page_config(page_title="CaptionEval", layout="wide")

# Initialize session state
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "checks" not in st.session_state:
    st.session_state.checks = {}
if "file_name" not in st.session_state:
    st.session_state.file_name = "evaluation_results.json"

def load_json(file):
    return json.load(file)

def save_results(checks, total_scores, file_name):
    output_data = {
        "checks": checks,
        "total_scores": total_scores
    }
    with open(file_name, "w") as f:
        json.dump(output_data, f, indent=4)

# File upload
st.title("CaptionEval Tool")
if "uploaded_json_file" not in st.session_state:
    uploaded_file = st.file_uploader("Upload a JSON file", type="json")
    if uploaded_file:
        st.session_state.uploaded_json_file = load_json(uploaded_file)

        for idx, item in enumerate(st.session_state.uploaded_json_file):
            if idx not in st.session_state.checks:
                st.session_state.checks[idx] = {
                    "video_id": item["video_id"],
                    "timestamp": item["timestamp"],
                    "frame_path": item["frame_image_path"],
                    "caption": item["caption"],
                    "caption_ko": item["caption_ko"],
                    "description_correct": 0,
                    "action_focused": 0,
                    "no_hallucination": 0,
                    "completed": 0
                }
        st.rerun()

if "uploaded_json_file" in st.session_state:
    data = st.session_state.uploaded_json_file
    
    # Navigation buttons
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Previous") and st.session_state.current_index > 0:
            st.session_state.current_index -= 1
    with col2:
        if st.button("Next") and st.session_state.current_index < len(data) - 1:
            st.session_state.current_index += 1

    # Display the current JSON element
    current_index = st.session_state.current_index
    current_item = data[current_index]

    st.subheader(f"Item {current_index + 1} / {len(data)}")

    # Show image
    if os.path.exists(current_item["frame_image_path"]):
        image = Image.open(current_item["frame_image_path"])
        width, height = image.size
        new_size = (int(width * 0.9), int(height * 0.9))
        image = image.resize(new_size)
        st.image(image, use_container_width=False)

    # Display metadata
    st.text(f"Video ID: {current_item['video_id']}")
    st.text(f"Timestamp: {current_item['timestamp']}")
    st.text(f"Frame Path: {current_item['frame_image_path']}")
    st.text(f"Caption: {current_item['caption']}")
    st.text(f"Caption (Korean): {current_item['caption_ko']}")

    # Evaluation questions
    st.markdown("### Evaluation")
    current_checks = st.session_state.checks[current_index]
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        description_correct = st.checkbox(
            "Is the scene description correct?",
            value=bool(current_checks["description_correct"]),
            key=f"description_correct_{current_index}"
        )
    with col2:
        action_focused = st.checkbox(
            "Is it action-focused?",
            value=bool(current_checks["action_focused"]),
            key=f"action_focused_{current_index}"
        )
    with col3:
        no_hallucination = st.checkbox(
            "Is there no hallucination?",
            value=bool(current_checks["no_hallucination"]),
            key=f"no_hallucination_{current_index}"
        )
    with col4:
        completed = st.checkbox(
            "Evaluation Completed",
            value=bool(current_checks["completed"]),
            key=f"completed_{current_index}"
        )

    # Update session state for current item
    st.session_state.checks[current_index]["description_correct"] = int(description_correct)
    st.session_state.checks[current_index]["action_focused"] = int(action_focused)
    st.session_state.checks[current_index]["no_hallucination"] = int(no_hallucination)
    st.session_state.checks[current_index]["completed"] = int(completed)

    # Progress tracking
    completed_count = sum(1 for v in st.session_state.checks.values() if v["completed"])
    total_items = len(data)
    progress = (completed_count / total_items) * 100

    st.progress(progress / 100)
    st.text(f"Progress: {completed_count} / {total_items} ({progress:.2f}%)")

    # Show Save Results section only if progress is 100%
    st.markdown("### Save Results")
    if progress == 100:
        st.session_state.file_name = st.text_input("Enter file name to save results", value=st.session_state.file_name)

        # Save button
        if st.button("Save Results"):
            total_scores = {
                "description_correct": sum(item["description_correct"] for item in st.session_state.checks.values()),
                "action_focused": sum(item["action_focused"] for item in st.session_state.checks.values()),
                "no_hallucination": sum(item["no_hallucination"] for item in st.session_state.checks.values()),
            }
            save_results(st.session_state.checks, total_scores, st.session_state.file_name)
            st.success(f"Results saved to {st.session_state.file_name}!")


# streamlit run app.py --server.address 0.0.0.0 --server.port 30846
