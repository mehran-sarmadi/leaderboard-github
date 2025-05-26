import gradio as gr
import csv
import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import io # To handle string as a file-like object for pandas
import logging

from huggingface_hub import HfApi, HfFolder
from huggingface_hub.utils import HfHubHTTPError, EntryNotFoundError # For specific error handling

# --- Logging Setup ---
# (Add this if not already present, or integrate with a central logging config)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Hugging Face Hub Configuration ---
# IMPORTANT: Replace with your actual repository details
TARGET_REPO_ID = "YOUR_USERNAME_OR_ORG/YOUR_DATASET_REPO_NAME"  # e.g., "MehranS/PULL_submissions"
TARGET_REPO_TYPE = "dataset"  # Recommended type for storing data
FILENAME_IN_REPO = "model_submissions.csv"  # The name of the CSV file within the Hub repository

# Define the header for your CSV file. This must be consistent.
CSV_HEADER = [
    'timestamp', 'model_name', 'base_model', 'revision',
    'precision', 'weight_type', 'model_type', 'status', 'submission_type'
]

def get_hf_token() -> str | None:
    """Retrieves the Hugging Face token from environment variables or HfFolder."""
    token = os.environ.get("HF_TOKEN")  # Standard for Spaces secrets
    if not token:
        try:
            token = HfFolder.get_token() # Fallback for local development after CLI login
        except Exception:
            logger.warning("Hugging Face token not found in HfFolder and HF_TOKEN env var is not set.")
            token = None
    return token

def add_new_eval_hf_to_hub(model_name_hf_id: str, revision_hf: str) -> gr.Markdown:
    """
    Handles new Hugging Face model evaluation requests by saving them to a CSV file
    in a specified Hugging Face Hub repository.
    """
    if not model_name_hf_id:
        return gr.Markdown("⚠️ **Model Name (Hugging Face ID) is required.** Please enter a valid Hugging Face model ID.")

    token = get_hf_token()
    if not token:
        error_html = "<div style='color:red; padding:10px; border:1px solid red; border-radius:5px;'>⚠️ **Configuration Error:** Hugging Face Token not found. Cannot save submission to the Hub. Please ensure the `HF_TOKEN` Space secret is set with write permissions to the target repository.</div>"
        return gr.Markdown(error_html)

    api = HfApi(token=token)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    submission_data = {
        'timestamp': timestamp,
        'model_name': model_name_hf_id.strip(),
        'base_model': 'N/A', # As per the simple form's design
        'revision': revision_hf.strip() if revision_hf else 'main',
        'precision': 'To be fetched/determined',
        'weight_type': 'To be fetched/determined',
        'model_type': 'To be fetched/determined',
        'status': 'pending_hub_submission', # New status indicating it's for Hub processing
        'submission_type': 'huggingface_simple_form_to_hub' # New type
    }

    try:
        # 1. Attempt to download the existing CSV from the Hub
        try:
            local_download_path = hf_hub_download(
                repo_id=TARGET_REPO_ID,
                filename=FILENAME_IN_REPO,
                repo_type=TARGET_REPO_TYPE,
                token=token,
                # force_download=True, # Consider this if caching becomes an issue
            )
            # Read the downloaded CSV into a pandas DataFrame
            df = pd.read_csv(local_download_path)
            # Ensure columns match CSV_HEADER, add missing ones with NaN if necessary
            for col in CSV_HEADER:
                if col not in df.columns:
                    df[col] = pd.NA
            df = df[CSV_HEADER] # Reorder/select columns to match header
            file_exists_on_hub = True
            logger.info(f"Successfully downloaded existing '{FILENAME_IN_REPO}' from '{TARGET_REPO_ID}'.")
        except EntryNotFoundError:
            logger.info(f"'{FILENAME_IN_REPO}' not found in '{TARGET_REPO_ID}'. A new file will be created.")
            df = pd.DataFrame(columns=CSV_HEADER) # Create an empty DataFrame with the correct headers
            file_exists_on_hub = False
        except HfHubHTTPError as e:
            logger.error(f"HTTP error downloading '{FILENAME_IN_REPO}' from '{TARGET_REPO_ID}': {e.status_code} - {e.hf_raise}")
            error_html = f"<div style='color:red; padding:10px; border:1px solid red; border-radius:5px;'>⚠️ **Hub Error:** Could not access the repository '{TARGET_REPO_ID}'. (HTTP {e.status_code}). Please check token permissions and repository ID.</div>"
            return gr.Markdown(error_html)

        # 2. Append the new submission data
        new_row_df = pd.DataFrame([submission_data])
        df = pd.concat([df, new_row_df], ignore_index=True)

        # 3. Convert the DataFrame back to CSV in-memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False, header=True) # Always include header
        csv_content_bytes = csv_buffer.getvalue().encode('utf-8')
        csv_buffer.close()

        # 4. Upload the updated CSV content to the Hub
        commit_message = f"Add submission: {submission_data['model_name']} (rev: {submission_data['revision']})"
        if not file_exists_on_hub:
            commit_message = f"Create '{FILENAME_IN_REPO}' and add first submission: {submission_data['model_name']}"

        api.upload_file(
            path_or_fileobj=csv_content_bytes, # Pass the bytes directly
            path_in_repo=FILENAME_IN_REPO,
            repo_id=TARGET_REPO_ID,
            repo_type=TARGET_REPO_TYPE,
            commit_message=commit_message
        )

        logger.info(f"Submission for '{submission_data['model_name']}' pushed to '{TARGET_REPO_ID}/{FILENAME_IN_REPO}'.")
        success_message_html = f"""
        <div style='color:green; padding:10px; border:1px solid green; border-radius:5px;'>
            ✅ Request for Hugging Face model '<strong>{submission_data['model_name']}</strong>' (Revision: {submission_data['revision']}) has been successfully submitted to the central repository on Hugging Face Hub!
        </div>
        """
        return gr.Markdown(success_message_html)

    except Exception as e:
        logger.error(f"An unexpected error occurred while processing submission to Hugging Face Hub: {e}", exc_info=True)
        error_html = f"<div style='color:red; padding:10px; border:1px solid red; border-radius:5px;'>⚠️ **System Error:** An unexpected error occurred: {e}. Please try again or contact support.</div>"
        return gr.Markdown(error_html)


def render_submit():
    # Text for Introduction and Option 1 (Hugging Face Form)
    intro_and_option1_guidance = """
# Request Model Evaluation for PULL

We're excited to evaluate new models for the **Persian Universal LLM Leaderboard (PULL)**!
Please choose the submission path that best fits how your model can be accessed for evaluation.

---

### **Option 1: Your model is publicly available on Hugging Face Hub**

If your model and its tokenizer can be loaded directly using their Hugging Face identifier (e.g., `username/model_name`), you can use the simplified form below to submit its key identifiers. Your submission will be added to our central tracking repository on the Hugging Face Hub. Our team will attempt to gather other necessary details from the Hub.
"""

    # Text for Option 2 (Email Submission)
    option2_email_guidance = """
---

### **Option 2: Your model is NOT on Hugging Face, is private, or requires custom setup**

If your model is hosted elsewhere, is private, requires specific access permissions, needs custom inference code, or involves a more complex setup for evaluation, please initiate your submission request via email.

**To submit via email, please send comprehensive details to:**
📧 **mehran.sarmadi99@sharif.edu**

Our team will review your email and work with you to facilitate the evaluation process.
    """

    with gr.Blocks() as submit_tab_interface:
        gr.Markdown(intro_and_option1_guidance)

        with gr.Group():
            gr.Markdown("### ✨ Form for Option 1: Submit a Hugging Face Model to the Hub")
            
            model_name_textbox_hf = gr.Textbox(
                label="Model Name (Hugging Face ID: e.g., username/model_name)",
                placeholder="bigscience/bloom-560m"
            )
            revision_name_textbox_hf = gr.Textbox(
                label="Revision/Commit (Optional, defaults to 'main' if left empty)",
                placeholder="e.g., main, or a specific commit hash"
            )
            
            request_hf_button = gr.Button("🚀 Request Evaluation & Submit to Hub", variant="primary")
        
        submission_result_hf_form = gr.Markdown()

        request_hf_button.click(
            fn=add_new_eval_hf_to_hub, # Use the new function
            inputs=[
                model_name_textbox_hf,
                revision_name_textbox_hf,
            ],
            outputs=submission_result_hf_form,
        )
        
        gr.Markdown(option2_email_guidance)

    return submit_tab_interface

# For direct testing of this file:
if __name__ == '__main__':
    # You would need to set TARGET_REPO_ID and have a valid HF_TOKEN env var or be logged in.
    # Example: os.environ["HF_TOKEN"] = "your_hf_write_token"
    # TARGET_REPO_ID = "your-user/your-test-dataset" # Make sure this repo exists
    
    if not TARGET_REPO_ID.startswith("YOUR_"): # Basic check to prevent running with placeholder
        print(f"Testing submission to Hub. Target repo: {TARGET_REPO_ID}")
        test_interface = render_submit()
        test_interface.launch(debug=True)
    else:
        print("Please update TARGET_REPO_ID in submission.py before running this test.")
