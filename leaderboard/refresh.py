# leaderboard/refresh.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import yaml

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Path Definitions ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# --- Default Input/Output Paths ---
DEFAULT_MODELS_FOLDER = PROJECT_ROOT.parent / "llm-leaderboard/models_info"
DEFAULT_RESULTS_FOLDER = PROJECT_ROOT.parent / "llm-leaderboard/results"
OUTPUT_FOLDER = SCRIPT_DIR / "boards_data"
CONFIG_FILE_PATH = SCRIPT_DIR / "leaderboard_config.yaml"
TEMPLATE_FOLDER = SCRIPT_DIR / "template_jsons"

# --- Constants for Subtask Processing ---
NLU_NLG_TASK_KEYS = ["persian_nlu", "persian_nlg"]

ALL_LEADERBOARD_COLUMNS = [
    'Model Name', 'model_url', 'parameters_count', 'source_type', 'Average',
    'IFEval-Fa', 'Persian MT-Bench', "MMLU-Fa",
    "PerCoR", "Persian NLU", "Persian NLG"
]


def load_tasks_from_config(config_path: Path) -> Dict[str, str]:
    
    if not config_path.exists():
        logger.error(f"Configuration file not found: {config_path}. Cannot load tasks.")
        return {}
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        tasks_from_config = config_data.get('task_display_names', {})
        if not isinstance(tasks_from_config, dict):
            logger.error(f"'task_display_names' in {config_path} is not a dictionary.")
            return {}
        processed_tasks = {k: v for k, v in tasks_from_config.items() if str(k).lower() != 'all'}
        if not processed_tasks:
            logger.warning(f"No tasks in {config_path} under 'task_display_names' (excluding 'all').")
        return processed_tasks
    except Exception as e:
        logger.error(f"Error loading config {config_path}: {e}")
        return {}

class ModelEvaluationProcessor:
    def __init__(
        self,
        models_info_path: Path,
        results_base_path: Path,
        output_path: Path,
        template_jsons_path: Path,
    ) -> None:
        
        self.models_info_path = models_info_path
        self.results_base_path = results_base_path
        self.output_path = output_path
        self.template_folder = template_jsons_path
        self.output_path.mkdir(parents=True, exist_ok=True)

        self.tasks_config = load_tasks_from_config(CONFIG_FILE_PATH)
        print(f"\n\n{self.tasks_config}\n\n")
        if not self.tasks_config:
            logger.error("Tasks config is empty. Processing might be affected.")

        self.main_scores_map = {
            "ifeval": "strict_instruction_accuracy",
            "mt_bench": "score_mean",
            "MMLU": "acc",
            "persian_csr": "acc",
            "persian_nlg": "nlg_score",
            "persian_nlu": "nlu_score",
        }
    def _load_template(self, task_key: str) -> Dict[str, Any]:
        
        path = self.template_folder / f"{task_key}.json"
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.warning(f"Template file not found for task_key {task_key} at {path}. Using empty template.")
            return {}
        except Exception as e:
            logger.error(f"Cannot load template for task_key {task_key} from {path}: {e}")
            return {}

    def _deep_override(self, base: Any, override: Any) -> Any:
        if isinstance(base, dict) and isinstance(override, dict):
            merged = {}
            for k, v_base in base.items():
                if k in override and override[k] is not None and override[k] != -1:
                    merged[k] = self._deep_override(v_base, override[k])
                else:
                    merged[k] = v_base
            for k, v_override in override.items():
                 if k not in merged:
                    merged[k] = v_override
            return merged
        elif override is not None and override != -1:
            return override
        else:
            return base


    def _load_model_raw_results(self, model_folder_name: str, task_key: str) -> Dict[str, Any]:
        
        results_filename = f"{model_folder_name}___{task_key}.json"
        results_file_path = self.results_base_path / results_filename

        if results_file_path.exists():
            try:
                with open(results_file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data if isinstance(data, dict) else {}
            except json.JSONDecodeError as e:
                logger.error(f"JSONDecodeError for model '{model_folder_name}', task_key '{task_key}' from {results_file_path}: {e}")
            except Exception as e:
                logger.error(f"Error loading results for model '{model_folder_name}', task_key '{task_key}' from {results_file_path}: {e}")
        else:
            logger.warning(f"Results file not found for model '{model_folder_name}', task_key '{task_key}' at {results_file_path}")
        return {}

    def load_and_fill_task_results(self, model_folder_name: str, task_key: str) -> Dict[str, Any]:
        
        template = self._load_template(task_key)
        raw_results = self._load_model_raw_results(model_folder_name, task_key)
        return self._deep_override(template, raw_results)

    def clean_previous_subtask_files(self) -> None:
        
        logger.info("Cleaning previous NLU/NLG subtask JSONL files...")
        for task_key_prefix in NLU_NLG_TASK_KEYS:
            for result_file in self.results_base_path.glob(f"*___{task_key_prefix}.json"):
                try:
                    task_data_content = result_file.read_text(encoding="utf-8")
                    if not task_data_content.strip():
                        logger.debug(f"Skipping empty result file for subtask cleaning: {result_file}")
                        continue
                    task_data = json.loads(task_data_content)

                    main_score_for_this_task_prefix = self.main_scores_map.get(task_key_prefix)

                    for subtask_name in task_data:
                        if subtask_name == main_score_for_this_task_prefix:
                            continue
                        if isinstance(task_data.get(subtask_name), dict):
                            subtask_output_path = self.output_path / f"{subtask_name}.jsonl"
                            if subtask_output_path.exists():
                                subtask_output_path.unlink()
                                logger.info(f"Deleted previous subtask file: {subtask_output_path}")
                except json.JSONDecodeError as e:
                    logger.warning(f"Failed to decode JSON for subtask cleaning from {result_file}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to inspect/delete subtask files based on {result_file}: {e}")

    def _process_subtask_data(self, task_results: Dict[str, Any], base_model_info: Dict[str, Any], parent_task_main_score_key: Optional[str], parent_task_key_for_log: str) -> None:
        
        parent_task_main_score_value = task_results.get(parent_task_main_score_key) if parent_task_main_score_key else None

        for subtask_name, subtask_scores_dict in task_results.items():
            if subtask_name == parent_task_main_score_key:
                continue
            if not isinstance(subtask_scores_dict, dict):
                logger.debug(f"Skipping entry '{subtask_name}' in '{parent_task_key_for_log}': not a dictionary of subtask scores.")
                continue

            row_data = base_model_info.copy()
            row_data.update(subtask_scores_dict)

            if parent_task_main_score_key:
                row_data[parent_task_main_score_key] = parent_task_main_score_value

            subtask_output_file = f"{subtask_name}.jsonl"
            subtask_output_path = self.output_path / subtask_output_file

            try:
                current_entries = []
                if subtask_output_path.exists():
                    existing_df = pd.read_json(subtask_output_path, lines=True)
                    if not existing_df.empty and 'Model Name' in existing_df.columns:
                         current_entries = existing_df[existing_df['Model Name'] != row_data['Model Name']].to_dict(orient='records')

                current_entries.append(row_data)
                updated_df = pd.DataFrame(current_entries)
                updated_df.to_json(subtask_output_path, orient="records", lines=True, force_ascii=False)
                logger.debug(f"Updated subtask file: {subtask_output_path} for model {base_model_info.get('Model Name')}, parent task {parent_task_key_for_log}")
            except Exception as e:
                logger.error(f"Error updating subtask file {subtask_output_path} for parent {parent_task_key_for_log}: {e}")
    def process_nlu_nlg_subtasks(self, model_details: Dict[str, Any], model_folder_name: str, canonical_model_name: str) -> None:
        
        common_subtask_model_info = {
            "Model Name": canonical_model_name,
            "model_url": model_details.get('model_url', model_details.get('link', model_details.get('homepage', 'https://google.com'))),
            "parameters_count": str(model_details.get('n_parameters', "N/A")),
            "source_type": "Open-Source" # Default, will be refined
        }
        parameters_count_raw = model_details.get('n_parameters', None)
        if parameters_count_raw is not None:
            is_open_source_candidate = False
            if isinstance(parameters_count_raw, (int, float)) and parameters_count_raw > 0:
                is_open_source_candidate = True
            elif isinstance(parameters_count_raw, str) and \
                    str(parameters_count_raw).strip().lower() not in ["", "n/a", "unknown", "private", "confidential", "tbd", "null", "closed"]:
                is_open_source_candidate = True
            common_subtask_model_info["source_type"] = "Open-Source" if is_open_source_candidate else "Closed-Source"

        for task_key_for_subtasks in NLU_NLG_TASK_KEYS:
            if task_key_for_subtasks not in self.tasks_config:
                logger.debug(f"Subtask processing for '{task_key_for_subtasks}' skipped: not in tasks_config.")
                continue

            logger.info(f"Processing subtasks for '{task_key_for_subtasks}' for model '{canonical_model_name}'...")
            parent_task_full_results = self.load_and_fill_task_results(model_folder_name, task_key_for_subtasks)
            main_score_key_for_parent_task = self.main_scores_map.get(task_key_for_subtasks)
            if not main_score_key_for_parent_task:
                logger.warning(f"No main score key in main_scores_map for parent task '{task_key_for_subtasks}'.")

            self._process_subtask_data(
                parent_task_full_results,
                common_subtask_model_info,
                main_score_key_for_parent_task,
                task_key_for_subtasks
            )
    def process_models(self) -> Dict[str, pd.DataFrame]:
        processed_task_data: Dict[str, List[Dict[str, Any]]] = {task_key: [] for task_key in self.tasks_config.keys()}
        all_models_summary_data: List[Dict[str, Any]] = []

        if not self.models_info_path.exists() or not self.models_info_path.is_dir():
            logger.critical(f"Configured MODELS_FOLDER path does not exist or is not a directory: {self.models_info_path}")
            empty_dfs = {key: pd.DataFrame() for key in self.tasks_config.keys()}
            empty_dfs["all"] = pd.DataFrame()
            return empty_dfs

        model_info_files = list(self.models_info_path.glob("*.json"))
        if not model_info_files:
            logger.warning(f"No model info files (*.json) found in {self.models_info_path}. No models will be processed.")
            empty_dfs = {key: pd.DataFrame() for key in self.tasks_config.keys()}
            empty_dfs["all"] = pd.DataFrame()
            return empty_dfs

        for model_info_file in model_info_files:
            model_folder_name = model_info_file.stem
            try:
                with open(model_info_file, 'r', encoding='utf-8') as f:
                    model_details = json.load(f)

                canonical_model_name = model_details.get('name_for_leaderboard',
                                      model_details.get('model_hf_id',
                                      model_details.get('name', model_folder_name)))
                model_url = model_details.get('model_url', model_details.get('link', model_details.get('homepage', 'https_google.com')))
                if not model_url: model_url = 'https_google.com'

                parameters_count_raw = model_details.get('n_parameters', None)
                parameters_count_display = str(parameters_count_raw) if parameters_count_raw is not None else "N/A"

                source_type = "Closed-Source"
                if parameters_count_raw is not None:
                    is_open_source_candidate = False
                    if isinstance(parameters_count_raw, (int, float)) and parameters_count_raw > 0:
                        is_open_source_candidate = True
                    elif isinstance(parameters_count_raw, str) and \
                         str(parameters_count_raw).strip().lower() not in ["", "n/a", "unknown", "private", "confidential", "tbd", "null", "closed"]:
                        is_open_source_candidate = True
                    source_type = "Open-Source" if is_open_source_candidate else "Closed-Source"

            except Exception as e:
                logger.error(f"Error loading/parsing model info from {model_info_file}: {e}. Skipping '{model_folder_name}'.")
                continue

            logger.info(f"Processing model: {canonical_model_name} (source ID: {model_folder_name})")

            current_model_scores_for_summary: Dict[str, Any] = {
                "Model Name": canonical_model_name,
                "model_url": model_url,
                "parameters_count": parameters_count_display,
                "source_type": source_type
            }

            for task_key, task_display_name in self.tasks_config.items():
                task_specific_results = self.load_and_fill_task_results(model_folder_name, task_key)
                main_score_metric_name = self.main_scores_map.get(task_key)
                task_data_entry_for_specific_jsonl: Dict[str, Any] = {
                    "Model Name": canonical_model_name,
                    "model_url": model_url,
                    "parameters_count": parameters_count_display,
                    "source_type": source_type
                }

                if isinstance(task_specific_results, dict) and task_specific_results:
                    for metric, value in task_specific_results.items():
                        task_data_entry_for_specific_jsonl[metric] = value

                    if main_score_metric_name and main_score_metric_name in task_specific_results:
                        score_value = task_specific_results[main_score_metric_name]
                        if task_key == "mt_bench" and score_value is not None:
                             try:
                                score_value = float(score_value) / 10.0
                             except (ValueError, TypeError):
                                logger.warning(f"Could not convert mt_bench score '{score_value}' to float for division for model {canonical_model_name}")
                                score_value = pd.NA
                        current_model_scores_for_summary[task_display_name] = score_value
                    elif main_score_metric_name:
                        logger.warning(f"Main score metric '{main_score_metric_name}' for task '{task_key}' (Display: {task_display_name}) not found for model '{canonical_model_name}'. Will be NA.")
                        current_model_scores_for_summary[task_display_name] = pd.NA
                        task_data_entry_for_specific_jsonl[main_score_metric_name] = pd.NA
                else:
                    logger.warning(f"No valid results data for model '{canonical_model_name}', task_key '{task_key}'. Scores will be NA.")
                    if main_score_metric_name:
                        task_data_entry_for_specific_jsonl[main_score_metric_name] = pd.NA
                    current_model_scores_for_summary[task_display_name] = pd.NA

                processed_task_data[task_key].append(task_data_entry_for_specific_jsonl)

            all_models_summary_data.append(current_model_scores_for_summary)
            self.process_nlu_nlg_subtasks(model_details, model_folder_name, canonical_model_name)

        final_dataframes: Dict[str, pd.DataFrame] = {}
        for task_key, data_list in processed_task_data.items():
            df = pd.DataFrame(data_list) if data_list else pd.DataFrame()
            main_score_col = self.main_scores_map.get(task_key)
            if not df.empty and main_score_col and main_score_col in df.columns:
                try:
                    df[main_score_col] = pd.to_numeric(df[main_score_col], errors='coerce')
                    # Sort by main score (NaNs will go last or first depending on na_position, default is last)
                    df = df.sort_values(by=main_score_col, ascending=False, na_position='last')
                except Exception as e:
                    logger.warning(f"Could not sort dataframe for task {task_key} by score {main_score_col}: {e}")
            final_dataframes[task_key] = df
            if df.empty:
                 logger.warning(f"No data processed for task '{task_key}'. Resulting DataFrame is empty.")

        if all_models_summary_data:
            all_df = pd.DataFrame(all_models_summary_data)
            score_cols_for_average = []
            for _, task_display_name_for_avg in self.tasks_config.items():
                if task_display_name_for_avg in all_df.columns:
                    numeric_col = pd.to_numeric(all_df[task_display_name_for_avg], errors='coerce')
                    if numeric_col.notna().any(): # Check if there is at least one non-NA numeric value
                        all_df[task_display_name_for_avg] = numeric_col
                        score_cols_for_average.append(task_display_name_for_avg)
                    else: # All values are NA or non-numeric
                        all_df[task_display_name_for_avg] = pd.NA # Ensure column is NA if not usable
                        logger.warning(f"Column '{task_display_name_for_avg}' for averaging in 'all' table is not numeric or all NaN. Excluding from average calculation and setting to NA.")
            if score_cols_for_average:
                try:
                    # Calculate mean; it will be NaN if any constituent score for a row is NaN.
                    all_df["Average"] = all_df[score_cols_for_average].mean(axis=1, skipna=False)
                    # Round only non-NaN averages
                    all_df.loc[all_df["Average"].notna(), "Average"] = all_df.loc[all_df["Average"].notna(), "Average"].round(4)
                except Exception as e:
                    logger.error(f"Error calculating 'Average' for 'all' table: {e}. Average column might be NA or incorrect.")
                    all_df["Average"] = pd.NA # Fallback to NA
            else:
                logger.warning("No valid numeric score columns found to calculate 'Average' for 'all' table.")
                all_df["Average"] = pd.NA # Assign pd.NA if no columns to average

            # Sort 'all' table by Average (NaNs will be placed last by default with ascending=False)
            if "Average" in all_df.columns: # Check if 'Average' column exists
                 # NaNs are typically sorted to the end by default when ascending=False or na_position='last'
                all_df = all_df.sort_values(by="Average", ascending=False, na_position='last')


            existing_cols_in_order = [col for col in ALL_LEADERBOARD_COLUMNS if col in all_df.columns]
            other_cols = [col for col in all_df.columns if col not in existing_cols_in_order]
            all_df = all_df[existing_cols_in_order + other_cols]

            final_dataframes["all"] = all_df
        else:
            final_dataframes["all"] = pd.DataFrame()
            logger.warning("No summary data collected for the 'all' table.")

        return final_dataframes

    def save_dataframe_as_jsonl(self, df: pd.DataFrame, filename_base: str) -> None:
        
        if df is None or df.empty:
            logger.warning(f"DataFrame for '{filename_base}.jsonl' is empty or None. Skipping save.")
            return
        output_file_path = self.output_path / f"{filename_base}.jsonl"
        try:
            df.to_json(output_file_path, orient="records", lines=True, force_ascii=False, index=False)
            logger.info(f"Saved data to {output_file_path}")
        except Exception as e:
            logger.error(f"Failed to save DataFrame to {output_file_path}: {e}")
    def run(self) -> None:
        
        logger.info("Starting data processing pipeline in ModelEvaluationProcessor...")
        self.clean_previous_subtask_files()
        processed_dataframes = self.process_models()
        for task_key_or_name, df in processed_dataframes.items():
            self.save_dataframe_as_jsonl(df, task_key_or_name)
        logger.info("Data processing pipeline completed successfully!")

def main() -> None:
    
    models_folder_to_use = DEFAULT_MODELS_FOLDER
    results_folder_to_use = DEFAULT_RESULTS_FOLDER
    template_folder_to_use = TEMPLATE_FOLDER

    logger.info(f"Refresh script running from: {SCRIPT_DIR}")
    logger.info(f"CONFIGURED Input 'models_info' Path: {models_folder_to_use}")
    logger.info(f"CONFIGURED Input 'results' Path: {results_folder_to_use}")
    logger.info(f"CONFIGURED Input 'template_jsons' Path: {template_folder_to_use}")
    logger.info(f"Outputting processed data to (inside 'leaderboard' dir): {OUTPUT_FOLDER}")
    logger.info(f"Using configuration file (inside 'leaderboard' dir): {CONFIG_FILE_PATH}")

    if not CONFIG_FILE_PATH.exists():
        logger.critical(f"CRITICAL: Config file not found at {CONFIG_FILE_PATH}. Ensure '{CONFIG_FILE_PATH.name}' exists in '{SCRIPT_DIR}'.")
        return
    if not models_folder_to_use.exists() or not models_folder_to_use.is_dir():
        logger.critical(f"CRITICAL: Input 'models_info' directory not found at {models_folder_to_use} or is not a directory.")
        return
    if not results_folder_to_use.exists() or not results_folder_to_use.is_dir():
        logger.critical(f"CRITICAL: Input 'results' directory not found at {results_folder_to_use} or is not a directory.")
        return
    if not template_folder_to_use.exists() or not template_folder_to_use.is_dir():
        logger.warning(f"WARNING: 'template_jsons' directory not found at {template_folder_to_use}. Template filling might not work as expected.")

    try:
        processor = ModelEvaluationProcessor(
            models_info_path=models_folder_to_use,
            results_base_path=results_folder_to_use,
            output_path=OUTPUT_FOLDER,
            template_jsons_path=template_folder_to_use,
        )
        processor.run()
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}", exc_info=True)

if __name__ == "__main__":
    main()