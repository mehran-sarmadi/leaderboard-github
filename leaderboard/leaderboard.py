# leaderboard/leaderboard.py
import gradio as gr
import pandas as pd
import logging
from pathlib import Path
import yaml
from typing import Dict, List, Union, Optional, Any
import numpy as np

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Path Definitions ---
LEADERBOARD_DIR = Path(__file__).resolve().parent
CONFIG_FILE_PATH = LEADERBOARD_DIR / "leaderboard_config.yaml"
DATA_DIR = LEADERBOARD_DIR / "boards_data"

class ColumnConfig:
    
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.column_display_names_map: Dict[str, str] = {}
        self.task_tab_names_map: Dict[str, str] = {}

        default_task_tab_names = {
            "all": "Overall", "mt_bench": "MT-Bench", "ifeval": "IFEval",
            "MMLU": "MMLU", "persian_csr": "PerCoR",
            "persian_nlg": "Persian NLG", "persian_nlu": "Persian NLU"
        }
        default_column_names = {
            "Model Name": "Model", "model_url": "URL",
            "parameters_count": "⚙️ Params", "source_type": "Source",
            "Average": "Average", "Rank": "🏆 Rank", "score_mean": "MT-Bench Score",
            "strict_instruction_accuracy": "IFEval Strict Acc.", "acc": "Accuracy",
            "nlg_score": "NLG Score", "nlu_score": "NLU Score",
        }

        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    loaded_column_names = config.get('column_names', {})
                    self.column_display_names_map = {**default_column_names, **loaded_column_names}
                    loaded_task_names = config.get('task_display_names', {})
                    self.task_tab_names_map = {**default_task_tab_names, **loaded_task_names}
            except Exception as e:
                logger.error(f"Error loading UI name configurations from {self.config_path}: {e}. Using defaults.")
                self.column_display_names_map = default_column_names
                self.task_tab_names_map = default_task_tab_names
        else:
            logger.warning(f"UI Name configuration file '{self.config_path.name}' not found. Using defaults.")
            self.column_display_names_map = default_column_names
            self.task_tab_names_map = default_task_tab_names

    def get_column_display_name(self, original_col_name: str) -> str:
        return self.column_display_names_map.get(original_col_name, original_col_name.replace("_", " ").title())

    def get_task_tab_name(self, task_key: str) -> str:
        return self.task_tab_names_map.get(task_key, task_key.replace("_", " ").title())

    def rename_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty: return df
        rename_dict = {col: self.get_column_display_name(col) for col in df.columns}
        return df.rename(columns=rename_dict)


class LeaderboardApp:
    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.column_config = ColumnConfig(config_path)
        self.raw_dataframes: Dict[str, pd.DataFrame] = {}
        self.model_display_configs: Dict[str, Dict[str, str]] = {}

        self.model_identifier_column: str = "Model Name"
        self.main_scores_map: Dict[str, str] = {}
        self.allowed_null_columns_in_average: List[str] = ["Model Name", "model_url", "parameters_count", "source_type"]
        self.tab_processing_order: List[str] = []
        self.numeric_score_columns_for_bolding: List[str] = []
        self.columns_to_hide: List[str] = ["model_url", "source_type"]
        self.parent_child_task_map: Dict[str, List[str]] = {}

        self._load_global_settings()
        self._load_model_display_configs()

    def _load_global_settings(self) -> None:
        # ... (بدون تغییر نسبت به نسخه قبلی شما) ...
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    settings = config.get('global_settings', {})
                    self.model_identifier_column = settings.get('model_identifier_column', self.model_identifier_column)
                    self.main_scores_map = settings.get('main_scores_map', self.main_scores_map)
                    self.allowed_null_columns_in_average = settings.get('allowed_null_columns_in_average', self.allowed_null_columns_in_average)
                    self.tab_processing_order = settings.get('tab_processing_order', [])
                    self.columns_to_hide = settings.get('columns_to_hide', self.columns_to_hide)
                    self.parent_child_task_map = settings.get('parent_child_task_map', {})

                    default_numeric_bold_cols = list(self.main_scores_map.values()) if self.main_scores_map else []
                    self.numeric_score_columns_for_bolding = settings.get('numeric_score_columns_for_bolding', default_numeric_bold_cols)
                    if not self.numeric_score_columns_for_bolding and default_numeric_bold_cols:
                        self.numeric_score_columns_for_bolding = default_numeric_bold_cols
                    if 'all' in self.main_scores_map and self.main_scores_map.get('all') and \
                       self.main_scores_map['all'] not in self.numeric_score_columns_for_bolding:
                        self.numeric_score_columns_for_bolding.append(self.main_scores_map['all'])
                    self.numeric_score_columns_for_bolding = list(set(self.numeric_score_columns_for_bolding))
            except Exception as e:
                logger.error(f"Error loading global settings from {self.config_path}: {e}. Using defaults.")
        else:
            logger.error(f"Main configuration file '{getattr(self.config_path, 'name', 'config_path')}' not found. Critical settings will use defaults.")


    def _load_model_display_configs(self) -> None:
        
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    self.model_display_configs = config.get('model_display_configs', {})
            except Exception as e:
                logger.error(f"Error loading model display configs: {e}")
        else:
            logger.warning(f"Model display config section not found in {self.config_path}.")

    def load_data(self) -> None:
        
        logger.info(f"Loading all data from: {DATA_DIR}")
        if not DATA_DIR.exists() or not DATA_DIR.is_dir():
            logger.error(f"Data directory {DATA_DIR} not found. Cannot load data.")
            return

        all_jsonl_files = list(DATA_DIR.glob("*.jsonl"))
        if not all_jsonl_files:
            logger.warning(f"No .jsonl files found in {DATA_DIR}. No data will be loaded.")
            return

        for file_path in all_jsonl_files:
            task_key = file_path.stem
            try:
                self.raw_dataframes[task_key] = pd.read_json(file_path, lines=True)
                logger.info(f"Successfully loaded '{file_path.name}' for task key '{task_key}'.")
            except Exception as e:
                self.raw_dataframes[task_key] = pd.DataFrame()
                logger.error(f"Error loading '{file_path.name}' for task '{task_key}': {e}")

        configured_task_keys = set(self.tab_processing_order)
        for parent, children in self.parent_child_task_map.items():
            configured_task_keys.add(parent)
            if children:
                configured_task_keys.update(children)

        for key in configured_task_keys:
            if key not in self.raw_dataframes:
                self.raw_dataframes[key] = pd.DataFrame()
                logger.warning(f"No data file found for configured task key '{key}'. Initialized as empty.")

    def _get_benchmark_columns(self, df: pd.DataFrame) -> List[str]:
        
        if df.empty: return []
        excluded_cols = self.allowed_null_columns_in_average + ["Rank", "model_url", "Average"]
        return [col for col in df.columns if col not in excluded_cols and pd.api.types.is_numeric_dtype(df[col])]

    def handle_nulls_in_averages(self) -> None:
        
        logger.info("Skipping handle_nulls_in_averages as refresh.py is expected to handle it.")
        pass

    def _calculate_non_null_benchmark_score_count(self, df_row: pd.Series, benchmark_cols: List[str]) -> int:
        
        return df_row[benchmark_cols].notna().sum()

    def generate_model_rankings(self) -> None:
        # ... (بدون تغییر - منطق فعلی به درستی مدل‌های بدون میانگین را به پایین می‌برد) ...
        logger.info("Generating model rankings for each tab.")
        if not self.model_identifier_column:
            logger.error("`model_identifier_column` is not set. Cannot perform ranking.")
            return
        for task_key, df in self.raw_dataframes.items():
            if df.empty: continue
            ranked_df = df.copy()
            main_score_col_for_tab = self.main_scores_map.get(task_key)
            if not main_score_col_for_tab or main_score_col_for_tab not in ranked_df.columns:
                logger.warning(f"No main score column for task '{task_key}'. Ranking skipped.")
                ranked_df["Rank"] = pd.NA # Use pd.NA for missing ranks
                self.raw_dataframes[task_key] = ranked_df
                continue
            ranked_df[main_score_col_for_tab] = pd.to_numeric(ranked_df[main_score_col_for_tab], errors='coerce')
            ranked_df['_has_main_score'] = ranked_df[main_score_col_for_tab].notna()
            ranked_df['_sortable_main_score'] = ranked_df[main_score_col_for_tab].fillna(-np.inf)
            sort_by_cols = ['_has_main_score', '_sortable_main_score', self.model_identifier_column]
            ascending_order = [False, False, True]
            ranked_df = ranked_df.sort_values(by=sort_by_cols, ascending=ascending_order, na_position='last')
            # Assign ranks only to rows that have a main score; others get NA
            ranked_df["Rank"] = pd.NA
            ranked_df.loc[ranked_df['_has_main_score'], "Rank"] = range(1, ranked_df['_has_main_score'].sum() + 1)

            ranked_df.drop(columns=['_has_main_score', '_sortable_main_score'], inplace=True)
            self.raw_dataframes[task_key] = ranked_df
            logger.info(f"Generated rankings for {task_key}.")


    @staticmethod
    def _format_value_as_percentage(value: Any, score_cutoff_for_percentage: float = 0.0) -> Any:
        # ... (بدون تغییر - این متد ممکن است جای دیگری استفاده شود) ...
        if pd.isna(value) or not isinstance(value, (int, float)): return value
        if value >= score_cutoff_for_percentage and 0 <= value <= 1.0: return f"{value * 100:.2f}%"
        return f"{value:.2f}" if isinstance(value, float) else value

    @staticmethod
    def _format_parameters_count(value: Any) -> str:
        
        if pd.isna(value) or str(value).lower() in ["n/a", "unknown", "", "none"]: return "Unknown"
        try:
            num_value = float(value)
            if num_value == 0: return "N/A"
            if num_value >= 1_000_000_000: return f"{num_value / 1_000_000_000:.1f}B"
            if num_value >= 1_000_000: return f"{num_value / 1_000_000:.1f}M"
            if num_value >= 1_000: return f"{num_value / 1_000:.1f}K"
            return str(int(num_value))
        except ValueError: return str(value)

    def _apply_general_formatting_to_cells(self, df_to_format: pd.DataFrame, task_key: str) -> pd.DataFrame:
        if df_to_format.empty:
            return df_to_format

        formatted_df = df_to_format.copy()
        is_mt_bench_tab = (str(task_key).lower() == "mt_bench")

        for col_name in formatted_df.columns:
            if col_name == "parameters_count":
                formatted_df[col_name] = formatted_df[col_name].apply(self._format_parameters_count)
                continue

            if col_name == "Rank": # Rank should typically be integer or NA, no special formatting here
                # Convert Rank to integer if possible, otherwise keep as is (e.g. for NA)
                try:
                    # Attempt to convert to Int64 to handle pd.NA
                    formatted_df[col_name] = formatted_df[col_name].astype(pd.Int64Dtype())
                except Exception:
                    pass # If conversion fails, leave as is
                continue


            new_col_values = []
            for x_cell_value in formatted_df[col_name]:
                original_value_for_cell = x_cell_value

                numeric_x = x_cell_value
                is_cell_numeric_type = isinstance(x_cell_value, (int, float, np.number))

                if not is_cell_numeric_type:
                    try:
                        numeric_x = pd.to_numeric(x_cell_value)
                        is_cell_numeric_type = True
                    except ValueError:
                        is_cell_numeric_type = False

                if pd.isna(numeric_x):
                    new_col_values.append("") # Display NaNs as empty strings
                    continue

                formatted_cell_value = original_value_for_cell

                if is_cell_numeric_type:
                    if is_mt_bench_tab: # Special handling for mt_bench tab
                        if isinstance(numeric_x, float):
                            formatted_cell_value = f"{numeric_x:.2f}"
                        else:
                            formatted_cell_value = numeric_x
                    else: # For all other tabs
                        if isinstance(numeric_x, (int, float)) and 0 <= numeric_x <= 1.0:
                            val_multiplied = numeric_x * 100
                            # If original was 0 or 1 (resulting in 0 or 100), format as integer
                            if numeric_x == 1.0 or numeric_x == 0.0:
                                formatted_cell_value = f"{val_multiplied:.0f}" # "100" or "0"
                            else:
                                # Otherwise, format to 2 decimal places (e.g., 88.00, 75.50)
                                formatted_cell_value = f"{val_multiplied:.2f}"
                        elif isinstance(numeric_x, float):
                            formatted_cell_value = f"{numeric_x:.2f}"
                        else: # Integers outside 0-1 range, etc.
                            formatted_cell_value = numeric_x

                new_col_values.append(formatted_cell_value)
            formatted_df[col_name] = new_col_values
        return formatted_df

    def _apply_markdown_and_bolding(self, df_with_general_formats: pd.DataFrame) -> pd.DataFrame:
        # ... (بدون تغییر نسبت به نسخه قبلی شما) ...
        if df_with_general_formats.empty: return df_with_general_formats
        formatted_df = df_with_general_formats.copy()

        model_id_col_original = self.model_identifier_column

        if model_id_col_original in formatted_df.columns and 'model_url' in formatted_df.columns:
            def create_markdown_link(row):
                model_id_val = row[model_id_col_original]
                url = row['model_url']

                display_conf = self.model_display_configs.get(str(model_id_val), {})
                display_name = display_conf.get('display_name', str(model_id_val))
                url_for_link = display_conf.get('url', url if pd.notna(url) else 'https://google.com')
                if not url_for_link or pd.isna(url_for_link): url_for_link = 'https://google.com'
                return f"[{display_name}]({url_for_link})"
            formatted_df[model_id_col_original] = formatted_df.apply(create_markdown_link, axis=1)

        for col_name_original in self.numeric_score_columns_for_bolding:
            if col_name_original in formatted_df.columns:
                def to_numeric_for_max(val):
                    if isinstance(val, str):
                        # Percentage sign is no longer added, so no need to check for it here
                        # if val.endswith('%'):
                        #     try: return float(val[:-1])
                        #     except ValueError: return -np.inf
                        try: return float(val) # Handles "88.00", "75.50", "100", "0"
                        except ValueError: return -np.inf
                    return val if pd.notna(val) else -np.inf

                numeric_series_for_max = formatted_df[col_name_original].apply(to_numeric_for_max)

                if not numeric_series_for_max.empty and numeric_series_for_max.notna().any() and \
                   pd.api.types.is_numeric_dtype(numeric_series_for_max) and not numeric_series_for_max.eq(-np.inf).all():
                    max_val_numeric = numeric_series_for_max.max(skipna=True) # Ensure skipna=True for max
                    if pd.notna(max_val_numeric) and max_val_numeric != -np.inf:
                        # Iterate using index to ensure correct .loc access
                        for i in numeric_series_for_max.index:
                            current_numeric_val = numeric_series_for_max.loc[i]
                            if pd.notna(current_numeric_val) and current_numeric_val == max_val_numeric:
                                display_val_to_bold = formatted_df.loc[i, col_name_original]
                                if not (isinstance(display_val_to_bold, str) and display_val_to_bold.startswith("**") and display_val_to_bold.endswith("**")):
                                    formatted_df.loc[i, col_name_original] = f"**{display_val_to_bold}**"
                            elif pd.isna(current_numeric_val) or current_numeric_val == -np.inf:
                                cell_content = formatted_df.loc[i, col_name_original]
                                if cell_content is None or \
                                   (isinstance(cell_content, str) and \
                                    cell_content.strip().lower() in ["n/a", "", "unknown", "nan"]): # Standardize NA display
                                    formatted_df.loc[i, col_name_original] = ""
        return formatted_df
    # ... (بقیه متدهای LeaderboardApp بدون تغییر باقی می‌مانند، از جمله _get_gr_datatypes, get_prepared_dataframe, make_update_fn_for_task_closure, _create_and_bind_dataframe_component, create_gradio_interface, run_standalone) ...

    @staticmethod
    def _get_gr_datatypes(df_with_original_cols: pd.DataFrame, model_id_col_original_name: str, score_cols_original_names: List[str]) -> List[str]:
        datatypes = []
        if df_with_original_cols.empty: return []

        markdown_cols_original_names = {model_id_col_original_name}
        markdown_cols_original_names.add("parameters_count")
        markdown_cols_original_names.update(score_cols_original_names)

        for col_name_original in df_with_original_cols.columns:
            if col_name_original == "Rank":
                datatypes.append("number") # Rank can be number or string if NA
            elif col_name_original in markdown_cols_original_names:
                datatypes.append("markdown")
            else:
                # Most other formatted cells become strings
                # Checking the dtype of the formatted column can be more robust
                # For now, default to str for non-markdown, non-rank
                datatypes.append("str")
        return datatypes

    def get_prepared_dataframe(self, task_key: str, source_filter: str = "All", name_filter_query: str = "") -> pd.DataFrame:
        original_df_for_task = self.raw_dataframes.get(task_key)
        if original_df_for_task is None or original_df_for_task.empty:
            return pd.DataFrame()

        processed_df = original_df_for_task.copy()

        parent_nlu_nlg_task_keys = ["persian_nlg", "persian_nlu"]
        if task_key in parent_nlu_nlg_task_keys:
            cols_to_drop_due_to_object = []
            for col_name in processed_df.columns:
                if processed_df[col_name].apply(lambda x: isinstance(x, dict)).any():
                    cols_to_drop_due_to_object.append(col_name)
            if cols_to_drop_due_to_object:
                logger.info(f"For overview task '{task_key}', dropping object columns: {cols_to_drop_due_to_object}")
                processed_df = processed_df.drop(columns=cols_to_drop_due_to_object, errors='ignore')

        if 'source_type' in processed_df.columns and source_filter != "All":
            processed_df = processed_df[processed_df['source_type'] == source_filter]
            if processed_df.empty: return pd.DataFrame()

        if name_filter_query and self.model_identifier_column in processed_df.columns:
            try:
                processed_df = processed_df[processed_df[self.model_identifier_column].astype(str).str.contains(name_filter_query, case=False, na=False)]
            except Exception as e: logger.error(f"Name filter error: {e}")
            if processed_df.empty: return pd.DataFrame()

        if processed_df.empty: return pd.DataFrame()

        # Apply cell formatting (this now includes the new number formatting rules)
        processed_df = self._apply_general_formatting_to_cells(processed_df, task_key)
        # Apply markdown and bolding
        processed_df = self._apply_markdown_and_bolding(processed_df)

        if self.columns_to_hide:
            columns_to_drop_existing = [col for col in self.columns_to_hide if col in processed_df.columns]
            if columns_to_drop_existing:
                processed_df = processed_df.drop(columns=columns_to_drop_existing, errors='ignore')

        if "Rank" in processed_df.columns:
            # Ensure Rank is first, if it exists
            cols_order = ["Rank"] + [col for col in processed_df.columns if col != "Rank"]
            processed_df = processed_df[cols_order]
        
        # Convert Rank to string for display after all operations, to handle NA consistently with other strings
        if "Rank" in processed_df.columns:
             processed_df["Rank"] = processed_df["Rank"].apply(lambda x: str(int(x)) if pd.notna(x) and isinstance(x, (float,int)) and x == int(x) else (str(x) if pd.notna(x) else ""))


        processed_df = processed_df.fillna("") # Final fillna for display
        return processed_df

    def make_update_fn_for_task_closure(self, task_key_for_df_data: str):
        
        def update_table_data(name_query_str, source_filter_str):
            logger.debug(f"Updating table for task_key '{task_key_for_df_data}' with name: '{name_query_str}', source: '{source_filter_str}'")

            df_original_cols_formatted_values = self.get_prepared_dataframe(
                task_key_for_df_data, source_filter_str, name_query_str
            )

            if df_original_cols_formatted_values.empty:
                base_raw_df = self.raw_dataframes.get(task_key_for_df_data, pd.DataFrame())
                base_raw_df_cols = list(base_raw_df.columns) if not base_raw_df.empty else []

                if base_raw_df_cols:
                    temp_empty_df_orig_cols = pd.DataFrame(columns=base_raw_df_cols)
                    if self.columns_to_hide:
                        cols_to_drop_now = [col for col in self.columns_to_hide if col in temp_empty_df_orig_cols.columns]
                        if cols_to_drop_now:
                            temp_empty_df_orig_cols = temp_empty_df_orig_cols.drop(columns=cols_to_drop_now)

                    if self.main_scores_map.get(task_key_for_df_data) and "Rank" not in temp_empty_df_orig_cols.columns:
                         temp_empty_df_orig_cols.insert(0, "Rank", [])


                    renamed_empty_df = self.column_config.rename_dataframe_columns(temp_empty_df_orig_cols)
                    display_headers = list(renamed_empty_df.columns)
                    gr_datatypes = ["str"] * len(display_headers) if display_headers else ["str"]
                    return gr.DataFrame(value=pd.DataFrame(columns=display_headers), headers=display_headers if display_headers else ["Info"], datatype=gr_datatypes)
                else:
                    info_message = f"No data available for {self.column_config.get_task_tab_name(task_key_for_df_data)} with current filters."
                    return gr.DataFrame(value=pd.DataFrame([{"Info": info_message}]), headers=["Info"], datatype=["str"])

            gr_datatypes = self._get_gr_datatypes(
                df_original_cols_formatted_values,
                self.model_identifier_column,
                self.numeric_score_columns_for_bolding
            )

            df_display_cols_formatted_values = self.column_config.rename_dataframe_columns(df_original_cols_formatted_values)
            display_headers = list(df_display_cols_formatted_values.columns)

            return gr.DataFrame(value=df_display_cols_formatted_values, headers=display_headers, datatype=gr_datatypes)
        return update_table_data


    def _create_and_bind_dataframe_component(self, current_task_key: str, name_search_textbox: gr.Textbox, source_filter_radio: gr.Radio):
        
        initial_df_original_cols = self.get_prepared_dataframe(current_task_key, "All", "")

        current_display_headers = []
        current_datatypes = None
        df_value_for_gr_display_cols = pd.DataFrame()

        if initial_df_original_cols.empty:
            base_df = self.raw_dataframes.get(current_task_key, pd.DataFrame())
            base_df_cols_original = list(base_df.columns) if not base_df.empty else []

            if base_df_cols_original:
                temp_empty_df_orig_cols = pd.DataFrame(columns=base_df_cols_original)
                if self.columns_to_hide:
                    cols_to_drop_now = [col for col in self.columns_to_hide if col in temp_empty_df_orig_cols.columns]
                    if cols_to_drop_now:
                        temp_empty_df_orig_cols = temp_empty_df_orig_cols.drop(columns=cols_to_drop_now)

                if self.main_scores_map.get(current_task_key) and "Rank" not in temp_empty_df_orig_cols.columns:
                    temp_empty_df_orig_cols.insert(0, "Rank", [])

                initial_df_display_cols = self.column_config.rename_dataframe_columns(temp_empty_df_orig_cols)
                current_display_headers = list(initial_df_display_cols.columns)
                current_datatypes = ["str"] * len(current_display_headers) if current_display_headers else ["str"]
                df_value_for_gr_display_cols = pd.DataFrame(columns=current_display_headers)
            else:
                current_display_headers = ["Info"]
                current_datatypes = ["str"]
                df_value_for_gr_display_cols = pd.DataFrame([{"Info":f"No data or columns configured for {self.column_config.get_task_tab_name(current_task_key)}."}])
        else:
            current_datatypes = self._get_gr_datatypes(
                initial_df_original_cols,
                self.model_identifier_column,
                self.numeric_score_columns_for_bolding
            )
            initial_df_display_cols = self.column_config.rename_dataframe_columns(initial_df_original_cols)
            current_display_headers = list(initial_df_display_cols.columns)
            df_value_for_gr_display_cols = initial_df_display_cols

        df_component = gr.DataFrame(
            value=df_value_for_gr_display_cols,
            headers=current_display_headers,
            datatype=current_datatypes,
            interactive=False,
            wrap=True,
            # height=700,
            # elem_id=f"dataframe_{current_task_key}"
        )

        update_fn = self.make_update_fn_for_task_closure(current_task_key)
        filter_inputs = [name_search_textbox, source_filter_radio]

        name_search_textbox.submit(fn=update_fn, inputs=filter_inputs, outputs=[df_component])
        source_filter_radio.change(fn=update_fn, inputs=filter_inputs, outputs=[df_component])

        return df_component

    def create_gradio_interface(self) -> gr.Blocks:
        
        logger.info("Creating Gradio interface with potentially nested tabs.")
        with gr.Blocks(theme=gr.themes.Soft(), elem_id="leaderboard_main_container") as leaderboard_ui_blocks:
            if not self.tab_processing_order and not self.parent_child_task_map:
                gr.Markdown("### Leaderboard Not Configured\n- `tab_processing_order` and `parent_child_task_map` are not defined or empty in `leaderboard_config.yaml`.")
                return leaderboard_ui_blocks
            if not self.raw_dataframes or all(df.empty for df in self.raw_dataframes.values()):
                 gr.Markdown("### No Data Loaded\n- No data loaded from `boards_data/`. Ensure `refresh.py` ran and JSONL files exist.")
                 return leaderboard_ui_blocks

            with gr.Row():
                name_search_textbox = gr.Textbox(label="Search by Model Name", placeholder="Type model name and press Enter...", interactive=True, scale=3)
                source_filter_radio = gr.Radio(choices=["All", "Open-Source", "Closed-Source"], value="All", label="Filter by Model Source", interactive=True, scale=1)

            with gr.Tabs(elem_id="main_benchmark_tabs") as main_tabs:
                processed_top_level_keys = set()

                for main_task_key in self.tab_processing_order:
                    if main_task_key in processed_top_level_keys: continue
                    processed_top_level_keys.add(main_task_key)

                    main_tab_display_label = self.column_config.get_task_tab_name(main_task_key)

                    with gr.TabItem(label=main_tab_display_label, id=f"main_tab_{main_task_key}"):
                        gr.Markdown(f"## {main_tab_display_label}")

                        child_task_keys_for_parent = self.parent_child_task_map.get(main_task_key, [])

                        if child_task_keys_for_parent:
                            with gr.Tabs(elem_id=f"sub_tabs_for_{main_task_key}") as sub_tabs_component:
                                for child_key in child_task_keys_for_parent:
                                    if child_key not in self.raw_dataframes or self.raw_dataframes[child_key].empty: # Check if df is empty
                                        logger.warning(f"Data for sub-task '{child_key}' under parent '{main_task_key}' not loaded or is empty. Skipping sub-tab.")
                                        child_tab_display_label_empty = self.column_config.get_task_tab_name(child_key)
                                        with gr.TabItem(label=child_tab_display_label_empty, id=f"sub_tab_{child_key}_empty"):
                                            gr.Markdown(f"Data for {child_tab_display_label_empty} is not available.")
                                        continue
                                    processed_top_level_keys.add(child_key)
                                    child_tab_display_label = self.column_config.get_task_tab_name(child_key)
                                    with gr.TabItem(label=child_tab_display_label, id=f"sub_tab_{child_key}"):
                                        self._create_and_bind_dataframe_component(child_key, name_search_textbox, source_filter_radio)
                        else: # This main_task_key is a STANDALONE tab
                            if main_task_key not in self.raw_dataframes or self.raw_dataframes[main_task_key].empty: # Check if df is empty
                                logger.warning(f"Data for standalone task '{main_task_key}' not loaded or is empty. Skipping tab content.")
                                gr.Markdown(f"Data for {main_tab_display_label} is not available.")
                                continue
                            self._create_and_bind_dataframe_component(main_task_key, name_search_textbox, source_filter_radio)
            return leaderboard_ui_blocks

    def run_standalone(self) -> None:
        
        logger.info("Running LeaderboardApp in standalone mode.")
        try:
            self.load_data()
            if not self.raw_dataframes or all(df.empty for df in self.raw_dataframes.values()):
                 logger.warning("No data loaded. Leaderboard might be empty or show 'No data' messages.")
            self.generate_model_rankings()
            demo_interface = self.create_gradio_interface()
            demo_interface.launch(server_name="0.0.0.0", server_port=7860, debug=True)
        except Exception as e:
            logger.error(f"Error during standalone run: {e}", exc_info=True)
            try:
                with gr.Blocks() as error_demo: gr.Error(f"Failed to launch LeaderboardApp: {e}")
                error_demo.launch(server_name="0.0.0.0", server_port=7860)
            except Exception as launch_err:
                 logger.error(f"CRITICAL: Failed even to launch the error Gradio page: {launch_err}")


def main():
    
    logger.info(f"Initializing LeaderboardApp with config: {CONFIG_FILE_PATH}")
    if not CONFIG_FILE_PATH.exists():
        logger.critical(f"CRITICAL: Config file '{CONFIG_FILE_PATH.name}' not found at {CONFIG_FILE_PATH}. App cannot start.")
        try:
            with gr.Blocks() as error_demo: gr.Error(f"Config File Not Found: {CONFIG_FILE_PATH}")
            error_demo.launch(server_name="0.0.0.0", server_port=7860)
        except Exception as launch_err:
            logger.error(f"CRITICAL: Failed to launch the error Gradio page for missing config: {launch_err}")
        return
    app = LeaderboardApp(config_path=CONFIG_FILE_PATH)
    app.run_standalone()

if __name__ == '__main__':
    main()