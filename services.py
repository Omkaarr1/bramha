# services.py
import io
import re
from typing import Optional, Tuple

import pandas as pd
from openai import OpenAI

client = OpenAI()  # uses OPENAI_API_KEY from env / secrets


# ----------------- Cleaning helpers ----------------- #
def to_number(x) -> Optional[float]:
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s == "" or s.lower() in {"nan", "none"}:
        return None
    s = re.sub(r"[,\s]", "", s)
    s = re.sub(r"[₹£$€]", "", s)
    if re.fullmatch(r"^\(.*\)$", s):
        s = "-" + s.strip("()")
    try:
        return float(s)
    except Exception:
        return None


def tidy_excel_to_long(excel: bytes, sheet_name: str = "BS"):
    """
    Normalise a BS/TB Excel sheet to long-form:
    columns: [Line Item, Year, Amount]
    """
    book = io.BytesIO(excel)
    df_raw = pd.read_excel(book, sheet_name=sheet_name, header=None)
    candidate = df_raw.iloc[:, 0:12].copy()

    header_idx = None
    year_cols = []
    for i in range(min(15, len(candidate))):
        row = candidate.iloc[i].astype(str).tolist()
        yrs = []
        for j, v in enumerate(row):
            if re.fullmatch(r"20\d{2}", v):
                yrs.append((j, int(v)))
        if len(yrs) >= 3:
            header_idx = i
            year_cols = [j for j, _ in yrs]
            break

    if header_idx is None:
        # Fallback layout
        df = df_raw.iloc[6:, [4, 5, 6, 7, 8, 9]].copy()
        years = ["2025", "2024", "2023", "2022", "2021"]
        df.columns = ["Line Item"] + years
    else:
        first_text_col = None
        for c in range(candidate.shape[1]):
            col_vals = candidate.iloc[header_idx + 1 :, c]
            if col_vals.astype(str).str.strip().str.len().max() > 0:
                first_text_col = c
                break
        if first_text_col is None:
            raise ValueError("Could not detect Line Item column.")
        years = [str(int(candidate.iloc[header_idx, c])) for c in year_cols]
        keep_cols = [first_text_col] + year_cols
        df = candidate.iloc[header_idx + 1 :, keep_cols].copy()
        df.columns = ["Line Item"] + years

    df["Line Item"] = df["Line Item"].astype(str).str.strip()
    df = df[~df["Line Item"].str.fullmatch(r"(?i)(nan|none)?\s*")]

    years_sorted = sorted([int(y) for y in df.columns[1:]])
    years_sorted_str = [str(y) for y in years_sorted]
    for y in years_sorted_str:
        df[y] = df[y].apply(to_number)
    if years_sorted_str:
        df = df.dropna(subset=years_sorted_str, how="all")
    df_long = (
        df.melt(
            id_vars=["Line Item"],
            value_vars=years_sorted_str,
            var_name="Year",
            value_name="Amount",
        )
        .dropna(subset=["Amount"])
    )
    df_long["Year"] = df_long["Year"].astype(int)
    df_long = df_long.sort_values(["Year", "Line Item"]).reset_index(drop=True)
    return df_long, years_sorted_str


# ----------------- AI chatbot helper ----------------- #
def ask_sheet_ai(question: str, data_df: pd.DataFrame) -> Tuple[str, bool]:
    """
    Call gpt-4.1-nano to answer a question using the current batch's data.

    Returns (answer, out_of_context_flag).
    """
    if data_df is None or data_df.empty:
        return "I don't see any data for this batch yet. Please upload a file first.", True

    sample_rows = min(300, len(data_df))
    try:
        context_table = data_df.head(sample_rows).to_markdown(index=False)
    except Exception:
        context_table = data_df.head(sample_rows).to_string(index=False)

    prompt = (
        "You are an expert accounting / finance data analyst.\n"
        "You are given a table with columns:\n"
        "- line_item: name of the accounting line item\n"
        "- year: financial year\n"
        "- amount: numeric value\n\n"
        f"DATA (first {sample_rows} rows of the current batch):\n"
        f"{context_table}\n\n"
        f"User question: {question}\n\n"
    )

    try:
        resp = client.responses.create(
            model="gpt-4.1-nano",
            input=prompt,
            max_output_tokens=400,
        )
        text = resp.output[0].content[0].text.strip()
    except Exception as e:
        return f"⚠️ Error calling OpenAI API: {e}", True

    if text == "OUT_OF_CONTEXT":
        return "", True

    return text, False
