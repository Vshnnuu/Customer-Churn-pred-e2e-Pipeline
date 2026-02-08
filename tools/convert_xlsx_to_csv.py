from pathlib import Path
import pandas as pd

xlsx_path = Path("data/raw/telco-customer-churn.xlsx")
csv_path = Path("data/raw/telco_churn.csv")

df = pd.read_excel(xlsx_path)          # reads the first sheet by default
df.to_csv(csv_path, index=False)       # writes CSV

print("Saved:", csv_path.resolve())
print("Rows, Cols:", df.shape)
print("Columns:", list(df.columns))

