import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load JSON data
with open("user-wallet-transactions.json", "r") as f:
    data = json.load(f)

# Step 2: Convert to DataFrame
df = pd.DataFrame(data)

# DEBUG: Print column names and first rows to verify structure
print("🔍 Columns:", df.columns)
print(df.head(2))

# Step 3: Check the wallet ID field name
wallet_column = 'userWallet' if 'userWallet' in df.columns else 'walletAddress' if 'walletAddress' in df.columns else None
if wallet_column is None:
    raise ValueError("No wallet column found in data.")

# Step 4: Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Step 5: Extract value from actionData
def extract_value(action_data):
    if isinstance(action_data, dict):
        # Try common keys for amount/value
        for key in ['amount', 'value', 'repayAmount', 'borrowAmount', 'depositAmount']:
            if key in action_data:
                return float(action_data[key])
    return 0.0

df['value'] = df['actionData'].apply(extract_value)

# Step 6: Check for required columns
required_columns = ['action', 'value', 'timestamp']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Step 7: Group by wallet
grouped = df.groupby(wallet_column)

wallets = []

# Step 8: Feature extraction per wallet
for wallet, user_df in grouped:
    deposit_total = user_df[user_df['action'] == 'deposit']['value'].sum()
    borrow_total = user_df[user_df['action'] == 'borrow']['value'].sum()
    repay_total = user_df[user_df['action'] == 'repay']['value'].sum()
    liquidation_count = user_df[user_df['action'] == 'liquidationcall'].shape[0]
    active_days = user_df['timestamp'].dt.date.nunique()
    repay_ratio = repay_total / borrow_total if borrow_total > 0 else 0

    wallets.append({
        'user': wallet,
        'deposit_total': deposit_total,
        'borrow_total': borrow_total,
        'repay_total': repay_total,
        'repay_ratio': repay_ratio,
        'liquidation_count': liquidation_count,
        'active_days': active_days
    })

# Step 9: Create DataFrame
wallet_df = pd.DataFrame(wallets)

# Step 10: Credit score calculation
def calculate_score(row):
    score = 0
    score += min(row['repay_ratio'], 1.0) * 300
    score += min(row['deposit_total'] / 10000, 1.0) * 200
    score += (1 - min(row['liquidation_count'] / 5, 1.0)) * 300
    score += min(row['active_days'] / 30, 1.0) * 200
    return round(score)

wallet_df['credit_score'] = wallet_df.apply(calculate_score, axis=1)

# Step 11: Save scores
wallet_df[['user', 'credit_score']].to_csv("wallet_scores.csv", index=False)
print("✅ wallet_scores.csv saved successfully.")

# Step 12: Plot credit score distribution
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))
sns.histplot(wallet_df['credit_score'], bins=10, kde=True, color='skyblue')
plt.title("Wallet Credit Score Distribution", fontsize=14)
plt.xlabel("Credit Score")
plt.ylabel("Number of Wallets")
plt.savefig("score_distribution.png")
plt.show()
print("✅ score_distribution.png saved successfully.")
