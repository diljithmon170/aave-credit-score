## Aave V2 Wallet Credit Scoring

### Overview
This project assigns a credit score (0-1000) to each wallet interacting with the Aave V2 protocol, based on historical transaction behavior. The score reflects reliability and risk, using only transaction-level data.

### Features Engineered
- **Total Deposits:** Sum of all deposit amounts.
- **Total Borrows:** Sum of all borrow amounts.
- **Total Repays:** Sum of all repay amounts.
- **Repay Ratio:** Repays divided by borrows.
- **Liquidation Count:** Number of liquidation events.
- **Active Days:** Number of unique days with activity.

### Scoring Logic
- **Repay Ratio:** Up to 300 points (higher is better).
- **Deposit Activity:** Up to 200 points (more deposits, more points).
- **Liquidation Penalty:** Up to 300 points (fewer liquidations, more points).
- **Active Days:** Up to 200 points (more active days, more points).
- **Total Score:** Sum of above, capped at 1000.

### Processing Flow
1. Load JSON data and convert to DataFrame.
2. Extract transaction features per wallet.
3. Calculate credit score using engineered features.
4. Save scores to wallet_scores.csv.
5. Plot score distribution.

### How to Run
`ash
python main.py
`
Outputs:
- wallet_scores.csv: Wallet and score.
- score_distribution.png: Score distribution plot.

### Extensibility
- Add more features (e.g., protocol, asset type).
- Tune scoring weights.
- Integrate with ML models for prediction.

---

