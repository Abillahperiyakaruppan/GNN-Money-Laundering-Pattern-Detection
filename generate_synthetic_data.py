# generate_synthetic_data.py
import csv, random
from datetime import datetime, timedelta

def make_accounts(n_accounts=1500):
    accounts = []
    for i in range(n_accounts):
        accounts.append({
            "account_id": f"A{i:06d}",
            "signup_days": random.randint(1, 365*5)
        })
    return accounts

def add_transactions(accounts, n_tx=8000, suspicious_fraction=0.02):
    account_ids = [a['account_id'] for a in accounts]
    now = datetime.now()
    transactions = []
    suspicious_accounts = set()

    # create a suspicious circular chain
    n_susp = max(2, int(len(accounts) * suspicious_fraction))
    chain = random.sample(account_ids, k=min(8, n_susp+3))
    tstamp = now - timedelta(hours=1)
    for i in range(len(chain)):
        src = chain[i]
        dst = chain[(i+1) % len(chain)]
        amt = random.uniform(9000, 50000)
        transactions.append((src, dst, round(amt,2), tstamp.isoformat()))
        suspicious_accounts.update([src,dst])
        tstamp -= timedelta(seconds=random.randint(10,300))

    # funnel pattern: many small -> mule
    mule = random.choice(account_ids)
    for _ in range(40):
        src = random.choice(account_ids)
        if src == mule: continue
        amt = random.uniform(500, 2000)
        transactions.append((src, mule, round(amt,2), (now - timedelta(days=random.randint(0,30))).isoformat()))
        suspicious_accounts.add(mule)

    # random normal transactions
    for _ in range(n_tx):
        src = random.choice(account_ids)
        dst = random.choice(account_ids)
        if dst == src: continue
        amt = max(1.0, min(100000.0, random.expovariate(1/2000)))
        t = now - timedelta(days=random.randint(0,365), seconds=random.randint(0,86400))
        transactions.append((src, dst, round(amt,2), t.isoformat()))

    return transactions, suspicious_accounts

def save_csv(transactions, accounts, tfile="transactions.csv", afile="accounts.csv", suspicious_accounts=set()):
    with open(tfile, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["src", "dst", "amount", "timestamp"])
        w.writerows(transactions)
    with open(afile, "w", newline='') as f:
        w = csv.writer(f)
        w.writerow(["account_id", "signup_days", "is_suspicious"])
        for a in accounts:
            w.writerow([a['account_id'], a['signup_days'], 1 if a['account_id'] in suspicious_accounts else 0])

if __name__ == "__main__":
    accounts = make_accounts(1500)
    txs, susp = add_transactions(accounts, n_tx=8000, suspicious_fraction=0.015)
    save_csv(txs, accounts, suspicious_accounts=susp)
    print("Saved transactions.csv and accounts.csv. Suspicious accounts:", len(susp))
