# Producer Placeholder

def generate_transaction():
    # Simulate PII-masked transaction data
    return {"tx_id": "uuid4", "amount": 100.00, "merchant": "STARBUCKS"}

# In a real scenario, this would publish to Kafka topic 'transactions'
print("Transaction data simulation initiated.")