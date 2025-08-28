from cryptography.fernet import Fernet

# Generate key
key = Fernet.generate_key()

# Save it to secret.key
with open("secret.key", "wb") as f:
    f.write(key)

print("✅ Key generated and saved to secret.key")
print(key.decode())
