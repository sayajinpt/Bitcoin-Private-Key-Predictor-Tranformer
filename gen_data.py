import os
import hashlib
import ecdsa
import base58
import csv

def generate_key_pair():
    private_key_bytes = os.urandom(32)
    
    sk = ecdsa.SigningKey.from_string(private_key_bytes, curve=ecdsa.SECP256k1)

    public_key_bytes = sk.get_verifying_key().to_string("compressed")

    sha256_hash = hashlib.sha256(public_key_bytes).digest()
    ripemd160_hash = hashlib.new('ripemd160', sha256_hash).digest()
    extended_hash = b'\x00' + ripemd160_hash
    checksum = hashlib.sha256(hashlib.sha256(extended_hash).digest()).digest()[:4]
    final_hash = extended_hash + checksum
    bitcoin_address = base58.b58encode(final_hash).decode()

    return private_key_bytes.hex(), bitcoin_address

with open('dataset.csv', 'w', newline='') as csvfile:
    fieldnames = ['target', 'input']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    for _ in range(1000000):
        private_key, bitcoin_address = generate_key_pair()
        

        if len(bitcoin_address) == 34:
            writer.writerow({'target': private_key, 'input': bitcoin_address})

print("dataset created successfully.")
