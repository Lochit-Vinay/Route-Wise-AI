import sys
sys.path.insert(0, '.')
from src.data_generator import generate_dataset
s, r = generate_dataset()
print(f"Shipments: {len(s)}, Routes: {len(r)}")
print(f"Routes per shipment: min={r.groupby('shipment_id').size().min()}, max={r.groupby('shipment_id').size().max()}")
print(s.head())
print(r.head())
