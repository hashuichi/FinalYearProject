from faker import Faker
import pandas as pd
import random

fake = Faker(["en_UK"])

hotel_ext = [
    ("Hotel", 0.7),
    ("Inn", 0.1),
    ("Lodge", 0.1),
    ("Resort", 0.1), 
]

room_types = [
    ("Double bed", 0.4),
    ("2 Single beds", 0.3),
    ("King bed", 0.2),
    ("Queen bed", 0.1),
]

def generate_fake_hotel_data():
    hotel_name = fake.first_name()
    ext, _ = random.choices(hotel_ext, weights=[weight for _, weight in hotel_ext])[0]
    room_type, _ = random.choices(room_types, weights=[weight for _, weight in room_types])[0]
    price = round(random.uniform(50, 499))

    return {
        "Hotel Name": hotel_name +" "+ str(ext),
        "Room Type": room_type,
        "Price": price,
    }

fake_rooms = {}

for i in range(50):
    fake_rooms[i] = generate_fake_hotel_data()

df = pd.DataFrame(fake_rooms).T
print(df)
df.to_csv('hotel_data.csv', index=False)
