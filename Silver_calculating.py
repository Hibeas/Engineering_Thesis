import json
import time
from confluent_kafka import Consumer, Producer

# --- KONFIGURACJA ---
KAFKA_SERVER = 'localhost:8081'
INPUT_TOPIC = 'face-landmarks'
OUTPUT_TOPIC = 'face-emotions'

# Config dla Konsumenta (odczyt Bronze)
consumer_conf = {
    'bootstrap.servers': KAFKA_SERVER,
    'group.id': 'silver-expert-v2',
    'auto.offset.reset': 'earliest' # Start from beginning if new group
}

# Config dla Producenta (zapis Silver)
producer_conf = {
    'bootstrap.servers': KAFKA_SERVER
}

consumer = Consumer(consumer_conf)
consumer.subscribe([INPUT_TOPIC])
producer = Producer(producer_conf)

def delivery_report(err, msg):
    if err is not None:
        print(f"❌ Delivery failed: {err}")

def get_expert_emotions(bs):
    """
    Zaawansowana ekstrakcja oparta na Action Units (FACS).
    """
    e = {}

    # 1. RADOŚĆ (Duchenne Smile)
    smile = (bs.get('mouthSmileLeft', 0) + bs.get('mouthSmileRight', 0)) / 2
    squint = (bs.get('cheekSquintLeft', 0) + bs.get('cheekSquintRight', 0)) / 2
    e['HAPPY'] = (smile * 0.7) + (squint * 0.3)

    # 2. SMUTEK
    brow_up = bs.get('browInnerUp', 0)
    frown = (bs.get('mouthFrownLeft', 0) + bs.get('mouthFrownRight', 0)) / 2
    e['SAD'] = (brow_up * 0.6) + (frown * 0.4)

    # 3. ZŁOŚĆ / FRUSTRACJA
    brows_down = (bs.get('browDownLeft', 0) + bs.get('browDownRight', 0)) / 2
    m_press = (bs.get('mouthPressLeft', 0) + bs.get('mouthPressRight', 0)) / 2
    e['ANGRY'] = (brows_down * 0.5) + (m_press * 0.3) + (bs.get('noseSneerLeft', 0) * 0.2)

    # 4. ZASKOCZENIE
    jaw = bs.get('jawOpen', 0)
    eyes_wide = (bs.get('eyeWideLeft', 0) + bs.get('eyeWideRight', 0)) / 2
    e['SURPRISE'] = (jaw * 0.5) + (eyes_wide * 0.5)

    # 5. POGARDA / SCEPTYCYZM
    e['CONTEMPT'] = abs(bs.get('mouthSmileLeft', 0) - bs.get('mouthSmileRight', 0))

    # 6. SKUPIENIE
    e['FOCUS'] = (bs.get('eyeSquintLeft', 0) + bs.get('eyeSquintRight', 0)) / 2 - (smile * 0.5)
    
    return {k: round(max(0, v), 2) for k, v in e.items()}

print(f"🚀 Expert Silver Layer started.")
print(f"📥 Reading: {INPUT_TOPIC} | 📤 Writing: {OUTPUT_TOPIC}")

try:
    while True:
        msg = consumer.poll(1.0)
        if msg is None: 
            continue
        if msg.error():
            print(f"Kafka Error: {msg.error()}")
            continue

        try:
            # 1. Dekodowanie danych z Bronze Layer
            data = json.loads(msg.value().decode('utf-8'))
            bs = data.get('blendshapes', {})
            
            if not bs:
                continue

            # 2. Obliczanie emocji
            emotions = get_expert_emotions(bs)
            dominant = max(emotions, key=emotions.get)
            
            # 3. Przygotowanie paczki Silver
            silver_payload = {
                "student_id": data.get("student_id", "unknown"),
                "ts": int(data.get("ts", time.time()) * 1000),
                "emotions": emotions,
                "dominant": dominant,
                "confidence": emotions[dominant]
            }

            # 4. Wysłanie z powrotem do Kafka (Topic: face-emotions)
            producer.produce(
                OUTPUT_TOPIC, 
                json.dumps(silver_payload).encode('utf-8'),
                callback=delivery_report
            )
            producer.poll(0) # Triggers callbacks

            # Logowanie do konsoli dla Ciebie
            if emotions[dominant] > 0.25:
                print(f"Sent: {dominant} ({emotions[dominant]})")
            else:
                print("Sent: NEUTRAL")

        except Exception as e:
            print(f"Processing error: {e}")

except KeyboardInterrupt:
    print("Stopping...")
finally:
    producer.flush()
    consumer.close()