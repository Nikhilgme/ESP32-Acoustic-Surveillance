import os
import re
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("=" * 75)
print("   ESP32-S3 EDGE AI: PCEN + 1D-CNN ACOUSTIC THREAT TRAINING PIPELINE")
print("=" * 75)

def parse_esp32_log(filepath):
    samples = []
    pattern = re.compile(r"DATA:\[([0-9.,\s]+)\]")
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                try:
                    vals = [float(x) for x in match.group(1).split(",") if x.strip() != ""]
                    if len(vals) == 128:
                        samples.append(vals)
                except ValueError:
                    continue
    return np.array(samples, dtype=np.float32)

print("\n[STEP 1/5] Loading Raw PCEN Audio Logs from ESP32-S3...")
raw_safe = parse_esp32_log("safe_data2.txt")
raw_threat = parse_esp32_log("threat_data.txt")
print(f"  -> Loaded {len(raw_safe)} raw 128-bin frames from 'safe_data2.txt' (20 mins Silence)")
print(f"  -> Loaded {len(raw_threat)} raw 128-bin frames from 'threat_data.txt' (20 mins Threats)")

print("\n[STEP 2/5] Filtering & Preprocessing PCEN Feature Tensors...")
safe_means = np.mean(raw_safe, axis=1)
safe_maxs = np.max(raw_safe, axis=1)
clean_safe = raw_safe[(safe_means > 0.05) & (safe_means < 0.32) & (safe_maxs < 0.85)]

threat_means = np.mean(raw_threat, axis=1)
threat_maxs = np.max(raw_threat, axis=1)
threat_p75 = np.percentile(raw_threat, 75, axis=1)
clean_threat = raw_threat[(threat_means > 0.32) | ((threat_maxs > 0.50) & (threat_p75 > 0.22))]

np.random.seed(42)
tf.random.set_seed(42)

def augment_dataset(data, target_count, noise_std, scale_range):
    out = list(data)
    while len(out) < target_count:
        sample = data[np.random.randint(0, len(data))].copy()
        sample = sample * np.random.uniform(*scale_range) + np.random.normal(0, noise_std, size=sample.shape)
        out.append(np.clip(sample, 0.0, None))
    return np.array(out[:target_count], dtype=np.float32)

X_safe = augment_dataset(clean_safe, 2000, 0.01, (0.85, 1.12))
X_threat = augment_dataset(clean_threat, 2000, 0.02, (0.85, 1.25))

X = np.vstack([X_safe, X_threat])[..., np.newaxis]
y = np.array([0] * len(X_safe) + [1] * len(X_threat), dtype=np.int32)
y_cat = tf.keras.utils.to_categorical(y, num_classes=2)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_cat, test_size=0.20, random_state=42, stratify=y
)

print(f"  -> Clean Safe Samples (Class 0):   {len(X_safe)} frames")
print(f"  -> Clean Threat Samples (Class 1): {len(X_threat)} frames (Claps, Metal, Drops)")
print(f"  -> Total Dataset Shape:            {X.shape} (4,000 samples x 128 PCEN bins x 1 channel)")
print(f"  -> Training Split (80%):           {len(X_train)} samples")
print(f"  -> Validation Split (20%):         {len(X_val)} samples")

print("\n[STEP 3/5] Building 1D-CNN Architecture...")
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 1), name="PCEN_Input_128x1"),
    tf.keras.layers.Conv1D(16, kernel_size=5, strides=2, padding="same", activation="relu", name="Conv1D_Layer1"),
    tf.keras.layers.MaxPooling1D(pool_size=2, name="MaxPool1D_Layer1"),
    tf.keras.layers.Conv1D(32, kernel_size=3, strides=1, padding="same", activation="relu", name="Conv1D_Layer2"),
    tf.keras.layers.GlobalAveragePooling1D(name="GlobalAvgPool1D"),
    tf.keras.layers.Dense(16, activation="relu", name="Dense_Hidden"),
    tf.keras.layers.Dropout(0.2, name="Dropout_Regularization"),
    tf.keras.layers.Dense(2, activation="softmax", name="Output_Softmax")
], name="ESP32_Acoustic_1D_CNN")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

total_params = model.count_params()
print("-" * 75)
print(f"  MODEL HYPERPARAMETERS SUMMARY (FOR VIVA / EVALUATION):")
print(f"  * Architecture:          1D Convolutional Neural Network (1D-CNN)")
print(f"  * Input Tensor Shape:    (128, 1) -> 128 PCEN Mel-Frequency/Time Bins")
print(f"  * Total Parameters:      {total_params:,} parameters (100% Trainable)")
print(f"  * Optimizer & Loss:      Adam (lr = 0.001) | Categorical Cross-Entropy")
print(f"  * Training Configuration: 30 Epochs | Batch Size = 32")
print("-" * 75)

print("\n[STEP 4/5] Training 1D-CNN Model (30 Epochs)...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=32,
    verbose=1
)

val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
print(f"\n  -> FINAL VALIDATION ACCURACY: {val_acc * 100:.2f}% | VALIDATION LOSS: {val_loss:.4f}")

print("\n[STEP 5/5] Performing Full INT8 Post-Training Quantization for ESP-NN...")
def representative_dataset():
    for i in range(200):
        idx = np.random.randint(0, len(X_train))
        yield [X_train[idx:idx+1]]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_bytes = converter.convert()
with open("intruder_model_int8.tflite", "wb") as f:
    f.write(tflite_bytes)

interpreter = tf.lite.Interpreter(model_content=tflite_bytes)
interpreter.allocate_tensors()
in_details = interpreter.get_input_details()[0]
out_details = interpreter.get_output_details()[0]
in_scale, in_zp = in_details["quantization"]
out_scale, out_zp = out_details["quantization"]

print(f"  -> Quantization Type:    Full Integer (INT8 Weights & Activations)")
print(f"  -> Input Quantization:   Scale = {in_scale:.8f}, Zero-Point = {in_zp}")
print(f"  -> Output Quantization:  Scale = {out_scale:.8f}, Zero-Point = {out_zp}")
print(f"  -> Final TFLite Size:    {len(tflite_bytes):,} bytes ({len(tflite_bytes)/1024:.2f} KB)")

print("\n  LIVE INT8 MODEL VERIFICATION ON TEST SAMPLES:")
def test_int8_sample(sample_float, label_desc):
    q_in = np.clip(np.round(sample_float / in_scale + in_zp), -128, 127).astype(np.int8).reshape(1, 128, 1)
    interpreter.set_tensor(in_details["index"], q_in)
    interpreter.invoke()
    q_out = interpreter.get_tensor(out_details["index"])[0]
    print(f"  * {label_desc:<28} -> Safe: {q_out[0]:>4d} | Threat: {q_out[1]:>4d}")

test_int8_sample(clean_safe[10], "Sample 1 (Ambient Silence)")
test_int8_sample(clean_safe[25], "Sample 2 (Quiet Room)")
test_int8_sample(clean_threat[0], "Sample 3 (Metal Clanking)")
test_int8_sample(clean_threat[5], "Sample 4 (Hand Clap / Drop)")

hex_lines = []
for i in range(0, len(tflite_bytes), 12):
    chunk = tflite_bytes[i:i+12]
    hex_lines.append("  " + ", ".join([f"0x{b:02x}" for b in chunk]))
hex_body = ",\n".join(hex_lines)

header_text = (
    "#ifndef MODEL_DATA_H\n"
    "#define MODEL_DATA_H\n\n"
    "// Auto-generated INT8 Quantized 1D-CNN Model for ESP32-S3\n"
    f"#define MODEL_INPUT_SCALE {in_scale:.8f}f\n"
    f"#define MODEL_INPUT_ZERO_POINT {in_zp}\n"
    f"#define MODEL_OUTPUT_SCALE {out_scale:.8f}f\n"
    f"#define MODEL_OUTPUT_ZERO_POINT {out_zp}\n\n"
    "alignas(16) const unsigned char g_intruder_model_data[] = {\n"
    f"{hex_body}\n"
    "};\n"
    f"const unsigned int g_intruder_model_data_len = {len(tflite_bytes)};\n\n"
    "#endif // MODEL_DATA_H\n"
)

with open("model_data.h", "w") as f:
    f.write(header_text)

print("\n" + "=" * 75)
print(f"  SUCCESS: Exported 'model_data.h' ({len(tflite_bytes):,} bytes) Ready for ESP32-S3!")
print("=" * 75)
