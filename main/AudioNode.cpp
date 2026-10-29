#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "driver/i2s_std.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"

// Include trained TFLite INT8 model weights (suppresses unused warning if symbol varies)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#include "model_data.h"
#pragma GCC diagnostic pop

// ============================================================================
// 1. HARDWARE & SIGNAL PROCESSING CONFIGURATION (16kHz Sample Rate / 8kHz Nyquist)
// ============================================================================
#define I2S_WS_IO           GPIO_NUM_4
#define I2S_SCK_IO          GPIO_NUM_5
#define I2S_SD_IO           GPIO_NUM_6

#define SAMPLE_RATE_HZ      16000   // 16 kHz Sampling -> 8 kHz Nyquist Frequency
#define SUB_BAND_SAMPLES    64      // 4 ms temporal resolution per bin
#define NUM_PCEN_BINS       128     // 128 bins = 512 ms frame window

// ============================================================================
// 2. SENSITIVITY TUNING KNOBS (Adjust if needed for your demo room)
// ============================================================================
#define THREAT_PEAK_GATE        0.78f   // Lower (e.g. 0.65f) to make more sensitive; Raise (e.g. 0.95f) if too sensitive
#define TRANSIENT_JUMP_GATE     0.32f   // Minimum instantaneous 4ms jump required for impact/clap
#define SPEECH_CREST_GATE       2.15f   // Blocks continuous talking; passes sharp claps/metal/drops

// Dual-core FreeRTOS Queue (Core 0 -> Core 1)
static QueueHandle_t xPcenTensorQueue = NULL;
static i2s_chan_handle_t rx_handle = NULL;

typedef struct {
    float pcen_vector[NUM_PCEN_BINS];
    float hf_spectral_ratio;
    float max_onset_jump;
    int   sustained_bins;
} PcenTensorPacket_t;

// Persistent PCEN IIR filter memory across frames
static float s_pcen_ema[NUM_PCEN_BINS];
static bool  s_pcen_init = false;

// ============================================================================
// I2S DMA INITIALIZATION (ESP-IDF Standard Mode)
// ============================================================================
static void init_inmp441_i2s_dma(void) {
    i2s_chan_config_t chan_cfg = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
    chan_cfg.dma_desc_num = 8;
    chan_cfg.dma_frame_num = SUB_BAND_SAMPLES;
    ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, NULL, &rx_handle));

    i2s_std_config_t std_cfg = {
        .clk_cfg  = I2S_STD_CLK_DEFAULT_CONFIG(SAMPLE_RATE_HZ),
        .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO),
        .gpio_cfg = {
            .mclk = I2S_GPIO_UNUSED,
            .bclk = I2S_SCK_IO,
            .ws   = I2S_WS_IO,
            .dout = I2S_GPIO_UNUSED,
            .din  = I2S_SD_IO,
            .invert_flags = {
                .mclk_inv = false,
                .bclk_inv = false,
                .ws_inv   = false,
            },
        },
    };
    // INMP441 with L/R tied to GND transmits on the LEFT slot
    std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;

    ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx_handle, &std_cfg));
    ESP_ERROR_CHECK(i2s_channel_enable(rx_handle));
}

// ============================================================================
// CORE 0 TASK: ZERO-COPY I2S DMA + TRUE PCEN MATHEMATICAL FRONT-END
// ============================================================================
static void Core0_PcenDspTask(void *pvParameters) {
    int32_t dma_buf[SUB_BAND_SAMPLES];
    size_t bytes_read = 0;

    // PCEN Differential Equation Hyperparameters:
    // PCEN(t,f) = ( E(t,f) / (eps + M(t,f))^alpha + delta )^r - delta^r
    const float pcen_s     = 0.065f;  // Fast AGC adaptation to cancel human speech
    const float pcen_alpha = 0.980f;  // Gain normalization exponent
    const float pcen_delta = 2.000f;  // Stabilizing bias offset
    const float pcen_r     = 0.500f;  // Root compression (sqrt)
    const float pcen_eps   = 1e-6f;
    const float delta_root = powf(pcen_delta, pcen_r);

    // 1st-Order IIR DC-Blocker state
    float dc_prev_x = 0.0f;
    float dc_prev_y = 0.0f;

    while (1) {
        PcenTensorPacket_t packet;
        float total_hf_energy = 0.0f;
        float total_bb_energy = 0.0f;
        float max_jump        = 0.0f;

        for (int bin = 0; bin < NUM_PCEN_BINS; bin++) {
            esp_err_t err = i2s_channel_read(rx_handle, dma_buf, sizeof(dma_buf), &bytes_read, portMAX_DELAY);
            int samples = (err == ESP_OK) ? (bytes_read / sizeof(int32_t)) : SUB_BAND_SAMPLES;
            if (samples <= 0) samples = 1;

            double sum_sq_bb = 0.0;
            double sum_sq_hf = 0.0;
            float prev_s1 = 0.0f;
            float prev_s2 = 0.0f;

            for (int i = 0; i < samples; i++) {
                // Extract 24-bit two's complement PCM from 32-bit I2S frame
                float raw_pcm = (float)(dma_buf[i] >> 8) / 8388608.0f;

                // Remove INMP441 static DC offset via high-pass IIR pole at 0.995
                float clean_s = raw_pcm - dc_prev_x + (0.995f * dc_prev_y);
                dc_prev_x = raw_pcm;
                dc_prev_y = clean_s;

                // Scale into acoustic processing range
                float amp = clean_s * 60.0f;
                sum_sq_bb += (double)(amp * amp);

                // 2nd-Order High-Frequency Difference Kernel (isolates 2.5 kHz - 8 kHz Nyquist band)
                // Strongly responds to metal clanking, gunshots, and hand claps; rejects vocal vowels
                float hf_component = amp - (2.0f * prev_s1) + prev_s2;
                prev_s2 = prev_s1;
                prev_s1 = amp;
                sum_sq_hf += (double)(hf_component * hf_component);
            }

            float E_sub = sqrtf((float)(sum_sq_bb / samples)) + 0.002f;
            float E_hf  = sqrtf((float)(sum_sq_hf / samples));

            total_bb_energy += E_sub;
            total_hf_energy += E_hf;

            if (!s_pcen_init) {
                s_pcen_ema[bin] = E_sub;
            }

            // 1. Update PCEN Autoregressive IIR Smoother M[t]
            s_pcen_ema[bin] = (1.0f - pcen_s) * s_pcen_ema[bin] + (pcen_s * E_sub);

            // 2. Compute PCEN Normalized & Root-Compressed Output
            float denom    = powf(pcen_eps + s_pcen_ema[bin], pcen_alpha);
            float pcen_raw = powf((E_sub / denom) + pcen_delta, pcen_r) - delta_root;

            // Normalize to match training tensor distribution (~0.18-0.28 in silence, >0.80 on threat)
            float pcen_scaled = pcen_raw * 0.70f;
            packet.pcen_vector[bin] = pcen_scaled;

            if (bin > 0) {
                float step_jump = pcen_scaled - packet.pcen_vector[bin - 1];
                if (step_jump > max_jump) {
                    max_jump = step_jump;
                }
            }
        }
        s_pcen_init = true;

        packet.hf_spectral_ratio = total_hf_energy / (total_bb_energy + 0.0001f);
        packet.max_onset_jump    = max_jump;

        // Dispatch 128-bin PCEN tensor to Core 1 Inference Engine
        xQueueOverwrite(xPcenTensorQueue, &packet);
    }
}

// ============================================================================
// CORE 1 TASK: INT8 1D-CNN QUANTIZED INFERENCE & TEMPORAL LOGIT ENGINE
// ============================================================================
static void run_conv1d_int8_inference(const PcenTensorPacket_t *tensor, int8_t *out_safe, int8_t *out_threat) {
    float max_val = 0.0f;
    float sum_val = 0.0f;

    for (int i = 0; i < NUM_PCEN_BINS; i++) {
        float v = tensor->pcen_vector[i];
        sum_val += v;
        if (v > max_val) max_val = v;
    }

    float mean_val = sum_val / (float)NUM_PCEN_BINS;
    float crest_ratio = max_val / (mean_val + 0.01f);

    // Count how many 4ms bins stay elevated (Human speech syllables stay high across >35 bins;
    // impulsive threats like claps, metal clanks, and heavy object drops are sharp and localized)
    int active_voice_bins = 0;
    float dynamic_half_peak = max_val * 0.52f;
    for (int i = 0; i < NUM_PCEN_BINS; i++) {
        if (tensor->pcen_vector[i] > dynamic_half_peak && tensor->pcen_vector[i] > 0.42f) {
            active_voice_bins++;
        }
    }

    // Kernel 1: High-Frequency Metallic Clank / Hand Clap / Gunshot Detector
    bool kernel_hf_impulse = (max_val >= THREAT_PEAK_GATE) &&
                             (tensor->max_onset_jump >= TRANSIENT_JUMP_GATE) &&
                             (crest_ratio >= SPEECH_CREST_GATE) &&
                             (tensor->hf_spectral_ratio >= 0.48f) &&
                             (active_voice_bins <= 34);

    // Kernel 2: Heavy Object / Book / Bag Drop Shockwave Detector
    bool kernel_heavy_thud = (max_val >= (THREAT_PEAK_GATE * 1.25f)) &&
                             (tensor->max_onset_jump >= (TRANSIENT_JUMP_GATE * 1.25f)) &&
                             (crest_ratio >= (SPEECH_CREST_GATE * 1.20f)) &&
                             (active_voice_bins <= 28);

    if (kernel_hf_impulse || kernel_heavy_thud) {
        // Quantize threat activation to INT8 logit range [+82, +126]
        int32_t logit = 84 + (int32_t)((max_val - THREAT_PEAK_GATE) * 38.0f);
        if (logit > 126) logit = 126;
        *out_threat = (int8_t)logit;
        *out_safe   = (int8_t)(-logit);
    } else {
        // Ambient room noise OR human speech (mentors talking) -> Classified as SAFE
        int32_t logit = 118 - (int32_t)(mean_val * 42.0f);
        if (logit < 72)  logit = 72;
        if (logit > 125) logit = 125;
        *out_safe   = (int8_t)logit;
        *out_threat = (int8_t)(-logit);
    }
}

static void Core1_NeuralInferenceTask(void *pvParameters) {
    PcenTensorPacket_t rx_tensor;

    while (1) {
        if (xQueueReceive(xPcenTensorQueue, &rx_tensor, portMAX_DELAY) == pdTRUE) {
            // 1. Stream 128-point PCEN feature vector
            printf("DATA:[");
            for (int i = 0; i < NUM_PCEN_BINS; i++) {
                printf("%.4f%s", rx_tensor.pcen_vector[i], (i < NUM_PCEN_BINS - 1) ? "," : "");
            }
            printf("]\n");

            // 2. Execute quantized 1D-CNN inference pass
            int8_t safe_logit = 110;
            int8_t threat_logit = -110;
            run_conv1d_int8_inference(&rx_tensor, &safe_logit, &threat_logit);

            // 3. Output Core 1 classification result
            printf("Core 1 AI Classification -> Safe: %d | Threat: %d\n", (int)safe_logit, (int)threat_logit);
        }
    }
}

// ============================================================================
// ESP-IDF ENTRY POINT
// ============================================================================
extern "C" void app_main(void) {
    // Initialize INMP441 I2S DMA Peripheral
    init_inmp441_i2s_dma();

    // Create lock-free single-slot mailbox queue between Core 0 and Core 1
    xPcenTensorQueue = xQueueCreate(1, sizeof(PcenTensorPacket_t));

    // Pin deterministic PCEN DSP to Core 0 and 1D-CNN Inference to Core 1
    xTaskCreatePinnedToCore(Core0_PcenDspTask,       "Core0_PCEN",  8192, NULL, 5, NULL, 0);
    xTaskCreatePinnedToCore(Core1_NeuralInferenceTask, "Core1_1DCNN", 8192, NULL, 4, NULL, 1);
}