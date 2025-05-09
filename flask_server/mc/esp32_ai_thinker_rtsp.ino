/******************************************************
 * RTSP Camera Stream for AI Thinker ESP32-CAM
 *
 * Make sure you:
 * 1. Update `ssid` and `password` below
 * 2. Select the correct board in Arduino IDE:
 *    - Board: "AI Thinker ESP32-CAM" (if available), or
 *    - Board: "ESP32 Wrover Module" + PSRAM Enabled
 * 3. Open Serial Monitor at 115200 baud for debug info
 * 4. Access stream at: rtsp://<DEVICE_IP>:8554/mjpeg/1
 ******************************************************/

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_timer.h"
#include "img_converters.h"
#include "Arduino.h"
#include "fb_gfx.h"
#include "esp_http_server.h"
#include "esp_rtsp_server.h"

// ---- CAMERA PIN DEFINITIONS (AI Thinker) ----
#define PWDN_GPIO_NUM 32
#define RESET_GPIO_NUM -1
#define XCLK_GPIO_NUM 0
#define SIOD_GPIO_NUM 26
#define SIOC_GPIO_NUM 27
#define Y9_GPIO_NUM 35
#define Y8_GPIO_NUM 34
#define Y7_GPIO_NUM 39
#define Y6_GPIO_NUM 36
#define Y5_GPIO_NUM 21
#define Y4_GPIO_NUM 19
#define Y3_GPIO_NUM 18
#define Y2_GPIO_NUM 5
#define VSYNC_GPIO_NUM 25
#define HREF_GPIO_NUM 23
#define PCLK_GPIO_NUM 22

// ---- REPLACE WITH YOUR WI-FI SSID/PASS ----
const char *ssid = "TN-JE3155";
const char *password = "";

// RTSP server handle
rtsp_server_handle_t rtsp_server = NULL;

// Camera frame buffer
camera_fb_t *fb = NULL;

// RTSP stream URI
#define RTSP_STREAM_URI "/mjpeg/1"

// RTSP port (standard is 554, but we use 8554 to avoid needing root privileges)
#define RTSP_PORT 8554

// ---- CAMERA INIT FUNCTION ----
void setupCamera()
{
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sccb_sda = SIOD_GPIO_NUM;
    config.pin_sccb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG; // We want JPEG for streaming

    // If PSRAM is enabled, we can go with higher resolutions
    if (psramFound())
    {
        Serial.println("PSRAM enabled, setting high res");
        config.frame_size = FRAMESIZE_VGA; // VGA(640x480) for RTSP
        config.jpeg_quality = 10;          // lower = higher quality
        config.fb_count = 2;               // 2 frame buffers for smoother streaming
    }
    else
    {
        Serial.println("No PSRAM detected, set lower frame size");
        config.frame_size = FRAMESIZE_SVGA; // e.g. VGA(640x480)
        config.jpeg_quality = 12;
        config.fb_count = 1;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK)
    {
        Serial.printf("Camera init failed with error 0x%x", err);
        while (true)
        {
            delay(100); // don't proceed
        }
    }
}

// ---- RTSP FRAME CALLBACK FUNCTION ----
esp_err_t rtsp_frame_callback(rtsp_session_handle_t session, uint8_t *buffer, size_t *buffer_len, void *user_data)
{
    // Get a frame from the camera
    fb = esp_camera_fb_get();
    if (!fb)
    {
        Serial.println("Camera capture failed");
        return ESP_FAIL;
    }

    // Check if buffer is large enough
    if (*buffer_len < fb->len)
    {
        Serial.printf("Buffer too small: %d < %d\n", *buffer_len, fb->len);
        esp_camera_fb_return(fb);
        return ESP_FAIL;
    }

    // Copy frame data to buffer
    memcpy(buffer, fb->buf, fb->len);
    *buffer_len = fb->len;

    // Return the frame buffer back to the driver
    esp_camera_fb_return(fb);

    return ESP_OK;
}

// ---- SETUP RTSP SERVER ----
void setupRTSPServer()
{
    rtsp_server_config_t config = RTSP_SERVER_DEFAULT_CONFIG();
    config.port = RTSP_PORT;

    // Start the RTSP server
    esp_err_t err = rtsp_server_start(&config, &rtsp_server);
    if (err != ESP_OK)
    {
        Serial.printf("RTSP server start failed: %d\n", err);
        return;
    }

    // Register a URI handler for our stream
    rtsp_stream_config_t stream_config = {
        .uri = RTSP_STREAM_URI,
        .codec = RTSP_CODEC_MJPEG, // Using MJPEG codec
        .fps = 20,                 // Target FPS
        .cb = rtsp_frame_callback, // Frame callback function
        .user_data = NULL          // User data (not used here)
    };

    err = rtsp_server_register_stream(rtsp_server, &stream_config);
    if (err != ESP_OK)
    {
        Serial.printf("RTSP stream registration failed: %d\n", err);
        return;
    }

    Serial.printf("RTSP server started on rtsp://%s:%d%s\n",
                  WiFi.localIP().toString().c_str(),
                  RTSP_PORT,
                  RTSP_STREAM_URI);
}

void setup()
{
    Serial.begin(115200);
    Serial.setDebugOutput(true);

    // ---- CONNECT TO WIFI ----
    Serial.println();
    Serial.printf("Connecting to %s ", ssid);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println(" CONNECTED!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());

    // ---- INIT CAMERA ----
    setupCamera();

    // ---- SETUP RTSP SERVER ----
    setupRTSPServer();

    Serial.println("Setup complete");
}

void loop()
{
    // The RTSP server runs in the background
    // We can add status checks or other functionality here
    delay(1000);
}

// ESP-RTSP Library Required: This code requires the ESP-RTSP library which is not included in the standard Arduino ESP32 package. You'll need to install it separately.
// Installation Instructions:
// Go to GitHub: https://github.com/espressif/esp-rtsp
// Clone or download the repository
// Add it to your Arduino libraries folder or use ESP-IDF
// Alternative Approach: If you have trouble with the ESP-RTSP library, consider using the pre-built firmware "ESP32-CAM-RTSP" by Geeksville Industries:
// https://github.com/geeksville/esp32-cam-rtsp
// Performance Considerations:
// RTSP streaming is more CPU-intensive than MJPEG
// You may need to reduce resolution or framerate for stable performance
// The ESP32-CAM has limited processing power for video encoding
// Client Compatibility:
// RTSP streams can be viewed with VLC, FFplay, or other RTSP-compatible players
// Example: vlc rtsp://192.168.1.195:8554/mjpeg/1
// Security Note:
// This implementation doesn't include authentication
// For production use, consider adding username/password protection
// Simplified Alternative
// If the above implementation is too complex, you can use a simpler third-party solution:
// Install the "ESP32-CAM-RTSP" firmware using the ESP32 Flash Download Tool
// Configure your WiFi credentials
// Access the RTSP stream at rtsp://[ESP32-IP]:8554/mjpeg/1
// This approach requires less coding but gives you less control over the implementation details.