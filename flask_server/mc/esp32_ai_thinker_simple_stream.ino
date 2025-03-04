/******************************************************
 * Minimal Camera Stream for AI Thinker ESP32-CAM
 *
 * Make sure you:
 * 1. Update `ssid` and `password` below
 * 2. Select the correct board in Arduino IDE:
 *    - Board: "AI Thinker ESP32-CAM" (if available), or
 *    - Board: "ESP32 Wrover Module" + PSRAM Enabled
 * 3. Open Serial Monitor at 115200 baud for debug info
 * 4. Access stream at: http://<DEVICE_IP>/stream
 ******************************************************/

#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

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

// Create a WebServer on port 80
WebServer server(80);

// ---- MJPEG STREAM HANDLER ----
void handle_jpg_stream()
{
    // We'll send a multipart response with frames in sequence
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "multipart/x-mixed-replace; boundary=frame");

    // Continuously capture frames until client disconnects
    while (true)
    {
        // Grab a frame
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb)
        {
            Serial.println("Camera capture failed");
            // Let's end the response on error
            break;
        }
        // Build the multipart boundaries + headers
        char partBuf[128];
        sprintf(partBuf,
                "--frame\r\n"
                "Content-Type: image/jpeg\r\n"
                "Content-Length: %u\r\n\r\n",
                fb->len);
        // Send boundary + frame
        server.sendContent(partBuf);
        server.sendContent((const char *)fb->buf, fb->len);
        // Return frame buffer back to driver
        esp_camera_fb_return(fb);

        // Small delay to reduce streaming overhead
        delay(30);

        // If client disconnects, break out of loop
        if (!server.client().connected())
        {
            break;
        }
    }
}

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
        config.frame_size = FRAMESIZE_SVGA; // e.g. SVGA(800x600), can go up to UXGA
        config.jpeg_quality = 10;           // lower = higher quality
        config.fb_count = 2;                // 2 frame buffers for smoother streaming
    }
    else
    {
        Serial.println("No PSRAM detected, set lower frame size");
        config.frame_size = FRAMESIZE_VGA; // e.g. VGA(640x480)
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

    // ---- SETUP HTTP SERVER ----
    // route /stream -> handle_jpg_stream
    server.on("/stream", HTTP_GET, handle_jpg_stream);

    // an optional root page
    server.on("/", HTTP_GET, []()
              { server.send(200, "text/plain",
                            "ESP32-CAM is alive! Go to /stream for MJPEG video."); });

    // start server
    server.begin();
    Serial.println("HTTP server started. Go to /stream to view camera.");
}

void loop()
{
    server.handleClient();
}
