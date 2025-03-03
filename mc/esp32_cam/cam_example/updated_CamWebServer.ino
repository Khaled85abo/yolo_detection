#include "esp_camera.h"
#include <WiFi.h>
#include <WebServer.h>

// ===============
// Camera Model
// ===============
#define CAMERA_MODEL_AI_THINKER // Only keep this line!
#include "camera_pins.h"

// ===============
// Wi-Fi Settings
// ===============
const char *ssid = "TN-JE3155";
const char *password = "";

// Start the web server on port 80
WebServer server(80);

// ===============
// MJPEG Stream Handler
// ===============
void handle_jpg_stream()
{
    server.sendHeader("Access-Control-Allow-Origin", "*");
    server.setContentLength(CONTENT_LENGTH_UNKNOWN);
    server.send(200, "multipart/x-mixed-replace; boundary=frame");

    while (true)
    {
        camera_fb_t *fb = esp_camera_fb_get();
        if (!fb)
        {
            Serial.println("Camera capture failed");
            break;
        }

        // Send multipart frame boundary
        server.sendContent("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + String(fb->len) + "\r\n\r\n");
        server.sendContent((const char *)fb->buf, fb->len);
        server.sendContent("\r\n");

        esp_camera_fb_return(fb);

        // Short delay to avoid overwhelming the connection
        delay(30);

        // Stop streaming if the client disconnects
        if (!server.client().connected())
        {
            break;
        }
    }
}

// ===============
// Camera Init Function
// ===============
void setup_camera()
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
    config.pixel_format = PIXFORMAT_JPEG; // JPEG for streaming

    if (psramFound())
    {
        Serial.println("PSRAM detected!");
        config.frame_size = FRAMESIZE_SVGA; // Higher resolution (SVGA = 800x600)
        config.jpeg_quality = 10;           // Lower = better quality
        config.fb_count = 2;
    }
    else
    {
        Serial.println("No PSRAM detected, reducing settings");
        config.frame_size = FRAMESIZE_VGA; // Lower resolution
        config.jpeg_quality = 12;
        config.fb_count = 1;
    }

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK)
    {
        Serial.printf("Camera init failed with error 0x%x", err);
        while (true)
        {
            delay(100);
        }
    }
    sensor_t *s = esp_camera_sensor_get();
    s->set_vflip(s, 1);   // ✅ Flip vertically
    s->set_hmirror(s, 1); // ✅ Mirror horizontally (optional)
}

// ===============
// Setup Function
// ===============
void setup()
{
    Serial.begin(115200);
    Serial.setDebugOutput(true);

    // Connect to Wi-Fi
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi...");
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }
    Serial.println("\nWiFi connected!");
    Serial.print("Stream available at: http://");
    Serial.println(WiFi.localIP());

    // Initialize the camera
    setup_camera();

    // Set up the streaming endpoint
    server.on("/stream", HTTP_GET, handle_jpg_stream);
    server.begin();
}

// ===============
// Loop Function
// ===============
void loop()
{
    server.handleClient();
}
