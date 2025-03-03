#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <lvgl.h>
#include <TFT_eSPI.h>

// Replace with your network credentials
const char *ssid = "TN-JE3155";
const char *password = "";

// WebSocket server details
const char *ws_server = "192.168.1.249";
const int ws_port = 5000;
const char *ws_url = "/ws";
unsigned long pingInterval = 25000;
unsigned long lastPing = 0;
bool connected = false;

// Status variables
bool plankStopActive = false;
bool plankOverlapActive = false;
bool plankIncorrectActive = false;
bool conveyorStopActive = false;

// LVGL display objects
TFT_eSPI tft = TFT_eSPI();
static lv_draw_buf_t draw_buf;
static uint32_t buf_size = TFT_WIDTH * 10 * sizeof(lv_color_t);
static uint8_t *buf_1;

// LVGL UI elements
lv_obj_t *connectionLabel;
lv_obj_t *conveyorLabel;
lv_obj_t *stoppedPanel;
lv_obj_t *stoppedLabel;
lv_obj_t *overlapPanel;
lv_obj_t *overlapLabel;
lv_obj_t *incorrectPanel;
lv_obj_t *incorrectLabel;

// WebSocket client
WebSocketsClient webSocket;

// Display flush function for LVGL
static void my_disp_flush(lv_display_t *disp, const lv_area_t *area, uint8_t *px_map)
{
    uint32_t w = (area->x2 - area->x1 + 1);
    uint32_t h = (area->y2 - area->y1 + 1);

    tft.startWrite();
    tft.setAddrWindow(area->x1, area->y1, w, h);
    tft.pushPixels((uint16_t *)px_map, w * h);
    tft.endWrite();

    lv_disp_flush_ready(disp);
}

// Ticker for LVGL timing
#include <Ticker.h>
Ticker lvglTicker;

void lvglTimerHandler()
{
    lv_tick_inc(5); // Increment by 5ms (adjust as needed)
}

void setup()
{
    Serial.begin(115200);
    Serial.println("ESP32 LVGL Plank Monitor");

    // Initialize LVGL
    lv_init();

    // Initialize TFT
    tft.begin();
    tft.setRotation(0); // Adjust based on your display orientation

    // Allocate buffer for LVGL
    buf_1 = (uint8_t *)heap_caps_malloc(buf_size, MALLOC_CAP_DMA);
    if (buf_1 == NULL)
    {
        Serial.println("Failed to allocate display buffer");
        while (1)
            ;
    }

    // Initialize display buffer for LVGL
    lv_draw_buf_init(&draw_buf, TFT_WIDTH, TFT_HEIGHT, LV_COLOR_FORMAT_RGB565, TFT_WIDTH * sizeof(lv_color_t), buf_1, buf_size);

    // Initialize display driver
    static lv_display_t *disp = lv_display_create(TFT_WIDTH, TFT_HEIGHT);
    lv_display_set_flush_cb(disp, my_disp_flush);
    lv_display_set_buffers(disp, &draw_buf, NULL, buf_size, LV_DISPLAY_RENDER_MODE_PARTIAL);

    // Setup timer for LVGL
    lvglTicker.attach_ms(5, lvglTimerHandler);

    // Create UI
    createUI();

    // Connect to Wi-Fi
    WiFi.begin(ssid, password);
    updateConnectionLabel("Connecting to WiFi...");

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
        lv_timer_handler(); // Keep UI responsive during connection
    }

    Serial.println("\nConnected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    updateConnectionLabel("WiFi: Connected");

    // Initialize WebSocket connection
    webSocket.begin(ws_server, ws_port, ws_url);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
    webSocket.enableHeartbeat(15000, 3000, 2);

    updateConnectionLabel("Connecting to server...");
}

void loop()
{
    webSocket.loop();
    lv_timer_handler(); // Handle LVGL tasks

    unsigned long currentMillis = millis();

    // Handle WebSocket ping to maintain connection
    if (connected && currentMillis - lastPing > pingInterval)
    {
        lastPing = currentMillis;
        sendPing();
        Serial.println("Sending ping");
    }
}

void createUI()
{
    // Create title
    lv_obj_t *titleLabel = lv_label_create(lv_scr_act());
    lv_label_set_text(titleLabel, "Plank Monitor");
    lv_obj_align(titleLabel, LV_ALIGN_TOP_MID, 0, 10);
    lv_obj_set_style_text_font(titleLabel, &lv_font_montserrat_16, 0);

    // Create connection status label
    connectionLabel = lv_label_create(lv_scr_act());
    lv_label_set_text(connectionLabel, "Connecting...");
    lv_obj_align_to(connectionLabel, titleLabel, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);

    // Create conveyor status label
    conveyorLabel = lv_label_create(lv_scr_act());
    lv_label_set_text(conveyorLabel, "Conveyor: Unknown");
    lv_obj_align_to(conveyorLabel, connectionLabel, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);

    // Create warning panels
    int panel_width = TFT_WIDTH - 40;
    int panel_height = 40;
    int panel_spacing = 20;
    int start_y = 80;

    // Stopped plank panel
    stoppedPanel = lv_obj_create(lv_scr_act());
    lv_obj_set_size(stoppedPanel, panel_width, panel_height);
    lv_obj_align(stoppedPanel, LV_ALIGN_TOP_MID, 0, start_y);
    lv_obj_set_style_bg_color(stoppedPanel, lv_color_hex(0x00FF00), 0); // Green

    stoppedLabel = lv_label_create(stoppedPanel);
    lv_label_set_text(stoppedLabel, "Stopped Plank");
    lv_obj_center(stoppedLabel);

    // Overlapped planks panel
    overlapPanel = lv_obj_create(lv_scr_act());
    lv_obj_set_size(overlapPanel, panel_width, panel_height);
    lv_obj_align_to(overlapPanel, stoppedPanel, LV_ALIGN_OUT_BOTTOM_MID, 0, panel_spacing);
    lv_obj_set_style_bg_color(overlapPanel, lv_color_hex(0x00FF00), 0); // Green

    overlapLabel = lv_label_create(overlapPanel);
    lv_label_set_text(overlapLabel, "Overlapped Planks");
    lv_obj_center(overlapLabel);

    // Incorrect orientation panel
    incorrectPanel = lv_obj_create(lv_scr_act());
    lv_obj_set_size(incorrectPanel, panel_width, panel_height);
    lv_obj_align_to(incorrectPanel, overlapPanel, LV_ALIGN_OUT_BOTTOM_MID, 0, panel_spacing);
    lv_obj_set_style_bg_color(incorrectPanel, lv_color_hex(0x00FF00), 0); // Green

    incorrectLabel = lv_label_create(incorrectPanel);
    lv_label_set_text(incorrectLabel, "Incorrect Orientation");
    lv_obj_center(incorrectLabel);
}

void updateConnectionLabel(const char *text)
{
    lv_label_set_text(connectionLabel, text);
    lv_timer_handler();
}

void updateConveyorLabel(bool stopped)
{
    if (stopped)
    {
        lv_label_set_text(conveyorLabel, "Conveyor: Stopped");
    }
    else
    {
        lv_label_set_text(conveyorLabel, "Conveyor: Running");
    }
}

void updateWarningPanel(lv_obj_t *panel, bool active)
{
    if (active)
    {
        lv_obj_set_style_bg_color(panel, lv_color_hex(0xFF0000), 0); // Red
    }
    else
    {
        lv_obj_set_style_bg_color(panel, lv_color_hex(0x00FF00), 0); // Green
    }
}

void sendPing()
{
    StaticJsonDocument<128> doc;
    doc["event"] = "ping";
    JsonObject data = doc.createNestedObject("data");

    String jsonString;
    serializeJson(doc, jsonString);
    webSocket.sendTXT(jsonString);
}

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length)
{
    StaticJsonDocument<512> doc;

    switch (type)
    {
    case WStype_DISCONNECTED:
        Serial.println("WebSocket Disconnected!");
        connected = false;
        updateConnectionLabel("Server: Disconnected");
        break;

    case WStype_CONNECTED:
        Serial.println("WebSocket Connected!");
        connected = true;
        lastPing = millis();
        updateConnectionLabel("Server: Connected");
        break;

    case WStype_TEXT:
        Serial.printf("WebSocket message received (%d bytes)\n", length);

        {
            DeserializationError error = deserializeJson(doc, payload);

            if (error)
            {
                Serial.print("deserializeJson() failed: ");
                Serial.println(error.c_str());
                return;
            }

            const char *event = doc["event"];
            Serial.print("Event name: ");
            Serial.println(event);

            if (strcmp(event, "status_update") == 0)
            {
                handleStatusUpdate(doc["data"]);
            }
            else if (strcmp(event, "control_conveyor") == 0)
            {
                handleConveyorControl(doc["data"]);
            }
        }
        break;

    case WStype_ERROR:
        Serial.println("WebSocket ERROR received");
        connected = false;
        updateConnectionLabel("Server: Error");
        break;

    default:
        break;
    }
}

void handleStatusUpdate(const JsonDocument &data)
{
    // Update status variables
    plankStopActive = data["stop"];
    plankOverlapActive = data["overlap"];
    plankIncorrectActive = data["incorrect"];

    Serial.println("Status update received:");
    Serial.println("  Stop: " + String(plankStopActive));
    Serial.println("  Overlap: " + String(plankOverlapActive));
    Serial.println("  Incorrect: " + String(plankIncorrectActive));

    // Update UI
    updateWarningPanel(stoppedPanel, plankStopActive);
    updateWarningPanel(overlapPanel, plankOverlapActive);
    updateWarningPanel(incorrectPanel, plankIncorrectActive);
}

void handleConveyorControl(const JsonDocument &data)
{
    if (data.containsKey("state"))
    {
        conveyorStopActive = data["state"];
        Serial.println("Conveyor state update: " + String(conveyorStopActive));
        updateConveyorLabel(conveyorStopActive);
    }
}
