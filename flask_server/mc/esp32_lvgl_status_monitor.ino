#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include <lvgl.h>
#include <TFT_eSPI.h>

// Replace with your network credentials
const char *ssid = "Pi-rise";
const char *password = "";

// WebSocket server details
const char *ws_server = "10.42.0.1";
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

// Rules configuration
typedef enum {
    RULE_IGNORE,
    RULE_STOP_CONVEYOR,
    RULE_ALERT
} rule_action_t;

rule_action_t overlapRule = RULE_IGNORE;
rule_action_t stopRule = RULE_IGNORE;
rule_action_t incorrectRule = RULE_IGNORE;

// LVGL display objects
TFT_eSPI tft = TFT_eSPI();
static lv_display_t *display;
static lv_draw_buf_t draw_buf;
static lv_color_t buf1[120 * 10]; // Reduced from 240*10

// LVGL UI elements - Global variables
lv_obj_t *tabview;
lv_obj_t *statusTab;
lv_obj_t *rulesTab;
lv_obj_t *cameraTab;

// Status tab elements
lv_obj_t *connectionLabel;
lv_obj_t *conveyorPanel;
lv_obj_t *conveyorLabel;
lv_obj_t *stoppedPanel;
lv_obj_t *stoppedLabel;
lv_obj_t *overlapPanel;
lv_obj_t *overlapLabel;
lv_obj_t *incorrectPanel;
lv_obj_t *incorrectLabel;
lv_obj_t *startBtn;
lv_obj_t *stopBtn;

// Rules tab elements
lv_obj_t *overlapDropdown;
lv_obj_t *stopDropdown;
lv_obj_t *incorrectDropdown;
lv_obj_t *saveRulesBtn;

// WebSocket client
WebSocketsClient webSocket;

// Display flush function for LVGL
static void my_disp_flush(lv_display_t *disp, const lv_area_t *area, uint8_t *buf)
{
    uint32_t w = (area->x2 - area->x1 + 1);
    uint32_t h = (area->y2 - area->y1 + 1);

    tft.startWrite();
    tft.setAddrWindow(area->x1, area->y1, w, h);
    tft.pushColors((uint16_t *)buf, w * h, true);
    tft.endWrite();

    lv_display_flush_ready(disp);
}

// Input read function for LVGL
static void my_touchpad_read(lv_indev_t *indev, lv_indev_data_t *data)
{
    uint16_t touchX, touchY;
    bool touched = false;
    
    // Simple direct call to getTouch - wrapped in try/catch to handle potential issues
    try {
        touched = tft.getTouch(&touchX, &touchY);
    } catch (...) {
        // If touch is not supported, it will fail silently
        touched = false;
    }
    
    if (touched) {
        data->state = LV_INDEV_STATE_PRESSED;
        data->point.x = touchX;
        data->point.y = touchY;
    } else {
        data->state = LV_INDEV_STATE_RELEASED;
    }
}

// Ticker for LVGL timing
#include <Ticker.h>
Ticker lvglTicker;

void lvglTimerHandler()
{
    lv_tick_inc(5); // Increment by 5ms
}

void setup()
{
    Serial.begin(115200);
    Serial.println("ESP32 LVGL Status Monitor");

    // Initialize LVGL
    lv_init();

    // Initialize TFT
    tft.begin();
    tft.setRotation(1); // Landscape orientation
    
    // Initialize display buffer for LVGL with smaller buffer size
    uint32_t buf_size = 120 * 10; // Reduced buffer size
    lv_draw_buf_init(&draw_buf, 320, 240, LV_COLOR_FORMAT_RGB565, 320 * 2,
                     buf1, buf_size * sizeof(lv_color_t));

    // Initialize display driver with horizontal and vertical resolution
    display = lv_display_create(320, 240); // Pass horizontal and vertical resolution
    lv_display_set_flush_cb(display, (lv_display_flush_cb_t)my_disp_flush);
    lv_display_set_draw_buffers(display, &draw_buf, NULL);

    // Initialize input device driver for touch
    lv_indev_t *indev = lv_indev_create();
    lv_indev_set_type(indev, LV_INDEV_TYPE_POINTER);
    lv_indev_set_read_cb(indev, my_touchpad_read);

    // Setup timer for LVGL
    lvglTicker.attach_ms(5, lvglTimerHandler);

    // Create UI
    createUI();

    // Update status tab to show connecting state
    lv_label_set_text(connectionLabel, "Connecting to WiFi...");
    lv_timer_handler();

    // Connect to Wi-Fi
    WiFi.begin(ssid, password);
    
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
        lv_timer_handler(); // Keep UI responsive during connection
    }

    Serial.println("\nConnected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    lv_label_set_text(connectionLabel, "WiFi: Connected");
    lv_timer_handler();

    // Initialize WebSocket connection
    webSocket.begin(ws_server, ws_port, ws_url);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
    webSocket.enableHeartbeat(15000, 3000, 2);

    lv_label_set_text(connectionLabel, "Connecting to server...");
    lv_timer_handler();
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
    }
}

void createUI()
{
    // Create tabview with three tabs
    tabview = lv_tabview_create(lv_scr_act());
    statusTab = lv_tabview_add_tab(tabview, "Status");
    rulesTab = lv_tabview_add_tab(tabview, "Rules");
    cameraTab = lv_tabview_add_tab(tabview, "Camera");

    // Create content for Status Tab
    createStatusTab();
    
    // Create content for Rules Tab
    createRulesTab();
    
    // Create content for Camera Tab (placeholder)
    createCameraTab();
}

void createStatusTab()
{
    // Create title
    lv_obj_t *titleLabel = lv_label_create(statusTab);
    lv_label_set_text(titleLabel, "Plank Monitor Status");
    lv_obj_align(titleLabel, LV_ALIGN_TOP_MID, 0, 5);
    lv_obj_set_style_text_font(titleLabel, &lv_font_montserrat_14, 0);

    // Create connection status label
    connectionLabel = lv_label_create(statusTab);
    lv_label_set_text(connectionLabel, "Connecting...");
    lv_obj_align_to(connectionLabel, titleLabel, LV_ALIGN_OUT_BOTTOM_MID, 0, 5);

    // Create conveyor status panel
    conveyorPanel = lv_obj_create(statusTab);
    lv_obj_set_size(conveyorPanel, 280, 35);
    lv_obj_align_to(conveyorPanel, connectionLabel, LV_ALIGN_OUT_BOTTOM_MID, 0, 10);
    lv_obj_set_style_bg_color(conveyorPanel, lv_color_hex(0x00FF00), 0); // Green

    conveyorLabel = lv_label_create(conveyorPanel);
    lv_label_set_text(conveyorLabel, "Conveyor: RUNNING");
    lv_obj_center(conveyorLabel);

    // Control buttons
    stopBtn = lv_btn_create(statusTab);
    lv_obj_set_size(stopBtn, 130, 35);
    lv_obj_align_to(stopBtn, conveyorPanel, LV_ALIGN_OUT_BOTTOM_LEFT, 10, 10);
    lv_obj_set_style_bg_color(stopBtn, lv_color_hex(0xFF4444), 0); // Red
    lv_obj_add_event_cb(stopBtn, stopBtnHandler, LV_EVENT_CLICKED, NULL);
    
    lv_obj_t *stopLabel = lv_label_create(stopBtn);
    lv_label_set_text(stopLabel, "Stop Conveyor");
    lv_obj_center(stopLabel);

    startBtn = lv_btn_create(statusTab);
    lv_obj_set_size(startBtn, 130, 35);
    lv_obj_align_to(startBtn, conveyorPanel, LV_ALIGN_OUT_BOTTOM_RIGHT, -10, 10);
    lv_obj_set_style_bg_color(startBtn, lv_color_hex(0x44FF44), 0); // Green
    lv_obj_add_event_cb(startBtn, startBtnHandler, LV_EVENT_CLICKED, NULL);
    
    lv_obj_t *startLabel = lv_label_create(startBtn);
    lv_label_set_text(startLabel, "Start Conveyor");
    lv_obj_center(startLabel);

    // Warning panels
    int panel_height = 32;
    int panel_spacing = 5;
    int start_y = 155;

    // Stopped plank panel
    stoppedPanel = lv_obj_create(statusTab);
    lv_obj_set_size(stoppedPanel, 280, panel_height);
    lv_obj_set_pos(stoppedPanel, 20, start_y);
    lv_obj_set_style_bg_color(stoppedPanel, lv_color_hex(0x00FF00), 0); // Green

    stoppedLabel = lv_label_create(stoppedPanel);
    lv_label_set_text(stoppedLabel, "Stopped Plank: Clear");
    lv_obj_center(stoppedLabel);

    // Overlapped planks panel
    overlapPanel = lv_obj_create(statusTab);
    lv_obj_set_size(overlapPanel, 280, panel_height);
    lv_obj_set_pos(overlapPanel, 20, start_y + panel_height + panel_spacing);
    lv_obj_set_style_bg_color(overlapPanel, lv_color_hex(0x00FF00), 0); // Green

    overlapLabel = lv_label_create(overlapPanel);
    lv_label_set_text(overlapLabel, "Overlapped Planks: Clear");
    lv_obj_center(overlapLabel);

    // Incorrect orientation panel
    incorrectPanel = lv_obj_create(statusTab);
    lv_obj_set_size(incorrectPanel, 280, panel_height);
    lv_obj_set_pos(incorrectPanel, 20, start_y + 2 * (panel_height + panel_spacing));
    lv_obj_set_style_bg_color(incorrectPanel, lv_color_hex(0x00FF00), 0); // Green

    incorrectLabel = lv_label_create(incorrectPanel);
    lv_label_set_text(incorrectLabel, "Incorrect Orientation: Clear");
    lv_obj_center(incorrectLabel);
}

void createRulesTab()
{
    // Title
    lv_obj_t *titleLabel = lv_label_create(rulesTab);
    lv_label_set_text(titleLabel, "Rule Configuration");
    lv_obj_align(titleLabel, LV_ALIGN_TOP_MID, 0, 5);
    lv_obj_set_style_text_font(titleLabel, &lv_font_montserrat_14, 0);

    // Create rule dropdowns
    // Overlap Rule
    lv_obj_t *overlapLabel = lv_label_create(rulesTab);
    lv_label_set_text(overlapLabel, "When Overlap Detected:");
    lv_obj_align_to(overlapLabel, titleLabel, LV_ALIGN_OUT_BOTTOM_MID, 0, 15);

    overlapDropdown = lv_dropdown_create(rulesTab);
    lv_dropdown_set_options(overlapDropdown, 
                           "Ignore\n"
                           "Stop Conveyor\n"
                           "Alert Only");
    lv_obj_set_width(overlapDropdown, 220);
    lv_obj_align_to(overlapDropdown, overlapLabel, LV_ALIGN_OUT_BOTTOM_MID, 0, 5);
    lv_obj_add_event_cb(overlapDropdown, ruleDropdownHandler, LV_EVENT_VALUE_CHANGED, NULL);

    // Stop Rule
    lv_obj_t *stopLabel = lv_label_create(rulesTab);
    lv_label_set_text(stopLabel, "When Stopped Plank Detected:");
    lv_obj_align_to(stopLabel, overlapDropdown, LV_ALIGN_OUT_BOTTOM_MID, 0, 15);

    stopDropdown = lv_dropdown_create(rulesTab);
    lv_dropdown_set_options(stopDropdown, 
                           "Ignore\n"
                           "Stop Conveyor\n"
                           "Alert Only");
    lv_obj_set_width(stopDropdown, 220);
    lv_obj_align_to(stopDropdown, stopLabel, LV_ALIGN_OUT_BOTTOM_MID, 0, 5);
    lv_obj_add_event_cb(stopDropdown, ruleDropdownHandler, LV_EVENT_VALUE_CHANGED, NULL);

    // Incorrect Rule
    lv_obj_t *incorrectLabel = lv_label_create(rulesTab);
    lv_label_set_text(incorrectLabel, "When Incorrect Plank Detected:");
    lv_obj_align_to(incorrectLabel, stopDropdown, LV_ALIGN_OUT_BOTTOM_MID, 0, 15);

    incorrectDropdown = lv_dropdown_create(rulesTab);
    lv_dropdown_set_options(incorrectDropdown, 
                           "Ignore\n"
                           "Stop Conveyor\n"
                           "Alert Only");
    lv_obj_set_width(incorrectDropdown, 220);
    lv_obj_align_to(incorrectDropdown, incorrectLabel, LV_ALIGN_OUT_BOTTOM_MID, 0, 5);
    lv_obj_add_event_cb(incorrectDropdown, ruleDropdownHandler, LV_EVENT_VALUE_CHANGED, NULL);

    // Save Button
    saveRulesBtn = lv_btn_create(rulesTab);
    lv_obj_set_size(saveRulesBtn, 140, 35);
    lv_obj_align_to(saveRulesBtn, incorrectDropdown, LV_ALIGN_OUT_BOTTOM_MID, 0, 20);
    lv_obj_set_style_bg_color(saveRulesBtn, lv_color_hex(0x3498DB), 0); // Blue
    lv_obj_add_event_cb(saveRulesBtn, saveRulesBtnHandler, LV_EVENT_CLICKED, NULL);
    
    lv_obj_t *saveLabel = lv_label_create(saveRulesBtn);
    lv_label_set_text(saveLabel, "Save Rules");
    lv_obj_center(saveLabel);
}

void createCameraTab()
{
    // Simple placeholder message for camera tab
    lv_obj_t *cameraMsg = lv_label_create(cameraTab);
    lv_label_set_text(cameraMsg, "Camera streaming not supported\non this device due to\nbandwidth limitations.");
    lv_obj_center(cameraMsg);
    
    // Create a placeholder frame/border to represent where the camera
    // feed would be displayed
    lv_obj_t *cameraFrame = lv_obj_create(cameraTab);
    lv_obj_set_size(cameraFrame, 240, 180);
    lv_obj_align(cameraFrame, LV_ALIGN_TOP_MID, 0, 20);
    lv_obj_set_style_border_width(cameraFrame, 2, 0);
    lv_obj_set_style_border_color(cameraFrame, lv_color_hex(0x3498DB), 0); // Blue
    lv_obj_set_style_bg_opa(cameraFrame, LV_OPA_0, 0); // Transparent background
}

static void startBtnHandler(lv_event_t *e)
{
    controlConveyor(false); // Start conveyor
}

static void stopBtnHandler(lv_event_t *e)
{
    controlConveyor(true); // Stop conveyor
}

static void ruleDropdownHandler(lv_event_t *e)
{
    lv_obj_t *dropdown = (lv_obj_t *)lv_event_get_target(e); // Cast to lv_obj_t*
    char buf[32];
    lv_dropdown_get_selected_str(dropdown, buf, sizeof(buf));
    
    Serial.print("Selected option: ");
    Serial.println(buf);
    
    // Just note the change here, actual save happens when Save button is pressed
}

static void saveRulesBtnHandler(lv_event_t *e)
{
    // Get selected values
    uint16_t overlapIdx = lv_dropdown_get_selected(overlapDropdown);
    uint16_t stopIdx = lv_dropdown_get_selected(stopDropdown);
    uint16_t incorrectIdx = lv_dropdown_get_selected(incorrectDropdown);
    
    // Convert dropdown indices to rule actions
    overlapRule = (rule_action_t)overlapIdx;
    stopRule = (rule_action_t)stopIdx;
    incorrectRule = (rule_action_t)incorrectIdx;
    
    // Send rules to server
    sendRulesToServer();
}

void sendRulesToServer()
{
    if (!connected) {
        Serial.println("Cannot save rules: Not connected to server");
        return;
    }
    
    StaticJsonDocument<256> doc;
    doc["event"] = "update_rules";
    JsonObject data = doc.createNestedObject("data");
    
    // Convert enum values to strings expected by server
    const char* actionNames[] = {"ignore", "stop_conveyor", "alert"};
    
    data["overlap"] = actionNames[overlapRule];
    data["stop"] = actionNames[stopRule];
    data["incorrect"] = actionNames[incorrectRule];
    
    String jsonString;
    serializeJson(doc, jsonString);
    webSocket.sendTXT(jsonString);
    
    Serial.println("Sent rules to server:");
    Serial.println(jsonString);
}

void controlConveyor(bool stopState)
{
    if (!connected) {
        Serial.println("Cannot control conveyor: Not connected to server");
        return;
    }
    
    StaticJsonDocument<128> doc;
    doc["event"] = "control_conveyor";
    JsonObject data = doc.createNestedObject("data");
    data["state"] = stopState;
    
    String jsonString;
    serializeJson(doc, jsonString);
    webSocket.sendTXT(jsonString);
    
    Serial.print("Sent conveyor command: ");
    Serial.println(stopState ? "STOP" : "START");
    
    // Update UI to reflect the requested state (will be confirmed when server responds)
    updateConveyorStatus(stopState);
}

void updateConveyorStatus(bool stopped)
{
    conveyorStopActive = stopped;
    
    if (stopped) {
        lv_obj_set_style_bg_color(conveyorPanel, lv_color_hex(0xFF0000), 0); // Red
        lv_label_set_text(conveyorLabel, "Conveyor: STOPPED");
    } else {
        lv_obj_set_style_bg_color(conveyorPanel, lv_color_hex(0x00FF00), 0); // Green
        lv_label_set_text(conveyorLabel, "Conveyor: RUNNING");
    }
}

void updateStatusPanels()
{
    // Update stopped plank panel
    if (plankStopActive) {
        lv_obj_set_style_bg_color(stoppedPanel, lv_color_hex(0xFF0000), 0); // Red
        lv_label_set_text(stoppedLabel, "Stopped Plank: Detected");
    } else {
        lv_obj_set_style_bg_color(stoppedPanel, lv_color_hex(0x00FF00), 0); // Green
        lv_label_set_text(stoppedLabel, "Stopped Plank: Clear");
    }
    
    // Update overlap panel
    if (plankOverlapActive) {
        lv_obj_set_style_bg_color(overlapPanel, lv_color_hex(0xFF0000), 0); // Red
        lv_label_set_text(overlapLabel, "Overlapped Planks: Detected");
    } else {
        lv_obj_set_style_bg_color(overlapPanel, lv_color_hex(0x00FF00), 0); // Green
        lv_label_set_text(overlapLabel, "Overlapped Planks: Clear");
    }
    
    // Update incorrect orientation panel
    if (plankIncorrectActive) {
        lv_obj_set_style_bg_color(incorrectPanel, lv_color_hex(0xFF0000), 0); // Red
        lv_label_set_text(incorrectLabel, "Incorrect Orientation: Detected");
    } else {
        lv_obj_set_style_bg_color(incorrectPanel, lv_color_hex(0x00FF00), 0); // Green
        lv_label_set_text(incorrectLabel, "Incorrect Orientation: Clear");
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
    Serial.println("Ping sent");
}

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length)
{
    StaticJsonDocument<512> doc;

    switch (type)
    {
    case WStype_DISCONNECTED:
        Serial.println("WebSocket Disconnected!");
        connected = false;
        lv_label_set_text(connectionLabel, "Server: Disconnected");
        break;

    case WStype_CONNECTED:
        Serial.println("WebSocket Connected!");
        connected = true;
        lastPing = millis();
        lv_label_set_text(connectionLabel, "Server: Connected");
        break;

    case WStype_TEXT:
        Serial.printf("WebSocket message received (%d bytes)\n", length);

        {
            DeserializationError error = deserializeJson(doc, payload);

            if (error) {
                Serial.print("deserializeJson() failed: ");
                Serial.println(error.c_str());
                return;
            }

            const char *event = doc["event"];
            Serial.print("Event name: ");
            Serial.println(event);

            if (strcmp(event, "status_update") == 0) {
                handleStatusUpdate(doc["data"]);
            }
            else if (strcmp(event, "control_conveyor") == 0) {
                handleConveyorControl(doc["data"]);
            }
            else if (strcmp(event, "rules_applied") == 0) {
                handleRulesApplied(doc["data"]);
            }
            else if (strcmp(event, "rules_update") == 0) {
                handleRulesUpdate(doc["data"]);
            }
        }
        break;

    case WStype_ERROR:
        Serial.println("WebSocket ERROR received");
        connected = false;
        lv_label_set_text(connectionLabel, "Server: Error");
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
    conveyorStopActive = data["conveyor_stop"];

    Serial.println("Status update received:");
    Serial.println("  Conveyor Stop: " + String(conveyorStopActive));
    Serial.println("  Stop: " + String(plankStopActive));
    Serial.println("  Overlap: " + String(plankOverlapActive));
    Serial.println("  Incorrect: " + String(plankIncorrectActive));

    // Update UI
    updateConveyorStatus(conveyorStopActive);
    updateStatusPanels();
}

void handleConveyorControl(const JsonDocument &data)
{
    if (data.containsKey("state")) {
        conveyorStopActive = data["state"];
        Serial.println("Conveyor state update: " + String(conveyorStopActive));
        updateConveyorStatus(conveyorStopActive);
    }
}

void handleRulesApplied(const JsonDocument &data)
{
    // Reset status
    plankStopActive = false;
    plankOverlapActive = false;
    plankIncorrectActive = false;
    
    // Update conveyor status if present
    if (data.containsKey("stop_conveyor")) {
        conveyorStopActive = data["stop_conveyor"];
        updateConveyorStatus(conveyorStopActive);
    }
    
    // Process alerts
    if (data.containsKey("alert")) {
        // For ArduinoJson v7, update to use JsonArrayConst or check if text
        if (data["alert"].is<const char *>()) {
            const char *alertType = data["alert"];
            processAlertType(alertType);
        }
        else {
            // For array, iterate differently with ArduinoJson v7
            JsonVariantConst alertVar = data["alert"];
            if (alertVar.is<JsonArrayConst>()) {
                JsonArrayConst alertArray = alertVar.as<JsonArrayConst>();
                for (JsonVariantConst v : alertArray) {
                    if (v.is<const char *>()) {
                        processAlertType(v.as<const char *>());
                    }
                }
            }
        }
    }
    
    // Update UI
    updateStatusPanels();
}

void handleRulesUpdate(const JsonDocument &data)
{
    if (data.containsKey("rules")) {
        // For ArduinoJson v7, directly access members
        JsonVariantConst rulesVar = data["rules"];
        
        // Update dropdown selections based on received rules
        if (rulesVar.containsKey("overlap")) {
            const char* value = rulesVar["overlap"].as<const char*>();
            if (strcmp(value, "ignore") == 0) {
                lv_dropdown_set_selected(overlapDropdown, RULE_IGNORE);
                overlapRule = RULE_IGNORE;
            } else if (strcmp(value, "stop_conveyor") == 0) {
                lv_dropdown_set_selected(overlapDropdown, RULE_STOP_CONVEYOR);
                overlapRule = RULE_STOP_CONVEYOR;
            } else if (strcmp(value, "alert") == 0) {
                lv_dropdown_set_selected(overlapDropdown, RULE_ALERT);
                overlapRule = RULE_ALERT;
            }
        }
        
        if (rulesVar.containsKey("stop")) {
            const char* value = rulesVar["stop"].as<const char*>();
            if (strcmp(value, "ignore") == 0) {
                lv_dropdown_set_selected(stopDropdown, RULE_IGNORE);
                stopRule = RULE_IGNORE;
            } else if (strcmp(value, "stop_conveyor") == 0) {
                lv_dropdown_set_selected(stopDropdown, RULE_STOP_CONVEYOR);
                stopRule = RULE_STOP_CONVEYOR;
            } else if (strcmp(value, "alert") == 0) {
                lv_dropdown_set_selected(stopDropdown, RULE_ALERT);
                stopRule = RULE_ALERT;
            }
        }
        
        if (rulesVar.containsKey("incorrect")) {
            const char* value = rulesVar["incorrect"].as<const char*>();
            if (strcmp(value, "ignore") == 0) {
                lv_dropdown_set_selected(incorrectDropdown, RULE_IGNORE);
                incorrectRule = RULE_IGNORE;
            } else if (strcmp(value, "stop_conveyor") == 0) {
                lv_dropdown_set_selected(incorrectDropdown, RULE_STOP_CONVEYOR);
                incorrectRule = RULE_STOP_CONVEYOR;
            } else if (strcmp(value, "alert") == 0) {
                lv_dropdown_set_selected(incorrectDropdown, RULE_ALERT);
                incorrectRule = RULE_ALERT;
            }
        }
        
        Serial.println("Rules updated from server");
    }
}

void processAlertType(const char *alertType)
{
    if (strcmp(alertType, "stop") == 0) {
        plankStopActive = true;
    }
    else if (strcmp(alertType, "overlap") == 0) {
        plankOverlapActive = true;
    }
    else if (strcmp(alertType, "incorrect") == 0) {
        plankIncorrectActive = true;
    }
} 