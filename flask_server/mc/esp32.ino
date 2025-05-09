#include <WiFi.h>
#include <WebServer.h>
// #include <SPIFFS.h>
#include <HTTPClient.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// esp32 ip address: http://192.168.1.202/

// Replace with your network credentials
const char *ssid = "TN-JE3155";
const char *password = "";

// Constants and configuration
#define VERSION "1.0.0"

// Pin definitions
struct Pins
{
    // LED pins
    const int PLANK_STOP = 19;       // red 1
    const int PLANK_OVERLAP = 4;     // yellow 3
    const int PLANK_INCORRECT = 5;   // green 4
    const int CONVEYOR_STOP = 22;    // blue 2
    const int CONVEYOR_START_1 = 27; // Connect to IN1 of L298N
    const int CONVEYOR_START_2 = 26; // Connect to IN2 of L298N
} pins;

// State management
struct State
{
    // LED states
    bool plankStopActive = false;
    bool plankOverlapActive = false;
    bool plankIncorrectActive = false;
    bool conveyorStopActive = false;

    // Connection states
    bool wsConnected = false;
    bool ledState = false;
    bool reconnecting = false;
} state;

// Timing variables
struct Timing
{
    unsigned long lastPing = 0;
    unsigned long lastReconnectAttempt = 0;
    unsigned long lastBlinkTime = 0;
    const unsigned long PING_INTERVAL = 25000;
    const unsigned long BLINK_INTERVAL = 250; // WebSocket ping interval
} timing;

struct Server_config
{
    const char *server_ip = "192.168.1.249";
    const int port = 5000;      // Flask's port
    const char *ws_url = "/ws"; // WebSocket endpoint
} server_config;

WebServer server(80);

// Replace SocketIOclient with WebSocketsClient
WebSocketsClient webSocket;

// HTML content as a string constant
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <title>Plank Monitor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        .warning { padding: 10px; margin: 5px; border-radius: 5px; }
        .active { background-color: #ff4444; animation: blink 1s infinite; }
        .inactive { background-color: #90EE90; }
        button { padding: 10px; margin: 5px; width: 100%; }
        @keyframes blink {
            0% { background-color: #90EE90; }
            50% { background-color: #ff4444; }
            100% { background-color: #90EE90; }
        }
        .blink { animation: blink 1s infinite; }
    </style>
</head>
<body>
    <h1>Plank Monitor</h1>
    <div class="connection-status">
        <h3>Connection Status!!</h3>
    </div>
    <div class="conveyor-status">
        <h3>Conveyor Status!!</h3>
    </div>
    <div id="warnings">
        <div id="stopped" class="warning inactive">
            <h3>Stopped Plank</h3>
        </div>
        <div id="overlap" class="warning inactive">
            <h3>Overlapped Planks</h3>
        </div>
        <div id="incorrect" class="warning inactive">
            <h3>Incorrect Orientation</h3>
        </div>
    </div>
    <script>
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                    updateWarning('stopped', data.stop);
                    updateWarning('overlap', data.overlap);
                    updateWarning('incorrect', data.incorrect);
                    updateConnectionStatus(data.connected);
                    updateConveyorStop(data.conveyor_stop);
                });
        }

        function updateWarning(type, active) {
            const element = document.getElementById(type);
            element.className = 'warning ' + (active ? 'active' : 'inactive');
        }
        function updateConnectionStatus(connected) {
            const element = document.querySelector('.connection-status > h3');
            element.className = 'connection-status ' + (connected ? 'inactive' : 'active');
            element.innerHTML = 'Connected to Flask server: ' + (connected ? 'Yes' : 'No');
        }
        function updateConveyorStop(conveyor_stop) {
            const element = document.querySelector('.conveyor-status > h3');
            element.className = 'conveyor-status ' + (conveyor_stop ? 'active' : 'inactive');
            element.innerHTML = 'Conveyor Status: ' + (conveyor_stop ? 'Stopped' : 'Running');
        }


        setInterval(updateStatus, 1000);

    </script>
</body>
</html>
)rawliteral";

void setup()
{
    Serial.begin(115200);
    Serial.println("\n\nPlank Monitor System v" VERSION);
    Serial.println("Initializing...");

    setupPins();
    connectToWiFi();
    setupWebSocket();
    setupWebServer();

    Serial.println("Setup complete");
}

void setupPins()
{
    // Set LED pins as outputs
    pinMode(pins.PLANK_STOP, OUTPUT);
    pinMode(pins.PLANK_OVERLAP, OUTPUT);
    pinMode(pins.PLANK_INCORRECT, OUTPUT);
    pinMode(pins.CONVEYOR_STOP, OUTPUT);
    pinMode(pins.CONVEYOR_START_1, OUTPUT);
    pinMode(pins.CONVEYOR_START_2, OUTPUT);

    // Initialize all LEDs to off
    turnAllLEDsOff();
}

void connectToWiFi()
{
    Serial.print("Connecting to WiFi network: ");
    Serial.println(ssid);

    WiFi.begin(ssid, password);

    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20)
    {
        delay(500);
        Serial.print(".");
        attempts++;
    }

    if (WiFi.status() == WL_CONNECTED)
    {
        Serial.println("\nConnected to WiFi");
        Serial.print("IP Address: ");
        Serial.println(WiFi.localIP());
    }
    else
    {
        Serial.println("\nFailed to connect to WiFi. Continuing anyway...");
    }
}

void setupWebSocket()
{
    webSocket.begin(server_config.server_ip, server_config.port, server_config.ws_url);
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);
    webSocket.enableHeartbeat(15000, 3000, 2);
    Serial.println("WebSocket client initialized");
}

void setupWebServer()
{
    // Route for root / web page
    server.on("/", HTTP_GET, []()
              { server.send(200, "text/html", index_html); });

    // Route for API endpoints
    server.on("/api/status", HTTP_GET, getStatus);

    server.begin();
    Serial.println("HTTP server started");
}

void loop()
{
    webSocket.loop();
    server.handleClient();

    unsigned long currentMillis = millis();

    handleWebSocketPing(currentMillis);
    handleLEDBlinking(currentMillis);

    // If not connected, keep all LEDs on as a visual indicator
    if (!state.wsConnected)
    {
        turnAllLEDsOn();
    }
}

void handleWebSocketPing(unsigned long currentMillis)
{
    if (state.wsConnected && currentMillis - timing.lastPing > timing.PING_INTERVAL)
    {
        timing.lastPing = currentMillis;
        sendPing();
        Serial.println("Sending ping to keep connection alive");
    }
}

void handleLEDBlinking(unsigned long currentMillis)
{
    if (currentMillis - timing.lastBlinkTime >= timing.BLINK_INTERVAL && state.wsConnected)
    {
        timing.lastBlinkTime = currentMillis;
        state.ledState = !state.ledState;

        updateLEDs();
    }
}

void updateLEDs()
{
    // Update LEDs based on their active state
    digitalWrite(pins.PLANK_STOP, state.plankStopActive ? (state.ledState ? HIGH : LOW) : LOW);
    digitalWrite(pins.PLANK_OVERLAP, state.plankOverlapActive ? (state.ledState ? HIGH : LOW) : LOW);
    digitalWrite(pins.PLANK_INCORRECT, state.plankIncorrectActive ? (state.ledState ? HIGH : LOW) : LOW);
    digitalWrite(pins.CONVEYOR_STOP, state.conveyorStopActive ? (state.ledState ? HIGH : LOW) : LOW);
}

void turnAllLEDsOff()
{
    digitalWrite(pins.PLANK_STOP, LOW);
    digitalWrite(pins.PLANK_OVERLAP, LOW);
    digitalWrite(pins.PLANK_INCORRECT, LOW);
    digitalWrite(pins.CONVEYOR_STOP, LOW);
}

void turnAllLEDsOn()
{
    digitalWrite(pins.PLANK_STOP, HIGH);
    digitalWrite(pins.PLANK_OVERLAP, HIGH);
    digitalWrite(pins.PLANK_INCORRECT, HIGH);
    digitalWrite(pins.CONVEYOR_STOP, HIGH);
}

void startConveyor()
{
    digitalWrite(pins.CONVEYOR_START_1, HIGH);
    digitalWrite(pins.CONVEYOR_START_2, LOW);
    state.conveyorStopActive = false;
    digitalWrite(pins.CONVEYOR_STOP, LOW);
    updateConveyorStatus();
}

void stopConveyor()
{
    digitalWrite(pins.CONVEYOR_START_1, LOW);
    digitalWrite(pins.CONVEYOR_START_2, LOW);
    state.conveyorStopActive = true;
    digitalWrite(pins.CONVEYOR_STOP, HIGH);
    updateConveyorStatus();
}

void getStatus()
{
    String payload = "{\"stop\": " + String(state.plankStopActive) +
                     ", \"overlap\": " + String(state.plankOverlapActive) +
                     ", \"incorrect\": " + String(state.plankIncorrectActive) +
                     ", \"connected\": " + String(state.wsConnected) +
                     ", \"conveyor_stop\": " + String(state.conveyorStopActive) + "}";
    server.send(200, "application/json", payload);
}

void webSocketEvent(WStype_t type, uint8_t *payload, size_t length)
{
    switch (type)
    {
    case WStype_DISCONNECTED:
        handleWebSocketDisconnect();
        break;

    case WStype_CONNECTED:
        handleWebSocketConnect();
        break;

    case WStype_TEXT:
        handleWebSocketMessage(payload, length);
        break;

    case WStype_BIN:
        Serial.println("WebSocket binary data received");
        break;

    case WStype_ERROR:
        Serial.println("WebSocket ERROR received");
        state.wsConnected = false;
        break;

    case WStype_PING:
        Serial.println("WebSocket PING received");
        break;

    case WStype_PONG:
        Serial.println("WebSocket PONG received");
        break;

    default:
        Serial.printf("WebSocket unknown type %d received\n", type);
        break;
    }
}

void handleWebSocketDisconnect()
{
    Serial.println("WebSocket Disconnected!");
    state.wsConnected = false;
    stopConveyor();
}

void handleWebSocketConnect()
{
    Serial.println("WebSocket Connected!");
    state.wsConnected = true;
    timing.lastPing = millis();

    // Initialize system after connection
    turnAllLEDsOff();
    startConveyor();
}

void handleWebSocketMessage(uint8_t *payload, size_t length)
{
    Serial.printf("WebSocket message received (%d bytes)\n", length);

    // Print readable form of the payload for debugging
    if (length < 256)
    { // Only print if not too large
        Serial.print("Message: ");
        for (size_t i = 0; i < length; i++)
        {
            Serial.print((char)payload[i]);
        }
        Serial.println();
    }

    // Parse JSON message
    StaticJsonDocument<512> doc;
    DeserializationError error = deserializeJson(doc, payload);

    if (error)
    {
        Serial.print("deserializeJson() failed: ");
        Serial.println(error.c_str());
        return;
    }

    // Process message based on event type
    const char *event = doc["event"];
    if (!event)
    {
        Serial.println("Error: No event field in message");
        return;
    }

    Serial.print("Event name: ");
    Serial.println(event);

    if (strcmp(event, "control_conveyor") == 0)
    {
        handleConveyorControl(doc["data"]);
    }
    else if (strcmp(event, "rules_applied") == 0)
    {
        handleRulesApplied(doc["data"]);
    }
    // Add other event types as needed
}

void handleConveyorControl(const JsonDocument &data)
{
    if (data.containsKey("state"))
    {
        Serial.println("Conveyor control command received: " + String(data["state"]));
        // if true, stop conveyor
        if (data["state"])
            stopConveyor();
        else
            startConveyor();
    }
}

void updateConveyorStatus()
{
    StaticJsonDocument<256> doc;
    doc["event"] = "update_conveyor_stop";
    JsonObject data = doc.createNestedObject("data");
    data["state"] = state.conveyorStopActive;

    String jsonString;
    serializeJson(doc, jsonString);

    if (state.wsConnected)
    {
        webSocket.sendTXT(jsonString);
        Serial.println("Sent conveyor status update: " + jsonString);
    }
}

void handleRulesApplied(const JsonDocument &data)
{
    Serial.println("Rules applied received");

    // Handle conveyor control
    if (data.containsKey("stop_conveyor"))
    {
        bool shouldStop = data["stop_conveyor"].as<bool>();
        Serial.println("  Stop conveyor: " + String(shouldStop));

        if (shouldStop)
        {
            stopConveyor();
        }
    }

    // Reset all LED states first
    state.plankStopActive = false;
    state.plankOverlapActive = false;
    state.plankIncorrectActive = false;

    // Process alerts
    processAlerts(data);
}

void processAlerts(const JsonDocument &data)
{
    if (!data.containsKey("alert"))
    {
        return;
    }

    // Handle single alert (string)
    if (data["alert"].is<const char *>())
    {
        const char *alertType = data["alert"].as<const char *>();
        processAlertType(alertType);
        return;
    }

    // Handle multiple alerts (array)
    if (data["alert"].is<JsonArray>())
    {
        // Fix: Use JsonArrayConst instead of JsonArray for reading
        JsonArrayConst alerts = data["alert"].as<JsonArrayConst>();
        for (JsonVariantConst alert : alerts)
        {
            if (alert.is<const char *>())
            {
                processAlertType(alert.as<const char *>());
            }
        }
    }
}

// Helper function to process alert types
void processAlertType(const char *alertType)
{
    Serial.print("Processing alert: ");
    Serial.println(alertType);

    if (strcmp(alertType, "stop") == 0)
    {
        state.plankStopActive = true;
    }
    else if (strcmp(alertType, "overlap") == 0)
    {
        state.plankOverlapActive = true;
    }
    else if (strcmp(alertType, "incorrect") == 0)
    {
        state.plankIncorrectActive = true;
    }
}

// Helper function to send ping
void sendPing()
{
    StaticJsonDocument<128> doc;
    doc["event"] = "ping";
    doc.createNestedObject("data"); // Empty data object

    String jsonString;
    serializeJson(doc, jsonString);
    webSocket.sendTXT(jsonString);
}