#include <WiFi.h>
#include <WebServer.h>
// #include <SPIFFS.h>
#include <HTTPClient.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// esp32 ip address: http://192.168.1.202/
// Flask server IP address
const char *flask_server_ip = "http://192.168.1.249:5000/api/status";

// Replace with your network credentials
const char *ssid = "TN-JE3155";
const char *password = "";

// LED pins
const int PLANK_STOP_LED = 19;     // stop
const int PLANK_OVERLAP_LED = 4;   // overlap
const int PLANK_INCORRECT_LED = 5; // incorrect
const int CONVEYOR_STOP_LED = 22;  // conveyor_stop

const int CONVEYOR_START_LED_1 = 27; // Connect to IN1 of L298N
const int CONVEYOR_START_LED_2 = 26; // Connect to IN2 of L298N

// Global variables to store LED states
bool plankStopLedActive = false;      // stop
bool plankOverlapLedActive = false;   // overlap
bool plankIncorrectLedActive = false; // incorrect
bool conveyorStopLedActive = false;   // conveyor_stop

WebServer server(80);

// Replace SocketIOclient with WebSocketsClient
WebSocketsClient webSocket;

// Update Flask server details
const char *ws_server = "192.168.1.249";
bool connected = false;
const int ws_port = 5000; // Flask's port
// Update to use standard WebSocket endpoint
const char *ws_url = "/ws";         // New WebSocket endpoint
unsigned long pingInterval = 25000; // WebSocket ping interval
unsigned long lastPing = 0;
bool reconnecting = false;
unsigned long lastReconnectAttempt = 0;
unsigned long lastBlinkTime = 0;
const unsigned long BLINK_INTERVAL = 250; // Blink
bool ledState = false;

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

    // Connect to Wi-Fi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED)
    {
        delay(1000);
        Serial.println("Connecting to WiFi...");
    }
    Serial.println("Connected to WiFi");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());

    // Initialize WebSocket connection
    webSocket.begin(ws_server, ws_port, ws_url);
    webSocket.onEvent(webSocketEvent);
    // Increase timeout/reconnect interval to reduce disconnections
    webSocket.setReconnectInterval(5000);
    // Enable auto-reconnect
    webSocket.enableHeartbeat(15000, 3000, 2);

    // Set LED pins as outputs
    pinMode(PLANK_STOP_LED, OUTPUT);
    pinMode(PLANK_OVERLAP_LED, OUTPUT);
    pinMode(PLANK_INCORRECT_LED, OUTPUT);
    pinMode(CONVEYOR_STOP_LED, OUTPUT);
    pinMode(CONVEYOR_START_LED_1, OUTPUT);
    pinMode(CONVEYOR_START_LED_2, OUTPUT);

    // Route for root / web page
    server.on("/", HTTP_GET, []()
              { server.send(200, "text/html", index_html); });

    // Route for API endpoints
    server.on("/api/status", HTTP_GET, getStatus);

    server.begin();
}

void loop()
{
    webSocket.loop();
    server.handleClient();

    unsigned long currentMillis = millis();

    // Handle WebSocket ping to maintain connection
    if (connected && currentMillis - lastPing > pingInterval)
    {
        lastPing = currentMillis;
        // Send ping to keep connection alive
        sendPing();
        Serial.println("Sending ping");
    }

    // Handle LED blinking
    if (currentMillis - lastBlinkTime >= BLINK_INTERVAL && connected)
    {
        lastBlinkTime = currentMillis;
        ledState = !ledState;

        // Update LEDs based on their active state
        if (plankStopLedActive)
            digitalWrite(PLANK_STOP_LED, ledState ? HIGH : LOW);
        else
            digitalWrite(PLANK_STOP_LED, LOW);
        if (plankOverlapLedActive)
            digitalWrite(PLANK_OVERLAP_LED, ledState ? HIGH : LOW);
        else
            digitalWrite(PLANK_OVERLAP_LED, LOW);
        if (plankIncorrectLedActive)
            digitalWrite(PLANK_INCORRECT_LED, ledState ? HIGH : LOW);
        else
            digitalWrite(PLANK_INCORRECT_LED, LOW);
        if (conveyorStopLedActive)
            digitalWrite(CONVEYOR_STOP_LED, ledState ? HIGH : LOW);
        else
            digitalWrite(CONVEYOR_STOP_LED, LOW);
    }
    if (!connected)
    {
        turnAllLEDsOn();
    }
}

void turnAllLEDsOff()
{
    digitalWrite(PLANK_STOP_LED, LOW);
    digitalWrite(PLANK_OVERLAP_LED, LOW);
    digitalWrite(PLANK_INCORRECT_LED, LOW);
    digitalWrite(CONVEYOR_STOP_LED, LOW);
}

void turnAllLEDsOn()
{
    digitalWrite(PLANK_STOP_LED, HIGH);
    digitalWrite(PLANK_OVERLAP_LED, HIGH);
    digitalWrite(PLANK_INCORRECT_LED, HIGH);
    digitalWrite(CONVEYOR_STOP_LED, HIGH);
}

void startConveyor()
{
    digitalWrite(CONVEYOR_START_LED_1, HIGH);
    digitalWrite(CONVEYOR_START_LED_2, LOW);
    conveyorStopLedActive = false;
    digitalWrite(CONVEYOR_STOP_LED, LOW);
    updateConveyorStatus();
}

void stopConveyor()
{
    digitalWrite(CONVEYOR_START_LED_1, LOW);
    digitalWrite(CONVEYOR_START_LED_2, LOW);
    conveyorStopLedActive = true;
    digitalWrite(CONVEYOR_STOP_LED, HIGH);
    updateConveyorStatus();
}

// Keep the API endpoint handler separate
void getStatus()
{
    // return the current status of the LEDs
    String payload = "{\"stop\": " + String(plankStopLedActive) + ", \"overlap\": " + String(plankOverlapLedActive) + ", \"incorrect\": " + String(plankIncorrectLedActive) + ", \"connected\": " + String(connected) + ", \"conveyor_stop\": " + String(conveyorStopLedActive) + "}";
    server.send(200, "application/json", payload);
}

// Update WebSocket event handler
void webSocketEvent(WStype_t type, uint8_t *payload, size_t length)
{
    // Document declaration outside the switch
    StaticJsonDocument<512> doc;

    switch (type)
    {
    case WStype_DISCONNECTED:
        Serial.println("WebSocket Disconnected!");
        connected = false;
        stopConveyor();
        break;

    case WStype_CONNECTED:
        Serial.println("WebSocket Connected!");
        connected = true;
        lastPing = millis();
        // Send initial status update after connection
        turnAllLEDsOff();
        startConveyor();
        updateConveyorStatus();
        break;

    case WStype_TEXT:
        Serial.printf("WebSocket message received (%d bytes): %s\n", length, payload);

        // Print readable form of the payload for debugging
        for (size_t i = 0; i < length; i++)
        {
            Serial.print((char)payload[i]);
        }
        Serial.println();

        // Parse JSON message
        {
            DeserializationError error = deserializeJson(doc, payload);

            if (error)
            {
                Serial.print("deserializeJson() failed: ");
                Serial.println(error.c_str());
                return;
            }

            // WebSocket messages come with event and data fields
            const char *event = doc["event"];
            Serial.print("Event name: ");
            Serial.println(event);

            // if (strcmp(event, "status_update") == 0)
            // {
            //     handleStatusUpdate(doc["data"]);
            // }
            if (strcmp(event, "control_conveyor") == 0)
            {
                handleConveyorControl(doc["data"]);
            }
            else if (strcmp(event, "rules_applied") == 0)
            {
                handleRulesApplied(doc["data"]);
            }
        }
        break;

    case WStype_BIN:
        Serial.println("WebSocket binary data received");
        break;

    case WStype_ERROR:
        Serial.println("WebSocket ERROR received");
        connected = false;
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

void handleConveyorControl(const JsonDocument &data)
{
    if (data.containsKey("state"))
    {
        Serial.println("Conveyor control command received: " + String(data["state"]));
        // Update the server about our new state
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
    data["state"] = conveyorStopLedActive;

    String jsonString;
    serializeJson(doc, jsonString);

    if (connected)
    {
        webSocket.sendTXT(jsonString);
        Serial.println("Sent conveyor status update: " + jsonString);
    }
}

void handleStatusUpdate(const JsonDocument &data)
{
    // Update LED active states
    plankStopLedActive = data["stop"];
    plankOverlapLedActive = data["overlap"];
    plankIncorrectLedActive = data["incorrect"];
    // conveyorStopLedActive = data["conveyor_stop"];

    Serial.println("Status update received:");
    Serial.println("  Stop: " + String(plankStopLedActive));
    Serial.println("  Overlap: " + String(plankOverlapLedActive));
    Serial.println("  Incorrect: " + String(plankIncorrectLedActive));
    // Serial.println("  Conveyor Stop: " + String(conveyorStopLedActive));

    // If not active, ensure LEDs are off
    if (!plankStopLedActive)
        digitalWrite(PLANK_STOP_LED, LOW);
    if (!plankOverlapLedActive)
        digitalWrite(PLANK_OVERLAP_LED, LOW);
    if (!plankIncorrectLedActive)
        digitalWrite(PLANK_INCORRECT_LED, LOW);
    // if (!conveyorStopLedActive)
    //     digitalWrite(CONVEYOR_STOP_LED, LOW);
}

void handleRulesApplied(const JsonDocument &data)
{
    Serial.println("Rules applied received:");
    Serial.println("  Stop: " + String(data["stop_conveyor"].as<bool>()));

    // If stop_conveyor is true, stop the conveyor
    if (data["stop_conveyor"].as<bool>())
    {
        stopConveyor();
    }

    // Reset all LED states first
    plankStopLedActive = false;
    plankOverlapLedActive = false;
    plankIncorrectLedActive = false;

    // Process alerts using a more compatible approach
    if (data.containsKey("alert"))
    {
        // Check if alert is a string (single value)
        if (data["alert"].is<const char *>())
        {
            const char *alertType = data["alert"].as<const char *>();
            processAlertType(alertType);
        }
        // Otherwise, try to process it as an array
        else
        {
            // Print the raw JSON for debugging
            Serial.print("Alert type: ");

            // Try to access array elements directly by index
            int i = 0;
            while (data["alert"][i])
            { // Check if element exists
                const char *alertType = data["alert"][i].as<const char *>();
                processAlertType(alertType);
                i++;
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
        plankStopLedActive = true;
    }
    else if (strcmp(alertType, "overlap") == 0)
    {
        plankOverlapLedActive = true;
    }
    else if (strcmp(alertType, "incorrect") == 0)
    {
        plankIncorrectLedActive = true;
    }
}

// Helper function to send ping
void sendPing()
{
    StaticJsonDocument<128> doc;
    doc["event"] = "ping";
    JsonObject data = doc.createNestedObject("data");

    String jsonString;
    serializeJson(doc, jsonString);
    webSocket.sendTXT(jsonString);
}