#include <WiFi.h>
#include <WebServer.h>
// #include <SPIFFS.h>
#include <HTTPClient.h>
#include <SocketIOclient.h>
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
// Global variables to store LED states
bool plankStopLedActive = true;      // stop
bool plankOverlapLedActive = true;   // overlap
bool plankIncorrectLedActive = true; // incorrect
bool conveyorStopLedActive = true;   // conveyor_stop
WebServer server(80);

// Replace WebSocketsClient with SocketIOclient
SocketIOclient socketIO;

// Update Flask server details
const char *ws_server = "192.168.1.249";
bool connected = false;
const int ws_port = 5000;                  // Flask's port
const char *ws_path = "/socket.io/?EIO=4"; // Socket.IO path with Engine.IO v4 protocol
unsigned long lastStatusCheck = 0;
unsigned long lastBlinkTime = 0;
// const unsigned long STATUS_CHECK_INTERVAL = 1000; // Check every 1 second
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
        // const ws = new WebSocket('ws://192.168.1.249:5000/ws');
        
        // ws.onmessage = function(event) {
        //     const data = JSON.parse(event.data);
        //     console.log(data);
        //     updateWarning('stopped', data.stop);
        //     updateWarning('overlap', data.overlap);
        //     updateWarning('incorrect', data.incorrect);
        // };

        // ws.onclose = function() {
        //     console.log('WebSocket connection closed');
        //     // Attempt to reconnect
        //     setTimeout(function() {
        //         location.reload();
        //     }, 5000);
        // };
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
            element.className = 'conveyor-status ' + (conveyor_stop ? 'inactive' : 'active');
            element.innerHTML = 'Conveyor Status: ' + (conveyor_stop ? 'Running' : 'Stopped');
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

    // Initialize Socket.IO connection
    socketIO.begin(ws_server, ws_port, ws_path);
    socketIO.onEvent(socketIOEvent);

    // Set LED pins as outputs
    pinMode(PLANK_STOP_LED, OUTPUT);
    pinMode(PLANK_OVERLAP_LED, OUTPUT);
    pinMode(PLANK_INCORRECT_LED, OUTPUT);
    pinMode(CONVEYOR_STOP_LED, OUTPUT);

    // Route for root / web page
    server.on("/", HTTP_GET, []()
              { server.send(200, "text/html", index_html); });

    // Route for API endpoints
    server.on("/api/status", HTTP_GET, getStatus);

    server.begin();
}

void loop()
{
    socketIO.loop(); // This handles incoming Socket.IO events
    server.handleClient();

    // Handle LED blinking
    unsigned long currentMillis = millis();
    if (currentMillis - lastBlinkTime >= BLINK_INTERVAL)
    {
        lastBlinkTime = currentMillis;
        ledState = !ledState;

        // Update LEDs based on their active state
        if (plankStopLedActive)
            digitalWrite(PLANK_STOP_LED, ledState ? HIGH : LOW);
        if (plankOverlapLedActive)
            digitalWrite(PLANK_OVERLAP_LED, ledState ? HIGH : LOW);
        if (plankIncorrectLedActive)
            digitalWrite(PLANK_INCORRECT_LED, ledState ? HIGH : LOW);
        if (conveyorStopLedActive)
            digitalWrite(CONVEYOR_STOP_LED, ledState ? HIGH : LOW);
    }
}

// Keep the API endpoint handler separate
void getStatus()
{
    // return the current status of the LEDs
    String payload = "{\"stop\": " + String(plankStopLedActive) + ", \"overlap\": " + String(plankOverlapLedActive) + ", \"incorrect\": " + String(plankIncorrectLedActive) + ", \"connected\": " + String(connected) + ", \"conveyor_stop\": " + String(conveyorStopLedActive) + "}";
    server.send(200, "application/json", payload);
}

// Update WebSocket event handler to handle Socket.IO events
void socketIOEvent(socketIOmessageType_t type, uint8_t *payload, size_t length)
{
    switch (type)
    {
    case sIOtype_DISCONNECT:
        Serial.println("Socket.IO Disconnected!");
        connected = false;
        break;
    case sIOtype_CONNECT:
        Serial.println("Socket.IO Connected!");
        // Join default namespace
        socketIO.send(sIOtype_CONNECT, "/");
        connected = true;
        break;
    case sIOtype_EVENT:
        Serial.printf("[IOc] Event: %s\n", payload);
        StaticJsonDocument<200> doc;
        DeserializationError error = deserializeJson(doc, payload);

        if (error)
        {
            Serial.print("deserializeJson() failed: ");
            Serial.println(error.c_str());
            return;
        }

        // Socket.IO events come as arrays where first element is event name
        const char *event = doc[0];
        if (strcmp(event, "status_update") == 0)
        {
            handleStatusUpdate(doc[1]);
        }
        break;
    }
}

void handleStatusUpdate(const JsonDocument &doc)
{
    // Update LED active states instead of directly controlling pins
    plankStopLedActive = doc["stop"];
    plankOverlapLedActive = doc["overlap"];
    plankIncorrectLedActive = doc["incorrect"];
    conveyorStopLedActive = doc["conveyor_stop"];

    // If not active, ensure LEDs are off
    if (!plankStopLedActive)
        digitalWrite(PLANK_STOP_LED, LOW);
    if (!plankOverlapLedActive)
        digitalWrite(PLANK_OVERLAP_LED, LOW);
    if (!plankIncorrectLedActive)
        digitalWrite(PLANK_INCORRECT_LED, LOW);
    if (!conveyorStopLedActive)
        digitalWrite(CONVEYOR_STOP_LED, LOW);
}
