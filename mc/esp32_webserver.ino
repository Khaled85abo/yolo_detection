#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>
#include <HTTPClient.h>
#include <WebSocketsClient.h>

// Flask server IP address
const char *flask_server_ip = "http://192.168.1.249:5000/api/status";

// Replace with your network credentials
const char *ssid = "TN-JE3155";
const char *password = "";

// LED pins
const int RED_LED = 2;
const int YELLOW_LED = 3;
const int GREEN_LED = 4;

WebServer server(80);

// Add WebSocket client instance
WebSocketsClient webSocket;

// Update Flask server details
const char *ws_server = "192.168.1.249";
const int ws_port = 8765; // Typical WebSocket port, adjust as needed

// Add WebSocket event handler
void webSocketEvent(WStype_t type, uint8_t *payload, size_t length)
{
    switch (type)
    {
    case WStype_DISCONNECTED:
        Serial.println("WebSocket Disconnected!");
        break;
    case WStype_CONNECTED:
        Serial.println("WebSocket Connected!");
        break;
    case WStype_TEXT:
        // Handle incoming WebSocket message
        String message = String((char *)payload);
        handleStatusUpdate(message);
        // Forward the status to connected web clients
        server.send(200, "application/json", message);
        break;
    }
}

void handleStatusUpdate(String payload)
{
    // Update LED states based on received payload
    digitalWrite(RED_LED, payload.indexOf("\"stop\": true") > -1 ? HIGH : LOW);
    digitalWrite(YELLOW_LED, payload.indexOf("\"overlap\": true") > -1 ? HIGH : LOW);
    digitalWrite(GREEN_LED, payload.indexOf("\"incorrect\": true") > -1 ? HIGH : LOW);
}

// HTML content as a string constant
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
    <title>Plank Monitor</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        .warning { padding: 10px; margin: 5px; border-radius: 5px; }
        .active { background-color: #ffcccc; animation: blink 1s infinite; }
        .inactive { background-color: #ccffcc; }
        button { padding: 10px; margin: 5px; width: 100%; }
        @keyframes blink {
            0% { background-color: #ccffcc; }
            50% { background-color: #ffcccc; }
            100% { background-color: #ccffcc; }
        }
        .blink { animation: blink 1s infinite; }
    </style>
</head>
<body>
    <h1>Plank Monitor</h1>
    <div id="warnings">
        <div id="stopped" class="warning inactive">
            <h3>Stopped Plank</h3>
            <button onclick="acknowledge('stopped')">Acknowledge</button>
        </div>
        <div id="overlap" class="warning inactive">
            <h3>Overlapped Planks</h3>
            <button onclick="acknowledge('overlap')">Acknowledge</button>
        </div>
        <div id="incorrect" class="warning inactive">
            <h3>Incorrect Orientation</h3>
            <button onclick="acknowledge('incorrect')">Acknowledge</button>
        </div>
    </div>
    <button onclick="stopConveyor()" style="background-color: #ff6666;">STOP CONVEYOR</button>
    <script>
        const ws = new WebSocket('ws://192.168.1.249:8765/ws');
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            console.log(data);
            updateWarning('stopped', data.stop);
            updateWarning('overlap', data.overlap);
            updateWarning('incorrect', data.incorrect);
        };

        ws.onclose = function() {
            console.log('WebSocket connection closed');
            // Attempt to reconnect
            setTimeout(function() {
                location.reload();
            }, 5000);
        };

        function updateWarning(type, active) {
            const element = document.getElementById(type);
            element.className = 'warning ' + (active ? 'active' : 'inactive');
        }
        function acknowledge(type) {
            fetch('/api/acknowledge?type=' + type, { method: 'POST' });
        }
        function stopConveyor() {
            fetch('/api/stop-conveyor', { method: 'POST' });
        }
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
    webSocket.begin(ws_server, ws_port, "/ws");
    webSocket.onEvent(webSocketEvent);
    webSocket.setReconnectInterval(5000);

    // Route for root / web page
    server.on("/", HTTP_GET, []()
              { server.send(200, "text/html", index_html); });

    // Route for API endpoints
    server.on("/api/status", HTTP_GET, handleStatus);
    server.on("/api/acknowledge", HTTP_POST, handleAcknowledge);
    server.on("/api/stop-conveyor", HTTP_POST, handleStopConveyor);

    server.begin();
}

void loop()
{
    webSocket.loop();
    server.handleClient();
}

void handleStatus()
{
    // Get status from Flask server
    // if server is not reachable, use fake data
    HTTPClient http;
    http.begin(flask_server_ip);
    int httpCode = http.GET();
    String payload = http.getString();
    if (httpCode != 200)
    {
        payload = "{\"stop\": false, \"overlap\": false, \"incorrect\": true}";
    }
    // Turn on/off the warning lights
    // Red: for stopped planks
    // Yellow: for overlapped planks
    // Green: for correct planks
    if (payload.indexOf("\"stop\": true") > -1)
    {
        digitalWrite(RED_LED, HIGH);
    }
    else
    {
        digitalWrite(RED_LED, LOW);
    }
    if (payload.indexOf("\"overlap\": true") > -1)
    {
        digitalWrite(YELLOW_LED, HIGH);
    }
    else
    {
        digitalWrite(YELLOW_LED, LOW);
    }
    if (payload.indexOf("\"incorrect\": true") > -1)
    {
        digitalWrite(GREEN_LED, HIGH);
    }
    else
    {
        digitalWrite(GREEN_LED, LOW);
    }
    http.end();
    server.send(200, "application/json", payload);
}

void handleAcknowledge()
{
    String type = server.arg("type");
    // Handle acknowledgment
    // TODO: Implement acknowledgment logic with Flask server
    Serial.println("Acknowledged: " + type);
    server.send(200, "text/plain", "OK");
}

void handleStopConveyor()
{
    // TODO: Implement conveyor stop logic
    server.send(200, "text/plain", "OK");
}