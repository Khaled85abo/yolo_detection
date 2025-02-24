#include <WiFi.h>
#include <WebServer.h>
#include <SPIFFS.h>
#include <HTTPClient.h>

// Flask server IP address
const char *flask_server_ip = "http://192.168.1.249:5000/api/status";

// Replace with your network credentials
const char *ssid = "TN-JE3155";
const char *password = "";

WebServer server(80);

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
        function updateStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    console.log(data);
                    updateWarning('stopped', data.stop);
                    updateWarning('overlap', data.overlap);
                    updateWarning('incorrect', data.incorrect);
                });
        }
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
        setInterval(updateStatus, 1000);
        updateStatus();
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
    server.handleClient();
}

void handleStatus()
{
    // Get status from Flask server
    HTTPClient http;
    http.begin(flask_server_ip);
    int httpCode = http.GET();
    String payload = http.getString();
    http.end();
    server.send(200, "application/json", payload);
}

void handleAcknowledge()
{
    String type = server.arg("type");
    // Handle acknowledgment
    server.send(200, "text/plain", "OK");
}

void handleStopConveyor()
{
    // Implement conveyor stop logic
    server.send(200, "text/plain", "OK");
}