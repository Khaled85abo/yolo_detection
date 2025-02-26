#include <WiFi.h>
#include <WebServer.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// esp32 ip address: http://192.168.1.202/
// Flask server IP address and endpoints
const char *flask_server_base = "http://192.168.1.249:5000";
const char *flask_status_endpoint = "http://192.168.1.249:5000/api/status";
const char *flask_conveyor_endpoint = "http://192.168.1.249:5000/api/conveyor";

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
bool connected = false;              // server connection status

WebServer server(80);

// Timing variables
unsigned long lastStatusCheck = 0;
unsigned long lastBlinkTime = 0;
const unsigned long STATUS_CHECK_INTERVAL = 1000; // Check every 1 second
const unsigned long BLINK_INTERVAL = 250;         // Blink interval
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
    <div class="controls">
        <button id="toggleConveyor">Toggle Conveyor</button>
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
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }

        function updateWarning(type, active) {
            const element = document.getElementById(type);
            if (element) {
                element.className = 'warning ' + (active ? 'active' : 'inactive');
            }
        }
        
        function updateConnectionStatus(connected) {
            const element = document.querySelector('.connection-status > h3');
            if (element) {
                element.className = connected ? 'inactive' : 'active';
                element.innerHTML = 'Connected to Flask server: ' + (connected ? 'Yes' : 'No');
            }
        }
        
        function updateConveyorStop(conveyor_stop) {
            const element = document.querySelector('.conveyor-status > h3');
            if (element) {
                element.className = conveyor_stop ? 'active' : 'inactive';
                element.innerHTML = 'Conveyor Status: ' + (conveyor_stop ? 'Stopped' : 'Running');
            }
        }

        // Set up toggle conveyor button
        document.getElementById('toggleConveyor').addEventListener('click', function() {
            fetch('/api/conveyor', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ toggle: true }),
            })
            .then(response => response.json())
            .then(data => {
                console.log('Conveyor toggled:', data);
                updateStatus();
            })
            .catch(error => {
                console.error('Error toggling conveyor:', error);
            });
        });

        // Update status every second
        setInterval(updateStatus, 1000);
        // Initial status update
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

    // Route to control conveyor
    server.on("/api/conveyor", HTTP_POST, handleConveyorControl);

    server.begin();
    Serial.println("HTTP server started");
}

void loop()
{
    server.handleClient();

    // Poll for status at regular intervals
    unsigned long currentMillis = millis();
    if (currentMillis - lastStatusCheck >= STATUS_CHECK_INTERVAL)
    {
        lastStatusCheck = currentMillis;
        pollStatusFromServer();
    }

    // Handle LED blinking
    if (currentMillis - lastBlinkTime >= BLINK_INTERVAL)
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
}

void getStatus()
{
    // Return the current status of the LEDs
    String payload = "{\"stop\": " + String(plankStopLedActive ? "true" : "false") +
                     ", \"overlap\": " + String(plankOverlapLedActive ? "true" : "false") +
                     ", \"incorrect\": " + String(plankIncorrectLedActive ? "true" : "false") +
                     ", \"connected\": " + String(connected ? "true" : "false") +
                     ", \"conveyor_stop\": " + String(conveyorStopLedActive ? "true" : "false") + "}";
    server.send(200, "application/json", payload);
}

void handleConveyorControl()
{
    // Handle POST requests to control the conveyor
    if (server.hasArg("plain"))
    {
        String body = server.arg("plain");
        StaticJsonDocument<200> doc;
        DeserializationError error = deserializeJson(doc, body);

        if (!error)
        {
            bool toggle = doc["toggle"];
            if (toggle)
            {
                // Toggle the conveyor state
                conveyorStopLedActive = !conveyorStopLedActive;

                // Send update to Flask server
                updateConveyorStateOnServer(conveyorStopLedActive);

                server.send(200, "application/json", "{\"success\": true, \"state\": " + String(conveyorStopLedActive ? "true" : "false") + "}");
            }
            else if (doc.containsKey("state"))
            {
                bool state = doc["state"];
                conveyorStopLedActive = state;

                // Send update to Flask server
                updateConveyorStateOnServer(conveyorStopLedActive);

                server.send(200, "application/json", "{\"success\": true, \"state\": " + String(conveyorStopLedActive ? "true" : "false") + "}");
            }
            else
            {
                server.send(400, "application/json", "{\"error\": \"Missing 'toggle' or 'state' parameter\"}");
            }
        }
        else
        {
            server.send(400, "application/json", "{\"error\": \"Invalid JSON\"}");
        }
    }
    else
    {
        server.send(400, "application/json", "{\"error\": \"No data provided\"}");
    }
}

void pollStatusFromServer()
{
    if (WiFi.status() == WL_CONNECTED)
    {
        HTTPClient http;
        http.begin(flask_status_endpoint);
        int httpCode = http.GET();

        if (httpCode > 0)
        {
            if (httpCode == HTTP_CODE_OK)
            {
                String payload = http.getString();
                Serial.println("Status from server: " + payload);

                // Parse JSON
                StaticJsonDocument<200> doc;
                DeserializationError error = deserializeJson(doc, payload);

                if (!error)
                {
                    // Update LED states
                    plankStopLedActive = doc["stop"];
                    plankOverlapLedActive = doc["overlap"];
                    plankIncorrectLedActive = doc["incorrect"];
                    conveyorStopLedActive = doc["conveyor_stop"];
                    connected = true;
                }
                else
                {
                    Serial.print("deserializeJson() failed: ");
                    Serial.println(error.c_str());
                    connected = false;
                }
            }
            else
            {
                Serial.printf("HTTP GET failed, error: %s\n", http.errorToString(httpCode).c_str());
                connected = false;
            }
        }
        else
        {
            Serial.printf("HTTP GET failed, error: %s\n", http.errorToString(httpCode).c_str());
            connected = false;
        }

        http.end();
    }
    else
    {
        Serial.println("WiFi not connected");
        connected = false;
    }
}

void updateConveyorStateOnServer(bool state)
{
    if (WiFi.status() == WL_CONNECTED)
    {
        HTTPClient http;
        http.begin(flask_conveyor_endpoint);
        http.addHeader("Content-Type", "application/json");

        String jsonPayload = "{\"state\": " + String(state ? "true" : "false") + "}";
        Serial.println("Sending to server: " + jsonPayload);

        int httpCode = http.POST(jsonPayload);

        if (httpCode > 0)
        {
            if (httpCode == HTTP_CODE_OK)
            {
                String response = http.getString();
                Serial.println("Server response: " + response);
            }
            else
            {
                Serial.printf("HTTP POST failed, error: %s\n", http.errorToString(httpCode).c_str());
            }
        }
        else
        {
            Serial.printf("HTTP POST failed, error: %s\n", http.errorToString(httpCode).c_str());
        }

        http.end();
    }
    else
    {
        Serial.println("WiFi not connected");
    }
}