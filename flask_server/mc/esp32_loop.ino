#include <WiFi.h>
#include <WebServer.h>
// #include <SPIFFS.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// define a bit-mask style system for the status of the conveyor
// incorrect =1
// overlap = 2
// stop= 4
// Each mask represents a bit position using binary notation
// const uint8_t INCORRECT_MASK = 0b001; // Binary: 001 (Decimal: 1)
// const uint8_t OVERLAP_MASK = 0b010;   // Binary: 010 (Decimal: 2)
// const uint8_t STOP_MASK = 0b100;      // Binary: 100 (Decimal: 4)

// void checkStatus()
// {
//     HTTPClient http;
//     http.begin(flask_server_ip);
//     int httpCode = http.GET();

//     // Get the status as a single number
//     uint8_t status = (httpCode != 200) ? 7 : http.getString().toInt(); // 7 = all flags set in error case
//     http.end();

//     // Use bitwise AND to check each state
//     greenLedActive = status & INCORRECT_MASK; // Check bit 0
//     yellowLedActive = status & OVERLAP_MASK;  // Check bit 1
//     redLedActive = status & STOP_MASK;        // Check bit 2

//     // Turn off LEDs if they're not active
//     if (!redLedActive)
//         digitalWrite(RED_LED, LOW);
//     if (!yellowLedActive)
//         digitalWrite(YELLOW_LED, LOW);
//     if (!greenLedActive)
//         digitalWrite(GREEN_LED, LOW);
// }

// Flask server IP address
const char *flask_server_ip = "http://192.168.1.249:5000/api/status";

// Replace with your network credentials
const char *ssid = "TN-JE3155";
const char *password = "";

// LED pins
const int RED_LED = 19;
const int YELLOW_LED = 4;
const int GREEN_LED = 5;

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

// Add these variables at the top with other globals
unsigned long lastStatusCheck = 0;
unsigned long lastBlinkTime = 0;
const unsigned long STATUS_CHECK_INTERVAL = 1000; // Check every 1 second
const unsigned long BLINK_INTERVAL = 250;         // Blink every 250ms
bool ledState = false;

// Global variables to store LED states
bool redLedActive = false;
bool yellowLedActive = false;
bool greenLedActive = false;

String fetchStatusFromServer()
{
    HTTPClient http;
    http.begin(flask_server_ip);
    int httpCode = http.GET();
    String payload = http.getString();
    http.end();

    if (httpCode != 200)
    {
        payload = "{\"stop\": true, \"overlap\": true, \"incorrect\": true}";
    }
    return payload;
}

void checkStatus()
{
    String payload = fetchStatusFromServer();

    // Create a JSON document
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, payload);

    if (error)
    {
        Serial.print("JSON parsing failed: ");
        Serial.println(error.c_str());
        return;
    }

    // Update LED states based on parsed JSON
    redLedActive = doc["stop"].as<bool>();
    yellowLedActive = doc["overlap"].as<bool>();
    greenLedActive = doc["incorrect"].as<bool>();

    // Turn off LEDs if they're not active
    if (!redLedActive)
        digitalWrite(RED_LED, LOW);
    if (!yellowLedActive)
        digitalWrite(YELLOW_LED, LOW);
    if (!greenLedActive)
        digitalWrite(GREEN_LED, LOW);
}

void handleStatus()
{
    String payload = fetchStatusFromServer();
    server.send(200, "application/json", payload);
}

void setup()
{
    Serial.begin(115200);

    // Set LED pins as outputs
    pinMode(RED_LED, OUTPUT);
    pinMode(YELLOW_LED, OUTPUT);
    pinMode(GREEN_LED, OUTPUT);

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

    // Check status periodically
    unsigned long currentMillis = millis();
    if (currentMillis - lastStatusCheck >= STATUS_CHECK_INTERVAL)
    {
        lastStatusCheck = currentMillis;
        checkStatus();
    }

    // Handle LED blinking
    if (currentMillis - lastBlinkTime >= BLINK_INTERVAL)
    {
        lastBlinkTime = currentMillis;
        ledState = !ledState;

        // Update LEDs based on their active state
        if (redLedActive)
            digitalWrite(RED_LED, ledState);
        if (yellowLedActive)
            digitalWrite(YELLOW_LED, ledState);
        if (greenLedActive)
            digitalWrite(GREEN_LED, ledState);
    }
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