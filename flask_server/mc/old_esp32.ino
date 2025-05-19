#include <WiFi.h>
#include <WebServer.h>
// #include <SPIFFS.h>
#include <HTTPClient.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>
#include "esp_wifi.h"
#include "esp_netif.h"

// esp32 ip address: http://192.168.1.202/
// Flask server IP address
// const char *flask_server_ip = "http://192.168.1.249:5000/api/status";

// Replace with your network credentials
const char *ssid = "Pi-rise";
const char *password = "";

const char *ap_ssid = "ESP32-AP";
const char *ap_password = "riserise";

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

// Update global variables to use newer API
wifi_sta_list_t stationList;
unsigned long lastStationCheckTime = 0;
const unsigned long STATION_CHECK_INTERVAL = 5000; // Check every 5 seconds

// Replace SocketIOclient with WebSocketsClient
WebSocketsClient webSocket;

// Update Flask server details - try just IP address
const char *ws_server = "10.42.0.1";
bool connected = false;
const int ws_port = 5000;
// Update WebSocket path - try with and without leading slash
const char *ws_url = "/ws";  // could also try "ws"
unsigned long pingInterval = 25000; // WebSocket ping interval
unsigned long lastPing = 0;
bool reconnecting = false;
unsigned long lastReconnectAttempt = 0;
unsigned long lastBlinkTime = 0;
const unsigned long BLINK_INTERVAL = 250; // Blink
bool ledState = false;

// Define the MAC address of your Raspberry Pi's WiFi interface
// You'll need to find this value from your Pi (use 'ifconfig wlan0' or 'ip addr')
uint8_t piMacAddress[] = {0x2C, 0xCF, 0x67, 0x4A, 0xF9, 0xDE}; // Replace with your Pi's actual MAC

// The static IP you want to assign to the Pi
IPAddress piStaticIP(192, 168, 4, 100);

// Address of ESP32 AP
IPAddress apIP(192, 168, 4, 1);
IPAddress netMask(255, 255, 255, 0);

// WebSocket client
WiFiClient client;

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

// Add these variables
bool piDetected = false;
IPAddress piActualIP;

void setup()
{
    Serial.begin(115200);

    // Connect to Wi-Fi
    // WiFi.mode(WIFI_AP_STA);

    // // Start AP with limited configuration
    // WiFi.softAPConfig(apIP, apIP, netMask);
    // // Set maxConnection to 1 to only allow the Pi
    // WiFi.softAP(ap_ssid, ap_password, 1, 0, 1); // channel 1, hidden=false, max_connection=1
    // Serial.println("AP started with max 1 connection");
    // Serial.print("AP IP address: ");
    // Serial.println(WiFi.softAPIP());

    // WiFi.onEvent(WiFiEvent);


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

    // Initialize WebSocket connection with more debug output
    Serial.println("Attempting to connect to WebSocket server:");
    Serial.print("Server: ");
    Serial.println(ws_server);
    Serial.print("Port: ");
    Serial.println(ws_port);
    Serial.print("URL: ");
    Serial.println(ws_url);
    
    // Set more verbose debug level
    webSocket.setReconnectInterval(5000);
    webSocket.enableHeartbeat(15000, 3000, 2);
    
    // CRITICAL: Register the event handler (this may be missing)
    webSocket.onEvent(webSocketEvent);
    
    // Try standard begin with explicit path
    webSocket.begin(ws_server, ws_port, ws_url);
    
    // OR try Socket.IO mode if the above doesn't work
    // webSocket.beginSocketIO(ws_server, ws_port);
    
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

    // Note: The DHCP static lease configuration has been removed as it's not
    // directly supported in the same way with the newer ESP-IDF.
    // Instead, we'll log the Pi's MAC address when it connects
    
    // Serial.print("We'll be looking for Pi's MAC: ");
    // for (int i = 0; i < 6; i++) {
    //     Serial.printf("%02X", piMacAddress[i]);
    //     if (i < 5) Serial.print(":");
    // }
    // Serial.println();
    // Serial.print("Expected Pi IP: ");
    // Serial.println(piStaticIP.toString());

    server.begin();

    HTTPClient http;
    http.begin("http://10.42.0.1:5000");
    int httpCode = http.GET();
    Serial.printf("HTTP test response: %d\n", httpCode);
    http.end();
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

    // Check for connected stations periodically
    // if (currentMillis - lastStationCheckTime >= STATION_CHECK_INTERVAL) {
    //     lastStationCheckTime = currentMillis;
    //     piDetected = printConnectedStations();
        
    //     // If we detected the Pi but aren't connected, try its last known IP
    //     if (piDetected && !connected && piActualIP != IPAddress(0,0,0,0)) {
    //         Serial.printf("Attempting to reconnect to last known Pi IP: %s\n", 
    //                      piActualIP.toString().c_str());
    //         webSocket.disconnect();
    //         delay(500);
    //         webSocket.begin(piActualIP.toString().c_str(), ws_port, ws_url);
    //     }
    // }

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

    // Check for connected clients - simplified version
    // if(WiFi.softAPgetStationNum() > 0) {
    //     // Get connected station count
    //     Serial.print("Number of connected stations: ");
    //     Serial.println(WiFi.softAPgetStationNum());
        
    //     // We can't easily get the detailed station info with the newer API
    //     // in the Arduino framework, so we'll just check for our WebSocket connection
    // }
    
    // Check if we can connect to the Pi's WebSocket server
    // if (!client.connected()) {
    //     Serial.println("Attempting to connect to Pi's WebSocket server...");
    //     if (client.connect(piStaticIP, ws_port)) { // Use ws_port (5000) instead of webSocketPort
    //         Serial.println("Connected to WebSocket server!");
    //         // Here you would implement your WebSocket protocol
    //     } else {
    //         Serial.println("Connection to WebSocket server failed");
    //         delay(5000); // Wait 5 seconds before retrying
    //     }
    // }
    
    // Handle WebSocket communication when connected
    // if (client.connected()) {
    //     // Your WebSocket communication code here
    // }
    
    delay(5000); // Check every 5 seconds
}

// void WiFiEvent(WiFiEvent_t event) {
//     switch(event) {
//         case ARDUINO_EVENT_WIFI_AP_STACONNECTED:
//             Serial.println("New station connected to ESP32 AP!");
//             printConnectedStations();
//             break;
//         case ARDUINO_EVENT_WIFI_AP_STADISCONNECTED:
//             Serial.println("Station disconnected from ESP32 AP!");
//             printConnectedStations();
//             break;
//         default:
//             break;
//     }
// }

// Update the printConnectedStations function to scan a smaller IP range
// bool printConnectedStations() {
//     bool piFound = false;
//     int stationCount = WiFi.softAPgetStationNum();
//     if (stationCount == 0) {
//         Serial.println("No stations connected");
//         return false;
//     }
    
//     Serial.print("Number of connected stations: ");
//     Serial.println(stationCount);
    
//     // Get the station list using the newer API
//     esp_err_t result = esp_wifi_ap_get_sta_list(&stationList);
    
//     if (result != ESP_OK) {
//         Serial.println("Failed to get station list");
//         return false;
//     }
    
//     for (int i = 0; i < stationList.num; i++) {
//         wifi_sta_info_t station = stationList.sta[i];
        
//         Serial.print("Station ");
//         Serial.print(i + 1);
//         Serial.print(" - MAC: ");
        
//         char macStr[18] = {0};
//         sprintf(macStr, "%02X:%02X:%02X:%02X:%02X:%02X",
//                 station.mac[0], station.mac[1], station.mac[2], 
//                 station.mac[3], station.mac[4], station.mac[5]);
//         Serial.print(macStr);
        
//         // Check if this is the Pi (based on MAC address)
//         bool isPi = true;
//         for (int j = 0; j < 6; j++) {
//             if (station.mac[j] != piMacAddress[j]) {
//                 isPi = false;
//                 break;
//             }
//         }
        
//         if (isPi) {
//             piFound = true;
//             Serial.println(" (Raspberry Pi detected!)");
            
//             // Only scan a small range since we're limiting to 1 client
//             // With a single client, the Pi should almost always get 192.168.4.2
//             IPAddress baseIP = WiFi.softAPIP();
//             IPAddress clientIP(baseIP[0], baseIP[1], baseIP[2], 2);
            
//             Serial.printf("Testing likely Pi IP: %s\n", clientIP.toString().c_str());
            
//             // Try to connect to the expected client IP
//             WiFiClient testClient;
//             if (testClient.connect(clientIP, ws_port)) {
//                 Serial.printf("*** Found active device at IP: %s ***\n", clientIP.toString().c_str());
//                 testClient.stop();
                
//                 // Let's try to establish a WebSocket connection
//                 Serial.printf("Attempting WebSocket connection to %s:%d%s\n", 
//                             clientIP.toString().c_str(), ws_port, ws_url);
                
//                 // Store the Pi's actual IP
//                 piActualIP = clientIP;
                
//                 // Try to connect via WebSocket
//                 webSocket.disconnect();
//                 delay(500);
//                 webSocket.begin(clientIP.toString().c_str(), ws_port, ws_url);
//             } else {
//                 // If the likely IP didn't work, try a small range
//                 Serial.println("Could not connect to expected IP, trying a small range...");
//                 for (int ip = 2; ip <= 5; ip++) {
//                     IPAddress testIP(baseIP[0], baseIP[1], baseIP[2], ip);
                    
//                     Serial.printf("Testing IP: %s\n", testIP.toString().c_str());
                    
//                     if (testClient.connect(testIP, ws_port)) {
//                         Serial.printf("*** Found active device at IP: %s ***\n", testIP.toString().c_str());
//                         testClient.stop();
                        
//                         // Store the Pi's actual IP
//                         piActualIP = testIP;
                        
//                         // Try to connect via WebSocket
//                         webSocket.disconnect();
//                         delay(500);
//                         webSocket.begin(testIP.toString().c_str(), ws_port, ws_url);
//                         break;
//                     }
//                 }
//             }
//         } else {
//             Serial.println();
//         }
//     }
    
//     return piFound;
// }

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

// Modify the webSocketEvent function to provide more debugging info
void webSocketEvent(WStype_t type, uint8_t *payload, size_t length)
{
    // Document declaration outside the switch
    StaticJsonDocument<512> doc;

    switch (type)
    {
    case WStype_DISCONNECTED:
        Serial.println("WebSocket Disconnected!");
        Serial.printf("Attempting reconnect to %s:%d%s\n", ws_server, ws_port, ws_url);
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