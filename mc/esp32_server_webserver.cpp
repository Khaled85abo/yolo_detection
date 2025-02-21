#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>
#include <SPIFFS.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

WebServer server(80);

// LED pins for different warnings
const int STOPPED_LED = 25;
const int OVERLAP_LED = 26;
const int INCORRECT_LED = 27;
const int BUZZER_PIN = 32;

// System state
struct SystemState {
    bool stoppedWarning = false;
    bool overlapWarning = false;
    bool incorrectWarning = false;
    String lastWarningMessage = "";
} state;

void setup() {
    Serial.begin(115200);
    
    // Initialize pins
    pinMode(STOPPED_LED, OUTPUT);
    pinMode(OVERLAP_LED, OUTPUT);
    pinMode(INCORRECT_LED, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    
    // Initialize SPIFFS for serving web files
    if(!SPIFFS.begin(true)) {
        Serial.println("SPIFFS Mount Failed");
        return;
    }
    
    // Connect to WiFi
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    
    Serial.println("");
    Serial.println("WiFi connected");
    Serial.println("IP address: ");
    Serial.println(WiFi.localIP());
    
    setupRoutes();
    server.begin();
}

void setupRoutes() {
    // Serve the web interface
    server.on("/", HTTP_GET, []() {
        File file = SPIFFS.open("/index.html", "r");
        server.streamFile(file, "text/html");
        file.close();
    });
    
    // API endpoints
    server.on("/api/status", HTTP_GET, handleGetStatus);
    server.on("/api/warning", HTTP_POST, handleWarning);
    server.on("/api/acknowledge", HTTP_POST, handleAcknowledge);
    server.on("/api/stop-conveyor", HTTP_POST, handleStopConveyor);
}

void handleGetStatus() {
    StaticJsonDocument<200> doc;
    doc["stopped"] = state.stoppedWarning;
    doc["overlap"] = state.overlapWarning;
    doc["incorrect"] = state.incorrectWarning;
    doc["lastWarning"] = state.lastWarningMessage;
    
    String response;
    serializeJson(doc, response);
    server.send(200, "application/json", response);
}

void handleWarning() {
    if (!server.hasArg("plain")) {
        server.send(400, "text/plain", "No data received");
        return;
    }
    
    String message = server.arg("plain");
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, message);
    
    if (error) {
        server.send(400, "text/plain", "Invalid JSON");
        return;
    }
    
    const char* status = doc["status"];
    int level = doc["level"];
    
    updateWarningState(status, level);
    server.send(200, "text/plain", "Warning processed");
}

void handleAcknowledge() {
    if (!server.hasArg("type")) {
        server.send(400, "text/plain", "Warning type not specified");
        return;
    }
    
    String warningType = server.arg("type");
    if (warningType == "stopped") {
        state.stoppedWarning = false;
        digitalWrite(STOPPED_LED, LOW);
    } else if (warningType == "overlap") {
        state.overlapWarning = false;
        digitalWrite(OVERLAP_LED, LOW);
    } else if (warningType == "incorrect") {
        state.incorrectWarning = false;
        digitalWrite(INCORRECT_LED, LOW);
    }
    
    server.send(200, "text/plain", "Warning acknowledged");
}

void handleStopConveyor() {
    // Add conveyor stop logic here
    server.send(200, "text/plain", "Conveyor stopped");
}

void updateWarningState(const char* status, int level) {
    if (strcmp(status, "STOPPED") == 0) {
        state.stoppedWarning = true;
        digitalWrite(STOPPED_LED, HIGH);
        if (level == 2) { // ERROR level
            tone(BUZZER_PIN, 1000, 500);
        }
    }
    else if (strcmp(status, "OVERLAPPED") == 0) {
        state.overlapWarning = true;
        digitalWrite(OVERLAP_LED, HIGH);
        tone(BUZZER_PIN, 2000, 1000);
    }
    else if (strcmp(status, "INCORRECT") == 0) {
        state.incorrectWarning = true;
        digitalWrite(INCORRECT_LED, HIGH);
    }
    
    state.lastWarningMessage = String(status);
}

void loop() {
    server.handleClient();
}