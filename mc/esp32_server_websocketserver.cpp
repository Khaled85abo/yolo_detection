#include <WiFi.h>
#include <WebSocketsServer.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

WebSocketsServer webSocket = WebSocketsServer(81);

// LED pins for different warnings
const int STOPPED_LED = 25;
const int OVERLAP_LED = 26;
const int INCORRECT_LED = 27;
const int BUZZER_PIN = 32;

void setup() {
    Serial.begin(115200);
    
    // Initialize pins
    pinMode(STOPPED_LED, OUTPUT);
    pinMode(OVERLAP_LED, OUTPUT);
    pinMode(INCORRECT_LED, OUTPUT);
    pinMode(BUZZER_PIN, OUTPUT);
    
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
    
    webSocket.begin();
    webSocket.onEvent(webSocketEvent);
}

void loop() {
    webSocket.loop();
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {
    switch(type) {
        case WStype_DISCONNECTED:
            Serial.printf("[%u] Disconnected!\n", num);
            break;
        case WStype_CONNECTED:
            Serial.printf("[%u] Connected!\n", num);
            break;
        case WStype_TEXT:
            handleMessage(payload, length);
            break;
    }
}

void handleMessage(uint8_t * payload, size_t length) {
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, payload);
    
    if (error) {
        Serial.println("Failed to parse JSON");
        return;
    }
    
    const char* type = doc["type"];
    if (strcmp(type, "warning") == 0) {
        const char* status = doc["status"];
        int level = doc["level"];
        
        // Handle different warnings
        if (strcmp(status, "STOPPED") == 0) {
            digitalWrite(STOPPED_LED, HIGH);
            if (level == 2) { // ERROR level
                tone(BUZZER_PIN, 1000, 500);
            }
        }
        else if (strcmp(status, "OVERLAPPED") == 0) {
            digitalWrite(OVERLAP_LED, HIGH);
            tone(BUZZER_PIN, 2000, 1000);
        }
        else if (strcmp(status, "INCORRECT") == 0) {
            digitalWrite(INCORRECT_LED, HIGH);
        }
    }
}