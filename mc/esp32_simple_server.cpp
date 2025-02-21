#include <WiFi.h>
#include <WebServer.h> // The lightweight server library

// Replace with your Wi-Fi network credentials
const char *ssid = "TN-JE3155";
const char *password = "";

// Pin connected to LED (built-in LED on many ESP32 dev boards is GPIO 2)
const int LED_PIN = 2;

// Create a WebServer object that listens on port 80
WebServer server(80);

void setup()
{
    Serial.begin(115200);
    delay(1000);

    // Initialize the LED pin as OUTPUT
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, LOW); // start with LED off

    // Connect to Wi-Fi
    WiFi.mode(WIFI_STA);
    WiFi.begin(ssid, password);
    Serial.print("Connecting to ");
    Serial.println(ssid);

    while (WiFi.status() != WL_CONNECTED)
    {
        delay(500);
        Serial.print(".");
    }

    Serial.println("\nWiFi connected!");
    Serial.print("ESP32 IP address: ");
    Serial.println(WiFi.localIP());

    // Route for turning LED on/off (example: /led?state=on or /led?state=off)
    server.on("/led", []()
              {
    if (server.hasArg("state")) {
      String state = server.arg("state");
      if (state == "on") {
        digitalWrite(LED_PIN, HIGH);
        server.send(200, "text/plain", "LED is now ON");
      } else if (state == "off") {
        digitalWrite(LED_PIN, LOW);
        server.send(200, "text/plain", "LED is now OFF");
      } else {
        server.send(400, "text/plain", "Invalid state. Use 'on' or 'off'.");
      }
    } else {
      server.send(400, "text/plain", "Missing 'state' parameter. Use ?state=on or ?state=off");
    } });

    // Optional: root page
    server.on("/", []()
              { server.send(200, "text/plain", "Hello from ESP32! Use /led?state=on or /led?state=off"); });

    // Start the server
    server.begin();
    Serial.println("HTTP server started");
}

void loop()
{
    // Handle incoming client requests
    server.handleClient();
}
